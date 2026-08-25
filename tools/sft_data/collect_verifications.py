#!/usr/bin/env python3
"""Step 2 — collect verification / critique via OpenAI-compatible API (GPT).

For each unique extracted answer, send the matching y0 (not just the boxed
number) and ask GPT for the same **freeform** diagnostic as
``tools/controlled_rectify`` ``teacher_freeform``: what went wrong and how to
fix the reasoning, no full solution, no ``\\boxed{}``, PAG closer.

Example:
  export OPENAI_API_KEY=...
  # optional: export OPENAI_BASE_URL=https://...
  python tools/sft_data/collect_verifications.py \\
    --solutions datasets/sft_collect/solutions.jsonl \\
    --output datasets/sft_collect/verifications.jsonl \\
    --model gpt-5.6-terra
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Value
from ctypes import c_int
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_CR = _REPO / "tools" / "controlled_rectify"
for p in (_HERE, _REPO, _CR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pag_verify_utils import (  # noqa: E402
    CORRECT_CLOSE,
    WRONG_CLOSE,
    ensure_wrong_close,
    parse_verdict as pag_tail_verdict,
    strip_verdict_close,
)
from sft_collect_utils import get_soft_answer_correction  # noqa: E402
from structured_feedback import FREEFORM_TEACHER_SYSTEM, FREEFORM_TEACHER_USER  # noqa: E402

_WRITE_LOCK = threading.Lock()
MAX_Y0_CHARS = 3500

# Gold-matching y0s still need a closer; allow the polar-style case
# (numeric answer ok, reasoning wrong → The answer is wrong.).
CORRECT_FREEFORM_SYSTEM = (
    "You are a generative math verifier. Check the given solution's reasoning, "
    "not by re-solving from scratch. "
    "If you find a mistake: diagnose what went wrong and how to fix the reasoning. "
    "Do NOT provide a full worked solution. Do NOT use \\boxed{}. "
    "End with exactly: The answer is wrong. "
    "If all steps are correct: do not propose edits. End with exactly: The answer is correct."
)
CORRECT_FREEFORM_USER = (
    "Problem:\n{problem}\n\n"
    "Student solution:\n{solution}\n\n"
    "Provide diagnostic feedback on the given steps (why/how to fix if wrong), "
    "without solving the problem."
)


def load_completed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["unique_id"])
            except Exception:  # noqa: BLE001
                continue
    return ids


def load_reference(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    if path is None or not path.exists():
        return {}
    ref: Dict[Tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            answers = data.get("round_1_extracted_answer") or []
            veris = data.get("verification") or []
            for ans, veri in zip(answers, veris):
                if veri:
                    ref[(data["problem"], ans)] = veri
    return ref


def _truncate(text: str, n: int = MAX_Y0_CHARS) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n] + "\n[... truncated ...]"


def finalize_freeform(text: str, *, force_wrong: bool) -> str:
    """Strip boxed leaks and pin a PAG closer, matching generate_structured_feedback."""
    text = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", text or "")
    text = text.strip()
    if force_wrong:
        return ensure_wrong_close(text)
    tail = pag_tail_verdict(text)
    body = strip_verdict_close(text)
    if tail == "wrong":
        return (body + "\n" + WRONG_CLOSE) if body else WRONG_CLOSE
    if tail == "correct":
        return (body + "\n" + CORRECT_CLOSE) if body else CORRECT_CLOSE
    if not body:
        return CORRECT_CLOSE
    return body + "\n" + CORRECT_CLOSE


def call_api(
    client: Any,
    model: str,
    user: str,
    temperature: float,
    max_retries: int = 50,
    system: Optional[str] = None,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "n": 1,
            }
            # Some GPT-5 / o-series reject temperature / want max_completion_tokens
            m = model.lower()
            if m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5"):
                kwargs.pop("temperature", None)
                kwargs["max_completion_tokens"] = 1024
            else:
                kwargs["max_tokens"] = 1024
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"[warn] API failed after retries: {e}")
                return None
            time.sleep(min(2 ** min(attempt, 5), 30))
    return None


def process_one(
    problem: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    reference: Dict[Tuple[str, str], str],
    output_file: Path,
    gpt_counter: Any,
    reuse_counter: Any,
) -> bool:
    all_answers: List[str] = list(problem.get("round_1_extracted_answer") or [])
    all_responses: List[str] = list(problem.get("round_1_response") or [])
    if not all_answers or len(all_answers) != len(all_responses):
        return False

    answer_indices: Dict[str, List[int]] = {}
    for i, ans in enumerate(all_answers):
        answer_indices.setdefault(ans, []).append(i)

    unique_answers: List[str] = []
    gold = problem.get("gold_extracted_answer") or ""
    if gold in answer_indices:
        unique_answers.append(gold)
    for ans in answer_indices:
        if ans not in unique_answers and len(unique_answers) < 5:
            unique_answers.append(ans)

    # One y0 per unique extracted answer so the critique names *this* solution's steps.
    kept_idx = [answer_indices[ans][0] for ans in unique_answers]
    kept_answers = [all_answers[i] for i in kept_idx]
    kept_responses = [all_responses[i] for i in kept_idx]

    verification: List[str] = []
    for pred_answer, y0 in zip(kept_answers, kept_responses):
        key = (problem["problem"], pred_answer)
        if key in reference:
            verification.append(reference[key])
            with reuse_counter.get_lock():
                reuse_counter.value += 1
            continue
        gold_ok = bool(gold) and get_soft_answer_correction(gold, pred_answer)
        y0_trunc = _truncate(y0)
        if gold_ok:
            system = CORRECT_FREEFORM_SYSTEM
            user = CORRECT_FREEFORM_USER.replace("{problem}", problem["problem"]).replace(
                "{solution}", y0_trunc
            )
        else:
            system = FREEFORM_TEACHER_SYSTEM
            user = FREEFORM_TEACHER_USER.replace("{problem}", problem["problem"]).replace(
                "{wrong_attempt}", y0_trunc
            )
        raw = call_api(client, model, user, temperature, system=system)
        if raw is None:
            return False
        with gpt_counter.get_lock():
            gpt_counter.value += 1
        verification.append(finalize_freeform(raw, force_wrong=not gold_ok))

    out = dict(problem)
    out["round_1_extracted_answer"] = kept_answers
    out["round_1_response"] = kept_responses
    out["verification"] = verification
    out["verify_model"] = model

    with _WRITE_LOCK:
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--model",
        default=os.environ.get("GPT_VERIFY_MODEL")
        or os.environ.get("GPT_CRITIQUE_MODEL", "gpt-5.6-terra"),
    )
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL") or None)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="optional prior verifications.jsonl to reuse (problem, answer) keys",
    )
    ap.add_argument("--max_rows", type=int, default=-1)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY / --api_key required for verification")

    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    problems: List[Dict[str, Any]] = []
    with args.solutions.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if args.max_rows > 0:
        problems = problems[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed_ids(args.output)
    problems = [p for p in problems if str(p["unique_id"]) not in done]
    reference = load_reference(args.reference)
    print(f"remaining: {len(problems)}; reference keys: {len(reference)}")

    gpt_counter = Value(c_int, 0)
    reuse_counter = Value(c_int, 0)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                process_one,
                p.copy(),
                client,
                args.model,
                args.temperature,
                reference,
                args.output,
                gpt_counter,
                reuse_counter,
            )
            for p in problems
        ]
        pbar = tqdm(as_completed(futs), total=len(futs), desc="verifications")
        for fut in pbar:
            ok = fut.result()
            if ok:
                pbar.set_description(
                    f"API={gpt_counter.value} reuse={reuse_counter.value}"
                )

    print(
        f"done. API calls={gpt_counter.value}, reuse={reuse_counter.value} → {args.output}"
    )


if __name__ == "__main__":
    main()
