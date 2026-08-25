#!/usr/bin/env python3
"""Step 2b — sample y1 from the *base* model given GPT verification.

For each unique wrong y0 whose GPT verdict is incorrect, roll the same local
policy in PAG format:

  user(problem) → assistant(y0) → user(VERIFY) → assistant(gpt_verify)
  → user(REGENERATE) → assistant(y1)

Gold is only used as a filter later (construct). No extra GPT cost.

Example:
  python tools/sft_data/collect_y1.py \\
    --verifications datasets/sft_collect/verifications.jsonl \\
    --output datasets/sft_collect/y1.jsonl \\
    --base_url http://127.0.0.1:8081/v1 \\
    --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sft_collect_utils import (  # noqa: E402
    append_jsonl,
    boxed_or_extract,
    completed_pair_keys,
    get_soft_answer_correction,
    load_jsonl,
    parse_verdict,
)

try:
    from verl.utils.pag_prompts import REGENERATE_USER, VERIFY_USER
except Exception:  # noqa: BLE001
    VERIFY_USER = (
        "Verify the previous solution without re-solving the problem from scratch. "
        "Do NOT write a full corrected solution, and do NOT put a final answer in \\boxed{}. "
        "Check only the given steps. "
        "If you find a mistake: in 1-4 sentences, name the wrong step, explain why it is wrong, "
        "and say what should be fixed. End your response with exactly: The answer is wrong. "
        "If all steps are correct: do not propose edits. End your response with exactly: The answer is correct."
    )
    REGENERATE_USER = (
        "You indicated that your previous answer was wrong. "
        "Please provide the correct solution to the math problem."
    )

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
_WRITE_LOCK = threading.Lock()


def _strip_stops(text: str) -> str:
    text = (text or "").strip()
    for stop in ("<|im_end|>", "<|eot_id|>", "</s>"):
        if text.endswith(stop):
            text = text[: -len(stop)].rstrip()
    return text


def iter_y1_jobs(veri_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    seen = set()
    for row in veri_rows:
        uid = str(row.get("unique_id", ""))
        gold = row.get("gold_extracted_answer") or ""
        responses = row.get("round_1_response") or []
        answers = row.get("round_1_extracted_answer") or []
        veris = row.get("verification") or []
        n = min(len(responses), len(answers), len(veris))
        for i in range(n):
            y0 = responses[i]
            pred = answers[i]
            verify = veris[i]
            if not y0 or not verify:
                continue
            verdict, verify_clean = parse_verdict(verify)
            if verdict != "incorrect":
                continue
            if get_soft_answer_correction(gold, pred):
                continue
            key = (uid, str(pred))
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                {
                    "unique_id": uid,
                    "problem": row["problem"],
                    "gold_extracted_answer": gold,
                    "y0": y0,
                    "y0_extracted_answer": pred,
                    "verification": verify_clean,
                }
            )
    return jobs


def generate_y1(
    client: Any,
    model: str,
    job: Dict[str, Any],
    n: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": job["problem"]},
        {"role": "assistant", "content": job["y0"]},
        {"role": "user", "content": VERIFY_USER},
        {"role": "assistant", "content": job["verification"]},
        {"role": "user", "content": REGENERATE_USER},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return [_strip_stops(ch.message.content) for ch in resp.choices]


def process_one(
    job: Dict[str, Any],
    client: Any,
    model: str,
    n: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    output_file: Path,
) -> bool:
    texts = generate_y1(client, model, job, n, temperature, max_tokens, top_p)
    gold = job.get("gold_extracted_answer") or ""
    y1_extracted = [boxed_or_extract(t) for t in texts]
    matches = [bool(gold) and get_soft_answer_correction(gold, a) for a in y1_extracted]
    out = dict(job)
    out["y1_list"] = texts
    out["y1_extracted_list"] = y1_extracted
    out["y1_matches_gold_list"] = matches
    # keep first gold-matching sample as the canonical y1, else first sample
    pick = next((i for i, m in enumerate(matches) if m), 0)
    out["y1"] = texts[pick] if texts else ""
    out["y1_extracted_answer"] = y1_extracted[pick] if y1_extracted else ""
    out["y1_matches_gold"] = bool(matches[pick]) if matches else False
    out["y1_model"] = model
    with _WRITE_LOCK:
        append_jsonl(output_file, out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifications", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--base_url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8081/v1"),
    )
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument(
        "--model",
        default=os.environ.get("SOLUTION_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    ap.add_argument("--n", type=int, default=1, help="y1 samples per unique wrong y0")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_rows", type=int, default=-1, help="cap #y1 jobs")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    jobs = iter_y1_jobs(load_jsonl(args.verifications))
    if args.max_rows > 0:
        jobs = jobs[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pair_keys(args.output)
    jobs = [
        j
        for j in jobs
        if (j["unique_id"], str(j["y0_extracted_answer"])) not in done
    ]
    print(f"y1 jobs remaining: {len(jobs)} (done keys={len(done)})")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                process_one,
                j,
                client,
                args.model,
                args.n,
                args.temperature,
                args.max_tokens,
                args.top_p,
                args.output,
            )
            for j in jobs
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="y1"):
            fut.result()

    rows = load_jsonl(args.output)
    n_ok = sum(1 for r in rows if r.get("y1_matches_gold"))
    print(f"done. {len(rows)} y1 rows, gold-match={n_ok} → {args.output}")


if __name__ == "__main__":
    main()
