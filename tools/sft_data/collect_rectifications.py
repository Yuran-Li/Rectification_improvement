#!/usr/bin/env python3
"""Step 2b — collect a corrected solution (rectify) after GPT verify.

Official S2R never sampled rectify: it reused another *correct first-round y0*
from the same problem. That drops items the local model never solved, and the
retry is not conditioned on the critique.

This step asks the same teacher API to write a full corrected solution given
(problem, wrong y0, verification). Gold is used only as a *filter*, not in the
prompt.

Example:
  export OPENAI_API_KEY=...
  python tools/sft_data/collect_rectifications.py \\
    --verifications datasets/sft_collect/verifications.jsonl \\
    --output datasets/sft_collect/rectifications.jsonl \\
    --model gpt-5.6-terra
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import c_int
from multiprocessing import Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sft_collect_utils import (  # noqa: E402
    append_jsonl,
    boxed_or_extract,
    call_chat,
    completed_pair_keys,
    get_soft_answer_correction,
    load_jsonl,
    parse_verdict,
)

_WRITE_LOCK = threading.Lock()
MAX_Y0_CHARS = 3500

RECTIFY_PROMPT = """You are a math teacher. The student solution below is incorrect.
Using the verification notes, write a complete corrected solution from scratch.
Put the final answer in \\boxed{{}}.
Do not mention that you are a teacher. Do not quote the student's boxed answer as if it were correct.

* Problem:
{problem}

* Student solution:
{y0}

* Verification:
{verify}

* Corrected solution:
"""


def iter_rectify_jobs(veri_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One job per unique (uid, wrong extracted answer) with TP rejection."""
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
                # false reject — do not collect a "correction"
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


def process_one(
    job: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    output_file: Path,
    gpt_counter: Any,
) -> bool:
    y0 = job["y0"]
    if len(y0) > MAX_Y0_CHARS:
        y0 = y0[:MAX_Y0_CHARS] + "\n...[truncated]"
    prompt = RECTIFY_PROMPT.format(
        problem=job["problem"],
        y0=y0,
        verify=job["verification"],
    )
    text = call_chat(
        client, model, prompt, temperature=temperature, max_tokens=max_tokens
    )
    if text is None:
        return False
    with gpt_counter.get_lock():
        gpt_counter.value += 1

    pred = boxed_or_extract(text)
    gold = job.get("gold_extracted_answer") or ""
    out = dict(job)
    out["rectify"] = text
    out["rectify_extracted_answer"] = pred
    out["rectify_matches_gold"] = bool(gold) and get_soft_answer_correction(gold, pred)
    out["rectify_model"] = model
    with _WRITE_LOCK:
        append_jsonl(output_file, out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifications", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--model",
        default=os.environ.get("GPT_RECTIFY_MODEL")
        or os.environ.get("GPT_VERIFY_MODEL")
        or os.environ.get("GPT_CRITIQUE_MODEL", "gpt-5.6-terra"),
    )
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL") or None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_rows", type=int, default=-1, help="cap #rectify jobs")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY / --api_key required for rectify")

    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    veri_rows = load_jsonl(args.verifications)
    jobs = iter_rectify_jobs(veri_rows)
    if args.max_rows > 0:
        jobs = jobs[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pair_keys(args.output)
    jobs = [j for j in jobs if (j["unique_id"], str(j["y0_extracted_answer"])) not in done]
    print(f"rectify jobs remaining: {len(jobs)} (done keys={len(done)})")

    gpt_counter = Value(c_int, 0)
    ok_gold = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                process_one,
                j,
                client,
                args.model,
                args.temperature,
                args.max_tokens,
                args.output,
                gpt_counter,
            )
            for j in jobs
        ]
        pbar = tqdm(as_completed(futs), total=len(futs), desc="rectifications")
        for fut in pbar:
            if fut.result():
                pbar.set_description(f"API={gpt_counter.value}")

    # recount gold hits on full file
    n_match = sum(1 for r in load_jsonl(args.output) if r.get("rectify_matches_gold"))
    n_all = len(load_jsonl(args.output))
    print(f"done. wrote {n_all} rectify rows, gold-match={n_match} → {args.output}")


if __name__ == "__main__":
    main()
