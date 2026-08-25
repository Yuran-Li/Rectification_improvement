#!/usr/bin/env python3
"""Step 1 — sample first-round solutions from a local (or OpenAI-compatible) LLM.

Migrated from S2R ``tools/1_collect_data_from_llm.py``.

Default target is a vLLM OpenAI-compatible server:
  python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2.5-1.5B-Instruct --port 8081

Example:
  python tools/sft_data/collect_solutions.py \\
    --seed_file datasets/sft_collect/seed_math7500.jsonl \\
    --output datasets/sft_collect/solutions.jsonl \\
    --base_url http://127.0.0.1:8081/v1 \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from answer_extraction import extract_answer  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
_WRITE_LOCK = threading.Lock()


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


def chat_generate(
    client: Any,
    model: str,
    problem: str,
    n: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
    ]
    # OpenAI Chat Completions: n>1 returns multiple choices
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    outs: List[str] = []
    for ch in resp.choices:
        text = (ch.message.content or "").strip()
        for stop in ("<|im_end|>", "<|eot_id|>", "</s>"):
            if text.endswith(stop):
                text = text[: -len(stop)].rstrip()
        outs.append(text)
    return outs


def process_one(
    row: Dict[str, Any],
    client: Any,
    model: str,
    n: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    output_file: Path,
) -> bool:
    problem = row["problem"]
    responses = chat_generate(
        client, model, problem, n, temperature, max_tokens, top_p
    )
    extracted = [extract_answer(r) for r in responses]
    gold = row.get("gold_extracted_answer") or ""
    if not gold and row.get("solution"):
        gold = extract_answer(row["solution"])

    out = {
        "problem": problem,
        "round_1_instruction": problem,
        "round_1_response": responses,
        "round_1_extracted_answer": extracted,
        "gold_extracted_answer": gold,
        "solution": row.get("solution") or "",
        "unique_id": str(row["unique_id"]),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "data_source": row.get("data_source"),
        "model": model,
    }
    with _WRITE_LOCK:
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--base_url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8081/v1"),
    )
    ap.add_argument(
        "--api_key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="vLLM usually ignores this; set EMPTY",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("SOLUTION_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    ap.add_argument("--n", type=int, default=5, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_rows", type=int, default=-1)
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    seeds: List[Dict[str, Any]] = []
    with args.seed_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    if args.max_rows > 0:
        seeds = seeds[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed_ids(args.output)
    seeds = [s for s in seeds if str(s["unique_id"]) not in done]
    print(f"remaining: {len(seeds)} (already done filtered)")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                process_one,
                row,
                client,
                args.model,
                args.n,
                args.temperature,
                args.max_tokens,
                args.top_p,
                args.output,
            )
            for row in seeds
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="solutions"):
            fut.result()

    print(f"wrote → {args.output}")


if __name__ == "__main__":
    main()
