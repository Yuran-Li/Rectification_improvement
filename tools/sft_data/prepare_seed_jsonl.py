#!/usr/bin/env python3
"""Export PAG parquet / jsonl into the seed format used by collect_solutions.py.

Seed schema (one jsonl line):
  problem, solution (optional), unique_id, gold_extracted_answer (optional),
  subject/level (optional)

Examples:
  python tools/sft_data/prepare_seed_jsonl.py \\
    --input datasets/math7500.parquet \\
    --output datasets/sft_collect/seed_math7500.jsonl

  python tools/sft_data/prepare_seed_jsonl.py \\
    --input datasets/math500.parquet \\
    --output datasets/sft_collect/seed_math500.jsonl \\
    --max_rows 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _problem_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    # verl parquet: list/ndarray of {role, content}
    msgs = list(prompt)
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    # fallback: last content
    if msgs:
        last = msgs[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
        return str(last).strip()
    return ""


def row_to_seed(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    rm = row.get("reward_model") or {}
    if hasattr(rm, "item"):
        rm = rm.item()
    if not isinstance(rm, dict):
        rm = {}
    extra = row.get("extra_info") or {}
    if hasattr(extra, "item"):
        extra = extra.item()
    if not isinstance(extra, dict):
        extra = {}

    problem = row.get("problem") or _problem_from_prompt(row.get("prompt"))
    gold = (
        row.get("gold_extracted_answer")
        or row.get("answer")
        or rm.get("ground_truth")
        or ""
    )
    unique_id = (
        row.get("unique_id")
        or extra.get("index")
        or row.get("idx")
        or idx
    )
    return {
        "problem": str(problem).strip(),
        "solution": str(row.get("solution") or "").strip(),
        "unique_id": str(unique_id),
        "gold_extracted_answer": str(gold).strip() if gold is not None else "",
        "subject": row.get("type") or row.get("subject"),
        "level": row.get("level"),
        "data_source": row.get("data_source"),
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return [df.iloc[i].to_dict() for i in range(len(df))]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON must be a list: {path}")
    # jsonl
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max_rows", type=int, default=-1)
    args = ap.parse_args()

    raw = load_rows(args.input)
    if args.max_rows > 0:
        raw = raw[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(raw):
            seed = row_to_seed(row, i)
            if not seed["problem"]:
                continue
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} seeds → {args.output}")


if __name__ == "__main__":
    main()
