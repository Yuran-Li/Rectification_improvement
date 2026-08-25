#!/usr/bin/env python3
"""Build oracle initially-wrong pool from Pre-RL (Instruct) on MATH-500.

1) Generate one y0 per problem with the Pre-RL model (vLLM).
2) Grade y0 with GT via S2R extract_answer + math_equal (oracle a1).
3) Keep only a1=False with parseable answers.

No model self-verification is used.

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1 python build_prerl_pool.py \\
    --model_path /path/to/Qwen2.5-1.5B-Instruct \\
    --parquet ../../datasets/math500.parquet \\
    --out data/fixed_wrong_pag_prerl.jsonl \\
    --tensor_parallel_size 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse S2R MATH grader for consistency with controlled_rectify tables
S2R_EVAL = Path("/data/yuranli/LLM/2026.04/github_references/S2R/tools/qwen_eval/eval")
sys.path.insert(0, str(S2R_EVAL))
from grader import math_equal  # noqa: E402
from parser import extract_answer, strip_string  # noqa: E402

SYSTEM_FREE = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "Separate each major reasoning step with a blank line."
)
SYSTEM_STRIDE = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "You MUST wrap EACH distinct logical step in its own <step>...</step> tags. "
    "Example:\n"
    "<step>\nFirst observation or calculation.\n</step>\n"
    "<step>\nNext deduction.\n</step>\n"
    "Final answer: \\boxed{...}"
)


def extract_problem(prompt) -> str:
    """PAG parquet prompt is list[{role, content}, ...] with system+user."""
    if isinstance(prompt, str):
        return prompt
    msgs = list(prompt)
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    if users:
        return users[-1]
    return msgs[-1]["content"] if msgs else ""


def grade_y0(y0: str, gt: str, data_name: str = "math"):
    if not y0 or not str(y0).strip():
        return None, "", "empty_text"
    pred = extract_answer(y0, data_name)
    pred_s = strip_string(pred, skip_unit=False) if pred is not None else ""
    if pred is None or str(pred_s).strip() == "":
        return None, str(pred or ""), "unparseable"
    try:
        ok = bool(math_equal(pred_s, gt))
    except Exception:
        ok = str(pred_s).strip() == str(gt).strip()
    return ok, pred_s, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--data_name", default="math")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument(
        "--gen_format",
        default="free",
        choices=["free", "stride_tags"],
        help="free: blank-line steps (SCOPE); stride_tags: <step> blocks (STRIDE)",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.max_samples > 0:
        df = df.iloc[: args.max_samples]
    problems = []
    for i, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"]
        problems.append(
            {
                "idx": int(row["extra_info"].get("index", i)),
                "unique_id": row.get("unique_id"),
                "problem": extract_problem(row["prompt"]),
                "gt": gt,
                "answer": gt,
                "level": int(row["level"]) if "level" in row else None,
            }
        )
    print(f"Loaded {len(problems)} MATH problems from {args.parquet}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    system = SYSTEM_STRIDE if args.gen_format == "stride_tags" else SYSTEM_FREE
    prompts = []
    for p in problems:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": p["problem"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        )

    print(f"Generating y0 gen_format={args.gen_format} ...")
    outs = llm.generate(prompts, sampling, use_tqdm=True)

    kept, excluded = [], []
    for p, out in zip(problems, outs):
        y0 = out.outputs[0].text
        ok, pred, status = grade_y0(y0, p["gt"], args.data_name)
        base = {
            **p,
            "y0": y0,
            "wrong_attempt": y0,
            "y0_pred": pred,
            "grade_status": status,
            "source_model": args.model_path,
            "gen_format": args.gen_format,
        }
        if status != "ok":
            excluded.append({**base, "a1_correct": None})
            continue
        if ok:
            continue
        kept.append({**base, "a1_correct": False})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    excl = out.with_name(out.stem + "_excluded.jsonl")
    with open(excl, "w") as f:
        for r in excluded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"scanned={len(problems)}")
    print(f"oracle_a1_wrong={len(kept)} -> {out}")
    print(f"excluded_unparseable={len(excluded)} -> {excl}")


if __name__ == "__main__":
    main()
