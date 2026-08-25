#!/usr/bin/env python3
"""Controlled one-round rectification eval for PAG Pre-RL / PPO.

Prompt format matches PAG RL rollout (vllm_pag_rollout_spmd.py)::

  system: Please reason step by step, and put your final answer within \\boxed{}.
  user:   {problem}
  assistant: {y0}
  user:   Check the math solution step-by-step. ... end with 'The answer is wrong/correct'.
  assistant: {generic | fixed critique}   # forced to end with ``The answer is wrong.``
  user:   You indicated that your previous answer was wrong. Please provide the correct solution...
  assistant: <model generates revised solution only>

Conditions (same fixed initially-wrong set for all checkpoints):
  ECR_gen   : generic verify body (no problem-specific critique)
  ECR_fix : GPT / teacher fixed critique as the verify turn
  Acc_regen : problem only (no y0 / verify / regenerate)

@1 : temperature=0, n=1
@8 : temperature>0, n=8, pass@8 = any of 8 correct

One revise round only (no second verify).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Grade with S2R MATH parser (shared scoring; prompts are PAG-native)
EVAL_DIR = Path("/data/yuranli/LLM/2026.04/github_references/S2R/tools/qwen_eval/eval")
CR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(CR_DIR))

from grader import math_equal  # noqa: E402
from parser import extract_answer, strip_string  # noqa: E402
from pag_verify_utils import GENERIC_VERIFY_ASSISTANT, ensure_wrong_close  # noqa: E402

MAX_WRONG_CHARS = 3500

# Same system string as PAG math parquet / Instruct
SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Exact strings from verl/workers/rollout/vllm_rollout/vllm_pag_rollout_spmd.py
VERIFY_USER = (
    "Verify the previous solution without re-solving the problem from scratch. "
    "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
    "explain why it is wrong, and end your response with 'The answer is wrong'. "
    "If all steps are correct, end your response with 'The answer is correct'."
)

REGENERATE_USER = (
    "You indicated that your previous answer was wrong. "
    "Please provide the correct solution to the math problem."
)


def truncate(text: str, n: int = MAX_WRONG_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "\n[... truncated ...]"


def grade(text: str, gt: str, data_name: str = "math") -> bool:
    if not text or not text.strip():
        return False
    pred = extract_answer(text, data_name)
    pred_s = strip_string(pred, skip_unit=False) if pred is not None else ""
    if not str(pred_s).strip():
        return False
    try:
        return bool(math_equal(pred_s, gt))
    except Exception:
        return str(pred_s).strip() == str(gt).strip()


def build_regen_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def build_pag_revise_prompt(
    tokenizer, problem: str, y0: str, verify_assistant: str
) -> str:
    """Full PAG multi-turn chat up to the regenerate user turn; open assistant."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": y0.strip()},
        {"role": "user", "content": VERIFY_USER},
        {"role": "assistant", "content": ensure_wrong_close(verify_assistant)},
        {"role": "user", "content": REGENERATE_USER},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def build_prompt(tokenizer, cond: str, row: dict) -> str:
    problem = row["problem"]
    y0 = truncate(row.get("y0") or row.get("wrong_attempt") or "")
    if cond == "regen":
        return build_regen_prompt(tokenizer, problem)
    if cond == "gen":
        return build_pag_revise_prompt(tokenizer, problem, y0, GENERIC_VERIFY_ASSISTANT)
    if cond == "fix":
        critique = (row.get("fixed_critique") or "").strip()
        if not critique:
            raise ValueError(f"missing fixed_critique for idx={row.get('idx')}")
        return build_pag_revise_prompt(tokenizer, problem, y0, critique)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["gen", "fix", "regen"],
        choices=["gen", "fix", "regen"],
    )
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--data_name", default="math")
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    if args.n_samples > 1 and args.temperature <= 0:
        print("Warning: n_samples>1 with temperature<=0 yields identical samples.")

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if "fix" in args.conditions:
        missing = [r.get("idx") for r in records if not (r.get("fixed_critique") or "").strip()]
        if missing:
            raise SystemExit(
                f"missing fixed_critique for idx={missing[:10]}... (n={len(missing)})"
            )
    print(f"Loaded {len(records)} examples; conditions={args.conditions}")
    print("Prompt style: PAG multi-turn (verify → regenerate)")

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
        top_p=1.0 if args.temperature <= 0 else 0.95,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    out_rows = [
        {**{k: r.get(k) for k in ("idx", "problem", "gt", "y0_pred")}, "results": {}}
        for r in records
    ]

    for cond in args.conditions:
        print(f"=== condition={cond} ===")
        prompts = [build_prompt(tokenizer, cond, r) for r in records]
        outs = llm.generate(prompts, sampling, use_tqdm=True)
        for i, out in enumerate(outs):
            texts = [o.text for o in out.outputs]
            corrects = [grade(t, records[i]["gt"], args.data_name) for t in texts]
            acc1 = float(corrects[0]) if corrects else 0.0
            pass_k = float(any(corrects)) if corrects else 0.0
            out_rows[i]["results"][cond] = {
                "texts": texts,
                "corrects": corrects,
                "acc@1": acc1,
                f"pass@{args.n_samples}": pass_k,
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_path": args.model_path,
        "data": args.data,
        "prompt_style": "pag_multiturn_verify_regenerate",
        "n": len(records),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "metrics": {},
    }
    for cond in args.conditions:
        acc1 = sum(r["results"][cond]["acc@1"] for r in out_rows) / len(out_rows)
        pk = sum(r["results"][cond][f"pass@{args.n_samples}"] for r in out_rows) / len(
            out_rows
        )
        summary["metrics"][cond] = {
            "acc@1_pct": round(100 * acc1, 2),
            f"pass@{args.n_samples}_pct": round(100 * pk, 2),
        }
        print(f"{cond}: acc@1={100*acc1:.2f}%  pass@{args.n_samples}={100*pk:.2f}%")

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
