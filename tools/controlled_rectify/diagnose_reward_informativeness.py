#!/usr/bin/env python3
"""Diagnose reward informativeness in PAG controlled-rectify setting.

Given fixed wrong (problem, y0) pool and M sampled verify texts per row:
  - specific: each verify_j -> sample K regenerate outcomes
  - generic: fixed generic verify -> sample K regenerate outcomes
  - none: minimal wrong-only verify -> sample K regenerate outcomes
  - independent: ignore y0/verify, solve from scratch -> sample K outcomes

Writes raw boolean outcomes for aggregation:
  results.specific.matrix: M x K bool
  results.{generic,none,independent}.samples: K bool
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path("/data/yuranli/LLM/2026.04/github_references/S2R/tools/qwen_eval/eval")
CR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(CR_DIR))

from grader import math_equal  # noqa: E402
from parser import extract_answer, strip_string  # noqa: E402
from pag_verify_utils import (
    GENERIC_VERIFY_ASSISTANT,
    ensure_wrong_close,
    parse_verdict,
)  # noqa: E402

MAX_WRONG_CHARS = 3500
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
REGENERATE_USER = (
    "You indicated that your previous answer was wrong. "
    "Please provide the correct solution to the math problem."
)
NONE_VERIFY_ASSISTANT = "The answer is wrong."


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


def build_pag_revise_prompt(tokenizer, problem: str, y0: str, verify_assistant: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": truncate(y0)},
        {
            "role": "user",
            "content": (
                "Verify the previous solution without re-solving the problem from scratch. "
                "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
                "explain why it is wrong, and end your response with 'The answer is wrong'. "
                "If all steps are correct, end your response with 'The answer is correct'."
            ),
        },
        {"role": "assistant", "content": ensure_wrong_close(verify_assistant)},
        {"role": "user", "content": REGENERATE_USER},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_independent_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="frozen rectifier model path")
    ap.add_argument("--pool", required=True, help="fixed wrong pool jsonl")
    ap.add_argument("--verifies", required=True, help="sample_verifies_multi.py output")
    ap.add_argument("--output", required=True)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--data_name", default="math")
    ap.add_argument(
        "--verify_source",
        default="raw_gated",
        choices=["raw_gated", "raw_forced_wrong", "for_regen"],
        help=(
            "Which verify text variant drives specific-condition regenerate.\n"
            "raw_gated: use verifies_raw and only regenerate when raw verdict is 'wrong'.\n"
            "raw_forced_wrong: use verifies_raw but force wrong close for all samples.\n"
            "for_regen: use verifies_for_regen/verifies_specific (legacy behavior)."
        ),
    )
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    pool_rows = [json.loads(l) for l in open(args.pool) if l.strip()]
    verify_rows = [json.loads(l) for l in open(args.verifies) if l.strip()]
    verify_by_idx = {r["idx"]: r for r in verify_rows}

    rows = []
    total_specific = 0
    active_specific = 0
    for r in pool_rows:
        v = verify_by_idx.get(r["idx"])
        if not v:
            continue
        raw_list = (v.get("verifies_raw") or [])[: args.M]
        regen_list = (v.get("verifies_for_regen") or v.get("verifies_specific") or [])[: args.M]
        if args.verify_source in ("raw_gated", "raw_forced_wrong"):
            base = raw_list
        else:
            base = regen_list
        if len(base) < args.M:
            continue
        raw_verdicts = (v.get("verifies_verdict_raw") or [])[: args.M]
        if len(raw_verdicts) < args.M:
            raw_verdicts = [parse_verdict(t) for t in raw_list[: args.M]]
        verifies_for_regen = [ensure_wrong_close(x) for x in base]
        if args.verify_source == "raw_gated":
            specific_active_mask = [rv == "wrong" for rv in raw_verdicts]
        else:
            specific_active_mask = [True] * args.M
        total_specific += args.M
        active_specific += sum(1 for x in specific_active_mask if x)
        rows.append(
            {
                **r,
                "verifies_for_regen": verifies_for_regen,
                "verifies_verdict_raw": raw_verdicts,
                "specific_active_mask": specific_active_mask,
            }
        )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    active_pct = 100.0 * active_specific / total_specific if total_specific else 0.0
    print(
        f"Loaded {len(rows)} rows M={args.M}; K={args.K}; "
        f"verify_source={args.verify_source}; "
        f"specific active={active_specific}/{total_specific} ({active_pct:.1f}%)"
    )

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
    stop_token_ids = []
    for tid in (
        tokenizer.eos_token_id,
        getattr(tokenizer, "pad_token_id", None),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ):
        if tid is not None and isinstance(tid, int) and tid >= 0 and tid not in stop_token_ids:
            stop_token_ids.append(tid)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0 if args.temperature <= 0 else 0.95,
        top_k=args.top_k if args.temperature > 0 else -1,
        max_tokens=args.max_new_tokens,
        n=args.K,
        repetition_penalty=args.repetition_penalty,
        stop=["<|im_end|>", "<|endoftext|>"],
        stop_token_ids=stop_token_ids or None,
    )

    results = {r["idx"]: {} for r in rows}

    # specific: M critiques x K rectifications
    flat_prompts = []
    flat_index = []
    matrices = [[[False] * args.K for _ in range(args.M)] for _ in rows]
    for ri, r in enumerate(rows):
        y0 = r.get("y0") or r.get("wrong_attempt") or ""
        for j, verify_text in enumerate(r["verifies_for_regen"]):
            if not r["specific_active_mask"][j]:
                continue
            flat_prompts.append(build_pag_revise_prompt(tokenizer, r["problem"], y0, verify_text))
            flat_index.append((ri, j))
    if flat_prompts:
        outs = llm.generate(flat_prompts, sampling, use_tqdm=True)
        for (ri, j), out in zip(flat_index, outs):
            texts = [o.text for o in out.outputs]
            corrects = [grade(t, rows[ri]["gt"], args.data_name) for t in texts]
            matrices[ri][j] = corrects
    for ri, r in enumerate(rows):
        results[r["idx"]]["specific"] = {
            "matrix": matrices[ri],
            "active_mask": r["specific_active_mask"],
        }

    # generic / none
    for cond, verify_text in (("generic", GENERIC_VERIFY_ASSISTANT), ("none", NONE_VERIFY_ASSISTANT)):
        prompts = []
        for r in rows:
            y0 = r.get("y0") or r.get("wrong_attempt") or ""
            prompts.append(build_pag_revise_prompt(tokenizer, r["problem"], y0, verify_text))
        outs = llm.generate(prompts, sampling, use_tqdm=True)
        for r, out in zip(rows, outs):
            texts = [o.text for o in out.outputs]
            corrects = [grade(t, r["gt"], args.data_name) for t in texts]
            results[r["idx"]][cond] = {"samples": corrects}

    # independent re-solving
    prompts = [build_independent_prompt(tokenizer, r["problem"]) for r in rows]
    outs = llm.generate(prompts, sampling, use_tqdm=True)
    for r, out in zip(rows, outs):
        texts = [o.text for o in out.outputs]
        corrects = [grade(t, r["gt"], args.data_name) for t in texts]
        results[r["idx"]]["independent"] = {"samples": corrects}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "idx": r["idx"],
                        "unique_id": r.get("unique_id"),
                        "M": args.M,
                        "K": args.K,
                        "verify_source": args.verify_source,
                        "verifies_verdict_raw": r.get("verifies_verdict_raw", []),
                        "results": results[r["idx"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
