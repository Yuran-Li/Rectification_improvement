#!/usr/bin/env python3
"""
SEC quadrant preliminary sanity check.

For N_SAMPLE random training prompts, run K=8 rollouts with the base model
to estimate per-prompt:
  g_i = #{y0=C} / K
  c_i = (#{y0=W, y1=C} + 1) / (#{y0=W} + 2)   [Laplace-smoothed]

Then classify each prompt into one of four quadrants:
  HH: g>=0.5, c>=0.5   HL: g>=0.5, c<0.5
  LH: g<0.5,  c>=0.5   LL: g<0.5,  c<0.5

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/sec_quadrant_check.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --n_sample 800 --K 8 --seed 42 \
      --output tools/sec_quadrant_check_1p5b.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verl.utils.pag_prompts import GENERIC_CRITIQUE, REGENERATE_USER, VERIFY_USER
from verl.utils.reward_score.math import compute_score


def grade(text: str, gt: str) -> bool:
    return compute_score(text, gt) >= 0.5


def build_turn1_prompt(messages: list[dict]) -> list[dict]:
    """System + user only (no assistant)."""
    return [m for m in messages if m["role"] != "assistant"]


def build_verify_prompt(messages: list[dict], y0: str) -> list[dict]:
    """Turn 1 answer appended, then verify user turn."""
    base = build_turn1_prompt(messages)
    return base + [
        {"role": "assistant", "content": y0},
        {"role": "user", "content": VERIFY_USER},
    ]


def build_regen_prompt(messages: list[dict], y0: str, verify_text: str) -> list[dict]:
    """Full 3-turn: problem / y0 / verify / regenerate user."""
    base = build_turn1_prompt(messages)
    return base + [
        {"role": "assistant", "content": y0},
        {"role": "user", "content": VERIFY_USER},
        {"role": "assistant", "content": verify_text},
        {"role": "user", "content": REGENERATE_USER},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="datasets/math7500.parquet")
    parser.add_argument("--n_sample", type=int, default=800)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g_thresh", type=float, default=0.5)
    parser.add_argument("--c_thresh", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_util", type=float, default=0.65)
    parser.add_argument("--output", default="tools/sec_quadrant_check.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── load data ──────────────────────────────────────────────────────────
    data_path = ROOT / args.data
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} training prompts from {data_path.name}")

    idxs = random.sample(range(len(df)), min(args.n_sample, len(df)))
    rows = [df.iloc[i] for i in idxs]
    print(f"Sampled {len(rows)} prompts (seed={args.seed})")

    # ── vLLM engine ────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=64,
        max_model_len=4096,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    def chat_to_text(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── Turn 1: K initial rollouts per prompt ──────────────────────────────
    print(f"\n=== Turn 1: {len(rows)} prompts × K={args.K} rollouts ===")
    t1_prompts = []
    for row in rows:
        msgs = list(row["prompt"])
        t1_prompts.append(chat_to_text(build_turn1_prompt(msgs)))

    sp1 = SamplingParams(n=args.K, temperature=0.7, max_tokens=args.max_tokens)
    t1_outputs = llm.generate(t1_prompts, sp1)

    # grade turn-1 answers
    gt_list = [row["reward_model"]["ground_truth"] for row in rows]
    t1_corrects: list[list[bool]] = []
    for out, gt in zip(t1_outputs, gt_list):
        t1_corrects.append([grade(o.text, gt) for o in out.outputs])

    # ── Turn 2: verify + regen only for wrong y0 samples ──────────────────
    # For each (prompt_idx, sample_idx) where y0=W, we need a verify text.
    # Use GENERIC_CRITIQUE as the fixed verify text (same as PAG generic mode).
    # Then build regen prompts and generate y1.
    print(f"\n=== Turn 2: verify+regen for wrong y0 samples ===")
    regen_index: list[tuple[int, int]] = []  # (prompt_idx, sample_idx)
    regen_prompts: list[str] = []

    for pi, (row, corrects) in enumerate(zip(rows, t1_corrects)):
        msgs = list(row["prompt"])
        for ki, (is_c, out_obj) in enumerate(zip(corrects, t1_outputs[pi].outputs)):
            if not is_c:
                y0 = out_obj.text
                verify_text = GENERIC_CRITIQUE
                regen_msg = build_regen_prompt(msgs, y0, verify_text)
                regen_prompts.append(chat_to_text(regen_msg))
                regen_index.append((pi, ki))

    print(f"  {len(regen_prompts)} wrong y0 samples → generating y1")
    sp2 = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
    t2_outputs = llm.generate(regen_prompts, sp2)

    # ── Aggregate per-prompt g_i and c_i ──────────────────────────────────
    y1_corrects: dict[tuple[int,int], bool] = {}
    for (pi, ki), out in zip(regen_index, t2_outputs):
        y1_corrects[(pi, ki)] = grade(out.outputs[0].text, gt_list[pi])

    results = []
    for pi, (row, corrects) in enumerate(zip(rows, t1_corrects)):
        K = len(corrects)
        n_c = sum(corrects)
        n_w = K - n_c
        g_i = n_c / K

        # c_i: among wrong samples, how many y1 are correct (Laplace)
        n_y1c = sum(y1_corrects.get((pi, ki), False)
                    for ki, c in enumerate(corrects) if not c)
        c_i = (n_y1c + 1) / (n_w + 2)

        # quadrant
        g_hi = g_i >= args.g_thresh
        c_hi = c_i >= args.c_thresh
        quad = ("H" if g_hi else "L") + ("H" if c_hi else "L")

        results.append(dict(
            prompt_idx=idxs[pi],
            g_i=round(g_i, 4),
            c_i=round(c_i, 4),
            n_c=n_c,
            n_w=n_w,
            n_y1c=n_y1c,
            quad=quad,
        ))

    # ── Report ─────────────────────────────────────────────────────────────
    N = len(results)
    from collections import Counter
    quad_counts = Counter(r["quad"] for r in results)

    print("\n" + "="*60)
    print(f"Model: {args.model}   K={args.K}   N={N}")
    print(f"Thresholds: g>={args.g_thresh}, c>={args.c_thresh}")
    print("="*60)
    for q in ["HH", "HL", "LH", "LL"]:
        cnt = quad_counts.get(q, 0)
        print(f"  {q}: {cnt:4d}  ({100*cnt/N:.1f}%)")
    print()

    g_vals = [r["g_i"] for r in results]
    c_vals = [r["c_i"] for r in results]
    print(f"g_i:  mean={np.mean(g_vals):.3f}  median={np.median(g_vals):.3f}  std={np.std(g_vals):.3f}")
    print(f"c_i:  mean={np.mean(c_vals):.3f}  median={np.median(c_vals):.3f}  std={np.std(c_vals):.3f}")

    # c_i conditioned on g_L (the "interesting" slice)
    gl_cidx = [r["c_i"] for r in results if r["g_i"] < args.g_thresh]
    if gl_cidx:
        print(f"c_i | g<{args.g_thresh}: mean={np.mean(gl_cidx):.3f}  n={len(gl_cidx)}")
    gh_cidx = [r["c_i"] for r in results if r["g_i"] >= args.g_thresh]
    if gh_cidx:
        print(f"c_i | g>={args.g_thresh}: mean={np.mean(gh_cidx):.3f}  n={len(gh_cidx)}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        model=args.model,
        K=args.K,
        N=N,
        seed=args.seed,
        g_thresh=args.g_thresh,
        c_thresh=args.c_thresh,
        quad_counts=dict(quad_counts),
        quad_pcts={q: round(100*quad_counts.get(q,0)/N, 2) for q in ["HH","HL","LH","LL"]},
        g_mean=round(float(np.mean(g_vals)), 4),
        c_mean=round(float(np.mean(c_vals)), 4),
        g_median=round(float(np.median(g_vals)), 4),
        c_median=round(float(np.median(c_vals)), 4),
        results=results,
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
