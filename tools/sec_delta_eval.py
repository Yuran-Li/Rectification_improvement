#!/usr/bin/env python3
"""
Compute Δg and Δc per SEC category for a final checkpoint,
given a base-checkpoint JSON produced by sec_quadrant_check.py.

Category labels (fixed at t=0):
  C1: g=1
  C2: g>=0.5, n_WC>0
  C3: g>=0.5, n_WC=0
  C4: g<0.5,  n_WC>0   ← corrective salvage region
  C5: g<0.5,  n_WC=0

Metrics:
  Δg(Cj)  = mean(g_T - g_0) over prompts in Cj
  Δc(Cj)  = opportunity-weighted: Σn_WC_T / Σn_W_T  -  Σn_WC_0 / Σn_W_0

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/sec_delta_eval.py \
    --base_json tools/sec_quadrant_check_1p5b.json \
    --model .../qwen1p5b_pag/global_step_400/actor_hf \
    --output tools/sec_delta_1p5b_ppo400.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verl.utils.pag_prompts import GENERIC_CRITIQUE, VERIFY_USER, REGENERATE_USER
from verl.utils.reward_score.math import compute_score


def grade(text: str, gt: str) -> bool:
    return compute_score(text, gt) >= 0.5


def category(r: dict) -> str:
    g = r['g_i']; nwc = r['n_y1c']
    if g == 1.0:           return 'C1'
    elif g >= 0.5 and nwc > 0: return 'C2'
    elif g >= 0.5:         return 'C3'
    elif nwc > 0:          return 'C4'
    else:                  return 'C5'


def build_turn1(messages): return [m for m in messages if m['role'] != 'assistant']

def build_regen(messages, y0, verify_text):
    return build_turn1(messages) + [
        {'role': 'assistant', 'content': y0},
        {'role': 'user',      'content': VERIFY_USER},
        {'role': 'assistant', 'content': verify_text},
        {'role': 'user',      'content': REGENERATE_USER},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_json', required=True)
    ap.add_argument('--model',     required=True)
    ap.add_argument('--data',      default='datasets/math7500.parquet')
    ap.add_argument('--K',         type=int, default=8)
    ap.add_argument('--max_tokens',type=int, default=2048)
    ap.add_argument('--gpu_util',  type=float, default=0.65)
    ap.add_argument('--output',    required=True)
    args = ap.parse_args()

    # ── load base data ────────────────────────────────────────────────────
    base = json.loads(Path(args.base_json).read_text())
    base_by_idx = {r['prompt_idx']: r for r in base['results']}
    prompt_idxs = sorted(base_by_idx.keys())
    N = len(prompt_idxs)
    print(f'Base JSON: {args.base_json}  N={N}  K_base={base["K"]}')

    # ── load parquet ──────────────────────────────────────────────────────
    df = pd.read_parquet(ROOT / args.data)
    rows = [df.iloc[i] for i in prompt_idxs]
    gt_list = [row['reward_model']['ground_truth'] for row in rows]

    # ── vLLM ─────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_num_seqs=64, max_model_len=4096, enforce_eager=False)
    tok = llm.get_tokenizer()
    def to_text(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ── Turn 1 ────────────────────────────────────────────────────────────
    K = args.K
    print(f'\n=== Turn 1: {N} prompts × K={K} ===')
    t1_prompts = [to_text(build_turn1(list(row['prompt']))) for row in rows]
    t1_out = llm.generate(t1_prompts, SamplingParams(n=K, temperature=0.7, max_tokens=args.max_tokens))
    t1_corrects = [[grade(o.text, gt) for o in out.outputs]
                   for out, gt in zip(t1_out, gt_list)]

    # ── Turn 2: regen for wrong y0 ────────────────────────────────────────
    regen_idx, regen_prompts = [], []
    for pi, (row, corrects) in enumerate(zip(rows, t1_corrects)):
        for ki, (c, o) in enumerate(zip(corrects, t1_out[pi].outputs)):
            if not c:
                regen_prompts.append(to_text(build_regen(list(row['prompt']), o.text, GENERIC_CRITIQUE)))
                regen_idx.append((pi, ki))
    print(f'Turn 2: {len(regen_prompts)} wrong y0 → regen')
    t2_out = llm.generate(regen_prompts, SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens))
    y1_correct = {(pi,ki): grade(o.outputs[0].text, gt_list[pi])
                  for (pi,ki), o in zip(regen_idx, t2_out)}

    # ── Aggregate per prompt ──────────────────────────────────────────────
    new_results = []
    for pi, (pidx, row, corrects) in enumerate(zip(prompt_idxs, rows, t1_corrects)):
        nc = sum(corrects); nw = K - nc
        nwc = sum(y1_correct.get((pi,ki), False) for ki,c in enumerate(corrects) if not c)
        new_results.append(dict(prompt_idx=pidx, g_i=nc/K, n_c=nc, n_w=nw, n_y1c=nwc))

    new_by_idx = {r['prompt_idx']: r for r in new_results}

    # ── Δg and Δc per category ────────────────────────────────────────────
    cats = ['C1','C2','C3','C4','C5']
    labels = {
        'C1': 'Mastered (g=1)',
        'C2': 'Gen-capable + salvageable  (g≥0.5, n_WC>0)',
        'C3': 'Gen-capable + no salvage   (g≥0.5, n_WC=0)',
        'C4': 'Gen-hard + salvageable     (g<0.5, n_WC>0)  ← key',
        'C5': 'Gen-hard + no salvage      (g<0.5, n_WC=0)',
    }

    # group by base category (fixed at t=0)
    groups = defaultdict(list)
    for pidx, base_r in base_by_idx.items():
        cat = category(base_r)
        new_r = new_by_idx[pidx]
        groups[cat].append((base_r, new_r))

    print(f'\n{"="*72}')
    print(f'Model (T): {args.model.split("/")[-3]}')
    print(f'{"="*72}')
    print(f'{"Cat":4s}  {"n":>5s}  {"g0":>6s}  {"gT":>6s}  {"Δg":>7s}  {"c0":>6s}  {"cT":>6s}  {"Δc":>7s}  label')

    summary = {}
    for cat in cats:
        grp = groups[cat]
        n = len(grp)
        if n == 0:
            continue

        g0s = [b['g_i'] for b,_ in grp]
        gTs = [t['g_i'] for _,t in grp]
        g0_mean = np.mean(g0s); gT_mean = np.mean(gTs)
        delta_g = gT_mean - g0_mean

        # opportunity-weighted c
        nW0  = sum(b['n_w']   for b,_ in grp)
        nWC0 = sum(b['n_y1c'] for b,_ in grp)
        nWT  = sum(t['n_w']   for _,t in grp)
        nWCT = sum(t['n_y1c'] for _,t in grp)
        c0 = nWC0/nW0 if nW0 > 0 else float('nan')
        cT = nWCT/nWT if nWT > 0 else float('nan')
        delta_c = (cT - c0) if (nW0 > 0 and nWT > 0) else float('nan')

        print(f'{cat:4s}  {n:5d}  {g0_mean:6.3f}  {gT_mean:6.3f}  {delta_g:+7.3f}  '
              f'{c0:6.3f}  {cT:6.3f}  {delta_c:+7.3f}  {labels[cat]}')

        summary[cat] = dict(n=n, g0=round(g0_mean,4), gT=round(gT_mean,4),
                            delta_g=round(float(delta_g),4),
                            c0=round(float(c0),4) if not np.isnan(c0) else None,
                            cT=round(float(cT),4) if not np.isnan(cT) else None,
                            delta_c=round(float(delta_c),4) if not np.isnan(delta_c) else None,
                            nW0=nW0, nWT=nWT, nWC0=nWC0, nWCT=nWCT)

    out = dict(model_T=args.model, base_json=args.base_json, K=K, N=N, summary=summary,
               results=new_results)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f'\nSaved → {args.output}')


if __name__ == '__main__':
    main()
