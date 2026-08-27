#!/usr/bin/env python3
"""PAG-style K-rollout for curriculum retrospective analysis.

For each prompt in the dataset, runs K independent rollouts:
  y0  →  verify  →  [rectify if model says "wrong"]

Saves a JSON compatible with validation_results/*.json so curriculum_analysis.py
can directly load it without re-running.

Usage
-----
# Base checkpoint (Qwen2.5-1.5B-Instruct):
CUDA_VISIBLE_DEVICES=0,1 python scripts/curriculum_rollout.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset datasets/math500.parquet \
    --K 4 --out curriculum_rollouts/base_1.5b_math500_k4.json

# PAG-400 checkpoint:
CUDA_VISIBLE_DEVICES=0,1 python scripts/curriculum_rollout.py \
    --model checkpoints/PAG-critique-utility/qwen1.5b_pag/global_step_210/actor_hf \
    --dataset datasets/math500.parquet \
    --K 4 --out curriculum_rollouts/pag400_1.5b_math500_k4.json

For 7B, set --tensor_parallel_size 4 and use 4 GPUs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# vLLM v1 workers use forking by default on Linux; force spawn to avoid CUDA re-init errors.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path("/data/yuranli/LLM/2026.04/github_references/S2R/tools/qwen_eval/eval")))

from grader import math_equal  # noqa: E402
from parser import extract_answer, strip_string  # noqa: E402
from verl.utils.pag_prompts import REGENERATE_USER, VERIFY_USER  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
VERDICT_RE = re.compile(r"The answer is (correct|wrong)\.?\s*$", re.I | re.M)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(text: str, gt: str) -> bool:
    if not text or not str(gt).strip():
        return False
    pred = extract_answer(text, "math")
    pred_s = strip_string(pred, skip_unit=False) if pred is not None else ""
    if not str(pred_s).strip():
        return False
    try:
        return bool(math_equal(pred_s, gt))
    except Exception:
        return str(pred_s).strip() == str(gt).strip()


def parse_verdict(text: str) -> Optional[str]:
    m = list(VERDICT_RE.finditer(text or ""))
    if m:
        return m[-1].group(1).lower()
    tail = (text or "")[-120:].lower()
    if "the answer is incorrect" in tail:
        return "wrong"
    if "the answer is correct" in tail:
        return "correct"
    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _extract_problem(raw) -> str:
    """Extract user-facing problem string from prompt field (may be list/array of dicts or string)."""
    if isinstance(raw, (list, np.ndarray)):
        for m in raw:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
        return ""
    if raw is None:
        return ""
    return str(raw).strip()


def load_dataset(path: str, max_n: int = -1, seed: int = 42) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
        rows: List[Dict[str, Any]] = []
        for idx_r, r in df.iterrows():
            prompt_raw = r["prompt"] if "prompt" in r.index else None
            problem_raw = r["problem"] if "problem" in r.index else None
            problem = _extract_problem(prompt_raw) or _extract_problem(problem_raw)
            if not problem:
                continue
            rm = r["reward_model"] if "reward_model" in r.index else {}
            if isinstance(rm, str):
                rm = json.loads(rm)
            gt = str((rm or {}).get("ground_truth", "")).strip() if isinstance(rm, dict) else ""
            if not gt:
                continue
            uid_val = r["unique_id"] if "unique_id" in r.index else (r["index"] if "index" in r.index else idx_r)
            uid = str(uid_val)
            ds_val = r["data_source"] if "data_source" in r.index else path
            data_source = str(ds_val)
            rows.append({"problem": problem, "gt": gt, "uid": uid, "data_source": data_source})
        if 0 < max_n < len(rows):
            rng = np.random.RandomState(seed)
            sel = sorted(rng.choice(len(rows), max_n, replace=False).tolist())
            rows = [rows[i] for i in sel]
        return rows
    raise ValueError(f"Unsupported dataset format: {path}")


# ---------------------------------------------------------------------------
# Batched vLLM generation
# ---------------------------------------------------------------------------

def batched_generate(llm, prompts: List[str], max_tokens: int, sp_kwargs: dict) -> List[str]:
    if not prompts:
        return []
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, **sp_kwargs)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    return [o.outputs[0].text.strip() for o in outs]


def chat_prompt(tokenizer, messages: List[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Core rollout for one repetition
# ---------------------------------------------------------------------------

def rollout_once(
    llm,
    tokenizer,
    rows: List[Dict[str, Any]],
    max_tokens: int,
    sp_kwargs: dict,
) -> List[Dict[str, Any]]:
    """Run one full PAG trajectory for each row. Returns per-row result dicts."""
    # Turn 1 — generation
    t1_prompts = [
        chat_prompt(tokenizer, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["problem"]},
        ])
        for r in rows
    ]
    y0s = batched_generate(llm, t1_prompts, max_tokens, sp_kwargs)

    # Turn 2 — verify
    v_prompts = [
        chat_prompt(tokenizer, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["problem"]},
            {"role": "assistant", "content": y0},
            {"role": "user", "content": VERIFY_USER},
        ])
        for r, y0 in zip(rows, y0s)
    ]
    verifies = batched_generate(llm, v_prompts, max_tokens, sp_kwargs)

    # Turn 3 — rectify only when verdict = "wrong"
    rect_idx: List[int] = []
    rect_prompts: List[str] = []
    for i, (r, y0, vtext) in enumerate(zip(rows, y0s, verifies)):
        if parse_verdict(vtext) == "wrong":
            rect_idx.append(i)
            rect_prompts.append(chat_prompt(tokenizer, [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["problem"]},
                {"role": "assistant", "content": y0},
                {"role": "user", "content": VERIFY_USER},
                {"role": "assistant", "content": vtext},
                {"role": "user", "content": REGENERATE_USER},
            ]))
    y1s: Dict[int, str] = {}
    for i, text in zip(rect_idx, batched_generate(llm, rect_prompts, max_tokens, sp_kwargs)):
        y1s[i] = text

    # Score
    results: List[Dict[str, Any]] = []
    for i, (r, y0, vtext) in enumerate(zip(rows, y0s, verifies)):
        acc_t1 = 1.0 if grade(y0, r["gt"]) else 0.0
        verdict = parse_verdict(vtext)
        revised = verdict == "wrong"
        y1 = y1s.get(i, "")
        acc_t2 = (1.0 if grade(y1, r["gt"]) else 0.0) if revised else -1.0
        acc_final = acc_t2 if (revised and acc_t2 >= 0) else acc_t1
        results.append({
            "prompt": t1_prompts[i],
            "data_source": r["data_source"],
            "acc_t1": acc_t1,
            "acc_t2": acc_t2,
            "revised": revised,
            "acc_final": acc_final,
            "genrm_score": 1.0 if verdict == "wrong" else (0.0 if verdict == "correct" else 0.5),
            "genrm_prediction": verdict or "unknown",
            "ground_truth": r["gt"],
            "uid": r["uid"],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model path or checkpoint dir")
    ap.add_argument("--dataset", required=True, help="path to .parquet dataset")
    ap.add_argument("--K", type=int, default=4, help="rollouts per prompt")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_n", type=int, default=-1, help="cap number of prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp_kwargs = {"temperature": args.temperature, "top_p": args.top_p}

    rows = load_dataset(args.dataset, max_n=args.max_n, seed=args.seed)
    print(f"Loaded {len(rows)} prompts, K={args.K}")

    # Repeat rows K times, then run rollout_once (batches efficiently)
    repeated = rows * args.K
    np.random.RandomState(args.seed).shuffle(repeated)  # shuffle for vLLM throughput

    # Actually keep order: repeat interleaved so prompt i appears K times consecutively
    repeated_ordered = []
    for r in rows:
        for _ in range(args.K):
            repeated_ordered.append(r)

    all_results = rollout_once(llm, tokenizer, repeated_ordered, args.max_tokens, sp_kwargs)

    # Re-group: result[i*K : (i+1)*K] are the K rollouts for prompt i
    output: Dict[str, Any] = {}
    data_source = rows[0]["data_source"] if rows else "unknown"
    per_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        key = r["problem"]
        per_prompt[key] = all_results[i * args.K : (i + 1) * args.K]

    output[data_source] = per_prompt

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(per_prompt)} prompts × K={args.K} rollouts → {out_path}")


if __name__ == "__main__":
    main()
