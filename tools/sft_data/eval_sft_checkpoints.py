#!/usr/bin/env python3
"""Compare base / verify-SFT / rectify-SFT on MATH500 + SFT train roles.

MATH500 (PAG multi-turn, model-own trajectory):
  generate (A1), verify (TPR/TNR/verify_acc), rectify (ECR_TP/EIR_FP/final_acc)

Train set (teacher-forced context, role competence):
  verify:  prompt = gold turns up to VERIFY_USER; score verdict vs gold + format
  rectify: prompt = gold turns up to REGENERATE_USER; score answer vs gold boxed

Example:
  CUDA_VISIBLE_DEVICES=0 python tools/sft_data/eval_sft_checkpoints.py \\
    --math500 datasets/math500.parquet \\
    --out_dir results/sft_ckpt_compare
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path("/data/yuranli/LLM/2026.04/github_references/S2R/tools/qwen_eval/eval")))

from grader import math_equal  # noqa: E402
from parser import extract_answer, strip_string  # noqa: E402
from verl.utils.pag_prompts import REGENERATE_USER, VERIFY_USER  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

VERDICT_RE = re.compile(r"The answer is (correct|wrong)\.?\s*$", re.I | re.M)


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
    """Return 'wrong' | 'correct' | None."""
    m = list(VERDICT_RE.finditer(text or ""))
    if m:
        return m[-1].group(1).lower()
    # soft fallback for S2R-style closers
    tail = (text or "")[-120:].lower()
    if "the answer is incorrect" in tail:
        return "wrong"
    if "the answer is correct" in tail:
        return "correct"
    return None


def chat_prompt(tokenizer, messages: List[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def batched_generate(llm: LLM, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
    if not prompts:
        return []
    sp = SamplingParams(
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
        n=1,
    )
    outs = llm.generate(prompts, sp, use_tqdm=True)
    return [o.outputs[0].text.strip() for o in outs]


def load_math500(path: str, n: int = -1, seed: int = 42) -> List[dict]:
    df = pd.read_parquet(path)
    if 0 < n < len(df):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df), size=n, replace=False)
        df = df.iloc[sorted(idx)]
    rows = []
    for _, r in df.iterrows():
        prompt = r["prompt"]
        problem = ""
        if isinstance(prompt, (list, np.ndarray)):
            for m in prompt:
                if isinstance(m, dict) and m.get("role") == "user":
                    problem = m.get("content") or ""
                    break
        rm = r["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = str(rm.get("ground_truth", "")).strip()
        if problem and gt:
            rows.append({"problem": problem, "gt": gt, "unique_id": r.get("unique_id")})
    return rows


def load_train_jsonl(path: str, n: int = -1, seed: int = 42) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    if 0 < n < len(rows):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def eval_math500_pag(
    llm: LLM,
    tokenizer,
    rows: List[dict],
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, float], List[dict]]:
    # Round 1: generate
    prompts = [
        chat_prompt(tokenizer, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["problem"]},
        ])
        for r in rows
    ]
    y0s = batched_generate(llm, prompts, max_tokens, temperature)

    # Round 2: verify
    v_prompts = []
    for r, y0 in zip(rows, y0s):
        v_prompts.append(chat_prompt(tokenizer, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["problem"]},
            {"role": "assistant", "content": y0},
            {"role": "user", "content": VERIFY_USER},
        ]))
    verifies = batched_generate(llm, v_prompts, max_tokens, temperature)

    # Round 3: rectify only when v=wrong
    rect_idx = []
    rect_prompts = []
    for i, (r, y0, vtext) in enumerate(zip(rows, y0s, verifies)):
        v = parse_verdict(vtext)
        if v == "wrong":
            rect_idx.append(i)
            rect_prompts.append(chat_prompt(tokenizer, [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["problem"]},
                {"role": "assistant", "content": y0},
                {"role": "user", "content": VERIFY_USER},
                {"role": "assistant", "content": vtext},
                {"role": "user", "content": REGENERATE_USER},
            ]))
    y1s_map = {}
    for i, text in zip(rect_idx, batched_generate(llm, rect_prompts, max_tokens, temperature)):
        y1s_map[i] = text

    # Aggregate
    n = len(rows)
    n_a1 = n_final = 0
    n_a1_wrong = n_a1_correct = 0
    n_tpr = n_tnr = 0
    n_verify_agree = 0
    n_fmt = 0
    n_v_wrong_a1w = n_ecr = 0
    n_v_wrong_a1c = n_eir = 0
    n_i2c = n_c2i = 0
    n_boxed = n_errw = 0
    n_vwords = 0
    details = []

    for i, r in enumerate(rows):
        y0, vtext = y0s[i], verifies[i]
        a1 = int(grade(y0, r["gt"]))
        v = parse_verdict(vtext)
        if v is not None:
            n_fmt += 1
        n_vwords += len((vtext or "").split())
        if "\\boxed" in (vtext or ""):
            n_boxed += 1
        if any(w in (vtext or "").lower() for w in ("error", "wrong", "incorrect", "mistake", "should")):
            n_errw += 1
        v_wrong = 1 if v == "wrong" else 0
        # verify agrees with GT correctness of a1
        if v is not None:
            agree = (v_wrong == 1 and a1 == 0) or (v_wrong == 0 and a1 == 1)
            n_verify_agree += int(agree)

        a2 = a1
        y1 = None
        if v == "wrong" and i in y1s_map:
            y1 = y1s_map[i]
            a2 = int(grade(y1, r["gt"]))
            if a1 == 0 and a2 == 1:
                n_i2c += 1
                n_ecr += 1
            if a1 == 1 and a2 == 0:
                n_c2i += 1
                n_eir += 1

        n_a1 += a1
        n_final += a2
        if a1 == 0:
            n_a1_wrong += 1
            if v_wrong:
                n_tpr += 1
                n_v_wrong_a1w += 1
        else:
            n_a1_correct += 1
            if not v_wrong:
                n_tnr += 1
            else:
                n_v_wrong_a1c += 1

        details.append({
            "unique_id": r.get("unique_id"),
            "a1": a1, "a2": a2, "verdict": v, "revised": y1 is not None,
            "verify_n_words": len((vtext or "").split()),
            "verify_boxed": "\\boxed" in (vtext or ""),
            "verify_text": (vtext or "")[:1200],
        })

    metrics = {
        "n": n,
        "A1_acc": safe_div(n_a1, n),
        "final_acc": safe_div(n_final, n),
        "verify_format_rate": safe_div(n_fmt, n),
        "verify_mean_words": safe_div(n_vwords, n),
        "verify_boxed_rate": safe_div(n_boxed, n),
        "verify_error_word_rate": safe_div(n_errw, n),
        "verify_acc": safe_div(n_verify_agree, n),
        "TPR": safe_div(n_tpr, n_a1_wrong),
        "TNR": safe_div(n_tnr, n_a1_correct),
        "ECR_TP": safe_div(n_ecr, n_v_wrong_a1w),
        "EIR_FP": safe_div(n_eir, n_v_wrong_a1c),
        "i_to_c_rate": safe_div(n_i2c, n_a1_wrong),
        "c_to_i_rate": safe_div(n_c2i, n_a1_correct),
        "revise_rate": safe_div(len(rect_idx), n),
        "delta_final_minus_A1": safe_div(n_final, n) - safe_div(n_a1, n),
    }
    return metrics, details


def eval_train_verify(
    llm: LLM,
    tokenizer,
    rows: List[dict],
    max_tokens: int,
    temperature: float,
) -> Dict[str, float]:
    prompts = []
    gold_verdicts = []
    for r in rows:
        msgs = r["messages"]
        # up to last user (VERIFY); drop gold assistant
        assert msgs[-1]["role"] == "assistant"
        ctx = [{k: m[k] for k in ("role", "content")} for m in msgs[:-1]]
        prompts.append(chat_prompt(tokenizer, ctx))
        gv = parse_verdict(msgs[-1]["content"])
        gold_verdicts.append(gv)

    preds = batched_generate(llm, prompts, max_tokens, temperature)
    n = len(rows)
    n_fmt = n_match = 0
    for pred, gold in zip(preds, gold_verdicts):
        pv = parse_verdict(pred)
        if pv is not None:
            n_fmt += 1
        if pv is not None and gold is not None and pv == gold:
            n_match += 1
    return {
        "n": n,
        "verify_format_rate": safe_div(n_fmt, n),
        "verify_verdict_match_gold": safe_div(n_match, n),
        "n_gold_wrong": sum(1 for g in gold_verdicts if g == "wrong"),
        "n_gold_correct": sum(1 for g in gold_verdicts if g == "correct"),
    }


def eval_train_rectify(
    llm: LLM,
    tokenizer,
    rows: List[dict],
    max_tokens: int,
    temperature: float,
) -> Dict[str, float]:
    prompts = []
    gts = []
    for r in rows:
        msgs = r["messages"]
        assert msgs[-1]["role"] == "assistant"
        ctx = [{k: m[k] for k in ("role", "content")} for m in msgs[:-1]]
        prompts.append(chat_prompt(tokenizer, ctx))
        gold = msgs[-1]["content"]
        # GT = boxed from gold rectify
        pred = extract_answer(gold, "math")
        gt = strip_string(pred, skip_unit=False) if pred else ""
        gts.append(gt)

    preds = batched_generate(llm, prompts, max_tokens, temperature)
    n = len(rows)
    n_ok = sum(1 for p, gt in zip(preds, gts) if gt and grade(p, gt))
    n_has_gt = sum(1 for gt in gts if gt)
    return {
        "n": n,
        "n_with_gt": n_has_gt,
        "rectify_acc_vs_gold": safe_div(n_ok, n_has_gt),
    }


def eval_one_model(
    model_path: str,
    tag: str,
    math_rows: List[dict],
    verify_rows: List[dict],
    rectify_rows: List[dict],
    max_tokens: int,
    temperature: float,
    tp: int,
    gpu_util: float,
    out_dir: Path,
) -> Dict[str, Any]:
    print(f"\n===== Evaluating {tag}: {model_path} =====", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="bfloat16",
    )
    summary: Dict[str, Any] = {"tag": tag, "model_path": model_path}

    if math_rows:
        m, details = eval_math500_pag(llm, tokenizer, math_rows, max_tokens, temperature)
        summary["math500"] = m
        (out_dir / f"{tag}_math500_details.jsonl").write_text(
            "\n".join(json.dumps(d) for d in details) + "\n"
        )
        print(f"[{tag}] MATH500", json.dumps(m, indent=2), flush=True)

    if verify_rows:
        mv = eval_train_verify(llm, tokenizer, verify_rows, max_tokens, temperature)
        summary["train_verify"] = mv
        print(f"[{tag}] train_verify", json.dumps(mv, indent=2), flush=True)

    if rectify_rows:
        mr = eval_train_rectify(llm, tokenizer, rectify_rows, max_tokens, temperature)
        summary["train_rectify"] = mr
        print(f"[{tag}] train_rectify", json.dumps(mr, indent=2), flush=True)

    # free engine (vLLM workers often linger; force exit of distributed workers)
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data/yuranli/LLM/2026.04/models/Qwen2.5-Math-7B-Instruct")
    ap.add_argument(
        "--verify_ckpt",
        default=str(REPO / "checkpoints/sft/qwen25math7b_pag_sft_verify/global_step_153"),
    )
    ap.add_argument(
        "--rectify_ckpt",
        default=str(REPO / "checkpoints/sft/qwen25math7b_pag_sft_rectify/global_step_75"),
    )
    ap.add_argument("--mixed_ckpt", default="", help="optional mixed SFT ckpt vs base")
    ap.add_argument("--math500", default=str(REPO / "datasets/math500.parquet"))
    ap.add_argument("--train_verify", default=str(REPO / "datasets/sft/Qwen-1.5B/sft_verify_train.jsonl"))
    ap.add_argument("--train_rectify", default=str(REPO / "datasets/sft/Qwen-1.5B/sft_rectify_train.jsonl"))
    ap.add_argument("--math_n", type=int, default=-1, help="-1 = all MATH500")
    ap.add_argument("--train_verify_n", type=int, default=512)
    ap.add_argument("--train_rectify_n", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.9)
    ap.add_argument("--out_dir", type=Path, default=REPO / "results/sft_ckpt_compare")
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--skip_rectify", action="store_true")
    ap.add_argument("--skip_math", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    math_rows = [] if args.skip_math else load_math500(args.math500, args.math_n)
    verify_rows = [] if args.skip_train else load_train_jsonl(args.train_verify, args.train_verify_n)
    rectify_rows = [] if args.skip_train else load_train_jsonl(args.train_rectify, args.train_rectify_n)
    print(f"MATH500 n={len(math_rows)} train_verify n={len(verify_rows)} train_rectify n={len(rectify_rows)}")

    models = []
    if not args.skip_base:
        models.append(("base_instruct", args.base))
    if args.mixed_ckpt:
        models.append(("sft_mixed", args.mixed_ckpt))
    if not args.skip_verify:
        models.append(("sft_verify_153", args.verify_ckpt))
    if not args.skip_rectify:
        models.append(("sft_rectify_75", args.rectify_ckpt))

    all_summaries = []
    for tag, path in models:
        s = eval_one_model(
            path, tag, math_rows, verify_rows, rectify_rows,
            args.max_tokens, args.temperature, args.tp, args.gpu_util, args.out_dir,
        )
        all_summaries.append(s)
        (args.out_dir / f"{tag}_summary.json").write_text(json.dumps(s, indent=2) + "\n")

    # comparison table
    table = {"models": [s["tag"] for s in all_summaries]}
    for split in ("math500", "train_verify", "train_rectify"):
        table[split] = {}
        keys = set()
        for s in all_summaries:
            if split in s:
                keys |= set(s[split].keys())
        for k in sorted(keys):
            table[split][k] = {s["tag"]: s.get(split, {}).get(k) for s in all_summaries}

    out_json = args.out_dir / "compare_summary.json"
    out_json.write_text(json.dumps({"summaries": all_summaries, "table": table}, indent=2) + "\n")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["# SFT checkpoint comparison", ""]
    for split, title in [
        ("math500", "MATH500 (PAG multi-turn)"),
        ("train_verify", "Train verify (teacher-forced context)"),
        ("train_rectify", "Train rectify (teacher-forced context)"),
    ]:
        if not table.get(split):
            continue
        lines += [f"## {title}", ""]
        hdr = "| metric | " + " | ".join(table["models"]) + " |"
        sep = "|" + "---|" * (len(table["models"]) + 1)
        lines += [hdr, sep]
        for k, vals in table[split].items():
            row = [k] + [fmt(vals[t]) for t in table["models"]]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    md_path = args.out_dir / "compare.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
