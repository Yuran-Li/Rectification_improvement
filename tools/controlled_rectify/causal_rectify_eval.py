#!/usr/bin/env python3
"""Causal rectify eval: Feedback specificity × Edit constraint.

Fixed: (problem, wrong y0), same rectifier, same decoding budget.
Vary only:
  feedback condition ∈ {regenerate, wrong_only, localization,
                        localization_analysis, localization_analysis_plan, freeform}
  edit mode ∈ {full_regen, prefix_rewrite}

Reports per-cell W2C and related diagnostics used by aggregate_causal_rectify.py.
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
from step_utils import (  # noqa: E402
    first_erroneous_step_text,
    prefix_before,
    prefix_preserved,
    segment_by_method,
    suffix_from,
)
from structured_feedback import (  # noqa: E402
    EDIT_MODES,
    FEEDBACK_CONDITIONS,
    StructuredVerify,
    feedback_text_for_condition,
    is_actionable,
)

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
MAX_WRONG_CHARS = 3500


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


def load_structured(row: dict) -> StructuredVerify:
    d = row.get("teacher_structured") or {}
    return StructuredVerify(
        verdict=d.get("verdict", "Incorrect"),
        first_error=int(d.get("first_error") or 0),
        error_analysis=d.get("error_analysis") or "",
        rectification_plan=d.get("rectification_plan") or "",
        raw_text=d.get("raw_text") or row.get("teacher_structured_text") or "",
    )


def build_full_regen_prompt(tokenizer, problem: str, y0: str, feedback: str, cond: str) -> str:
    """Unconstrained regenerate / revise (may ignore prefix)."""
    if cond == "regenerate" or not feedback.strip():
        user = (
            f"Solve the following math problem. Put the final answer in \\boxed{{}}.\n\n"
            f"Problem:\n{problem}"
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": truncate(y0)},
        {
            "role": "user",
            "content": (
                f"{feedback.strip()}\n\n"
                "Please provide a corrected solution to the math problem. "
                "Put the final answer in \\boxed{}."
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_prefix_rewrite_prompt(
    tokenizer,
    problem: str,
    steps: list[str],
    t_star: int,
    feedback: str,
) -> tuple[str, str]:
    """Return (prompt, correct_prefix_text). Model should only produce suffix."""
    prefix = prefix_before(steps, t_star)
    err_step = first_erroneous_step_text(steps, t_star)
    err_suffix = suffix_from(steps, t_star)

    user = (
        f"Problem:\n{problem}\n\n"
        f"You must REWRITE ONLY the suffix starting at the first error step.\n"
        f"Do NOT modify or repeat the correct prefix.\n"
        f"Your output should continue from Step {t_star} onward and end with \\boxed{{}}.\n\n"
        f"=== Correct prefix (DO NOT CHANGE; already accepted) ===\n"
        f"{prefix if prefix else '(empty — error begins at Step 1)'}\n\n"
        f"=== First erroneous step ===\n{err_step}\n\n"
        f"=== Original erroneous suffix ===\n{err_suffix}\n\n"
        f"=== Feedback ===\n{feedback.strip() if feedback.strip() else 'The previous solution is incorrect.'}\n\n"
        f"Now write the corrected suffix starting at Step {t_star}:"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    # Soft prefix forcing: prefill assistant with correct prefix so generation continues after it.
    if prefix.strip():
        messages.append({"role": "assistant", "content": prefix.rstrip() + "\n\n"})
        try:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, continue_final_message=True, tokenize=False
            )
        except TypeError:
            # Older transformers: fall back to user-only prompt without continue
            messages = messages[:-1]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
    else:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt, prefix


def stitch_answer(prefix: str, suffix: str) -> str:
    suffix = (suffix or "").strip()
    prefix = (prefix or "").rstrip()
    if not prefix:
        return suffix
    # Avoid duplicating prefix if model echoed it
    if suffix.startswith(prefix[: min(80, len(prefix))]):
        return suffix
    return prefix + "\n\n" + suffix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="frozen rectifier HF path")
    ap.add_argument("--data", required=True, help="pool + teacher_structured jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--feedbacks",
        nargs="+",
        default=FEEDBACK_CONDITIONS,
        choices=FEEDBACK_CONDITIONS,
    )
    ap.add_argument(
        "--edit_modes",
        nargs="+",
        default=EDIT_MODES,
        choices=EDIT_MODES,
    )
    ap.add_argument("--n_samples", type=int, default=1, help="K rectifier samples per cell")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--data_name", default="math")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument(
        "--segment_method",
        default="",
        help="If set (and row has no steps), re-segment y0 with this method",
    )
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    missing = [r.get("idx") for r in records if not r.get("teacher_structured")]
    if missing:
        raise SystemExit(f"missing teacher_structured for idx={missing[:10]}... n={len(missing)}")
    print(f"Loaded {len(records)}; feedbacks={args.feedbacks}; edits={args.edit_modes}")

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
        n=args.n_samples,
        repetition_penalty=args.repetition_penalty,
        stop=["<|im_end|>", "<|endoftext|>"],
        stop_token_ids=stop_token_ids or None,
    )

    # Precompute steps / structured / feedback bodies
    prepared = []
    for r in records:
        y0 = r.get("y0") or r.get("wrong_attempt") or ""
        steps = r.get("steps")
        if not steps:
            method = args.segment_method or r.get("segment_method") or "legacy"
            steps = segment_by_method(y0, method).steps
        v = load_structured(r)
        freeform = r.get("teacher_freeform") or ""
        prepared.append({"row": r, "y0": y0, "steps": steps, "v": v, "freeform": freeform})

    out_rows = [
        {
            "idx": p["row"].get("idx"),
            "unique_id": p["row"].get("unique_id"),
            "gt": p["row"].get("gt"),
            "n_steps": len(p["steps"]),
            "t_star": p["v"].first_error,
            "correct_prefix": prefix_before(p["steps"], max(1, p["v"].first_error or 1)),
            "actionable": is_actionable(p["v"]),
            "results": {},
        }
        for p in prepared
    ]

    for feedback in args.feedbacks:
        for edit in args.edit_modes:
            key = f"{feedback}__{edit}"
            print(f"=== {key} ===")
            prompts = []
            prefixes = []
            for p in prepared:
                fb = feedback_text_for_condition(feedback, p["v"], p["freeform"])
                t_star = max(1, p["v"].first_error or 1)
                if edit == "full_regen":
                    prompts.append(
                        build_full_regen_prompt(
                            tokenizer, p["row"]["problem"], p["y0"], fb, feedback
                        )
                    )
                    prefixes.append(prefix_before(p["steps"], t_star))
                else:
                    prompt, pref = build_prefix_rewrite_prompt(
                        tokenizer, p["row"]["problem"], p["steps"], t_star, fb
                    )
                    prompts.append(prompt)
                    prefixes.append(pref)

            outs = llm.generate(prompts, sampling, use_tqdm=True)
            for i, (out, pref) in enumerate(zip(outs, prefixes)):
                texts_raw = [o.text for o in out.outputs]
                if edit == "prefix_rewrite":
                    texts = [stitch_answer(pref, t) for t in texts_raw]
                else:
                    texts = texts_raw
                corrects = [grade(t, prepared[i]["row"]["gt"], args.data_name) for t in texts]
                pprs = [prefix_preserved(t, pref) for t in texts]
                out_rows[i]["results"][key] = {
                    "texts_raw": texts_raw,
                    "texts": texts,
                    "correct_prefix": pref,
                    "corrects": corrects,
                    "pprs": pprs,
                    "acc@1": float(corrects[0]) if corrects else 0.0,
                    "pass@k": float(any(corrects)) if corrects else 0.0,
                    "ppr@1": float(pprs[0]) if pprs else 0.0,
                    "t_star": prepared[i]["v"].first_error,
                    "actionable": is_actionable(prepared[i]["v"]),
                }

    # Shuffled critique control for plan + freeform under both edits (optional but key metric)
    # Shuffle structured fields across rows for Delta_critique
    import random

    rng = random.Random(0)
    idxs = list(range(len(prepared)))
    shuffled = idxs[:]
    rng.shuffle(shuffled)
    for edit in args.edit_modes:
        if "localization_analysis_plan" not in args.feedbacks:
            continue
        key = f"shuffled_plan__{edit}"
        print(f"=== {key} ===")
        prompts, prefixes = [], []
        for i, p in enumerate(prepared):
            donor = prepared[shuffled[i]]
            fb = feedback_text_for_condition("localization_analysis_plan", donor["v"])
            # Keep THIS example's true t*/prefix for rewrite axis fairness on edit constraint,
            # but feed SHUFFLED feedback content (critique dependence).
            t_star = max(1, p["v"].first_error or 1)
            if edit == "full_regen":
                prompts.append(
                    build_full_regen_prompt(
                        tokenizer, p["row"]["problem"], p["y0"], fb, "localization_analysis_plan"
                    )
                )
                prefixes.append(prefix_before(p["steps"], t_star))
            else:
                prompt, pref = build_prefix_rewrite_prompt(
                    tokenizer, p["row"]["problem"], p["steps"], t_star, fb
                )
                prompts.append(prompt)
                prefixes.append(pref)
        outs = llm.generate(prompts, sampling, use_tqdm=True)
        for i, (out, pref) in enumerate(zip(outs, prefixes)):
            texts_raw = [o.text for o in out.outputs]
            texts = (
                [stitch_answer(pref, t) for t in texts_raw]
                if edit == "prefix_rewrite"
                else texts_raw
            )
            corrects = [grade(t, prepared[i]["row"]["gt"], args.data_name) for t in texts]
            pprs = [prefix_preserved(t, pref) for t in texts]
            out_rows[i]["results"][key] = {
                "texts_raw": texts_raw,
                "texts": texts,
                "correct_prefix": pref,
                "corrects": corrects,
                "pprs": pprs,
                "acc@1": float(corrects[0]) if corrects else 0.0,
                "pass@k": float(any(corrects)) if corrects else 0.0,
                "ppr@1": float(pprs[0]) if pprs else 0.0,
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Quick console summary
    keys = sorted({k for r in out_rows for k in r["results"]})
    summary = {"n": len(out_rows), "n_samples": args.n_samples, "metrics": {}}
    for key in keys:
        w2c = sum(r["results"][key]["acc@1"] for r in out_rows) / len(out_rows)
        ppr = sum(r["results"][key]["ppr@1"] for r in out_rows) / len(out_rows)
        summary["metrics"][key] = {
            "W2C@1_pct": round(100 * w2c, 2),
            "PPR@1_pct": round(100 * ppr, 2),
        }
        print(f"{key}: W2C@1={100*w2c:.1f}%  PPR@1={100*ppr:.1f}%")

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
