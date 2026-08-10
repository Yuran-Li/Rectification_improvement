# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Periodic PAG multi-turn generative eval for FSDP SFT.

Runs on a MATH500 subset with HF ``generate`` (inside FSDP ``summon_full_params``).
Metrics follow the paper definitions:

  TPR = P(v=1 | a1=0)   error recall
  TNR = P(v=0 | a1=1)   correct-answer retention
  ECR_TP = P(a2=1 | a1=0, v=1)
  EIR_FP = P(a2=0 | a1=1, v=1)

where v=1 means verify predicts wrong, v=0 means correct.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer

from verl.utils.reward_score.genrm_verify import get_verification_score
from verl.utils.reward_score.math_verify import compute_score as math_compute_score

# Keep in sync with vllm_pag_rollout_spmd.py
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
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def _is_correct(solution: str, gt: str) -> int:
    try:
        out = math_compute_score(solution, gt)
        return int(float(out.get("acc", out.get("score", 0))) > 0.5)
    except Exception:
        return 0


def _parse_problem_and_gt(row: dict) -> tuple[str, str]:
    prompt = row.get("prompt")
    problem = ""
    if isinstance(prompt, (list, np.ndarray)):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                problem = m.get("content") or ""
                break
    elif isinstance(prompt, str):
        problem = prompt
    rm = row.get("reward_model") or {}
    if isinstance(rm, str):
        try:
            rm = json.loads(rm)
        except Exception:
            rm = {}
    gt = str(rm.get("ground_truth", "")).strip()
    return problem.strip(), gt


def load_math500_subset(
    path: str,
    n: int,
    seed: int = 42,
) -> List[dict]:
    df = pd.read_parquet(path)
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(df))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(df))]
    rows = []
    for i in idxs:
        r = df.iloc[int(i)].to_dict()
        problem, gt = _parse_problem_and_gt(r)
        if not problem or not gt:
            continue
        rows.append({"problem": problem, "gt": gt, "unique_id": r.get("unique_id")})
    return rows


@torch.no_grad()
def _generate_one(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[dict],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_pag_multiturn_subset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    rows: Sequence[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, float]:
    """Evaluate a list of {problem, gt} with 1 gen → 1 verify → optional rectify."""
    # Counters for paper metrics
    n = 0
    n_a1_wrong = n_a1_correct = 0
    n_tpr_num = n_tnr_num = 0  # numerators
    n_v_wrong_on_a1_wrong = 0  # for ECR denom
    n_ecr_num = 0
    n_v_wrong_on_a1_correct = 0  # for EIR denom
    n_eir_num = 0

    n_verify_agree = 0
    n_revised = 0
    n_i_to_c = n_c_to_i = 0
    n_a1 = n_a2_final = 0  # a2_final = a2 if revised else a1

    for row in rows:
        problem, gt = row["problem"], row["gt"]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem},
        ]
        y0 = _generate_one(model, tokenizer, messages, max_new_tokens, temperature)
        a1 = _is_correct(y0, gt)
        n_a1 += a1

        messages.append({"role": "assistant", "content": y0})
        messages.append({"role": "user", "content": VERIFY_USER})
        verify_text = _generate_one(model, tokenizer, messages, max_new_tokens, temperature)
        vr = get_verification_score(verify_text, gt_judge=bool(a1))
        # v=1 means "wrong", v=0 means "correct" (from model prediction)
        v_wrong = 1 if vr["genrm_pred"] == "wrong" else 0
        n_verify_agree += int(vr["genrm_score"] > 0.5)

        n += 1
        if a1 == 0:
            n_a1_wrong += 1
            if v_wrong:
                n_tpr_num += 1
                n_v_wrong_on_a1_wrong += 1
        else:
            n_a1_correct += 1
            if not v_wrong:
                n_tnr_num += 1
            else:
                n_v_wrong_on_a1_correct += 1

        a2 = a1
        if v_wrong:
            messages.append({"role": "assistant", "content": verify_text})
            messages.append({"role": "user", "content": REGENERATE_USER})
            y1 = _generate_one(model, tokenizer, messages, max_new_tokens, temperature)
            a2 = _is_correct(y1, gt)
            n_revised += 1
            if a1 == 0 and a2 == 1:
                n_i_to_c += 1
                n_ecr_num += 1
            if a1 == 1 and a2 == 0:
                n_c_to_i += 1
                n_eir_num += 1

        n_a2_final += a2

    return {
        "_n": float(n),
        "_n_a1": float(n_a1),
        "_n_a2_final": float(n_a2_final),
        "_n_verify_agree": float(n_verify_agree),
        "_n_a1_wrong": float(n_a1_wrong),
        "_n_a1_correct": float(n_a1_correct),
        "_n_tpr_num": float(n_tpr_num),
        "_n_tnr_num": float(n_tnr_num),
        "_n_v_wrong_on_a1_wrong": float(n_v_wrong_on_a1_wrong),
        "_n_ecr_num": float(n_ecr_num),
        "_n_v_wrong_on_a1_correct": float(n_v_wrong_on_a1_correct),
        "_n_eir_num": float(n_eir_num),
        "_n_revised": float(n_revised),
        "_n_i_to_c": float(n_i_to_c),
        "_n_c_to_i": float(n_c_to_i),
    }


_COUNT_KEYS = [
    "_n",
    "_n_a1",
    "_n_a2_final",
    "_n_verify_agree",
    "_n_a1_wrong",
    "_n_a1_correct",
    "_n_tpr_num",
    "_n_tnr_num",
    "_n_v_wrong_on_a1_wrong",
    "_n_ecr_num",
    "_n_v_wrong_on_a1_correct",
    "_n_eir_num",
    "_n_revised",
    "_n_i_to_c",
    "_n_c_to_i",
]


def _finalize_from_counts(c: Dict[str, float]) -> Dict[str, float]:
    def _safe(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    n = c["_n"]
    return {
        "n": n,
        "a1_acc": _safe(c["_n_a1"], n),
        "final_acc": _safe(c["_n_a2_final"], n),
        "verify_acc": _safe(c["_n_verify_agree"], n),
        "TPR": _safe(c["_n_tpr_num"], c["_n_a1_wrong"]),
        "TNR": _safe(c["_n_tnr_num"], c["_n_a1_correct"]),
        "ECR_TP": _safe(c["_n_ecr_num"], c["_n_v_wrong_on_a1_wrong"]),
        "EIR_FP": _safe(c["_n_eir_num"], c["_n_v_wrong_on_a1_correct"]),
        "i_to_c_rate": _safe(c["_n_i_to_c"], c["_n_a1_wrong"]),
        "c_to_i_rate": _safe(c["_n_c_to_i"], c["_n_a1_correct"]),
        "revise_rate": _safe(c["_n_revised"], n),
    }


@torch.no_grad()
def run_distributed_pag_eval(
    fsdp_model: FSDP,
    tokenizer: PreTrainedTokenizer,
    rows: List[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, float]:
    """All ranks enter summon_full_params; each rank evals a data shard; reduce counts."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rows = [r for i, r in enumerate(rows) if i % world == rank]

    fsdp_model.eval()
    with FSDP.summon_full_params(fsdp_model, writeback=False, recurse=True):
        # Prefer unwrapped module if present
        module = fsdp_model.module if hasattr(fsdp_model, "module") else fsdp_model
        local = evaluate_pag_multiturn_subset(
            module, tokenizer, local_rows, max_new_tokens=max_new_tokens, temperature=temperature
        )

    # Reduce raw counts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vec = torch.tensor([local[k] for k in _COUNT_KEYS], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    counts = {k: float(vec[i].item()) for i, k in enumerate(_COUNT_KEYS)}
    metrics = _finalize_from_counts(counts)
    # prefix for wandb
    return {f"val/pag/{k}": v for k, v in metrics.items()}
