#!/usr/bin/env python3
"""Sample M verify texts per fixed wrong (problem, y0) using PAG prompts.

Prompt matches PAG verify turn:
  system: Please reason step by step...
  user:   problem
  assistant: y0
  user:   VERIFY_USER
  assistant: <sample M verify texts>

Outputs one JSONL row per example with fields:
  verifies_raw: list[str]         model output as sampled (length M)
  verifies_for_regen: list[str]   strip verdict + force wrong close (for PAG regenerate)
  verifies_specific: list[str]    alias of verifies_for_regen (backward compat)
  verifies_verdict_raw: list[str] per-sample verdict parsed from raw ('correct'|'wrong'|'none')
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CR_DIR))

from pag_verify_utils import (  # noqa: E402
    GENERIC_VERIFY_ASSISTANT,
    ensure_wrong_close,
    parse_verdict,
)

MAX_WRONG_CHARS = 3500
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
VERIFY_USER = (
    "Verify the previous solution without re-solving the problem from scratch. "
    "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
    "explain why it is wrong, and end your response with 'The answer is wrong'. "
    "If all steps are correct, end your response with 'The answer is correct'."
)


def truncate(text: str, n: int = MAX_WRONG_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "\n[... truncated ...]"


def build_verify_prompt(tokenizer, problem: str, y0: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": truncate(y0)},
        {"role": "user", "content": VERIFY_USER},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="verify sampler model path")
    ap.add_argument("--input", required=True, help="fixed wrong pool jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--M", type=int, default=8, help="verifies per y0")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    print(f"Loaded {len(rows)} rows from {args.input}; sampling M={args.M}")

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
        top_p=0.95 if args.temperature > 0 else 1.0,
        top_k=args.top_k if args.temperature > 0 else -1,
        max_tokens=args.max_new_tokens,
        n=args.M,
        repetition_penalty=args.repetition_penalty,
        stop=["<|im_end|>", "<|endoftext|>"],
        stop_token_ids=stop_token_ids or None,
    )

    prompts = [
        build_verify_prompt(tokenizer, r["problem"], r.get("y0") or r.get("wrong_attempt") or "")
        for r in rows
    ]
    outs = llm.generate(prompts, sampling, use_tqdm=True)

    out_rows = []
    n_correct = n_wrong = n_none = 0
    for r, out in zip(rows, outs):
        raw = [(o.text or "").strip() for o in out.outputs][: args.M]
        while len(raw) < args.M:
            raw.append("")
        verdicts = [parse_verdict(t) for t in raw]
        for v in verdicts:
            if v == "correct":
                n_correct += 1
            elif v == "wrong":
                n_wrong += 1
            else:
                n_none += 1
        for_regen = [ensure_wrong_close(t) for t in raw]
        out_rows.append(
            {
                **r,
                "verifies_raw": raw,
                "verifies_for_regen": for_regen,
                "verifies_specific": for_regen,
                "verifies_verdict_raw": verdicts,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = len(out_rows) * args.M
    print(f"Wrote {out} n={len(out_rows)} M={args.M}")
    print(
        f"raw verdicts: correct={n_correct}/{total} ({100*n_correct/total:.1f}%) "
        f"wrong={n_wrong}/{total} none={n_none}/{total}"
    )


if __name__ == "__main__":
    main()
