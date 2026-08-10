#!/usr/bin/env python3
"""Generate teacher structured feedback v=(c,t*,e,p) for causal rectify eval.

Supports OpenAI-compatible GPT API (preferred) or a local HF/vLLM model.

Writes per-row:
  steps, n_steps, numbered_y0
  teacher_structured (dict)
  teacher_freeform (str)
  feedback texts are derived at eval time from structured fields
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

CR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CR_DIR))

from step_utils import segment_by_method  # noqa: E402
from structured_feedback import (  # noqa: E402
    FREEFORM_TEACHER_SYSTEM,
    FREEFORM_TEACHER_USER,
    STRUCTURED_TEACHER_SYSTEM,
    STRUCTURED_TEACHER_USER,
    parse_structured,
    render_structured,
)

MAX_WRONG_CHARS = 3500


def truncate(text: str, n: int = MAX_WRONG_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "\n[... truncated ...]"


def uses_completion_tokens(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in m


def chat_create(client, model: str, messages: list, temperature: float, max_tokens: int):
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if uses_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e)
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            return client.chat.completions.create(**kwargs)
        if "temperature" in msg.lower() and "unsupported" in msg.lower():
            kwargs.pop("temperature", None)
            return client.chat.completions.create(**kwargs)
        raise


def call_gpt(client, model, system, user, temperature, max_tokens, max_retries=4):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = chat_create(client, model, messages, temperature, max_tokens)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"GPT call failed: {last_err}")


def gen_with_gpt(args, rows):
    from openai import OpenAI

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    def work(i, r):
        if args.skip_existing and r.get("teacher_structured") and r.get("teacher_freeform"):
            return i, r, False
        wrong = truncate(r.get("y0") or r.get("wrong_attempt") or "")
        stepped = segment_by_method(wrong, args.segment_method)
        user_s = (
            STRUCTURED_TEACHER_USER.replace("{problem}", r["problem"])
            .replace("{numbered_solution}", stepped.numbered_text)
            .replace("{gt}", str(r.get("gt") or r.get("answer") or ""))
        )
        raw_s = call_gpt(
            client, args.model, STRUCTURED_TEACHER_SYSTEM, user_s, args.temperature, args.max_tokens
        )
        structured = parse_structured(raw_s, stepped.n_steps)
        if structured.verdict != "Incorrect" or structured.first_error < 1:
            # Pool is oracle-wrong; force Incorrect with mid-step fallback
            structured.verdict = "Incorrect"
            if structured.first_error < 1:
                structured.first_error = max(1, (stepped.n_steps + 1) // 2)

        user_f = (
            FREEFORM_TEACHER_USER.replace("{problem}", r["problem"]).replace(
                "{wrong_attempt}", wrong
            )
        )
        freeform = call_gpt(
            client, args.model, FREEFORM_TEACHER_SYSTEM, user_f, args.temperature, args.max_tokens
        )
        freeform = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", freeform)
        if "The answer is wrong" not in freeform:
            freeform = freeform.rstrip() + "\nThe answer is wrong."

        out = dict(r)
        out["steps"] = stepped.steps
        out["n_steps"] = stepped.n_steps
        out["numbered_y0"] = stepped.numbered_text
        out["segment_method"] = args.segment_method
        out["teacher_structured"] = structured.to_dict()
        out["teacher_structured_text"] = render_structured(structured)
        out["teacher_freeform"] = freeform
        out["teacher_model"] = args.model
        return i, out, True

    out_rows = [None] * len(rows)
    n_new = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, r) for i, r in enumerate(rows)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="teacher_gpt"):
            i, row, is_new = fut.result()
            out_rows[i] = row
            n_new += int(is_new)
    return out_rows, n_new


def gen_with_local_vllm(args, rows):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=40,
        max_tokens=args.max_tokens,
        n=1,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    def build_prompt(system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Prepare stepped solutions once
    prepared = []
    for r in rows:
        wrong = truncate(r.get("y0") or r.get("wrong_attempt") or "")
        stepped = segment_by_method(wrong, args.segment_method)
        prepared.append((r, wrong, stepped))

    # Structured pass
    prompts_s = []
    for r, wrong, stepped in prepared:
        user_s = (
            STRUCTURED_TEACHER_USER.replace("{problem}", r["problem"])
            .replace("{numbered_solution}", stepped.numbered_text)
            .replace("{gt}", str(r.get("gt") or r.get("answer") or ""))
        )
        prompts_s.append(build_prompt(STRUCTURED_TEACHER_SYSTEM, user_s))
    outs_s = llm.generate(prompts_s, sampling, use_tqdm=True)

    # Freeform pass
    prompts_f = []
    for r, wrong, stepped in prepared:
        user_f = FREEFORM_TEACHER_USER.replace("{problem}", r["problem"]).replace(
            "{wrong_attempt}", wrong
        )
        prompts_f.append(build_prompt(FREEFORM_TEACHER_SYSTEM, user_f))
    outs_f = llm.generate(prompts_f, sampling, use_tqdm=True)

    out_rows = []
    for (r, wrong, stepped), os_, of_ in zip(prepared, outs_s, outs_f):
        raw_s = os_.outputs[0].text
        structured = parse_structured(raw_s, stepped.n_steps)
        structured.verdict = "Incorrect"
        if structured.first_error < 1:
            structured.first_error = max(1, (stepped.n_steps + 1) // 2)
        freeform = (of_.outputs[0].text or "").strip()
        freeform = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", freeform)
        if "The answer is wrong" not in freeform:
            freeform = freeform.rstrip() + "\nThe answer is wrong."
        out = dict(r)
        out["steps"] = stepped.steps
        out["n_steps"] = stepped.n_steps
        out["numbered_y0"] = stepped.numbered_text
        out["segment_method"] = args.segment_method
        out["teacher_structured"] = structured.to_dict()
        out["teacher_structured_text"] = render_structured(structured)
        out["teacher_freeform"] = freeform
        out["teacher_model"] = args.model
        out_rows.append(out)
    return out_rows, len(out_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "gpt", "local"],
        help="auto: gpt if OPENAI_API_KEY else local",
    )
    ap.add_argument(
        "--model",
        default="",
        help="GPT model id or local HF path",
    )
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL") or None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument(
        "--segment_method",
        default="legacy",
        choices=["legacy", "scope_nn", "stride_tags"],
        help="How to split y0 into steps for structured teacher",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    backend = args.backend
    if backend == "auto":
        backend = "gpt" if args.api_key else "local"

    if backend == "gpt":
        if not args.api_key:
            raise SystemExit("OPENAI_API_KEY required for --backend gpt")
        if not args.model:
            args.model = os.environ.get("GPT_CRITIQUE_MODEL", "gpt-5o")
        print(f"Teacher backend=gpt model={args.model} n={len(rows)}")
        out_rows, n_new = gen_with_gpt(args, rows)
    else:
        if not args.model:
            # Default: PPO HF checkpoint
            root = Path(__file__).resolve().parents[2]
            args.model = str(root / "checkpoints/PAG/qwen1p5b_pag/global_step_400/actor_hf")
        print(f"Teacher backend=local model={args.model} n={len(rows)}")
        out_rows, n_new = gen_with_local_vllm(args, rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out} n={len(out_rows)} new={n_new}")


if __name__ == "__main__":
    main()
