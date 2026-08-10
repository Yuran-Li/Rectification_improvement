#!/usr/bin/env python3
"""Generate fixed critiques via OpenAI-compatible GPT API for ECR_fix.

Writes ``fixed_critique`` (and ``critique_model``) onto each row so
``controlled_rectify_eval.py`` can consume the file directly.

Protocol matches S2R / Self-correction: diagnose only, no full solution,
no \\boxed{}.

Usage
-----
  export OPENAI_API_KEY=...
  # optional: OPENAI_BASE_URL=...  GPT_CRITIQUE_MODEL=gpt-5o
  python generate_fixed_critiques_gpt.py \\
    --input data/fixed_wrong_pag_prerl.jsonl \\
    --output data/fixed_wrong_pag_prerl_with_critique.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

MAX_WRONG_CHARS = 3500
MAX_CRITIQUE_CHARS = 1200

CRITIQUE_SYSTEM = (
    "You are a math error analyst acting as a generative verifier. "
    "Given a problem and an incorrect solution, identify the SPECIFIC error "
    "(wrong step and why). "
    "Do NOT provide the full correct solution. "
    "Do NOT put a final answer in \\boxed{}. "
    "Write 1-4 sentences describing what went wrong and what should be fixed, "
    "and end your response with exactly: The answer is wrong."
)

CRITIQUE_USER = (
    "Problem:\n{problem}\n\n"
    "Incorrect solution:\n{wrong_attempt}\n\n"
    "Identify the specific error. Do not solve the problem."
)

FALLBACK = "The previous solution contains an error in its reasoning."


def truncate(text: str, n: int = MAX_WRONG_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "\n[... truncated ...]"


def fill(template: str, problem: str, wrong: str) -> str:
    return template.replace("{problem}", problem).replace("{wrong_attempt}", wrong)


def sanitize(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", text)
    if len(text) > MAX_CRITIQUE_CHARS:
        text = text[:MAX_CRITIQUE_CHARS] + "\n[... truncated ...]"
    return text or FALLBACK


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


def call_one(client, model, problem, wrong, temperature, max_tokens, n_samples, max_retries):
    wrong = truncate(wrong)
    messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {"role": "user", "content": fill(CRITIQUE_USER, problem, wrong)},
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            candidates = []
            for _ in range(max(1, n_samples)):
                resp = chat_create(client, model, messages, temperature, max_tokens)
                text = sanitize(resp.choices[0].message.content or "")
                if text:
                    candidates.append(text)
            if not candidates:
                return FALLBACK
            ok = [c for c in candidates if len(c) >= 40]
            pool = ok or candidates
            return sorted(pool, key=lambda t: abs(len(t) - 300))[0]
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"GPT critique failed after retries: {last_err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--model",
        default=os.environ.get("GPT_CRITIQUE_MODEL", "gpt-5o"),
        help="OpenAI-compatible model id (default GPT_CRITIQUE_MODEL or gpt-5o)",
    )
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL") or None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. export OPENAI_API_KEY=... "
            "(optional OPENAI_BASE_URL / GPT_CRITIQUE_MODEL)"
        )

    from openai import OpenAI

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} from {args.input}")
    print(f"Model={args.model} workers={args.workers} base_url={args.base_url or 'default'}")

    def work(i: int, r: dict):
        if args.skip_existing and (r.get("fixed_critique") or "").strip():
            return i, r["fixed_critique"].strip(), False
        wrong = r.get("wrong_attempt") or r.get("y0") or ""
        text = call_one(
            client,
            args.model,
            r["problem"],
            wrong,
            args.temperature,
            args.max_tokens,
            args.n_samples,
            args.max_retries,
        )
        return i, text, True

    critiques = [None] * len(records)
    n_new = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, r) for i, r in enumerate(records)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="gpt_critique"):
            i, text, is_new = fut.result()
            critiques[i] = text
            n_new += int(is_new)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r, c in zip(records, critiques):
            row = dict(r)
            row["fixed_critique"] = c
            row["critique_model"] = args.model
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows ({n_new} new) -> {out}")


if __name__ == "__main__":
    main()
