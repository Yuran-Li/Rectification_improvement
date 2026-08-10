#!/usr/bin/env python3
"""Convert S2R monologue SFT JSON into PAG multi-turn messages for MultiTurnSFTDataset.

S2R trajectory (single assistant turn):
  y0 → \"Wait, let me recheck my solution.\" → verify → [\"Let me try again.\" → y1]

PAG trajectory (aligned with vllm_pag_rollout_spmd templates):
  system + user(problem)
  → assistant(y0)
  → user(VERIFY_TEMPLATE)
  → assistant(verify)          # loss target for verify-SFT
  → user(REGENERATE_TEMPLATE)  # only if incorrect
  → assistant(y1)              # loss target for rectify-SFT

MultiTurnSFTDataset only applies loss on the *last* assistant turn, so we emit
separate rows for verify and rectify (and an optional mixed concat).

Example:
  python tools/sft_data/convert_s2r_to_pag_multiturn.py \\
    --input /path/to/sft_qwen2.5_math_7B.json \\
    --out_dir datasets/sft
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

WAIT_BRIDGE = "Wait, let me recheck my solution."
RETRY_BRIDGE = "Let me try again."

# Must match verl/workers/rollout/vllm_rollout/vllm_pag_rollout_spmd.py
VERIFY_TEMPLATE = (
    "Verify the previous solution without re-solving the problem from scratch. "
    "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
    "explain why it is wrong, and end your response with 'The answer is wrong'. "
    "If all steps are correct, end your response with 'The answer is correct'."
)
REGENERATE_TEMPLATE = (
    "You indicated that your previous answer was wrong. "
    "Please provide the correct solution to the math problem."
)
SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.S)


def extract_problem(prompt: str) -> Tuple[str, str]:
    sm = SYSTEM_RE.search(prompt)
    um = USER_RE.search(prompt)
    if um is None:
        raise ValueError("failed to parse user problem from prompt")
    system = sm.group(1).strip() if sm else SYSTEM
    return system, um.group(1).strip()


def normalize_verify(verify: str, verdict: str) -> str:
    """Rewrite closing verdict to PAG wording (wrong/correct)."""
    text = verify.strip()
    # Drop trailing chatml if present
    text = text.replace("<|im_end|>", "").strip()
    # Strip existing closing lines, then append canonical PAG closer
    text = re.sub(
        r"(?:\n|^)\s*(?:Therefore,\s*)?the answer is (?:incorrect|correct|wrong)\.?\s*$",
        "",
        text,
        flags=re.I | re.M,
    ).rstrip()
    if verdict == "incorrect":
        closer = "The answer is wrong."
    elif verdict == "correct":
        closer = "The answer is correct."
    else:
        closer = "The answer is wrong."
    if text:
        return f"{text}\n\n{closer}"
    return closer


def detect_verdict(verify: str) -> str:
    vl = verify.lower()
    if "the answer is incorrect" in vl or re.search(r"\banswer is incorrect\b", vl):
        return "incorrect"
    if "the answer is wrong" in vl or re.search(r"\banswer is wrong\b", vl):
        return "incorrect"
    if "cannot verify" in vl:
        return "cannot"
    if "the answer is correct" in vl or re.search(r"\banswer is correct\b", vl):
        return "correct"
    return "unknown"


def parse_s2r_answer(answer: str) -> Dict[str, Any]:
    a = answer.replace("<|im_end|>", "").strip()
    if WAIT_BRIDGE not in a:
        raise ValueError("missing Wait bridge")
    y0, rest = a.split(WAIT_BRIDGE, 1)
    y0 = y0.strip()
    rest = rest.strip()
    if RETRY_BRIDGE in rest:
        verify_raw, after_retry = rest.split(RETRY_BRIDGE, 1)
        verify_raw = verify_raw.strip()
        # S2R answers may nest further Wait/retry loops; keep the *final* revision
        # solution only (drop trailing self-verify monologue).
        rectify = after_retry.strip()
        if RETRY_BRIDGE in rectify:
            rectify = rectify.rsplit(RETRY_BRIDGE, 1)[-1].strip()
        if WAIT_BRIDGE in rectify:
            rectify = rectify.split(WAIT_BRIDGE, 1)[0].strip()
        path = "incorrect_path"
    else:
        verify_raw = rest
        rectify = None
        path = "correct_path"
    verdict = detect_verdict(verify_raw)
    # Prefer path signal if verdict ambiguous
    if verdict == "unknown":
        verdict = "incorrect" if path == "incorrect_path" else "correct"
    verify = normalize_verify(verify_raw, verdict)
    return {
        "y0": y0,
        "verify_raw": verify_raw,
        "verify": verify,
        "rectify": rectify,
        "verdict": verdict,
        "path": path,
    }


def msg(role: str, content: str, loss: Optional[int] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"role": role, "content": content}
    if loss is not None:
        d["loss_mask"] = [loss]
    return d


def build_verify_messages(system: str, problem: str, y0: str, verify: str) -> List[Dict[str, Any]]:
    return [
        msg("system", system),
        msg("user", problem),
        msg("assistant", y0, loss=0),
        msg("user", VERIFY_TEMPLATE),
        msg("assistant", verify, loss=1),
    ]


def build_rectify_messages(
    system: str, problem: str, y0: str, verify: str, rectify: str
) -> List[Dict[str, Any]]:
    return [
        msg("system", system),
        msg("user", problem),
        msg("assistant", y0, loss=0),
        msg("user", VERIFY_TEMPLATE),
        msg("assistant", verify, loss=0),
        msg("user", REGENERATE_TEMPLATE),
        msg("assistant", rectify, loss=1),
    ]


def convert_one(ex: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    try:
        system, problem = extract_problem(ex["prompt"])
        parsed = parse_s2r_answer(ex["answer"])
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "error": str(e)}
    row = {
        "idx": idx,
        "problem": problem,
        "system": system,
        "y0": parsed["y0"],
        "verify": parsed["verify"],
        "verify_raw": parsed["verify_raw"],
        "rectify": parsed["rectify"],
        "verdict": parsed["verdict"],
        "path": parsed["path"],
    }
    return row


def to_verify_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "idx": row["idx"],
        "split": "verify",
        "verdict": row["verdict"],
        "messages": build_verify_messages(
            row["system"], row["problem"], row["y0"], row["verify"]
        ),
    }


def to_rectify_record(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not row.get("rectify"):
        return None
    if row["verdict"] not in ("incorrect", "unknown"):
        # Still allow incorrect_path with normalized wrong closer
        if row["path"] != "incorrect_path":
            return None
    return {
        "idx": row["idx"],
        "split": "rectify",
        "verdict": row["verdict"],
        "messages": build_rectify_messages(
            row["system"], row["problem"], row["y0"], row["verify"], row["rectify"]
        ),
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def split_train_val(
    rows: List[Dict[str, Any]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n_val = int(round(len(rows) * val_ratio))
    val_set = set(idxs[:n_val])
    train, val = [], []
    for i, r in enumerate(rows):
        (val if i in val_set else train).append(r)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/data/yuranli/LLM/2026.04/github_references/S2R/data/train_data/"
            "sft_qwen2.5_math_7B.json"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("datasets/sft"),
    )
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    parsed_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw):
        row = convert_one(ex, i)
        assert row is not None
        if "error" in row:
            errors.append(row)
        else:
            parsed_rows.append(row)

    verify_rows = [to_verify_record(r) for r in parsed_rows]
    rectify_rows = []
    for r in parsed_rows:
        rec = to_rectify_record(r)
        if rec is not None:
            rectify_rows.append(rec)
    mixed_rows = verify_rows + rectify_rows

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Full dumps
    write_jsonl(out / "parsed_s2r.jsonl", parsed_rows)
    write_jsonl(out / "sft_verify.jsonl", verify_rows)
    write_jsonl(out / "sft_rectify.jsonl", rectify_rows)
    write_jsonl(out / "sft_mixed.jsonl", mixed_rows)
    write_parquet(out / "sft_verify.parquet", verify_rows)
    write_parquet(out / "sft_rectify.parquet", rectify_rows)
    write_parquet(out / "sft_mixed.parquet", mixed_rows)

    # Train/val splits (by row, independent per split)
    for name, rows in [
        ("verify", verify_rows),
        ("rectify", rectify_rows),
        ("mixed", mixed_rows),
    ]:
        train, val = split_train_val(rows, args.val_ratio, args.seed)
        write_parquet(out / f"sft_{name}_train.parquet", train)
        write_parquet(out / f"sft_{name}_val.parquet", val)
        write_jsonl(out / f"sft_{name}_train.jsonl", train)
        write_jsonl(out / f"sft_{name}_val.jsonl", val)

    stats = {
        "input": str(args.input),
        "n_raw": len(raw),
        "n_parsed": len(parsed_rows),
        "n_errors": len(errors),
        "n_verify": len(verify_rows),
        "n_rectify": len(rectify_rows),
        "n_mixed": len(mixed_rows),
        "verdict_counts": {},
        "path_counts": {},
        "errors": errors[:20],
        "templates": {
            "verify": VERIFY_TEMPLATE,
            "regenerate": REGENERATE_TEMPLATE,
        },
    }
    for r in parsed_rows:
        stats["verdict_counts"][r["verdict"]] = stats["verdict_counts"].get(r["verdict"], 0) + 1
        stats["path_counts"][r["path"]] = stats["path_counts"].get(r["path"], 0) + 1

    (out / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if errors:
        write_jsonl(out / "conversion_errors.jsonl", errors)

    print(json.dumps({k: stats[k] for k in [
        "n_raw", "n_parsed", "n_errors", "n_verify", "n_rectify", "n_mixed",
        "verdict_counts", "path_counts",
    ]}, indent=2))
    print(f"wrote outputs under {out.resolve()}")


if __name__ == "__main__":
    main()
