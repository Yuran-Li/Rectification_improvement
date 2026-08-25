#!/usr/bin/env python3
"""Compare full_regen openings vs correct_prefix / prefix_rewrite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def head(s: str, n: int = 400) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "\n...[truncated]..."


def prefix_match_stats(prefix: str, text: str) -> dict:
    p = (prefix or "").strip()
    t = (text or "").lstrip()
    if not p:
        return {"exact_start": False, "overlap_chars": 0, "overlap_ratio": 0.0}
    exact = t.startswith(p) or t.startswith(p + "\n")
    # longest common prefix length
    m = 0
    for a, b in zip(p, t):
        if a != b:
            break
        m += 1
    return {
        "exact_start": exact,
        "overlap_chars": m,
        "overlap_ratio": round(m / max(len(p), 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--feedback", default="localization_analysis_plan")
    ap.add_argument("--idx", type=int, default=None, help="print one example in detail")
    ap.add_argument("--n_show", type=int, default=3)
    ap.add_argument("--head_chars", type=int, default=500)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    fr_key = f"{args.feedback}__full_regen"
    pr_key = f"{args.feedback}__prefix_rewrite"

    stats = {"n": 0, "exact_start": 0, "overlap_ge_50": 0, "overlap_ge_80": 0, "overlap_ge_95": 0}
    details = []
    for r in rows:
        res = r.get("results") or {}
        if fr_key not in res:
            continue
        pref = r.get("correct_prefix") or res[fr_key].get("correct_prefix") or ""
        fr_text = (res[fr_key].get("texts") or [""])[0]
        pr_text = (res.get(pr_key, {}).get("texts") or [""])[0] if pr_key in res else ""
        pr_raw = (res.get(pr_key, {}).get("texts_raw") or [""])[0] if pr_key in res else ""
        m = prefix_match_stats(pref, fr_text)
        stats["n"] += 1
        stats["exact_start"] += int(m["exact_start"])
        stats["overlap_ge_50"] += int(m["overlap_ratio"] >= 0.5)
        stats["overlap_ge_80"] += int(m["overlap_ratio"] >= 0.8)
        stats["overlap_ge_95"] += int(m["overlap_ratio"] >= 0.95)
        details.append(
            {
                "idx": r.get("idx"),
                "t_star": r.get("t_star"),
                "fr_acc": (res[fr_key].get("corrects") or [False])[0],
                "pr_acc": (res.get(pr_key, {}).get("corrects") or [False])[0] if pr_key in res else None,
                "fr_ppr": (res[fr_key].get("pprs") or [False])[0],
                **m,
                "prefix": pref,
                "full_regen": fr_text,
                "prefix_rewrite_full": pr_text,
                "prefix_rewrite_raw_suffix": pr_raw,
            }
        )

    n = max(stats["n"], 1)
    print(f"feedback={args.feedback}  n={stats['n']}")
    print(
        "full_regen starts with correct_prefix: "
        f"{100*stats['exact_start']/n:.1f}%  "
        f"(overlap≥50%: {100*stats['overlap_ge_50']/n:.1f}%, "
        f"≥80%: {100*stats['overlap_ge_80']/n:.1f}%, "
        f"≥95%: {100*stats['overlap_ge_95']/n:.1f}%)"
    )

    show = details
    if args.idx is not None:
        show = [d for d in details if d["idx"] == args.idx]
    else:
        show = details[: args.n_show]

    for d in show:
        print("\n" + "=" * 80)
        print(
            f"idx={d['idx']}  t*={d['t_star']}  "
            f"fr_acc={d['fr_acc']} pr_acc={d['pr_acc']}  "
            f"fr_exact_prefix={d['exact_start']} overlap={d['overlap_ratio']}"
        )
        print("\n--- correct_prefix ---")
        print(head(d["prefix"], args.head_chars))
        print("\n--- full_regen[0] opening ---")
        print(head(d["full_regen"], args.head_chars))
        print("\n--- prefix_rewrite texts_raw[0] (suffix only) ---")
        print(head(d["prefix_rewrite_raw_suffix"], args.head_chars))
        print("\n--- prefix_rewrite texts[0] (stitched) opening ---")
        print(head(d["prefix_rewrite_full"], args.head_chars))


if __name__ == "__main__":
    main()
