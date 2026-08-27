#!/usr/bin/env python3
"""Curriculum hypothesis retrospective analysis.

Computes per-prompt competence statistics (g, c) from K-rollout JSON dumps
(produced by curriculum_rollout.py OR the existing validation_results/*.json),
then tests whether high curriculum-score prompts at time 0 gain more by time T.

Competences (per prompt, across K rollouts)
-------------------------------------------
  g_i  = #{y0 correct} / K                  generation competence
  c_i  = #{y0=W AND e2e-corrected} / #{y0=W}  end-to-end correction competence
         (NaN when all K y0s are correct; treated as 1.0 for scoring)

Curriculum score
----------------
  s_i = α * g_i*(1-g_i)  +  β * (1-g_i)*c_i*(1-c_i)
  default α=β=1; c_i clamped to 0 when g_i=1.

Output
------
  * 2-D bin table: (g_0 bin) × (c_0 bin)  →  E[Δg], E[Δc], n
  * Score bins: s_0 low/mid/high  →  E[Δg], E[Δc]
  * Spearman ρ(s_0, Δg), ρ(s_0, Δc)
  * JSON + console

Usage
-----
python scripts/curriculum_analysis.py \
    --base  curriculum_rollouts/base_1.5b_math500_k4.json \
    --final curriculum_rollouts/pag400_1.5b_math500_k4.json \
    --out   results/curriculum/1.5b_retrospective.json

# Also accepts the training validation JSON (n=8):
python scripts/curriculum_analysis.py \
    --base  curriculum_rollouts/base_1.5b_math500_k4.json \
    --final validation_results/qwen1.5b_pag_step210_math500_n8_20260823_102937.json \
    --out   results/curriculum/1.5b_step210.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_dump(path: str) -> Dict[str, List[dict]]:
    """Load validation JSON → dict of prompt_key → list[rollout_dict].

    Accepts both curriculum_rollout.py format and validation_results/*.json.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level dict")
    # Top-level may be {data_source: {prompt_key: [rollouts]}} or {prompt_key: [rollouts]}
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict):
        # nested: {data_source: {prompt_key: [rollouts]}}
        merged: Dict[str, List[dict]] = {}
        for ds_dict in data.values():
            for k, v in ds_dict.items():
                if k not in merged:
                    merged[k] = v
                else:
                    merged[k].extend(v)
        return merged
    elif isinstance(first_val, list):
        return data
    else:
        raise ValueError(f"{path}: unexpected structure")


# ---------------------------------------------------------------------------
# Per-prompt competence
# ---------------------------------------------------------------------------

def prompt_competence(rollouts: List[dict]) -> Tuple[float, float, int]:
    """Return (g, c, K).

    g = fraction of y0 correct.
    c = fraction of wrong y0 that were end-to-end self-corrected;
        NaN (→ returned as np.nan) when g=1 (no wrong y0 to correct).
    K = number of rollouts used.
    """
    K = len(rollouts)
    if K == 0:
        return (np.nan, np.nan, 0)

    y0_correct = np.array([float(r["acc_t1"]) >= 0.5 for r in rollouts], dtype=float)
    g = float(y0_correct.mean())

    # c: among wrong y0s, did end-to-end self-correction succeed?
    # "revised" = model said wrong and produced y1; "acc_t2 >= 0.5" = y1 correct
    wrong_mask = y0_correct < 0.5
    n_wrong = int(wrong_mask.sum())
    if n_wrong == 0:
        c = np.nan
    else:
        corrected = 0
        for i, r in enumerate(rollouts):
            if not wrong_mask[i]:
                continue
            revised = bool(r.get("revised", False))
            acc_t2 = float(r.get("acc_t2", -1.0))
            if revised and acc_t2 >= 0.5:
                corrected += 1
        c = float(corrected) / float(n_wrong)

    return (g, c, K)


def curriculum_score(g: float, c: float, alpha: float = 1.0, beta: float = 1.0) -> float:
    """s = α·g(1-g) + β·(1-g)·c(1-c).  c=NaN (g=1) → c clamped to 1 → second term=0."""
    c_eff = 1.0 if np.isnan(c) else float(c)
    return alpha * g * (1.0 - g) + beta * (1.0 - g) * c_eff * (1.0 - c_eff)


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

G_BINS = [
    ("g=0",   lambda g: g == 0.0),
    ("g_low", lambda g: 0.0 < g < 0.5),
    ("g=0.5", lambda g: g == 0.5),
    ("g_hi",  lambda g: 0.5 < g < 1.0),
    ("g=1",   lambda g: g == 1.0),
]

C_BINS = [
    ("no_wrong",  lambda c: np.isnan(c)),     # g=1, no wrongs
    ("c=0",       lambda c: (not np.isnan(c)) and c == 0.0),
    ("c_low",     lambda c: (not np.isnan(c)) and 0.0 < c < 0.5),
    ("c=0.5",     lambda c: (not np.isnan(c)) and c == 0.5),
    ("c_hi",      lambda c: (not np.isnan(c)) and 0.5 < c < 1.0),
    ("c=1",       lambda c: (not np.isnan(c)) and c == 1.0),
]

S_TERTILES = ("s_low", "s_mid", "s_high")


def _mean_or_none(arr: List[float]) -> Optional[float]:
    return None if not arr else float(np.mean(arr))


def bin_2d(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """2D table: g_bin × c_bin → {E_delta_g, E_delta_c, n}."""
    table: Dict[str, Dict[str, Any]] = {}
    for g_label, g_fn in G_BINS:
        for c_label, c_fn in C_BINS:
            subset = [rec for rec in records if g_fn(rec["g0"]) and c_fn(rec["c0"])]
            n = len(subset)
            entry: Dict[str, Any] = {"n": n}
            if n > 0:
                entry["E_delta_g"] = _mean_or_none([r["delta_g"] for r in subset])
                entry["E_delta_c_exc_nan"] = _mean_or_none(
                    [r["delta_c"] for r in subset if not np.isnan(r["delta_c"])]
                )
                entry["E_g0"] = _mean_or_none([r["g0"] for r in subset])
                entry["E_c0_exc_nan"] = _mean_or_none(
                    [r["c0"] for r in subset if not np.isnan(r["c0"])]
                )
            table[f"{g_label}|{c_label}"] = entry
    return table


def score_tertile_analysis(records: List[Dict[str, Any]], alpha: float, beta: float) -> Dict[str, Any]:
    s0 = np.array([curriculum_score(r["g0"], r["c0"], alpha, beta) for r in records])
    if len(s0) < 3:
        return {}
    t33, t66 = float(np.percentile(s0, 33.3)), float(np.percentile(s0, 66.6))
    bins: Dict[str, List[Dict[str, Any]]] = {"s_low": [], "s_mid": [], "s_high": []}
    for rec, sv in zip(records, s0):
        if sv <= t33:
            bins["s_low"].append(rec)
        elif sv <= t66:
            bins["s_mid"].append(rec)
        else:
            bins["s_high"].append(rec)
    out: Dict[str, Any] = {"thresholds": {"t33": t33, "t66": t66}}
    for label, subset in bins.items():
        n = len(subset)
        dg = [r["delta_g"] for r in subset]
        dc = [r["delta_c"] for r in subset if not np.isnan(r["delta_c"])]
        out[label] = {
            "n": n,
            "E_s0": _mean_or_none([float(sv) for rec, sv in zip(records, s0) if rec in subset]),
            "E_delta_g": _mean_or_none(dg),
            "E_delta_c": _mean_or_none(dc),
        }
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",  required=True, help="JSON dump for base checkpoint")
    ap.add_argument("--final", required=True, help="JSON dump for final checkpoint")
    ap.add_argument("--out",   required=True, help="output JSON path")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--label_base",  default="base",  help="label for base ckpt")
    ap.add_argument("--label_final", default="final", help="label for final ckpt")
    args = ap.parse_args()

    print(f"Loading base:  {args.base}")
    base_dump  = load_dump(args.base)
    print(f"Loading final: {args.final}")
    final_dump = load_dump(args.final)

    # Match prompts
    common_keys = sorted(set(base_dump.keys()) & set(final_dump.keys()))
    print(f"Matched prompts: {len(common_keys)}  "
          f"(base={len(base_dump)}, final={len(final_dump)})")
    if not common_keys:
        print("ERROR: no common prompt keys. Check that both dumps use the same dataset.")
        return

    records: List[Dict[str, Any]] = []
    for key in common_keys:
        g0, c0, K0 = prompt_competence(base_dump[key])
        gT, cT, KT = prompt_competence(final_dump[key])
        if np.isnan(g0) or np.isnan(gT):
            continue
        delta_g = gT - g0
        delta_c = (cT - c0) if (not np.isnan(cT) and not np.isnan(c0)) else np.nan
        records.append({
            "prompt_key": key[:80],
            "g0": g0, "c0": c0, "K0": K0,
            "gT": gT, "cT": cT, "KT": KT,
            "delta_g": delta_g, "delta_c": delta_c,
            "s0": curriculum_score(g0, c0, args.alpha, args.beta),
        })

    print(f"Records with valid g0+gT: {len(records)}")

    # -----------------------------------------------------------------------
    # 2D bin table
    # -----------------------------------------------------------------------
    table_2d = bin_2d(records)

    # -----------------------------------------------------------------------
    # Score tertile analysis
    # -----------------------------------------------------------------------
    tertile_stats = score_tertile_analysis(records, args.alpha, args.beta)

    # -----------------------------------------------------------------------
    # Spearman correlations
    # -----------------------------------------------------------------------
    s0_arr  = np.array([r["s0"] for r in records])
    dg_arr  = np.array([r["delta_g"] for r in records])
    dc_arr  = np.array([r["delta_c"] for r in records], dtype=float)

    try:
        rho_g = spearman_corr(s0_arr, dg_arr)
        rho_c = spearman_corr(s0_arr, dc_arr)
    except ImportError:
        rho_g = rho_c = None
        print("scipy not found; skipping Spearman. Install with: pip install scipy")

    # -----------------------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------------------
    agg = {
        "n_prompts": len(records),
        "label_base":  args.label_base,
        "label_final": args.label_final,
        "alpha": args.alpha,
        "beta":  args.beta,
        "overall": {
            "E_g0": _mean_or_none([r["g0"] for r in records]),
            "E_gT": _mean_or_none([r["gT"] for r in records]),
            "E_delta_g": _mean_or_none([r["delta_g"] for r in records]),
            "E_c0": _mean_or_none([r["c0"] for r in records if not np.isnan(r["c0"])]),
            "E_cT": _mean_or_none([r["cT"] for r in records if not np.isnan(r.get("cT", np.nan))]),
            "E_delta_c": _mean_or_none([r["delta_c"] for r in records if not np.isnan(r["delta_c"])]),
            "E_s0": _mean_or_none([r["s0"] for r in records]),
        },
        "spearman": {
            "rho_s0_delta_g": rho_g,
            "rho_s0_delta_c": rho_c,
            "interpretation": (
                "> 0 means higher curriculum score at base → more gain after RL. "
                "Monotonic trend validates the recoverable-frontier hypothesis."
            ),
        },
        "score_tertiles": tertile_stats,
        "bin_2d": table_2d,
    }

    # -----------------------------------------------------------------------
    # Console pretty-print
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  Curriculum retrospective: {args.label_base}  →  {args.label_final}")
    print("=" * 70)
    ov = agg["overall"]
    print(f"  n_prompts={agg['n_prompts']}")
    print(f"  E[g0]={ov['E_g0']:.3f}  E[gT]={ov['E_gT']:.3f}  E[Δg]={ov['E_delta_g']:+.3f}")
    c0_str = f"{ov['E_c0']:.3f}" if ov['E_c0'] is not None else "N/A"
    cT_str = f"{ov['E_cT']:.3f}" if ov['E_cT'] is not None else "N/A"
    dc_str = f"{ov['E_delta_c']:+.3f}" if ov['E_delta_c'] is not None else "N/A"
    print(f"  E[c0]={c0_str}  E[cT]={cT_str}  E[Δc]={dc_str}")
    if rho_g is not None:
        print(f"\n  Spearman ρ(s0, Δg) = {rho_g:+.3f}   ρ(s0, Δc) = {rho_c:+.3f}")

    print("\n  Score tertiles (s0 = α·g(1-g) + β·(1-g)·c(1-c)):")
    for label in S_TERTILES:
        t = tertile_stats.get(label, {})
        n = t.get("n", 0)
        edg = t.get("E_delta_g")
        edc = t.get("E_delta_c")
        dg_s = f"{edg:+.3f}" if edg is not None else "N/A"
        dc_s = f"{edc:+.3f}" if edc is not None else "N/A"
        print(f"    {label:8s}  n={n:4d}  E[Δg]={dg_s}  E[Δc]={dc_s}")

    print("\n  2-D bin table  (g0 bin × c0 bin  →  E[Δg], n):")
    g_labels_short = ["g=0", "g_low", "g=0.5", "g_hi", "g=1"]
    c_labels_short = ["no_wrong", "c=0", "c_low", "c=0.5", "c_hi", "c=1"]
    header = f"  {'':10s}" + "".join(f"{cl:>12s}" for cl in c_labels_short)
    print(header)
    for gl in g_labels_short:
        row_str = f"  {gl:10s}"
        for cl in c_labels_short:
            cell = table_2d.get(f"{gl}|{cl}", {})
            n = cell.get("n", 0)
            edg = cell.get("E_delta_g")
            if n == 0:
                row_str += f"{'—':>12s}"
            else:
                edg_s = f"{edg:+.2f}" if edg is not None else "N/A"
                row_str += f"  {edg_s}({n:3d})"
        print(row_str)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
