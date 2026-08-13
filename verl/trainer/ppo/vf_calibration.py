"""Branching calibration for V_F (Step-4).

Protocol
--------
1. Sample decision states s ∈ {s^V, s^R} (50–100 is enough).
2. Record critic V_F(s)=σ(logit).
3. Freeze the prefix at s; draw K∈[4,8] on-policy continuations.
4. Each branch yields G_F=1[z_final=0] (final answer wrong).
5. Empirical p̂_fail(s) = (1/K) Σ_k G_F^{(k)}.
6. Compare V_F(s) vs p̂_fail(s): ECE, Brier, correlation, reliability bins.

Training does **not** need per-state branching; this audit is diagnostic only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BranchState:
    """One decision state with critic score and branch failure labels."""

    state_id: str
    role: str  # "V" or "R"
    vf: float  # σ(V_F logit) ∈ [0,1]
    branch_fail: Sequence[int]  # 1 = final answer wrong on that continuation
    meta: Optional[Dict[str, Any]] = None

    def p_hat(self) -> float:
        arr = np.asarray(self.branch_fail, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    def n_branch(self) -> int:
        return int(len(self.branch_fail))


def reliability_bins(
    vf: np.ndarray,
    p_hat: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Equal-width bins on V_F; report mean V_F and mean p̂_fail per bin."""
    assert vf.shape == p_hat.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            m = (vf >= lo) & (vf < hi)
        else:
            m = (vf >= lo) & (vf <= hi)
        if not m.any():
            rows.append(
                {
                    "bin": float(b),
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0.0,
                    "vf_mean": float("nan"),
                    "p_hat_mean": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        vm = float(vf[m].mean())
        pm = float(p_hat[m].mean())
        rows.append(
            {
                "bin": float(b),
                "lo": float(lo),
                "hi": float(hi),
                "n": float(m.sum()),
                "vf_mean": vm,
                "p_hat_mean": pm,
                "gap": abs(vm - pm),
            }
        )
    return rows


def expected_calibration_error(
    vf: np.ndarray,
    p_hat: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE = Σ_b (n_b/N) |mean V_F - mean p̂| (p̂ as 'accuracy' of fail event)."""
    bins = reliability_bins(vf, p_hat, n_bins=n_bins)
    n = float(len(vf))
    if n <= 0:
        return float("nan")
    ece = 0.0
    for row in bins:
        if row["n"] <= 0:
            continue
        ece += (row["n"] / n) * abs(row["vf_mean"] - row["p_hat_mean"])
    return float(ece)


def brier_score(vf: np.ndarray, p_hat: np.ndarray) -> float:
    """Treat p̂ as soft label for the Bernoulli mean; Brier = mean (V_F - p̂)^2."""
    return float(np.mean((vf - p_hat) ** 2))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def gate_empirical_rates(
    vf: np.ndarray,
    p_hat: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    """Sanity for gate: E[p̂|V_F>ε] should exceed E[p̂|V_F≤ε]."""
    above = vf > eps
    below = ~above
    out = {
        "eps": float(eps),
        "n_above": float(above.sum()),
        "n_below": float(below.sum()),
        "p_fail_given_gate": float(p_hat[above].mean()) if above.any() else float("nan"),
        "p_fail_given_nogate": float(p_hat[below].mean()) if below.any() else float("nan"),
    }
    if above.any() and below.any():
        out["gate_gap"] = out["p_fail_given_gate"] - out["p_fail_given_nogate"]
    else:
        out["gate_gap"] = float("nan")
    return out


def role_split_metrics(states: Sequence[BranchState]) -> Dict[str, Dict[str, float]]:
    by_role: Dict[str, List[BranchState]] = {"V": [], "R": []}
    for s in states:
        key = "V" if str(s.role).upper().startswith("V") else "R"
        by_role[key].append(s)
    out: Dict[str, Dict[str, float]] = {}
    for role, rows in by_role.items():
        if not rows:
            out[role] = {"n": 0.0}
            continue
        vf = np.asarray([r.vf for r in rows], dtype=np.float64)
        ph = np.asarray([r.p_hat() for r in rows], dtype=np.float64)
        out[role] = {
            "n": float(len(rows)),
            "vf_mean": float(vf.mean()),
            "p_hat_mean": float(ph.mean()),
            "corr": _safe_corr(vf, ph),
            "brier": brier_score(vf, ph),
            "ece": expected_calibration_error(vf, ph, n_bins=5),
        }
    return out


@dataclass
class CalibrationReport:
    n_states: int
    n_sv: int
    n_sr: int
    mean_branches: float
    vf_mean: float
    p_hat_mean: float
    corr: float
    brier: float
    ece: float
    vf_std: float
    p_hat_std: float
    gate: Dict[str, float]
    by_role: Dict[str, Dict[str, float]]
    bins: List[Dict[str, float]]
    pass_checks: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_branching_calibration(
    states: Sequence[BranchState],
    *,
    eps: float = 0.3,
    n_bins: int = 10,
    ece_max: float = 0.15,
    corr_min: float = 0.3,
    gate_gap_min: float = 0.05,
) -> CalibrationReport:
    """Aggregate branching audit + pass/fail checklist for Step-4."""
    assert states, "need at least one BranchState"
    vf = np.asarray([s.vf for s in states], dtype=np.float64)
    ph = np.asarray([s.p_hat() for s in states], dtype=np.float64)
    n_branch = np.asarray([s.n_branch() for s in states], dtype=np.float64)
    n_sv = sum(1 for s in states if str(s.role).upper().startswith("V"))
    n_sr = len(states) - n_sv

    ece = expected_calibration_error(vf, ph, n_bins=n_bins)
    brier = brier_score(vf, ph)
    corr = _safe_corr(vf, ph)
    gate = gate_empirical_rates(vf, ph, eps)
    bins = reliability_bins(vf, ph, n_bins=n_bins)
    by_role = role_split_metrics(states)

    checks = {
        "has_sv_and_sr": n_sv > 0 and n_sr > 0,
        "enough_states": len(states) >= 20,
        "enough_branches": float(n_branch.mean()) >= 3.5,
        "corr_ok": (not np.isnan(corr)) and corr >= corr_min,
        "ece_ok": (not np.isnan(ece)) and ece <= ece_max,
        "gate_gap_ok": (not np.isnan(gate["gate_gap"])) and gate["gate_gap"] >= gate_gap_min,
        "not_collapsed": float(np.std(vf)) > 0.05,
    }
    return CalibrationReport(
        n_states=len(states),
        n_sv=n_sv,
        n_sr=n_sr,
        mean_branches=float(n_branch.mean()),
        vf_mean=float(vf.mean()),
        p_hat_mean=float(ph.mean()),
        corr=float(corr),
        brier=float(brier),
        ece=float(ece),
        vf_std=float(vf.std()),
        p_hat_std=float(ph.std()),
        gate=gate,
        by_role=by_role,
        bins=bins,
        pass_checks=checks,
    )


def simulate_branch_states(
    true_p: Sequence[float],
    *,
    vf: Optional[Sequence[float]] = None,
    k: int = 8,
    roles: Optional[Sequence[str]] = None,
    seed: int = 0,
    noise: float = 0.0,
) -> List[BranchState]:
    """Synthetic branching for unit tests / dry-run.

    If vf is None, use calibrated VF = clip(true_p + N(0,noise)).
    """
    rng = np.random.default_rng(seed)
    true_p = np.asarray(true_p, dtype=np.float64)
    n = len(true_p)
    if vf is None:
        vf_arr = np.clip(true_p + rng.normal(0.0, noise, size=n), 0.0, 1.0)
    else:
        vf_arr = np.asarray(vf, dtype=np.float64)
        assert len(vf_arr) == n
    if roles is None:
        roles = ["V" if i % 2 == 0 else "R" for i in range(n)]
    states: List[BranchState] = []
    for i in range(n):
        fails = rng.binomial(1, float(true_p[i]), size=k).tolist()
        states.append(
            BranchState(
                state_id=f"s{i}",
                role=str(roles[i]),
                vf=float(vf_arr[i]),
                branch_fail=fails,
                meta={"true_p": float(true_p[i])},
            )
        )
    return states


def states_from_jsonl_rows(rows: Sequence[Dict[str, Any]]) -> List[BranchState]:
    """Parse dump rows: {state_id, role, vf, branch_fail: [...]}."""
    out: List[BranchState] = []
    for i, r in enumerate(rows):
        out.append(
            BranchState(
                state_id=str(r.get("state_id", f"row{i}")),
                role=str(r.get("role", "V")),
                vf=float(r["vf"]),
                branch_fail=list(r["branch_fail"]),
                meta=r.get("meta"),
            )
        )
    return out


def format_report(report: CalibrationReport) -> str:
    lines = [
        "=== Step-4 branching calibration ===",
        f"  n_states={report.n_states} (sV={report.n_sv}, sR={report.n_sr})  "
        f"mean_K={report.mean_branches:.1f}",
        f"  vf_mean={report.vf_mean:.3f}±{report.vf_std:.3f}  "
        f"p̂_mean={report.p_hat_mean:.3f}±{report.p_hat_std:.3f}",
        f"  corr(V_F,p̂)={report.corr:.3f}  Brier={report.brier:.4f}  ECE={report.ece:.4f}",
        f"  gate ε={report.gate['eps']}: "
        f"P̂(fail|g=1)={report.gate['p_fail_given_gate']:.3f}  "
        f"P̂(fail|g=0)={report.gate['p_fail_given_nogate']:.3f}  "
        f"gap={report.gate['gate_gap']:.3f}",
        f"  by_role: {report.by_role}",
        "  reliability bins (non-empty):",
    ]
    for b in report.bins:
        if b["n"] <= 0:
            continue
        lines.append(
            f"    [{b['lo']:.1f},{b['hi']:.1f}] n={b['n']:.0f} "
            f"vf={b['vf_mean']:.3f} p̂={b['p_hat_mean']:.3f} |gap|={b['gap']:.3f}"
        )
    lines.append(f"  pass_checks: {report.pass_checks}")
    all_ok = all(report.pass_checks.values())
    lines.append("  VERDICT: " + ("PASS" if all_ok else "FAIL (see pass_checks)"))
    return "\n".join(lines)
