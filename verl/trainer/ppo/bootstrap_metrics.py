"""Step-6 self-bootstrap coverage (GPT off).

Self-bootstrap is **problem-conditioned positive replay**:
  same uid / problem x, sibling traj s_B provides a_B^+.
It is **not** exact state-conditioned expert action a_E | s_A.

Coverage:
  P(∃ successful sibling | V_F(s)>ε)
split by verify-state s^V vs rectify-state s^R.

Successful sibling (primary) = z_final=1 (G_F=0 on that sibling).
Transferable expert (secondary) = sibling already has role expert_token_mask
(true-reject + I→C in pag.py) — what transfer_same_uid_bootstrap actually copies.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


def _row_valid(n: int, window_valid: Optional[np.ndarray], device) -> torch.Tensor:
    if window_valid is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    return torch.as_tensor(np.asarray(window_valid).astype(np.bool_), device=device)


def _row_vf(values_f: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is None:
        return None
    m = mask.bool()
    has = m.any(dim=-1)
    denom = m.float().sum(dim=-1).clamp(min=1.0)
    v = (values_f.float() * m.float()).sum(dim=-1) / denom
    return torch.where(has, v, torch.zeros_like(v)), has


def _success_final(
    returns_f: Optional[torch.Tensor],
    route: Optional[torch.Tensor],
    acc_final: Optional[np.ndarray],
    n: int,
    device,
) -> torch.Tensor:
    """z_final=1 ⇔ G_F=0 on routing states (shared on a traj)."""
    out = torch.zeros(n, dtype=torch.bool, device=device)
    if acc_final is not None:
        out = torch.as_tensor(
            np.asarray(acc_final, dtype=np.float32) >= 0.5, device=device
        )
        return out
    if returns_f is None or route is None:
        return out
    m = route.bool()
    has = m.any(dim=-1)
    # G_F is constant on routing tokens; mean==0 → success
    g = torch.zeros(n, device=device, dtype=returns_f.dtype)
    for i in range(n):
        if has[i]:
            g[i] = returns_f[i][m[i]].mean()
    return has & (g < 0.5)


def compute_bootstrap_coverage(
    uids,
    *,
    gated_v: torch.Tensor,
    gated_r: torch.Tensor,
    success_final: torch.Tensor,
    success_expert_v: Optional[torch.Tensor] = None,
    success_expert_r: Optional[torch.Tensor] = None,
    window_valid: Optional[np.ndarray] = None,
    window_index: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """P(∃ sibling with z_final=1 | g=1) by role. Sibling ≠ self, same uid."""
    n = int(gated_v.numel())
    device = gated_v.device
    valid = _row_valid(n, window_valid, device)
    gated_v = gated_v.bool() & valid
    gated_r = gated_r.bool() & valid
    success_final = success_final.bool() & valid
    if success_expert_v is None:
        success_expert_v = torch.zeros(n, dtype=torch.bool, device=device)
    if success_expert_r is None:
        success_expert_r = torch.zeros(n, dtype=torch.bool, device=device)
    success_expert_v = success_expert_v.bool() & valid
    success_expert_r = success_expert_r.bool() & valid

    nan = float("nan")
    out: Dict[str, float] = {
        "bootstrap/coverage": nan,
        "bootstrap/coverage_v": nan,
        "bootstrap/coverage_r": nan,
        "bootstrap/coverage_expert_v": nan,
        "bootstrap/coverage_expert_r": nan,
        "bootstrap/n_gated_v": float(gated_v.sum().item()),
        "bootstrap/n_gated_r": float(gated_r.sum().item()),
        "bootstrap/n_gated": float((gated_v | gated_r).sum().item()),
        "bootstrap/n_uncovered_v": 0.0,
        "bootstrap/n_uncovered_r": 0.0,
        "bootstrap/gpt_needed_v": nan,  # 1-coverage_v among gated s^V
        "bootstrap/gpt_needed_r": nan,
        "bootstrap/n_success_final": float(success_final.sum().item()),
        "bootstrap/n_expert_v": float(success_expert_v.sum().item()),
        "bootstrap/n_expert_r": float(success_expert_r.sum().item()),
    }

    groups: Dict[object, List[int]] = defaultdict(list)
    uid_list = list(uids)
    for i in range(n):
        if not bool(valid[i]):
            continue
        key = uid_list[i]
        if window_index is not None:
            key = (key, int(window_index[i]))
        groups[key].append(i)

    def _cov(gated: torch.Tensor, sib_ok: torch.Tensor) -> tuple[float, float]:
        idx = torch.where(gated)[0].tolist()
        if not idx:
            return nan, 0.0
        hit = 0
        for i in idx:
            key = uid_list[i]
            if window_index is not None:
                key = (key, int(window_index[i]))
            sibs = [j for j in groups.get(key, []) if j != i]
            if any(bool(sib_ok[j]) for j in sibs):
                hit += 1
        return float(hit / len(idx)), float(len(idx) - hit)

    cov_v, un_v = _cov(gated_v, success_final)
    cov_r, un_r = _cov(gated_r, success_final)
    cov_all, _ = _cov(gated_v | gated_r, success_final)
    cov_ev, _ = _cov(gated_v, success_expert_v)
    cov_er, _ = _cov(gated_r, success_expert_r)

    out["bootstrap/coverage"] = cov_all
    out["bootstrap/coverage_v"] = cov_v
    out["bootstrap/coverage_r"] = cov_r
    out["bootstrap/coverage_expert_v"] = cov_ev
    out["bootstrap/coverage_expert_r"] = cov_er
    out["bootstrap/n_uncovered_v"] = un_v
    out["bootstrap/n_uncovered_r"] = un_r
    if gated_v.any():
        out["bootstrap/gpt_needed_v"] = 1.0 - cov_v
    if gated_r.any():
        out["bootstrap/gpt_needed_r"] = 1.0 - cov_r
    return out


def bootstrap_coverage_from_batch(batch, eps: float = 0.3) -> Dict[str, float]:
    """Read V_F / G_F / uid from a DataProto-like batch (after gates)."""
    empty = compute_bootstrap_coverage(
        [],
        gated_v=torch.zeros(0, dtype=torch.bool),
        gated_r=torch.zeros(0, dtype=torch.bool),
        success_final=torch.zeros(0, dtype=torch.bool),
    )
    if "uid" not in batch.non_tensor_batch:
        return empty
    if "values_f" not in batch.batch:
        return empty

    vf = batch.batch["values_f"]
    device = vf.device
    B = vf.size(0)
    wv = batch.non_tensor_batch.get("window_valid", None)
    wi = batch.non_tensor_batch.get("window_index", None)

    # Role V_F: prefer row scalars from _compute_feasibility_gates
    if "v_f_state_v" in batch.batch and "feasibility_mask_v" in batch.batch:
        v_v = batch.batch["v_f_state_v"].float()
        has_v = batch.batch["feasibility_mask_v"].bool().any(dim=-1)
    else:
        pair = _row_vf(vf, batch.batch.get("feasibility_mask_v"))
        if pair is None:
            v_v = torch.zeros(B, device=device)
            has_v = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            v_v, has_v = pair

    if "v_f_state_r" in batch.batch and "feasibility_mask_r" in batch.batch:
        v_r = batch.batch["v_f_state_r"].float()
        has_r = batch.batch["feasibility_mask_r"].bool().any(dim=-1)
    else:
        pair = _row_vf(vf, batch.batch.get("feasibility_mask_r"))
        if pair is None:
            v_r = torch.zeros(B, device=device)
            has_r = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            v_r, has_r = pair

    gated_v = has_v & (v_v > eps)
    gated_r = has_r & (v_r > eps)

    route = batch.batch.get("feasibility_mask") if "feasibility_mask" in batch.batch else None
    returns_f = None
    if "feasibility_returns" in batch.batch:
        returns_f = batch.batch["feasibility_returns"]
    elif "returns_f" in batch.batch:
        returns_f = batch.batch["returns_f"]
    acc_final = batch.non_tensor_batch.get("acc_final", None)
    success_final = _success_final(returns_f, route, acc_final, B, device)

    ev = batch.batch.get("expert_token_mask_v")
    er = batch.batch.get("expert_token_mask_r")
    success_expert_v = ev.reshape(B, -1).any(dim=-1) if ev is not None else None
    success_expert_r = er.reshape(B, -1).any(dim=-1) if er is not None else None

    return compute_bootstrap_coverage(
        batch.non_tensor_batch["uid"],
        gated_v=gated_v,
        gated_r=gated_r,
        success_final=success_final,
        success_expert_v=success_expert_v,
        success_expert_r=success_expert_r,
        window_valid=wv,
        window_index=wi,
    )


def format_bootstrap_coverage(m: Dict[str, float]) -> str:
    def _f(k: str) -> str:
        v = m.get(k, float("nan"))
        if v != v:  # nan
            return "nan"
        return f"{v:.3f}"

    return (
        "[bootstrap] s_B→a_B+ (NOT s_A→a_E) "
        f"cov={_f('bootstrap/coverage')} "
        f"cov_v={_f('bootstrap/coverage_v')} "
        f"cov_r={_f('bootstrap/coverage_r')} "
        f"n_g_v={m.get('bootstrap/n_gated_v', 0):.0f} "
        f"n_g_r={m.get('bootstrap/n_gated_r', 0):.0f} "
        f"uncovered_v={m.get('bootstrap/n_uncovered_v', 0):.0f} "
        f"uncovered_r={m.get('bootstrap/n_uncovered_r', 0):.0f} "
        f"gpt_needed_v={_f('bootstrap/gpt_needed_v')} "
        f"gpt_needed_r={_f('bootstrap/gpt_needed_r')} "
        f"expert_cov_v={_f('bootstrap/coverage_expert_v')} "
        f"expert_cov_r={_f('bootstrap/coverage_expert_r')}"
    )
