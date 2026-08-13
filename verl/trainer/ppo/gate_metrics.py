"""Step-5 gate sanity: g(s)=1[V_F(s)>ε] routes PPO+BC vs PPO-only.

Notation
--------
Paper / new design:
  F(s)=V_F(s)-ε
  g(s)=1[F(s)>0]=1[V_F(s)>ε]   → need recovery BC
  F≤0 → PPO only; F>0 → PPO + BC

Implementation store (legacy name, inverted):
  feas_gate_* = 1[V_F ≤ ε]       → “feasible / no BC”

This module logs the paper g(·) rates and the critical check
  P(G_F=1 | g=1) > P(G_F=1 | g=0).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def compute_gate_sanity_metrics(
    values_f: torch.Tensor,
    returns_f: torch.Tensor,
    feasibility_mask: torch.Tensor,
    *,
    feasibility_mask_v: Optional[torch.Tensor] = None,
    feasibility_mask_r: Optional[torch.Tensor] = None,
    eps: float = 0.3,
    window_valid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """State-level gate diagnostics on routing positions.

    Args:
        values_f: σ(V_F logits), same shape as responses [B, L]
        returns_f: G_F ∈ {0,1} targets on routing states
        feasibility_mask: s^V ∪ s^R
        feasibility_mask_v / _r: role masks (optional)
        eps: threshold ε in g=1[V_F>ε]
    """
    device = values_f.device
    route = feasibility_mask.bool()
    if window_valid is not None:
        valid = torch.as_tensor(window_valid.astype(np.bool_), device=device)
        route = route & valid.unsqueeze(-1)

    out: Dict[str, float] = {
        "gate/eps": float(eps),
        "gate/n_routing": float(route.sum().item()),
        "gate/gate_verify_rate": float("nan"),
        "gate/gate_rectify_rate": float("nan"),
        "gate/gate_rate": float("nan"),
        "gate/vf_mean_gated": float("nan"),
        "gate/vf_mean_ungated": float("nan"),
        "gate/fail_rate_gated": float("nan"),
        "gate/fail_rate_ungated": float("nan"),
        "gate/fail_gap": float("nan"),
        "gate/sanity_Pfail_g1_gt_g0": 0.0,
        "gate/n_gated": 0.0,
        "gate/n_ungated": 0.0,
        "gate/n_sv": 0.0,
        "gate/n_sr": 0.0,
    }
    if not route.any():
        return out

    vf = values_f.float()
    g_f = returns_f.float()
    g = route & (vf > eps)  # paper g(s)=1[V_F>ε]
    ung = route & (vf <= eps)

    out["gate/gate_rate"] = float(g.sum().item() / route.sum().item())
    out["gate/n_gated"] = float(g.sum().item())
    out["gate/n_ungated"] = float(ung.sum().item())

    if g.any():
        out["gate/vf_mean_gated"] = float(vf[g].mean().item())
        out["gate/fail_rate_gated"] = float(g_f[g].mean().item())
    if ung.any():
        out["gate/vf_mean_ungated"] = float(vf[ung].mean().item())
        out["gate/fail_rate_ungated"] = float(g_f[ung].mean().item())
    if g.any() and ung.any():
        gap = out["gate/fail_rate_gated"] - out["gate/fail_rate_ungated"]
        out["gate/fail_gap"] = float(gap)
        out["gate/sanity_Pfail_g1_gt_g0"] = float(gap > 0.0)

    if feasibility_mask_v is not None:
        mv = feasibility_mask_v.bool()
        if window_valid is not None:
            mv = mv & valid.unsqueeze(-1)
        out["gate/n_sv"] = float(mv.sum().item())
        if mv.any():
            out["gate/gate_verify_rate"] = float((vf[mv] > eps).float().mean().item())
            gv = mv & (vf > eps)
            uv = mv & (vf <= eps)
            if gv.any():
                out["gate/fail_rate_gated_v"] = float(g_f[gv].mean().item())
                out["gate/vf_mean_gated_v"] = float(vf[gv].mean().item())
            if uv.any():
                out["gate/fail_rate_ungated_v"] = float(g_f[uv].mean().item())
                out["gate/vf_mean_ungated_v"] = float(vf[uv].mean().item())

    if feasibility_mask_r is not None:
        mr = feasibility_mask_r.bool()
        if window_valid is not None:
            mr = mr & valid.unsqueeze(-1)
        out["gate/n_sr"] = float(mr.sum().item())
        if mr.any():
            out["gate/gate_rectify_rate"] = float((vf[mr] > eps).float().mean().item())
            gr = mr & (vf > eps)
            ur = mr & (vf <= eps)
            if gr.any():
                out["gate/fail_rate_gated_r"] = float(g_f[gr].mean().item())
                out["gate/vf_mean_gated_r"] = float(vf[gr].mean().item())
            if ur.any():
                out["gate/fail_rate_ungated_r"] = float(g_f[ur].mean().item())
                out["gate/vf_mean_ungated_r"] = float(vf[ur].mean().item())

    return out


def format_gate_sanity(m: Dict[str, float]) -> str:
    return (
        "[gate] "
        f"ε={m.get('gate/eps', float('nan')):.3f} "
        f"gate_v={m.get('gate/gate_verify_rate', float('nan')):.3f} "
        f"gate_r={m.get('gate/gate_rectify_rate', float('nan')):.3f} "
        f"gate_all={m.get('gate/gate_rate', float('nan')):.3f} "
        f"vf|g=1={m.get('gate/vf_mean_gated', float('nan')):.3f} "
        f"vf|g=0={m.get('gate/vf_mean_ungated', float('nan')):.3f} "
        f"P(fail|g=1)={m.get('gate/fail_rate_gated', float('nan')):.3f} "
        f"P(fail|g=0)={m.get('gate/fail_rate_ungated', float('nan')):.3f} "
        f"gap={m.get('gate/fail_gap', float('nan')):.3f} "
        f"ok={bool(m.get('gate/sanity_Pfail_g1_gt_g0', 0.0))}"
    )
