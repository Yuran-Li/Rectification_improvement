"""Replay buffer + same-uid bootstrap transfer for expert BC (no API)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto


def log_problem_level_fail_rate(batch: DataProto) -> Dict[str, float]:
    """Diagnostic only: same-UID mean of traj fails ≈ P(fail|x). Does NOT train V_F.

    State-level V_F uses per-state on-policy G_i^F; do not overwrite those targets
    with this aggregate. Optional branching is for calibration, not this helper.
    """
    metrics = {
        "feasibility/problem_p_groups": 0.0,
        "feasibility/problem_p_mean": 0.0,
        "feasibility/problem_p_trajs_per_group": 0.0,
    }
    if "feasibility_mask" not in batch.batch or "feasibility_returns" not in batch.batch:
        return metrics
    if "uid" not in batch.non_tensor_batch:
        return metrics

    fm = batch.batch["feasibility_mask"].bool()
    fr = batch.batch["feasibility_returns"]
    B = fr.size(0)
    uids = batch.non_tensor_batch["uid"]
    window_index = batch.non_tensor_batch.get("window_index", None)
    window_valid = batch.non_tensor_batch.get("window_valid", None)

    row_mc = torch.zeros(B, device=fr.device, dtype=fr.dtype)
    for i in range(B):
        m = fm[i]
        if m.any():
            row_mc[i] = fr[i][m].mean()

    groups: Dict[object, List[int]] = defaultdict(list)
    for i in range(B):
        groups[uids[i]].append(i)

    p_vals = []
    trajs_per = []
    for idxs in groups.values():
        if window_index is not None:
            traj_rows = [i for i in idxs if int(window_index[i]) == 0]
            if not traj_rows:
                traj_rows = list(idxs)
        else:
            traj_rows = list(idxs)
        if window_valid is not None:
            valid_traj = [i for i in traj_rows if bool(window_valid[i])]
            if valid_traj:
                traj_rows = valid_traj
        p = float(row_mc[traj_rows].mean().item()) if traj_rows else 0.0
        p_vals.append(p)
        trajs_per.append(float(len(traj_rows)))
    metrics["feasibility/problem_p_groups"] = float(len(groups))
    metrics["feasibility/problem_p_mean"] = float(np.mean(p_vals)) if p_vals else 0.0
    metrics["feasibility/problem_p_trajs_per_group"] = (
        float(np.mean(trajs_per)) if trajs_per else 0.0
    )
    return metrics


def aggregate_recovery_failure_prob(batch: DataProto) -> Tuple[DataProto, Dict[str, float]]:
    """Deprecated for V_F training. Kept as alias that only logs problem-level diagnostics.

    Previously overwrote state-level G_i^F with P̂(fail|x). That mixes problem-level
    into a state-level critic. Training must keep per-state MC targets.
    """
    return batch, log_problem_level_fail_rate(batch)


def transfer_same_uid_bootstrap(batch: DataProto) -> Tuple[DataProto, Dict[str, float]]:
    """Copy successful expert rows onto infeasible same-uid siblings.

    With rollout.n>1, the same problem shares `uid` across n samples × windows.
    When one sample is a final-correct positive sibling (first-shot y^C→v^accept
    and/or W→true reject→C) and another is infeasible without expert mask,
    overwrite the failing row with the expert trajectory so actor BC can run
    without GPT. Masks stay type-specific: first-shot has no rectifier span.

    Prefer matching `window_index` when available. Call after feasibility gates /
    after critic, before actor (and before online GPT).
    """
    metrics = {
        "feasibility/bootstrap_transfer_candidates": 0.0,
        "feasibility/bootstrap_transfer_filled": 0.0,
        # Full-row copy (input_ids+responses+masks): keeps expert conditioning intact.
        # NOT cross-splicing y_{i+1}^B onto state (x,y_i^A).
        "feasibility/bootstrap_transfer_same_window": 0.0,
    }
    if "expert_token_mask" not in batch.batch or "feas_gate" not in batch.batch:
        return batch, metrics
    if "uid" not in batch.non_tensor_batch:
        return batch, metrics

    B = batch.batch["responses"].size(0)
    has_expert = batch.batch["expert_token_mask"].reshape(B, -1).any(dim=-1)
    for _mk in ("expert_token_mask_y", "expert_token_mask_v", "expert_token_mask_r"):
        if _mk in batch.batch:
            has_expert = has_expert | batch.batch[_mk].reshape(B, -1).any(dim=-1)
    need = batch.batch["feas_gate"] < 0.5
    if "window_valid" in batch.non_tensor_batch:
        valid = torch.tensor(
            batch.non_tensor_batch["window_valid"].astype(np.bool_),
            device=need.device,
        )
        need = need & valid
    need = need & (~has_expert)
    metrics["feasibility/bootstrap_transfer_candidates"] = float(need.sum().item())
    if not need.any():
        return batch, metrics

    uids = batch.non_tensor_batch["uid"]
    window_index = batch.non_tensor_batch.get("window_index", None)
    experts_by_uid: Dict[object, List[int]] = defaultdict(list)
    for j in torch.where(has_expert)[0].tolist():
        experts_by_uid[uids[j]].append(j)

    copy_keys = [
        k
        for k in (
            "responses",
            "expert_token_mask",
            "expert_token_mask_y",
            "expert_token_mask_v",
            "expert_token_mask_r",
            "multiturn_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        )
        if k in batch.batch.keys()
    ]
    filled = 0
    for i in torch.where(need)[0].tolist():
        cands = experts_by_uid.get(uids[i], [])
        if not cands:
            continue
        if window_index is not None:
            wi = int(window_index[i])
            same_w = [j for j in cands if int(window_index[j]) == wi]
            if same_w:
                cands = same_w
        src = int(cands[np.random.randint(len(cands))])
        if window_index is not None and int(window_index[src]) == int(window_index[i]):
            metrics["feasibility/bootstrap_transfer_same_window"] += 1.0
        for k in copy_keys:
            batch.batch[k][i] = batch.batch[k][src].clone()
        if "advantages" in batch.batch:
            batch.batch["advantages"][i].zero_()
        if "old_log_probs" in batch.batch:
            batch.batch["old_log_probs"][i].zero_()
        if "ref_log_prob" in batch.batch:
            batch.batch["ref_log_prob"][i].zero_()
        # Destination needed recovery: keep row infeasible for generator BC.
        # Only open the role gates that τ_B+ actually has — first-shot has no
        # rectifier, so do not force feas_gate_r=0 (would invent GPT/BC rectify).
        batch.batch["feas_gate"][i] = 0.0
        src_has_y = False
        src_has_v = False
        src_has_r = False
        if "expert_token_mask_y" in batch.batch:
            src_has_y = bool(batch.batch["expert_token_mask_y"][i].any().item())
        if "expert_token_mask_v" in batch.batch:
            src_has_v = bool(batch.batch["expert_token_mask_v"][i].any().item())
        if "expert_token_mask_r" in batch.batch:
            src_has_r = bool(batch.batch["expert_token_mask_r"][i].any().item())
        elif "expert_token_mask" in batch.batch:
            src_has_v = bool(batch.batch["expert_token_mask"][i].any().item())
        if "feas_gate_v" in batch.batch:
            batch.batch["feas_gate_v"][i] = 0.0 if (src_has_v or src_has_y) else 1.0
        if "feas_gate_r" in batch.batch:
            batch.batch["feas_gate_r"][i] = 0.0 if src_has_r else 1.0
        if "feas_weight" in batch.batch:
            batch.batch["feas_weight"][i] = torch.clamp(
                batch.batch["feas_weight"][i], min=0.5
            )
        if "feas_weight_v" in batch.batch:
            if src_has_v or src_has_y:
                batch.batch["feas_weight_v"][i] = torch.clamp(
                    batch.batch["feas_weight_v"][i], min=0.5
                )
            else:
                batch.batch["feas_weight_v"][i] = 0.0
        if "feas_weight_r" in batch.batch:
            if src_has_r:
                batch.batch["feas_weight_r"][i] = torch.clamp(
                    batch.batch["feas_weight_r"][i], min=0.5
                )
            else:
                batch.batch["feas_weight_r"][i] = 0.0
        filled += 1

    metrics["feasibility/bootstrap_transfer_filled"] = float(filled)
    return batch, metrics


class ExpertBootstrapBuffer:
    """CPU-side ring buffer of expert rows for optional BC mixing / logging."""

    def __init__(self, capacity: int = 256):
        self.capacity = int(capacity)
        self._buf: Deque[Dict[str, torch.Tensor]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push_from_batch(self, batch: DataProto) -> int:
        """Push rows with any expert_token_mask tokens. Returns number added."""
        if "expert_token_mask" not in batch.batch:
            return 0
        mask = batch.batch["expert_token_mask"]
        row_has = mask.reshape(mask.size(0), -1).any(dim=-1)
        idxs = torch.where(row_has)[0].tolist()
        keys = [
            "input_ids",
            "responses",
            "attention_mask",
            "position_ids",
            "multiturn_mask",
            "expert_token_mask",
        ]
        keys = [k for k in keys if k in batch.batch.keys()]
        n_add = 0
        for i in idxs:
            item = {k: batch.batch[k][i].detach().cpu().clone() for k in keys}
            self._buf.append(item)
            n_add += 1
        return n_add

    def sample_tensors(self, n: int, device=None) -> Optional[Dict[str, torch.Tensor]]:
        """Sample up to n expert rows stacked as a mini batch dict."""
        if not self._buf or n <= 0:
            return None
        n = min(n, len(self._buf))
        idxs = np.random.choice(len(self._buf), size=n, replace=False)
        keys = self._buf[0].keys()
        out = {}
        for k in keys:
            out[k] = torch.stack([self._buf[i][k] for i in idxs], dim=0)
            if device is not None:
                out[k] = out[k].to(device)
        return out
