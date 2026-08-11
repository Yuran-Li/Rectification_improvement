"""Replay buffer + same-uid bootstrap transfer for expert BC (no API)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto


def transfer_same_uid_bootstrap(batch: DataProto) -> Tuple[DataProto, Dict[str, float]]:
    """Copy successful expert rows onto infeasible same-uid siblings.

    With rollout.n>1, the same problem shares `uid` across n samples × windows.
    When one sample bootstraps (verify+rectify success) and another is infeasible
    without expert mask, overwrite the failing row with the expert trajectory so
    actor BC can run without GPT.

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
        # keep infeasible so actor applies high BC weight
        batch.batch["feas_gate"][i] = 0.0
        if "feas_weight" in batch.batch:
            batch.batch["feas_weight"][i] = torch.clamp(
                batch.batch["feas_weight"][i], min=0.5
            )
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
