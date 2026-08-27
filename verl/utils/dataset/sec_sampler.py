"""SEC (Self-Evolving Curriculum) sampler for PAG training.

Reference: "Self-Evolving Curriculum for LLM Reasoning"

The sampler maintains one Q-value per MATH difficulty level (1–5) and uses
a softmax policy P_t(c) to weight category selection.  Batch construction is
per-slot multinomial: for each of the B slots in the batch, independently draw
a category c ~ P_t then uniformly draw one prompt from that category.

Key invariant:
    P_t  →  sample batch t
         →  rollout / advantages
         →  normal PAG optimizer step
         →  compute r_t(c), update Q_t → Q_{t+1}
         →  P_{t+1} samples batch t+1

Q is updated AFTER the optimizer step so that batch t is never contaminated by
its own Q signal.

Integration notes
-----------------
When sec.enabled=True the training loop bypasses StatefulDataLoader entirely
and calls sec_sampler.sample_batch() at the top of each step.  This guarantees
zero prefetch lag and that Q_{t+1} affects batch t+1.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np


_VALID_LEVELS = (1, 2, 3, 4, 5)


class SECSampler:
    """Online curriculum sampler using MATH difficulty levels as categories.

    Parameters
    ----------
    level_to_indices : dict[int, list[int]]
        Mapping from MATH level (1-5) to the list of dataset row indices
        belonging to that level.  Indices correspond to the rows of the
        RLHFDataset after filtering.
    q_alpha : float
        EMA learning rate for Q-value update (0 < q_alpha <= 1).
    temperature : float
        Softmax temperature τ.  Higher → more uniform; lower → greedier.
    seed : int
        RNG seed for reproducibility.
    log_path : str | None
        JSONL file path for per-step SEC diagnostics.  None → no file log.
    """

    def __init__(
        self,
        level_to_indices: Dict[int, List[int]],
        q_alpha: float = 0.1,
        temperature: float = 1.0,
        seed: int = 42,
        log_path: Optional[str] = None,
    ) -> None:
        assert 0.0 < q_alpha <= 1.0, f"q_alpha must be in (0, 1], got {q_alpha}"
        assert temperature > 0.0, f"temperature must be > 0, got {temperature}"
        for lvl in _VALID_LEVELS:
            assert lvl in level_to_indices, f"Missing level {lvl} in level_to_indices"

        self.level_to_indices: Dict[int, np.ndarray] = {
            lvl: np.array(idxs, dtype=np.int64)
            for lvl, idxs in level_to_indices.items()
        }
        self.q_alpha = float(q_alpha)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.log_path = log_path

        # Q values and cumulative sampling counts (5 categories)
        self.Q = np.zeros(5, dtype=np.float64)          # Q[c-1] for level c
        self.cumulative_counts = np.zeros(5, dtype=np.int64)
        self.step: int = 0

    # ── policy ──────────────────────────────────────────────────────────────

    def softmax_policy(self) -> np.ndarray:
        """Return P_t(c) = softmax(Q / τ), shape (5,), sums to 1."""
        logits = self.Q / self.temperature
        logits -= logits.max()          # numerical stability
        probs = np.exp(logits)
        return probs / probs.sum()

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample *batch_size* dataset indices using the current P_t.

        Procedure
        ---------
        1. Draw category counts  (n_1,...,n_5) ~ Multinomial(B, P_t).
        2. For each category c, draw n_c indices uniformly with replacement
           from level_to_indices[c].
        3. Concatenate and shuffle.

        Returns
        -------
        np.ndarray of shape (batch_size,) with dataset row indices.
        """
        P = self.softmax_policy()
        # Multinomial category count vector
        counts = self.rng.multinomial(batch_size, P)   # shape (5,)

        parts: List[np.ndarray] = []
        for lvl_idx, n in enumerate(counts):
            lvl = lvl_idx + 1
            pool = self.level_to_indices[lvl]
            if n > 0:
                chosen = self.rng.choice(pool, size=n, replace=True)
                parts.append(chosen)
                self.cumulative_counts[lvl_idx] += n

        indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        self.rng.shuffle(indices)
        return indices

    # ── Q update ─────────────────────────────────────────────────────────────

    def update(
        self,
        r_per_level: Dict[int, float],
        step: int,
    ) -> None:
        """EMA update Q_{t+1}(c) = (1-α)·Q_t(c) + α·r_t(c).

        Categories absent from *r_per_level* are unchanged.

        Parameters
        ----------
        r_per_level : dict[int, float]
            Immediate utility r_t(c) for each category observed in the batch.
        step : int
            Current global training step (for logging).
        """
        self.step = step
        for lvl, r in r_per_level.items():
            idx = lvl - 1
            self.Q[idx] = (1.0 - self.q_alpha) * self.Q[idx] + self.q_alpha * float(r)

        if self.log_path:
            record = {
                "step":  step,
                "Q":     self.Q.tolist(),
                "P":     self.softmax_policy().tolist(),
                "r":     {str(k): v for k, v in r_per_level.items()},
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    # ── diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        P = self.softmax_policy()
        total = max(1, int(self.cumulative_counts.sum()))
        return {
            "step": self.step,
            "Q": self.Q.copy(),
            "P": P.copy(),
            "cumulative_counts": self.cumulative_counts.copy(),
            "cumulative_fracs":  self.cumulative_counts / total,
        }

    # ── checkpoint ────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "Q":                self.Q.tolist(),
            "cumulative_counts": self.cumulative_counts.tolist(),
            "step":             self.step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.Q                 = np.array(state["Q"],                 dtype=np.float64)
        self.cumulative_counts = np.array(state["cumulative_counts"], dtype=np.int64)
        self.step              = int(state["step"])

    # ── utility: compute r_t(c) from batch advantages ─────────────────────────

    @staticmethod
    def compute_r_per_level(
        advantages:    "torch.Tensor",  # (B*K, seq_len) — post-GAE-norm
        turn1_mask:    "torch.Tensor",  # (B*K, seq_len) — True on y0 tokens
        levels:        "np.ndarray",    # (B*K,)         — MATH level per row
        num_repeat:    int,             # K rollouts per original prompt
    ) -> Dict[int, float]:
        """Compute r_t(c) = mean_c( u_i^G ) for each category in this batch.

        u_i^G = mean_k( mean_{t ∈ y0_ik}( |A^G_{ik,t}| ) )

        Parameters
        ----------
        advantages  : post-normalization advantage tensor from batch['advantages']
        turn1_mask  : boolean mask for y0 (turn-1) response tokens
        levels      : MATH level label for every row in the (B*K)-sized batch
        num_repeat  : rollout fan-out K (rows per original prompt = K)

        Returns
        -------
        dict mapping level → scalar r_t(c)
        """
        import torch

        B_K = advantages.size(0)
        assert B_K % num_repeat == 0, "B*K must be divisible by K"
        B = B_K // num_repeat

        t1_len = turn1_mask.float().sum(dim=1).clamp(min=1.0)          # (B*K,)
        u_traj  = (advantages.abs() * turn1_mask.float()).sum(dim=1) / t1_len  # (B*K,)
        u_prompt = u_traj.view(B, num_repeat).mean(dim=1)               # (B,)

        # levels is (B*K,); take every K-th entry for the original prompt
        prompt_levels = levels[::num_repeat]                             # (B,)

        r_per_level: Dict[int, float] = {}
        for lvl in _VALID_LEVELS:
            mask = (prompt_levels == lvl)
            if mask.any():
                r_per_level[int(lvl)] = float(u_prompt[torch.tensor(mask)].mean().item())

        return r_per_level
