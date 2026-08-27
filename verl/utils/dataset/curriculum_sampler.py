"""
GenerationFrontierSampler — alternating Uniform-Refresh / Curriculum epochs.

Epoch schedule
──────────────
  Even epoch (0, 2, 4, …) — "uniform refresh"
      • Samples prompts uniformly WITHOUT replacement (= standard RandomSampler).
      • After each step the trainer calls update() to refresh g_i from acc_t1.

  Odd  epoch (1, 3, 5, …) — "curriculum freeze"
      • Samples prompts WITH replacement, weighted by the scores frozen at the
        end of the preceding refresh epoch.
      • update() calls during this epoch are a no-op (g is frozen).

Score / probability
───────────────────
  s_i = g_i · (1 − g_i)          if prompt i has been observed at least once
      = 0                          otherwise   ← never treat "unseen" as frontier

  p_i = (1−ε) · s_i/Σs  +  ε/N  ← ε-floor keeps every prompt reachable

Assumptions
───────────
  extra_info.index == dataset row position (0-based).
  Verified for math7500, math500, minervamath in this repo.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class GenerationFrontierSampler(Sampler):
    """Drop-in replacement for ``RandomSampler`` that alternates between a
    uniform refresh epoch and a curriculum freeze epoch.

    Parameters
    ----------
    n_total  : total number of prompts in the training dataset
    epsilon  : ε-floor in [0, 1]  (default 0.3)
    seed     : RNG seed
    log_path : JSONL file; one record per (step, prompt_idx) update.
               None → no file logging.
    """

    def __init__(
        self,
        n_total: int,
        epsilon: float = 0.3,
        seed: int = 42,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.N       = int(n_total)
        self.epsilon = float(epsilon)
        self.rng     = np.random.default_rng(seed)
        self.log_path = log_path

        # per-prompt competence estimates — only valid once n_seen > 0
        self.g      = np.full(self.N, 0.5, dtype=np.float32)   # placeholder
        self.n_seen = np.zeros(self.N, dtype=np.int32)          # 0 = never seen

        # epoch parity controls behaviour
        self.mode  : str = "uniform"   # "uniform" | "curriculum"
        self.epoch : int = 0

    # ── epoch control (called once per epoch by the trainer) ──────────────

    def set_epoch(self, epoch: int) -> None:
        """Switch mode based on epoch parity.

        epoch % 2 == 0  →  uniform refresh  (sample uniformly, update g)
        epoch % 2 == 1  →  curriculum freeze (weighted sample, g frozen)
        """
        self.epoch = epoch
        self.mode  = "uniform" if (epoch % 2 == 0) else "curriculum"

    # ── sampling weights (used only in curriculum mode) ───────────────────

    def compute_weights(self) -> np.ndarray:
        """Return normalised p_i.  Unseen prompts (n_seen==0) get s_i=0."""
        # Never treat an unseen prompt as frontier: mask by n_seen > 0
        s     = np.where(self.n_seen > 0, self.g * (1.0 - self.g), 0.0)
        total = float(s.sum())
        if total < 1e-9:
            # all unseen OR all g=0/1 → pure uniform fallback
            curriculum = np.ones(self.N, dtype=np.float64) / self.N
        else:
            curriculum = s.astype(np.float64) / total
        uniform = np.ones(self.N, dtype=np.float64) / self.N
        p = (1.0 - self.epsilon) * curriculum + self.epsilon * uniform
        return p / p.sum()     # re-normalise for fp safety

    # ── PyTorch Sampler interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return self.N

    def __iter__(self) -> Iterator[int]:
        if self.mode == "uniform":
            # without-replacement shuffle — identical to RandomSampler behaviour
            perm = self.rng.permutation(self.N)
            return iter(perm.tolist())
        else:
            # curriculum freeze: weighted WITH replacement
            p = self.compute_weights()
            indices = self.rng.choice(self.N, size=self.N, replace=True, p=p)
            return iter(indices.tolist())

    # ── online competence update ──────────────────────────────────────────

    def update(
        self,
        prompt_indices: Sequence[int],
        g_values: Sequence[float],
        step: int,
    ) -> None:
        """Refresh g[i] and write a JSONL snapshot.

        Called by the trainer after each step.  In curriculum (odd) epochs
        this method is a no-op: g is frozen so the competence table is not
        modified (though the JSONL could still record observations if desired).

        Parameters
        ----------
        prompt_indices : dataset positions seen in this step (unique prompts)
        g_values       : mean(acc_t1) over n rollouts for each prompt
        step           : global training step
        """
        if self.mode == "curriculum":
            # g frozen — do NOT update the competence table
            return

        records = []
        for idx, g_new in zip(prompt_indices, g_values):
            idx   = int(idx)
            g_old = float(self.g[idx])
            self.g[idx]      = float(g_new)
            self.n_seen[idx] += 1
            records.append(
                {
                    "epoch":      self.epoch,
                    "step":       step,
                    "prompt_idx": idx,
                    "g_old":      round(g_old, 5),
                    "g_new":      round(float(g_new), 5),
                    "n_seen":     int(self.n_seen[idx]),
                }
            )

        if self.log_path and records:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")

    # ── convenience ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        seen = self.n_seen > 0
        return {
            "mode":          self.mode,
            "epoch":         self.epoch,
            "n_seen":        int(seen.sum()),
            "n_unseen":      int((~seen).sum()),
            "g_mean_seen":   float(self.g[seen].mean()) if seen.any() else float("nan"),
            "g_std_seen":    float(self.g[seen].std())  if seen.any() else 0.0,
            "frontier_frac": float(((self.g >= 0.1) & (self.g <= 0.9) & seen).mean()),
            # per-bin counts over seen prompts (K=4 → g ∈ {0, 0.25, 0.5, 0.75, 1.0})
            "n_g0":   int(((self.g == 0.00) & seen).sum()),
            "n_g025": int(((self.g == 0.25) & seen).sum()),
            "n_g050": int(((self.g == 0.50) & seen).sum()),
            "n_g075": int(((self.g == 0.75) & seen).sum()),
            "n_g100": int(((self.g == 1.00) & seen).sum()),
        }

    def state_dict(self) -> dict:
        """Return serialisable state for checkpoint save."""
        return {
            "g":      self.g.tolist(),
            "n_seen": self.n_seen.tolist(),
            "epoch":  self.epoch,
            "mode":   self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        import numpy as np
        self.g      = np.array(state["g"],      dtype=np.float32)
        self.n_seen = np.array(state["n_seen"], dtype=np.int32)
        self.epoch  = int(state["epoch"])
        self.mode   = state["mode"]
