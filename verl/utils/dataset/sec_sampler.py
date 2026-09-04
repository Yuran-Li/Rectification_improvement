"""SEC (Self-Evolving Curriculum) sampler for PAG training.

Reference: "Self-Evolving Curriculum for LLM Reasoning"

This branch uses five *dynamic* policy-dependent categories C1–C5, refreshed
from a synchronized full-train-set PAG measurement of the current policy:

    C1: g=1
    C2: 0.5 ≤ g < 1, n_WC > 0
    C3: 0.5 ≤ g < 1, n_WC = 0
    C4: g < 0.5,     n_WC > 0
    C5: g < 0.5,     n_WC = 0

g = n_correct_y0 / K_refresh, n_WC = # {y0 wrong and rectified y2 correct}.
Per-prompt state is keyed by the stable prompt id (extra_info.index).

Sampling and the generation-only Q update are unchanged from fixed-category SEC:

    P_t(c) = softmax(Q_t / τ)   (empty categories masked)
    sample batch t  →  PAG rollout / advantages / optimizer
    r_t(c) = mean u_i^G         (y0 |A^G| only)
    Q_{t+1}(c) = (1-α) Q_t(c) + α r_t(c)

Q persists across membership refreshes. A refresh replaces prompt→category only.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CATEGORIES = (1, 2, 3, 4, 5)
_VALID_LEVELS = CATEGORIES  # alias kept for compute_r_per_level / tests

# Protocol keys that must match between a precomputed stats file and the run.
REFRESH_PROTOCOL_FIELDS = (
    "refresh_rollouts",
    "model_path",
    "rollout_type",
    "num_turns",
    "temperature",
    "top_k",
    "top_p",
    "revise_gate",
    "do_sample",
)


def assign_category(g: float, n_wc: int) -> int:
    """Map (g, n_WC) to C1–C5. No Laplace, no EMA."""
    g_f = float(g)
    n = int(n_wc)
    if n < 0:
        raise ValueError(f"n_WC must be >= 0, got {n}")
    if g_f > 1.0 + 1e-12 or g_f < -1e-12:
        raise ValueError(f"g must be in [0, 1], got {g_f}")
    if g_f >= 1.0 - 1e-12:
        return 1
    if g_f >= 0.5:
        return 2 if n > 0 else 3
    return 4 if n > 0 else 5


def refresh_protocol_from_mapping(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a protocol dict to comparable JSON-friendly scalars."""
    out: Dict[str, Any] = {}
    out["refresh_rollouts"] = int(protocol["refresh_rollouts"])
    out["model_path"] = str(protocol.get("model_path", ""))
    out["rollout_type"] = str(protocol.get("rollout_type", "pag"))
    out["num_turns"] = int(protocol.get("num_turns", 2))
    out["temperature"] = float(protocol.get("temperature", 1.0))
    out["top_k"] = int(protocol.get("top_k", -1))
    top_p = protocol.get("top_p", 1.0)
    out["top_p"] = float(top_p) if top_p is not None else 1.0
    out["revise_gate"] = str(protocol.get("revise_gate", "pag"))
    out["do_sample"] = bool(protocol.get("do_sample", True))
    return out


def assert_refresh_protocol_match(
    file_protocol: Mapping[str, Any],
    run_protocol: Mapping[str, Any],
    source: str = "initial_category_stats_path",
) -> None:
    """Reject precomputed stats that do not match this run's refresh protocol.

    In particular a K=8 C1–C5 dump cannot be loaded into a K=4 refresh run.
    """
    file_p = refresh_protocol_from_mapping(file_protocol)
    run_p = refresh_protocol_from_mapping(run_protocol)
    mismatches = []
    for key in REFRESH_PROTOCOL_FIELDS:
        if file_p[key] != run_p[key]:
            mismatches.append(f"{key}: file={file_p[key]!r} run={run_p[key]!r}")
    if mismatches:
        raise ValueError(
            f"Precomputed SEC category stats at {source} do not match the "
            f"current refresh protocol:\n  " + "\n  ".join(mismatches) +
            "\nDo not reuse stats from a different K / model / sampling / "
            "PAG correction protocol."
        )


def _index_from_extra_info(extra: Any) -> Optional[int]:
    """Read extra_info.index from a dict / mapping-like cell. None if absent."""
    if extra is None:
        return None
    if not isinstance(extra, dict):
        if hasattr(extra, "keys") and hasattr(extra, "__getitem__"):
            try:
                extra = {k: extra[k] for k in extra.keys()}
            except Exception:
                return None
        else:
            return None
    idx = extra.get("index")
    if idx is None:
        return None
    return int(idx)


def prompt_ids_from_rlhf_dataset(dataset: Any) -> List[int]:
    """Stable extra_info.index for every filtered train row.

    Never uses the filtered dataset row position. After overlong-prompt
    filtering, row 0 is not necessarily prompt 0.
    """
    frame = getattr(dataset, "dataframe", dataset)
    n = len(dataset)
    ids: List[int] = []
    for i in range(n):
        row = frame[i]
        if isinstance(row, dict):
            extra = row.get("extra_info")
        elif hasattr(row, "get"):
            extra = row.get("extra_info")
        else:
            try:
                extra = row["extra_info"]
            except Exception as exc:
                raise ValueError(
                    f"train row {i} has no extra_info cell ({type(row)}): {exc}"
                ) from exc
        pid = _index_from_extra_info(extra)
        if pid is None:
            raise ValueError(
                f"train row {i} is missing extra_info.index; dynamic SEC "
                "refuses to key categories by filtered dataset positions."
            )
        ids.append(pid)
    if len(ids) != len(set(ids)):
        raise ValueError("extra_info.index values are not unique in the train set")
    return ids


def load_initial_category_stats(
    path: str,
    run_protocol: Mapping[str, Any],
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Load per-prompt g / n_WC keyed by stable prompt id.

    Expected JSON::

        {
          "protocol": {refresh_rollouts, model_path, rollout_type, ...},
          "prompts": {"<prompt_id>": {"g": float, "n_WC": int}, ...}
        }
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "protocol" not in payload:
        raise ValueError(
            f"{path} has no 'protocol' field. Refusing to load (would allow "
            "K-mismatched C1–C5 dumps such as K=8 stats on a K=4 run)."
        )
    if "prompts" not in payload:
        raise ValueError(f"{path} has no per-prompt 'prompts' map")
    assert_refresh_protocol_match(payload["protocol"], run_protocol, source=path)
    g_map: Dict[int, float] = {}
    nwc_map: Dict[int, int] = {}
    for raw_pid, rec in payload["prompts"].items():
        pid = int(raw_pid)
        g_map[pid] = float(rec["g"])
        nwc_map[pid] = int(rec["n_WC"])
    return g_map, nwc_map


def aggregate_prompt_refresh_stats(
    prompt_ids: Sequence[int],
    acc_t1: Sequence[float],
    acc_t2: Sequence[float],
    revised: Sequence[bool],
    k_refresh: int,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Reduce K trajectories per prompt to g and n_WC using PAG definitions.

    turn-1 correct  := acc_t1 >= 0.5
    W→C             := acc_t1 < 0.5 and revised and acc_t2 >= 0.5
    """
    by_pid: Dict[int, List[int]] = {}
    for i, pid in enumerate(prompt_ids):
        by_pid.setdefault(int(pid), []).append(i)

    g_map: Dict[int, float] = {}
    nwc_map: Dict[int, int] = {}
    k = int(k_refresh)
    if k <= 0:
        raise ValueError(f"k_refresh must be > 0, got {k}")
    for pid, idxs in by_pid.items():
        if len(idxs) != k:
            raise ValueError(
                f"prompt_id={pid} has {len(idxs)} trajectories, expected K={k}"
            )
        n_correct = 0
        n_wc = 0
        for j in idxs:
            a1 = float(acc_t1[j])
            a2 = float(acc_t2[j])
            rev = bool(revised[j])
            if a1 >= 0.5:
                n_correct += 1
            if a1 < 0.5 and rev and a2 >= 0.5:
                n_wc += 1
        g_map[pid] = n_correct / float(k)
        nwc_map[pid] = n_wc
    return g_map, nwc_map


class SECSampler:
    """Online curriculum sampler with dynamic C1–C5 membership.

    Parameters
    ----------
    prompt_ids : sequence of stable prompt ids, aligned with dataset rows
        ``prompt_ids[row]`` is ``extra_info.index`` for that filtered row.
        Required for production. Tests may omit this and pass
        ``level_to_indices`` only (row index is then used as the prompt id).
    level_to_indices : optional initial C → dataset-row-index map
        Tests may pass a full 5-category map. Production starts empty and
        fills via ``replace_membership``.
    q_alpha, temperature, seed, log_path : same as fixed-category SEC.
    """

    def __init__(
        self,
        prompt_ids: Optional[Sequence[int]] = None,
        level_to_indices: Optional[Dict[int, List[int]]] = None,
        q_alpha: float = 0.1,
        temperature: float = 1.0,
        seed: int = 42,
        log_path: Optional[str] = None,
    ) -> None:
        assert 0.0 < q_alpha <= 1.0, f"q_alpha must be in (0, 1], got {q_alpha}"
        assert temperature > 0.0, f"temperature must be > 0, got {temperature}"

        self.q_alpha = float(q_alpha)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.log_path = log_path

        self.Q = np.zeros(5, dtype=np.float64)
        self.cumulative_counts = np.zeros(5, dtype=np.int64)
        self.step: int = 0
        self.last_refresh_step: int = -1

        if prompt_ids is None:
            if level_to_indices is None:
                raise ValueError("SECSampler requires prompt_ids or level_to_indices")
            n_rows = 1 + max(int(i) for idxs in level_to_indices.values() for i in idxs)
            prompt_ids = list(range(n_rows))
            # rows not in any category still get a pid == row for tests
            covered = {int(i) for idxs in level_to_indices.values() for i in idxs}
            for r in range(n_rows):
                if r not in covered:
                    prompt_ids[r] = r

        self.prompt_ids = np.asarray(list(prompt_ids), dtype=np.int64)
        if self.prompt_ids.ndim != 1:
            raise ValueError("prompt_ids must be 1-D")
        uniq, counts = np.unique(self.prompt_ids, return_counts=True)
        if np.any(counts > 1):
            dups = uniq[counts > 1][:5]
            raise ValueError(f"prompt_ids must be unique, duplicates e.g. {dups.tolist()}")
        self.pid_to_row: Dict[int, int] = {
            int(pid): int(row) for row, pid in enumerate(self.prompt_ids)
        }
        self.n_rows = int(self.prompt_ids.size)

        # Per-prompt dynamic state, keyed by stable prompt id.
        self.prompt_to_category: Dict[int, int] = {}
        self.g: Dict[int, float] = {}
        self.n_WC: Dict[int, int] = {}

        self.level_to_indices: Dict[int, np.ndarray] = {
            c: np.array([], dtype=np.int64) for c in CATEGORIES
        }
        if level_to_indices is not None:
            # Test / bootstrap path: build pid state from the provided pools.
            g_map: Dict[int, float] = {}
            nwc_map: Dict[int, int] = {}
            for c, idxs in level_to_indices.items():
                c_int = int(c)
                if c_int not in CATEGORIES:
                    raise ValueError(f"invalid category {c}")
                for row in idxs:
                    pid = int(self.prompt_ids[int(row)])
                    # dummy stats consistent with the assigned category so tests
                    # that only care about pools still have a mapping.
                    g_map[pid], nwc_map[pid] = _dummy_stats_for_category(c_int)
            if g_map:
                self.replace_membership(g_map, nwc_map, step=-1, log_transition=False)

    # ── membership ────────────────────────────────────────────────────────────

    def has_membership(self) -> bool:
        return any(arr.size > 0 for arr in self.level_to_indices.values())

    def category_of_row(self, row: int) -> int:
        pid = int(self.prompt_ids[int(row)])
        if pid not in self.prompt_to_category:
            raise KeyError(f"row {row} (prompt_id={pid}) has no category; unmeasured")
        return int(self.prompt_to_category[pid])

    def replace_membership(
        self,
        g_map: Mapping[int, float],
        nwc_map: Mapping[int, int],
        step: int,
        log_transition: bool = True,
    ) -> Dict[str, float]:
        """Atomically replace prompt_id → category from a synchronized snapshot.

        Does not modify Q, cumulative_counts, step (training), or the RNG.
        Unmeasured prompts (absent from g_map) are not assigned and are not
        sampled. Prompt ids unknown to this dataset are ignored.
        """
        prev_cat = dict(self.prompt_to_category)
        new_cat: Dict[int, int] = {}
        new_g: Dict[int, float] = {}
        new_nwc: Dict[int, int] = {}
        pools: Dict[int, List[int]] = {c: [] for c in CATEGORIES}

        for raw_pid, g_val in g_map.items():
            pid = int(raw_pid)
            if pid not in self.pid_to_row:
                continue
            if pid not in nwc_map:
                raise KeyError(f"prompt_id={pid} in g_map but missing from n_WC map")
            cat = assign_category(float(g_val), int(nwc_map[pid]))
            new_cat[pid] = cat
            new_g[pid] = float(g_val)
            new_nwc[pid] = int(nwc_map[pid])
            pools[cat].append(self.pid_to_row[pid])

        self.prompt_to_category = new_cat
        self.g = new_g
        self.n_WC = new_nwc
        self.level_to_indices = {
            c: np.array(pools[c], dtype=np.int64) for c in CATEGORIES
        }
        self.last_refresh_step = int(step)

        metrics = self._membership_metrics()
        if log_transition:
            metrics.update(self._transition_metrics(prev_cat, new_cat))
        for c in CATEGORIES:
            if self.level_to_indices[c].size == 0:
                warnings.warn(
                    f"[SEC] category C{c} is empty after refresh at step={step}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return metrics

    def _membership_metrics(self) -> Dict[str, float]:
        n = max(1, len(self.prompt_to_category))
        out: Dict[str, float] = {}
        for c in CATEGORIES:
            rows = self.level_to_indices[c]
            out[f"dynamic/category_count_C{c}"] = float(rows.size)
            out[f"dynamic/category_frac_C{c}"] = float(rows.size) / float(n)
            pids = [int(self.prompt_ids[int(r)]) for r in rows]
            if pids:
                gs = np.array([self.g[p] for p in pids], dtype=np.float64)
                nw = np.array([self.n_WC[p] for p in pids], dtype=np.float64)
                out[f"dynamic/g_mean_C{c}"] = float(gs.mean())
                out[f"dynamic/nWC_positive_frac_C{c}"] = float((nw > 0).mean())
            else:
                out[f"dynamic/g_mean_C{c}"] = float("nan")
                out[f"dynamic/nWC_positive_frac_C{c}"] = float("nan")
        return out

    @staticmethod
    def _transition_metrics(
        prev_cat: Mapping[int, int],
        new_cat: Mapping[int, int],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # Counts over prompts present in both snapshots.
        both = [pid for pid in new_cat if pid in prev_cat]
        for src in CATEGORIES:
            src_pids = [pid for pid in both if prev_cat[pid] == src]
            denom = max(1, len(src_pids))
            for dst in CATEGORIES:
                n = sum(1 for pid in src_pids if new_cat[pid] == dst)
                out[f"dynamic/transition_C{src}_to_C{dst}"] = float(n) / float(denom) if src_pids else float("nan")
        return out

    # ── policy ──────────────────────────────────────────────────────────────

    def nonempty_mask(self) -> np.ndarray:
        return np.array(
            [self.level_to_indices[c].size > 0 for c in CATEGORIES],
            dtype=bool,
        )

    def softmax_policy(self) -> np.ndarray:
        """P_t(c) = softmax(Q/τ) with empty categories masked, sums to 1."""
        nonempty = self.nonempty_mask()
        if not nonempty.any():
            raise RuntimeError(
                "SEC softmax_policy: all categories empty. "
                "Run a synchronized refresh before sampling."
            )
        logits = self.Q / self.temperature
        finite = np.where(nonempty, logits, -np.inf)
        finite = finite - np.nanmax(finite[nonempty])
        probs = np.exp(finite)
        probs = np.where(nonempty, probs, 0.0)
        s = float(probs.sum())
        if s <= 0.0:
            raise RuntimeError("SEC softmax_policy: masked probabilities sum to 0")
        return probs / s

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample *batch_size* dataset row indices using the current P_t.

        Procedure (unchanged from fixed-category SEC)
        ---------------------------------------------
        1. Draw category counts  (n_1,...,n_5) ~ Multinomial(B, P_t).
        2. For each category c, draw n_c indices uniformly with replacement
           from level_to_indices[c].
        3. Concatenate and shuffle.
        """
        P = self.softmax_policy()
        counts = self.rng.multinomial(batch_size, P)
        nonempty = self.nonempty_mask()
        if np.any((counts > 0) & ~nonempty):
            leftover = int(counts[~nonempty].sum())
            warnings.warn(
                f"[SEC] multinomial allocated {leftover} slots to empty "
                "categories; redistributing (should have been masked)",
                RuntimeWarning,
                stacklevel=2,
            )
            counts = counts.copy()
            counts[~nonempty] = 0
            counts = counts + self.rng.multinomial(leftover, P)

        parts: List[np.ndarray] = []
        for lvl_idx, n in enumerate(counts):
            lvl = lvl_idx + 1
            pool = self.level_to_indices[lvl]
            if n > 0:
                chosen = self.rng.choice(pool, size=int(n), replace=True)
                parts.append(chosen)
                self.cumulative_counts[lvl_idx] += int(n)

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
        P = self.softmax_policy() if self.has_membership() else np.full(5, np.nan)
        total = max(1, int(self.cumulative_counts.sum()))
        return {
            "step": self.step,
            "Q": self.Q.copy(),
            "P": P.copy(),
            "cumulative_counts": self.cumulative_counts.copy(),
            "cumulative_fracs":  self.cumulative_counts / total,
            "last_refresh_step": self.last_refresh_step,
        }

    # ── checkpoint ────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "Q":                 self.Q.tolist(),
            "cumulative_counts": self.cumulative_counts.tolist(),
            "step":              self.step,
            "last_refresh_step": self.last_refresh_step,
            "prompt_ids":        self.prompt_ids.tolist(),
            "prompt_to_category": {str(k): int(v) for k, v in self.prompt_to_category.items()},
            "g":                 {str(k): float(v) for k, v in self.g.items()},
            "n_WC":              {str(k): int(v) for k, v in self.n_WC.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if "prompt_to_category" not in state:
            raise ValueError(
                "SEC checkpoint is missing prompt_to_category. This branch "
                "cannot resume a fixed MATH-level sampler ckpt; start a new "
                "run (or provide initial_category_stats_path) so C1–C5 are "
                "measured under the current protocol."
            )
        ckpt_ids = np.array(state.get("prompt_ids", self.prompt_ids.tolist()), dtype=np.int64)
        if ckpt_ids.shape != self.prompt_ids.shape or not np.array_equal(ckpt_ids, self.prompt_ids):
            raise ValueError(
                "SEC checkpoint prompt_ids do not match the current dataset. "
                "Refusing to resume with misaligned prompt identity."
            )
        self.Q                 = np.array(state["Q"],                 dtype=np.float64)
        self.cumulative_counts = np.array(state["cumulative_counts"], dtype=np.int64)
        self.step              = int(state["step"])
        self.last_refresh_step = int(state.get("last_refresh_step", -1))

        g_map = {int(k): float(v) for k, v in state["g"].items()}
        nwc_map = {int(k): int(v) for k, v in state["n_WC"].items()}
        # Rebuild pools from stored stats; keep Q / counts already restored.
        q_saved = self.Q.copy()
        counts_saved = self.cumulative_counts.copy()
        step_saved = self.step
        self.replace_membership(g_map, nwc_map, step=self.last_refresh_step, log_transition=False)
        self.Q = q_saved
        self.cumulative_counts = counts_saved
        self.step = step_saved
        # Stored mapping must agree with recomputed C1–C5.
        stored_cat = {int(k): int(v) for k, v in state["prompt_to_category"].items()}
        if stored_cat != self.prompt_to_category:
            raise ValueError(
                "SEC checkpoint prompt_to_category disagrees with (g, n_WC) "
                "under assign_category."
            )

    # ── utility: compute r_t(c) from batch advantages ─────────────────────────

    @staticmethod
    def compute_r_per_level(
        advantages:    "torch.Tensor",  # (B*K, seq_len) — post-GAE-norm
        turn1_mask:    "torch.Tensor",  # (B*K, seq_len) — True on y0 tokens
        levels:        "np.ndarray",    # (B*K,)         — category id per row
        num_repeat:    int,             # K rollouts per original prompt
    ) -> Dict[int, float]:
        """Compute r_t(c) = mean_c( u_i^G ) for each category in this batch.

        u_i^G = mean_k( mean_{t ∈ y0_ik}( |A^G_{ik,t}| ) )

        Unchanged from fixed-category SEC: generation-only |A_G|.
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

    @staticmethod
    def compute_turn_absA_metrics(
        advantages: "torch.Tensor",
        turn_idx: "torch.Tensor",
        mt_mask: "torch.Tensor",
        levels_BK: np.ndarray,
        num_repeat: int,
    ) -> Dict[str, float]:
        """Per-category |A| diagnostics for generate / verify / rectify.

        ``sec/A_{role}_C*`` is the historical *unconditional* mean: trajectories
        with no tokens on that turn contribute 0 (``clamp(min=1)`` on length).
        For a category C that is equivalent to

            E[u^r | C] = P(has turn r | C) · E[|A_r| | has turn r, C].

        Rectify additionally logs the two factors separately (trajectory-level):

            sec/rectify_trigger_rate_C*     = P(has y2 | C)
            sec/A_rectify_conditional_C*    = E[mean_t |A| | has y2, C]
        """
        import torch

        out: Dict[str, float] = {}
        levels = np.asarray(levels_BK, dtype=np.int32)
        n = int(num_repeat)
        if advantages.size(0) % n != 0:
            raise ValueError("B*K must be divisible by K")
        if levels.shape[0] != advantages.size(0):
            raise ValueError(
                f"levels_BK length {levels.shape[0]} != B*K {advantages.size(0)}"
            )

        for turn_num, turn_name in [(1, "generate"), (2, "verify"), (3, "rectify")]:
            tmask = (turn_idx == turn_num) & mt_mask
            t_len = tmask.float().sum(dim=1)  # (B*K,)
            has_turn = t_len > 0
            u_t = (advantages.abs() * tmask.float()).sum(dim=1) / t_len.clamp(min=1.0)
            u_p = u_t.view(-1, n).mean(dim=1)
            levels_B = levels[::n]
            has_np = has_turn.detach().cpu().numpy()
            u_t_np = u_t.detach().cpu().numpy()

            for lvl in _VALID_LEVELS:
                m_prompt = levels_B == lvl
                if m_prompt.any():
                    out[f"sec/A_{turn_name}_C{lvl}"] = float(
                        u_p[torch.as_tensor(m_prompt, device=u_p.device)].mean().item()
                    )
                if turn_name != "rectify":
                    continue
                m_traj = levels == lvl
                if not m_traj.any():
                    continue
                out[f"sec/rectify_trigger_rate_C{lvl}"] = float(has_np[m_traj].mean())
                cond = m_traj & has_np
                if cond.any():
                    out[f"sec/A_rectify_conditional_C{lvl}"] = float(u_t_np[cond].mean())
                else:
                    out[f"sec/A_rectify_conditional_C{lvl}"] = float("nan")
        return out


def _dummy_stats_for_category(cat: int) -> Tuple[float, int]:
    """Invert C1–C5 to a legal (g, n_WC) pair for test bootstrap only."""
    if cat == 1:
        return 1.0, 0
    if cat == 2:
        return 0.75, 1
    if cat == 3:
        return 0.75, 0
    if cat == 4:
        return 0.25, 1
    if cat == 5:
        return 0.25, 0
    raise ValueError(cat)
