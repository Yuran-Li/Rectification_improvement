"""SEC (Self-Evolving Curriculum) sampler for PAG training.

Prevalence-aware C1–C5 with U/C interleaved epochs (no extra measurement refresh).

    C1: g=1
    C2: 0.5 ≤ g < 1, n_WC > 0
    C3: 0.5 ≤ g < 1, n_WC = 0
    C4: g < 0.5,     n_WC > 0
    C5: g < 0.5,     n_WC = 0

g = n_correct_y0 / K, n_WC = # {y0 wrong and rectified y2 correct}.
Per-prompt state is keyed by the stable prompt id (extra_info.index).

    prevalence_aware=True:  P(c) ∝ |C_c| exp(Q_c / τ)   (equal Q ⇒ prompt-uniform)
    prevalence_aware=False: P(c) ∝ exp(Q_c / τ)          (original softmax)
    U epoch: prompt-uniform, without replacement, one pass; membership frozen
    C epoch: multinomial under the selected P; membership frozen
    Q_{t+1}(c) = (1-α) Q_t(c) + α r_t(c)   generation-only |A_G|
"""
from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CATEGORIES = (1, 2, 3, 4, 5)
_VALID_LEVELS = CATEGORIES

# Kept so leftover protocol JSON loaders still parse; this experiment does not
# run a measurement-only refresh.
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
    out["num_turns"] = int(protocol["num_turns"] if "num_turns" in protocol else 2)
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
    file_p = refresh_protocol_from_mapping(file_protocol)
    run_p = refresh_protocol_from_mapping(run_protocol)
    mismatches = []
    for key in REFRESH_PROTOCOL_FIELDS:
        if file_p[key] != run_p[key]:
            mismatches.append(f"{key}: file={file_p[key]!r} run={run_p[key]!r}")
    if mismatches:
        raise ValueError(
            f"Precomputed SEC category stats at {source} do not match the "
            f"current refresh protocol:\n  " + "\n  ".join(mismatches)
        )


def _index_from_extra_info(extra: Any) -> Optional[int]:
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
    """Stable extra_info.index for every filtered train row."""
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
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "protocol" not in payload:
        raise ValueError(f"{path} has no 'protocol' field.")
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
    """Reduce K trajectories per prompt to g and n_WC using PAG definitions."""
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
    """Online curriculum sampler: prevalence-aware P(c) and U/C epochs."""

    def __init__(
        self,
        prompt_ids: Optional[Sequence[int]] = None,
        level_to_indices: Optional[Dict[int, List[int]]] = None,
        q_alpha: float = 0.1,
        temperature: float = 0.1,
        prevalence_aware: bool = True,
        seed: int = 42,
        log_path: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        assert 0.0 < q_alpha <= 1.0, f"q_alpha must be in (0, 1], got {q_alpha}"
        assert temperature > 0.0, f"temperature must be > 0, got {temperature}"

        self.q_alpha = float(q_alpha)
        self.temperature = float(temperature)
        self.prevalence_aware = bool(prevalence_aware)
        self.rng = np.random.default_rng(seed)
        self.log_path = log_path

        self.Q = np.zeros(5, dtype=np.float64)
        self.cumulative_counts = np.zeros(5, dtype=np.int64)
        self.step: int = 0
        self.last_refresh_step: int = -1  # last U-epoch membership rebuild

        if prompt_ids is None:
            if level_to_indices is None:
                raise ValueError("SECSampler requires prompt_ids or level_to_indices")
            n_rows = 1 + max(int(i) for idxs in level_to_indices.values() for i in idxs)
            prompt_ids = list(range(n_rows))
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

        self.prompt_to_category: Dict[int, int] = {}
        self.g: Dict[int, float] = {}
        self.n_WC: Dict[int, int] = {}
        self.level_to_indices: Dict[int, np.ndarray] = {
            c: np.array([], dtype=np.int64) for c in CATEGORIES
        }

        # U/C phase
        self.phase: str = "U"
        self.epoch_index: int = 0
        self.epoch_position: int = 0
        self.batch_size: int = int(batch_size) if batch_size is not None else 1
        self.n_steps_per_epoch: int = 0
        self.u_perm: Optional[np.ndarray] = None
        self.u_cursor: int = 0
        self.c_batches_done: int = 0
        self._u_acc1: Dict[int, List[float]] = {}
        self._u_acc2: Dict[int, List[float]] = {}
        self._u_rev: Dict[int, List[bool]] = {}
        self._u_ug: Dict[int, List[float]] = {}

        if level_to_indices is not None:
            g_map: Dict[int, float] = {}
            nwc_map: Dict[int, int] = {}
            for c, idxs in level_to_indices.items():
                c_int = int(c)
                if c_int not in CATEGORIES:
                    raise ValueError(f"invalid category {c}")
                for row in idxs:
                    pid = int(self.prompt_ids[int(row)])
                    g_map[pid], nwc_map[pid] = _dummy_stats_for_category(c_int)
            if g_map:
                self.replace_membership(g_map, nwc_map, step=-1, log_transition=False)

    # ── membership ────────────────────────────────────────────────────────────

    def has_membership(self) -> bool:
        return any(arr.size > 0 for arr in self.level_to_indices.values())

    def category_sizes(self) -> np.ndarray:
        return np.array(
            [self.level_to_indices[c].size for c in CATEGORIES],
            dtype=np.float64,
        )

    def category_frac(self) -> np.ndarray:
        sizes = self.category_sizes()
        n = float(sizes.sum())
        if n <= 0.0:
            return np.full(5, np.nan)
        return sizes / n

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
        """Atomically replace prompt_id → category. Does not modify Q / RNG / phase."""
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
                    f"[SEC] category C{c} is empty after U-epoch rebuild at step={step}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return metrics

    def _membership_metrics(self) -> Dict[str, float]:
        n = max(1, len(self.prompt_to_category))
        out: Dict[str, float] = {}
        sizes = self.category_sizes()
        frac = self.category_frac()
        for c in CATEGORIES:
            rows = self.level_to_indices[c]
            out[f"sec/category_count_C{c}"] = float(sizes[c - 1])
            out[f"sec/category_frac_C{c}"] = float(frac[c - 1]) if n else float("nan")
            # keep previous dynamic/ keys for continuity
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
        both = [pid for pid in new_cat if pid in prev_cat]
        for src in CATEGORIES:
            src_pids = [pid for pid in both if prev_cat[pid] == src]
            denom = max(1, len(src_pids))
            for dst in CATEGORIES:
                n = sum(1 for pid in src_pids if new_cat[pid] == dst)
                val = float(n) / float(denom) if src_pids else float("nan")
                out[f"dynamic/transition_C{src}_to_C{dst}"] = val
                out[f"sec/transition_C{src}_to_C{dst}"] = val
        return out

    # ── policy ──────────────────────────────────────────────────────────────

    def nonempty_mask(self) -> np.ndarray:
        return self.category_sizes() > 0

    def category_policy(self) -> np.ndarray:
        """C-phase category distribution. Empty categories masked. Sums to 1.

        prevalence_aware=True:  P(c) ∝ |C_c| exp(Q_c / τ)
            Equal Q ⇒ P(c) = |C_c| / N (prompt-uniform).
        prevalence_aware=False: P(c) ∝ exp(Q_c / τ)
            Equal Q ⇒ uniform over nonempty categories.
        """
        sizes = self.category_sizes()
        nonempty = sizes > 0
        if not nonempty.any():
            raise RuntimeError(
                "SEC category_policy: all categories empty. "
                "Finish a U epoch before C-phase sampling."
            )
        logits = self.Q / self.temperature
        finite = np.where(nonempty, logits, -np.inf)
        finite = finite - np.nanmax(finite[nonempty])
        weight = sizes if self.prevalence_aware else np.ones_like(sizes)
        unnorm = np.where(nonempty, weight * np.exp(finite), 0.0)
        s = float(unnorm.sum())
        if s <= 0.0:
            raise RuntimeError("SEC category_policy: masked probabilities sum to 0")
        return unnorm / s

    def softmax_policy(self) -> np.ndarray:
        """Alias: prevalence-aware policy (name kept for older call sites)."""
        return self.category_policy()

    def exposure_multiplier(self) -> np.ndarray:
        """P(c) / category_frac(c); nan for empty categories."""
        P = self.category_policy() if self.has_membership() else np.full(5, np.nan)
        frac = self.category_frac()
        out = np.full(5, np.nan)
        for i in range(5):
            if frac[i] > 0.0 and np.isfinite(P[i]):
                out[i] = float(P[i]) / float(frac[i])
        return out

    # ── U / C epochs ──────────────────────────────────────────────────────────

    def begin_uniform_epoch(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        self.phase = "U"
        self.epoch_position = 0
        self.u_perm = self.rng.permutation(self.n_rows).astype(np.int64)
        self.u_cursor = 0
        self.n_steps_per_epoch = int(math.ceil(self.n_rows / float(self.batch_size)))
        self.c_batches_done = 0
        self._u_acc1 = {}
        self._u_acc2 = {}
        self._u_rev = {}
        self._u_ug = {}

    def begin_curriculum_epoch(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self.batch_size = int(batch_size)
        if not self.has_membership():
            raise RuntimeError(
                "C epoch requires membership from a completed U epoch."
            )
        self.phase = "C"
        self.epoch_position = 0
        self.c_batches_done = 0
        if self.n_steps_per_epoch <= 0:
            self.n_steps_per_epoch = int(math.ceil(self.n_rows / float(self.batch_size)))

    def uniform_epoch_complete(self) -> bool:
        return self.phase == "U" and self.u_perm is not None and self.u_cursor >= self.n_rows

    def curriculum_epoch_complete(self) -> bool:
        return self.phase == "C" and self.c_batches_done >= self.n_steps_per_epoch

    def next_batch(self, batch_size: int) -> np.ndarray:
        """Next training-row index batch for the current phase."""
        self.batch_size = int(batch_size)
        if self.phase == "U":
            if self.u_perm is None:
                self.begin_uniform_epoch(batch_size)
            return self._next_uniform_batch(batch_size)
        return self.sample_batch(batch_size)

    def _next_uniform_batch(self, batch_size: int) -> np.ndarray:
        assert self.u_perm is not None
        if self.u_cursor >= self.n_rows:
            raise RuntimeError("U epoch already exhausted; finalize before sampling again")
        start = self.u_cursor
        end = min(start + int(batch_size), self.n_rows)
        chosen = self.u_perm[start:end].copy()
        self.u_cursor = end
        self.epoch_position += 1
        self._count_sampled(chosen)
        return chosen

    def _count_sampled(self, rows: np.ndarray) -> None:
        if not self.has_membership():
            return
        for row in rows:
            pid = int(self.prompt_ids[int(row)])
            cat = self.prompt_to_category.get(pid)
            if cat is not None:
                self.cumulative_counts[cat - 1] += 1

    def record_u_trajectories(
        self,
        prompt_ids: Sequence[int],
        acc_t1: Sequence[float],
        acc_t2: Sequence[float],
        revised: Sequence[bool],
        u_g: Sequence[float],
        k: int,
    ) -> None:
        """Accumulate U-epoch measurements. Does not change membership.

        Skip a prompt once K trajectories are already stored (padding duplicates).
        """
        if self.phase != "U":
            raise RuntimeError("record_u_trajectories only valid in U phase")
        k = int(k)
        by_pid: Dict[int, List[int]] = {}
        for i, pid in enumerate(prompt_ids):
            by_pid.setdefault(int(pid), []).append(i)
        for pid, idxs in by_pid.items():
            if pid in self._u_acc1 and len(self._u_acc1[pid]) >= k:
                continue
            if len(idxs) < k:
                raise ValueError(
                    f"U-epoch prompt_id={pid} has {len(idxs)} trajectories, expected K={k}"
                )
            use = idxs[:k]
            self._u_acc1[pid] = [float(acc_t1[j]) for j in use]
            self._u_acc2[pid] = [float(acc_t2[j]) for j in use]
            self._u_rev[pid] = [bool(revised[j]) for j in use]
            self._u_ug[pid] = [float(u_g[j]) for j in use]

    def finalize_uniform_epoch(self, step: int) -> Dict[str, float]:
        """Rebuild C1–C5 from the full U pass, EMA-update Q, then start C."""
        if self.phase != "U":
            raise RuntimeError("finalize_uniform_epoch called outside U")
        if not self.uniform_epoch_complete():
            raise RuntimeError(
                f"U epoch incomplete: cursor={self.u_cursor} n_rows={self.n_rows}"
            )
        expected = set(int(x) for x in self.prompt_ids.tolist())
        got = set(self._u_acc1)
        if got != expected:
            raise RuntimeError(
                f"U epoch coverage mismatch: missing={len(expected - got)} "
                f"extra={len(got - expected)}"
            )
        k = len(next(iter(self._u_acc1.values())))
        pids: List[int] = []
        a1: List[float] = []
        a2: List[float] = []
        rev: List[bool] = []
        ug_prompt: Dict[int, float] = {}
        for pid in expected:
            accs = self._u_acc1[pid]
            if len(accs) != k:
                raise RuntimeError(f"prompt_id={pid} has {len(accs)} U trajs, expected {k}")
            pids.extend([pid] * k)
            a1.extend(self._u_acc1[pid])
            a2.extend(self._u_acc2[pid])
            rev.extend(self._u_rev[pid])
            ug_prompt[pid] = float(np.mean(self._u_ug[pid]))

        g_map, nwc_map = aggregate_prompt_refresh_stats(pids, a1, a2, rev, k_refresh=k)
        metrics = self.replace_membership(g_map, nwc_map, step=step, log_transition=True)

        r_per_level: Dict[int, float] = {}
        for c in CATEGORIES:
            rows = self.level_to_indices[c]
            if rows.size == 0:
                continue
            vals = [ug_prompt[int(self.prompt_ids[int(r)])] for r in rows]
            r_per_level[c] = float(np.mean(vals))
        self.update(r_per_level, step=step, log_policy=True)
        metrics.update({f"sec/reward_C{c}": float(r_per_level.get(c, float("nan"))) for c in CATEGORIES})
        metrics.update(self._policy_log_metrics())

        self.epoch_index += 1
        self.begin_curriculum_epoch()
        return metrics

    def finalize_curriculum_epoch(self) -> None:
        if self.phase != "C":
            raise RuntimeError("finalize_curriculum_epoch called outside C")
        if not self.curriculum_epoch_complete():
            raise RuntimeError(
                f"C epoch incomplete: {self.c_batches_done}/{self.n_steps_per_epoch}"
            )
        self.epoch_index += 1
        self.begin_uniform_epoch()

    # ── sampling (C phase) ────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Curriculum batch: multinomial over prevalence-aware P, then within-cat uniform w/ replacement."""
        P = self.category_policy()
        counts = self.rng.multinomial(int(batch_size), P)
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
        self.epoch_position += 1
        self.c_batches_done += 1
        return indices

    # ── Q update ─────────────────────────────────────────────────────────────

    def update(
        self,
        r_per_level: Dict[int, float],
        step: int,
        log_policy: bool = True,
    ) -> None:
        """EMA update Q_{t+1}(c) = (1-α)·Q_t(c) + α·r_t(c)."""
        self.step = step
        for lvl, r in r_per_level.items():
            idx = lvl - 1
            self.Q[idx] = (1.0 - self.q_alpha) * self.Q[idx] + self.q_alpha * float(r)

        if self.log_path and log_policy:
            record = {
                "step":  step,
                "phase": self.phase,
                "epoch_index": self.epoch_index,
                "Q":     self.Q.tolist(),
                "P":     (self.category_policy().tolist() if self.has_membership() else [None] * 5),
                "r":     {str(k): v for k, v in r_per_level.items()},
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def _policy_log_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "sec/phase": 0.0 if self.phase == "U" else 1.0,
            "sec/epoch_index": float(self.epoch_index),
            "sec/prevalence_aware": 1.0 if self.prevalence_aware else 0.0,
        }
        sizes = self.category_sizes()
        frac = self.category_frac()
        for c in CATEGORIES:
            out[f"sec/category_count_C{c}"] = float(sizes[c - 1])
            out[f"sec/category_frac_C{c}"] = float(frac[c - 1])
            out[f"sec/Q_C{c}"] = float(self.Q[c - 1])
        if not self.has_membership():
            return out
        P = self.category_policy()
        expo = self.exposure_multiplier()
        for c in CATEGORIES:
            out[f"sec/Q_C{c}"] = float(self.Q[c - 1])
            out[f"sec/category_prob_C{c}"] = float(P[c - 1])
            out[f"sec/P_C{c}"] = float(P[c - 1])
            out[f"sec/exposure_multiplier_C{c}"] = float(expo[c - 1])
        return out

    def stats(self) -> dict:
        P = self.category_policy() if self.has_membership() else np.full(5, np.nan)
        total = max(1, int(self.cumulative_counts.sum()))
        return {
            "step": self.step,
            "Q": self.Q.copy(),
            "P": P.copy(),
            "phase": self.phase,
            "epoch_index": self.epoch_index,
            "cumulative_counts": self.cumulative_counts.copy(),
            "cumulative_fracs":  self.cumulative_counts / total,
            "last_refresh_step": self.last_refresh_step,
        }

    # ── checkpoint ────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "Q":                  self.Q.tolist(),
            "cumulative_counts":  self.cumulative_counts.tolist(),
            "step":               self.step,
            "last_refresh_step":  self.last_refresh_step,
            "prompt_ids":         self.prompt_ids.tolist(),
            "prompt_to_category": {str(k): int(v) for k, v in self.prompt_to_category.items()},
            "g":                  {str(k): float(v) for k, v in self.g.items()},
            "n_WC":               {str(k): int(v) for k, v in self.n_WC.items()},
            "phase":              self.phase,
            "epoch_index":        self.epoch_index,
            "epoch_position":     self.epoch_position,
            "batch_size":         self.batch_size,
            "n_steps_per_epoch":  self.n_steps_per_epoch,
            "u_perm":             None if self.u_perm is None else self.u_perm.tolist(),
            "u_cursor":           self.u_cursor,
            "c_batches_done":     self.c_batches_done,
            "u_acc1":             {str(k): v for k, v in self._u_acc1.items()},
            "u_acc2":             {str(k): v for k, v in self._u_acc2.items()},
            "u_rev":              {str(k): v for k, v in self._u_rev.items()},
            "u_ug":               {str(k): v for k, v in self._u_ug.items()},
            "rng_state":          self.rng.bit_generator.state,
            "prevalence_aware":   self.prevalence_aware,
        }

    def load_state_dict(self, state: dict) -> None:
        if "phase" not in state:
            raise ValueError(
                "SEC checkpoint is missing U/C phase. This branch cannot resume "
                "a refresh-interval sampler ckpt; start a new run."
            )
        if "prompt_to_category" not in state:
            raise ValueError(
                "SEC checkpoint is missing prompt_to_category. This branch "
                "cannot resume a fixed MATH-level sampler ckpt."
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
        self.phase             = str(state["phase"])
        if self.phase not in ("U", "C"):
            raise ValueError(f"invalid SEC phase {self.phase!r}")
        self.epoch_index       = int(state["epoch_index"])
        self.epoch_position    = int(state["epoch_position"])
        self.batch_size        = int(state.get("batch_size", self.batch_size))
        self.n_steps_per_epoch = int(state.get("n_steps_per_epoch", 0))
        self.u_cursor          = int(state.get("u_cursor", 0))
        self.c_batches_done    = int(state.get("c_batches_done", 0))
        perm = state.get("u_perm")
        self.u_perm = None if perm is None else np.array(perm, dtype=np.int64)
        self._u_acc1 = {int(k): list(v) for k, v in state.get("u_acc1", {}).items()}
        self._u_acc2 = {int(k): list(v) for k, v in state.get("u_acc2", {}).items()}
        self._u_rev  = {int(k): [bool(x) for x in v] for k, v in state.get("u_rev", {}).items()}
        self._u_ug   = {int(k): list(v) for k, v in state.get("u_ug", {}).items()}
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        if "prevalence_aware" in state:
            self.prevalence_aware = bool(state["prevalence_aware"])

        g_map = {int(k): float(v) for k, v in state["g"].items()}
        nwc_map = {int(k): int(v) for k, v in state["n_WC"].items()}
        q_saved = self.Q.copy()
        counts_saved = self.cumulative_counts.copy()
        step_saved = self.step
        phase_saved = self.phase
        if g_map:
            self.replace_membership(g_map, nwc_map, step=self.last_refresh_step, log_transition=False)
            stored_cat = {int(k): int(v) for k, v in state["prompt_to_category"].items()}
            if stored_cat != self.prompt_to_category:
                raise ValueError(
                    "SEC checkpoint prompt_to_category disagrees with (g, n_WC) "
                    "under assign_category."
                )
        else:
            self.prompt_to_category = {}
            self.g = {}
            self.n_WC = {}
            self.level_to_indices = {c: np.array([], dtype=np.int64) for c in CATEGORIES}
        self.Q = q_saved
        self.cumulative_counts = counts_saved
        self.step = step_saved
        self.phase = phase_saved

    @staticmethod
    def compute_r_per_level(
        advantages:    "torch.Tensor",
        turn1_mask:    "torch.Tensor",
        levels:        "np.ndarray",
        num_repeat:    int,
    ) -> Dict[int, float]:
        """r_t(c) = mean_c( u_i^G ), u_i^G = mean_k mean_{t ∈ y0} |A^G|."""
        import torch

        B_K = advantages.size(0)
        assert B_K % num_repeat == 0, "B*K must be divisible by K"
        B = B_K // num_repeat

        t1_len = turn1_mask.float().sum(dim=1).clamp(min=1.0)
        u_traj  = (advantages.abs() * turn1_mask.float()).sum(dim=1) / t1_len
        u_prompt = u_traj.view(B, num_repeat).mean(dim=1)

        prompt_levels = levels[::num_repeat]

        r_per_level: Dict[int, float] = {}
        for lvl in _VALID_LEVELS:
            mask = (prompt_levels == lvl)
            if mask.any():
                r_per_level[int(lvl)] = float(u_prompt[torch.tensor(mask)].mean().item())

        return r_per_level

    @staticmethod
    def compute_u_traj(
        advantages: "torch.Tensor",
        turn1_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Per-trajectory generation utility (y0 |A_G| mean)."""
        t1_len = turn1_mask.float().sum(dim=1).clamp(min=1.0)
        return (advantages.abs() * turn1_mask.float()).sum(dim=1) / t1_len


def _dummy_stats_for_category(cat: int) -> Tuple[float, int]:
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
