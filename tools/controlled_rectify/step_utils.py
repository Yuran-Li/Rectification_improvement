#!/usr/bin/env python3
"""Step segmentation and prefix helpers for causal rectify eval."""
from __future__ import annotations

import re
from dataclasses import dataclass

SEGMENT_METHODS = ("legacy", "scope_nn", "stride_tags")


@dataclass
class SteppedSolution:
    steps: list[str]  # 1-indexed semantically; list[0] is Step 1
    numbered_text: str

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def segment_scope_double_newline(text: str) -> SteppedSolution:
    """SCOPE-style: split on blank lines (\\n\\n) only. No numbered-marker heuristics."""
    text = (text or "").strip()
    if not text:
        return _pack([""])
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) >= 2:
        return _pack(paras)
    # Single block: keep intact (do NOT shatter by lines / Step markers)
    return _pack([text])


def segment_stride_tags(text: str) -> SteppedSolution:
    """STRIDE-style: parse <step>...</step> (or <step N>...</step>) blocks.

    Falls back to scope_nn if fewer than 2 tags are found.
    """
    text = (text or "").strip()
    if not text:
        return _pack([""])
    # Prefer explicit XML-like step tags
    tagged = re.findall(
        r"<step(?:\s+[^>]*)?>\s*(.*?)\s*</step>",
        text,
        flags=re.I | re.S,
    )
    steps = [s.strip() for s in tagged if s and s.strip()]
    if len(steps) >= 2:
        return _pack(steps)
    # Also accept ```step ... ``` fences if present
    fenced = re.findall(r"```step\s*(.*?)```", text, flags=re.I | re.S)
    steps = [s.strip() for s in fenced if s and s.strip()]
    if len(steps) >= 2:
        return _pack(steps)
    return segment_scope_double_newline(text)


def segment_solution(text: str) -> SteppedSolution:
    """Legacy splitter (original causal eval). Prefer markers → paras → line chunks."""
    text = (text or "").strip()
    if not text:
        return SteppedSolution(steps=[""], numbered_text="Step 1:\n")

    # Explicit "Step k" / "k." / "k)" markers
    marked = re.split(r"(?=\n?\s*(?:Step\s+\d+|\d+[\.\)])\s+)", text)
    marked = [m.strip() for m in marked if m and m.strip()]
    if len(marked) >= 2:
        steps = []
        for m in marked:
            cleaned = re.sub(r"^(?:Step\s+\d+|\d+[\.\)])\s*", "", m, flags=re.I).strip()
            if cleaned:
                steps.append(cleaned)
        if steps:
            return _pack(steps)

    # Paragraph / blank-line chunks
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) >= 2:
        return _pack(paras)

    # Line-group chunks (every ~3 non-empty lines)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4:
        chunk_size = max(2, len(lines) // 4)
        steps = []
        for i in range(0, len(lines), chunk_size):
            steps.append("\n".join(lines[i : i + chunk_size]))
        return _pack(steps)

    return _pack([text])


def segment_by_method(text: str, method: str = "legacy") -> SteppedSolution:
    method = (method or "legacy").strip().lower()
    if method in ("scope_nn", "scope", "double_newline", "nn"):
        return segment_scope_double_newline(text)
    if method in ("stride_tags", "stride", "tags"):
        return segment_stride_tags(text)
    return segment_solution(text)


def _pack(steps: list[str]) -> SteppedSolution:
    numbered = "\n\n".join(f"Step {i+1}:\n{s}" for i, s in enumerate(steps))
    return SteppedSolution(steps=steps, numbered_text=numbered)


def prefix_before(steps: list[str], t_star: int) -> str:
    """Return concatenated steps strictly before t* (1-indexed)."""
    if t_star <= 1:
        return ""
    keep = steps[: max(0, t_star - 1)]
    return "\n\n".join(f"Step {i+1}:\n{s}" for i, s in enumerate(keep))


def suffix_from(steps: list[str], t_star: int) -> str:
    """Return concatenated steps from t* onward (1-indexed)."""
    if t_star < 1:
        t_star = 1
    keep = steps[t_star - 1 :]
    return "\n\n".join(f"Step {t_star + i}:\n{s}" for i, s in enumerate(keep))


def first_erroneous_step_text(steps: list[str], t_star: int) -> str:
    if not steps:
        return ""
    idx = min(max(t_star, 1), len(steps)) - 1
    return f"Step {idx + 1}:\n{steps[idx]}"


def normalize_for_ppr(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"step\s+\d+\s*:\s*", "", text)
    return text.strip()


def prefix_preserved(y_prime: str, prefix: str, min_chars: int = 40) -> bool:
    """Heuristic PPR: does y' begin with / contain the correct prefix?"""
    pref = normalize_for_ppr(prefix)
    yp = normalize_for_ppr(y_prime)
    if not pref or len(pref) < min_chars:
        # Vacuous prefix: count as preserved
        return True
    # Exact head match or near-head containment
    if yp.startswith(pref[: min(len(pref), 200)]):
        return True
    # Soft: first 120 chars of prefix appear early in y'
    head = pref[:120]
    return head in yp[: max(300, len(head) + 100)]
