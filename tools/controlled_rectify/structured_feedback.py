#!/usr/bin/env python3
"""Structured verifier schema v=(c, t*, e, p) and feedback condition builders."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class StructuredVerify:
    verdict: str  # Correct / Incorrect
    first_error: int  # 1-indexed step id; 0 if Correct
    error_analysis: str
    rectification_plan: str
    raw_text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


STRUCTURED_TEACHER_SYSTEM = (
    "You are a math error localizer. Given a problem and an INCORRECT student "
    "solution already segmented into numbered steps, identify the FIRST erroneous step.\n"
    "Do NOT provide a full correct solution.\n"
    "Do NOT put any final answer in \\boxed{}.\n"
    "Output EXACTLY this format:\n"
    "Verdict: Incorrect\n"
    "First error: Step <k>\n"
    "Error analysis:\n"
    "<why that step is wrong; 1-3 sentences>\n"
    "Rectification plan:\n"
    "<what to do from the correct prefix; do not solve the whole problem>"
)

STRUCTURED_TEACHER_USER = (
    "Problem:\n{problem}\n\n"
    "Student solution (numbered steps):\n{numbered_solution}\n\n"
    "Ground-truth final answer (for localization only; do NOT reveal it):\n{gt}\n\n"
    "Locate the first error step and fill the required format."
)

FREEFORM_TEACHER_SYSTEM = (
    "You are a generative math verifier. Diagnose the incorrect solution: "
    "point out what went wrong and how to fix the reasoning. "
    "Do NOT provide a full worked solution. Do NOT use \\boxed{}. "
    "End with exactly: The answer is wrong."
)

FREEFORM_TEACHER_USER = (
    "Problem:\n{problem}\n\n"
    "Incorrect solution:\n{wrong_attempt}\n\n"
    "Provide diagnostic feedback (why/how), without solving the problem."
)


def render_structured(v: StructuredVerify) -> str:
    return (
        f"Verdict: {v.verdict}\n"
        f"First error: Step {v.first_error}\n"
        f"Error analysis:\n{v.error_analysis.strip()}\n"
        f"Rectification plan:\n{v.rectification_plan.strip()}"
    )


def parse_structured(text: str, n_steps: int) -> StructuredVerify:
    text = (text or "").strip()
    verdict = "Incorrect"
    m = re.search(r"Verdict:\s*(Correct|Incorrect)", text, re.I)
    if m:
        verdict = m.group(1).title()
        if verdict.lower() == "correct":
            verdict = "Correct"
        else:
            verdict = "Incorrect"

    first_error = 0
    m = re.search(r"First error:\s*Step\s*(\d+)", text, re.I)
    if m:
        first_error = int(m.group(1))
    elif verdict == "Incorrect":
        first_error = 1

    if n_steps > 0 and first_error > n_steps:
        first_error = n_steps
    if verdict == "Correct":
        first_error = 0

    ea = ""
    m = re.search(
        r"Error analysis:\s*(.*?)(?:\n\s*Rectification plan:|\Z)",
        text,
        re.I | re.S,
    )
    if m:
        ea = m.group(1).strip()

    rp = ""
    m = re.search(r"Rectification plan:\s*(.*)\Z", text, re.I | re.S)
    if m:
        rp = m.group(1).strip()

    # Strip leaked boxed answers
    ea = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", ea)
    rp = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", rp)

    if not ea:
        ea = "The indicated step contains an error in its reasoning."
    if not rp:
        rp = "Preserve the correct prefix and revise from the first error step."

    return StructuredVerify(
        verdict=verdict,
        first_error=first_error,
        error_analysis=ea,
        rectification_plan=rp,
        raw_text=text,
    )


def is_actionable(v: StructuredVerify) -> bool:
    if v.verdict != "Incorrect":
        return False
    if v.first_error < 1:
        return False
    if len(v.error_analysis.strip()) < 20:
        return False
    return True


def feedback_text_for_condition(cond: str, v: StructuredVerify, freeform: str = "") -> str:
    """Build the feedback body shown to the rectifier (not including edit protocol)."""
    cond = cond.lower()
    if cond in ("regenerate", "regen"):
        return ""
    if cond in ("wrong_only", "wrong"):
        return "The previous solution is incorrect."
    if cond in ("localization", "loc"):
        return (
            f"Verdict: Incorrect\n"
            f"First error: Step {v.first_error}"
        )
    if cond in ("localization_analysis", "loc_analysis", "analysis"):
        return (
            f"Verdict: Incorrect\n"
            f"First error: Step {v.first_error}\n"
            f"Error analysis:\n{v.error_analysis}"
        )
    if cond in ("localization_analysis_plan", "loc_analysis_plan", "plan"):
        return render_structured(v)
    if cond in ("freeform", "verification_why_how", "why_how"):
        return (freeform or "").strip() or (
            f"The solution is wrong. {v.error_analysis} {v.rectification_plan}\n"
            "The answer is wrong."
        )
    raise ValueError(f"unknown feedback condition: {cond}")


FEEDBACK_CONDITIONS = [
    "regenerate",
    "wrong_only",
    "localization",
    "localization_analysis",
    "localization_analysis_plan",
    "freeform",
]

EDIT_MODES = ["full_regen", "prefix_rewrite"]
