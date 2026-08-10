"""Shared PAG verify post-processing helpers."""
from __future__ import annotations

import re

WRONG_CLOSE = "The answer is wrong."
CORRECT_CLOSE = "The answer is correct."
GENERIC_VERIFY_ASSISTANT = (
    "The previous solution contains an error in its reasoning.\n"
    "The answer is wrong."
)

_VERDICT_TAIL = re.compile(
    r"(?:\n|\A)The answer is (?:correct|wrong)\.?\s*$",
    re.IGNORECASE,
)


def strip_verdict_close(text: str) -> str:
    """Remove trailing PAG verdict line(s) from verify text."""
    text = (text or "").strip()
    while text:
        new = _VERDICT_TAIL.sub("", text).rstrip()
        if new == text:
            break
        text = new
    return text


def parse_verdict(text: str) -> str:
    """Return 'correct', 'wrong', or 'none' from the model's trailing verdict."""
    text = (text or "").strip()
    if re.search(r"The answer is wrong\.?\s*$", text, re.IGNORECASE):
        return "wrong"
    if re.search(r"The answer is correct\.?\s*$", text, re.IGNORECASE):
        return "correct"
    return "none"


def ensure_wrong_close(text: str) -> str:
    """Strip any existing verdict, then force wrong close for regenerate gating."""
    body = strip_verdict_close(text)
    if not body:
        return GENERIC_VERIFY_ASSISTANT
    return body + "\n" + WRONG_CLOSE
