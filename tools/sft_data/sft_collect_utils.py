"""Shared helpers for S2R-style SFT collection."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from answer_extraction import answer_corrected_match, extract_answer, extract_boxed_answers

INCORRECT_PATTERNS = (
    "is incorrect",
    "is likely incorrect",
    "is unlikely correct",
    "answer is wrong",
    "incorrect answer",
    "not the correct answer",
    "answer is not correct",
    "solution is incorrect",
    "calculation is wrong",
    "result is incorrect",
)
CORRECT_PATTERNS = (
    "is correct",
    "appears to be correct",
    "answer is reasonable",
    "solution is correct",
    "calculation is correct",
    "result is correct",
    "correctly solved",
    "answer checks out",
)


def get_soft_answer_correction(gold_answer: str, output_answer: str) -> bool:
    gold_answer = str(gold_answer or "")
    output_answer = str(output_answer or "")
    if "=" in gold_answer:
        gold_answer = gold_answer.strip().split("=")[-1].strip()
    if "=" in output_answer:
        output_answer = output_answer.strip().split("=")[-1].strip()
    if gold_answer == output_answer:
        return True
    if gold_answer and output_answer and (
        answer_corrected_match(gold_answer, output_answer)
        or answer_corrected_match(output_answer, gold_answer)
    ):
        return True
    return False


def is_incorrect_verification(sentence: str) -> bool:
    sl = sentence.lower()
    return any(p in sl for p in INCORRECT_PATTERNS)


def is_correct_verification(sentence: str) -> bool:
    sl = sentence.lower()
    return any(p in sl for p in CORRECT_PATTERNS)


# Luna often writes "The answer **4 is correct**." / "The answer is **correct**."
# at the top, then a boxed proof. Strip markdown before matching.


def _plain_verify(text: str) -> str:
    text = re.sub(r"\*\*", "", text or "")
    text = re.sub(r"\\boxed\{[^{}]*\}", " ", text)
    return text


def _verdict_from_span(span: str) -> str:
    sl = _plain_verify(span).lower()
    if not sl.strip():
        return ""
    if re.search(r"\bis\s+(?:not\s+correct|incorrect|wrong)\b", sl) or is_incorrect_verification(sl):
        return "incorrect"
    if is_correct_verification(sl) or re.search(r"\bis\s+correct\b", sl):
        return "correct"
    if "plausible" in sl:
        return "correct"
    if "cannot" in sl and "verif" in sl:
        return ""
    m = re.search(
        r"the answer\b.{0,80}?\bis\s+(not\s+correct|incorrect|wrong|correct)\b",
        sl,
        re.S,
    )
    if not m:
        return ""
    return "incorrect" if m.group(1).startswith(("not", "incorrect", "wrong")) else "correct"


def parse_verdict(verification: str) -> Tuple[str, str]:
    """Return (verdict, possibly trimmed verification). verdict in {correct,incorrect,''}."""
    verification = (verification or "").strip()
    if not verification:
        return "", verification
    sentences = _plain_verify(verification).lower().strip().split(".")
    last_sentence = sentences[-2] if verification.endswith(".") else sentences[-1]

    v = _verdict_from_span(last_sentence)
    if v:
        return v, verification
    if len(sentences) >= 3:
        second_last = sentences[-3] if verification.endswith(".") else sentences[-2]
        v = _verdict_from_span(second_last)
        if v:
            return v, verification
    first_para = verification.split("\n\n", 1)[0]
    v = _verdict_from_span(first_para)
    if v:
        return v, verification
    v = _verdict_from_span(verification)
    if v:
        return v, verification
    return "", verification


def boxed_or_extract(text: str) -> str:
    boxed = extract_boxed_answers(text or "")
    if boxed:
        return boxed[-1].split("=")[-1].strip()
    return extract_answer(text or "") or ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_pair_keys(path: Path) -> set:
    """Resume key: (unique_id, y0_extracted_answer)."""
    keys = set()
    for row in load_jsonl(path):
        uid = str(row.get("unique_id", ""))
        ans = str(row.get("y0_extracted_answer", ""))
        if uid:
            keys.add((uid, ans))
    return keys


def call_chat(
    client: Any,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    max_retries: int = 50,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
            }
            m = model.lower()
            if m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"[warn] API failed after retries: {e}")
                return None
            time.sleep(min(2 ** min(attempt, 5), 30))
    return None
