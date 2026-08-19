import re
from typing import Optional, Sequence, Union

_VERDICT_LINE = re.compile(r"The answer is (correct|wrong)\.\s*$")
_VERDICT_ANY = re.compile(r"The answer is (correct|wrong)\.")


def find_answer_status(text):
    # Match trailing "The answer is correct/wrong." (allow whitespace after the period)
    matches = []
    try:
        for match in re.finditer(_VERDICT_LINE, text):
            phrase = match.group(0).strip()
            start_pos = match.start()
            matches.append((phrase, start_pos))
    except Exception:
        matches = []
    return matches


def find_feedback_last_token_index(
    tokenizer,
    token_ids: Union[Sequence[int], None],
) -> Optional[int]:
    """Index of the last feedback token in ``token_ids`` (before the verdict).

    Verify format: ``<feedback> The answer is (correct|wrong).``
    Returns None if there is no verdict, or the verdict starts at token 0.
    """
    if token_ids is None:
        return None
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    token_ids = [int(t) for t in token_ids]
    if not token_ids:
        return None
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    matches = list(_VERDICT_ANY.finditer(text or ""))
    if not matches:
        return None
    char_start = matches[-1].start()
    if char_start <= 0:
        return None
    for i in range(len(token_ids)):
        prefix = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        if len(prefix) > char_start:
            return i - 1 if i > 0 else None
        if len(prefix) == char_start:
            return i
    return None


def get_verification_score(solution_str: str, gt_judge: bool) -> float:
    status_list = find_answer_status(solution_str)
    if len(status_list) == 0 or len(status_list) > 1:
        return {"genrm_score": 0, "genrm_pred": "wrong"}
    else:
        status, _ = status_list[-1]
        if "correct" in status and gt_judge:
            return {"genrm_score": 1, "genrm_pred": "correct"}
        elif "correct" in status and not gt_judge:
            return {"genrm_score": 0, "genrm_pred": "correct"}
        elif "wrong" in status and not gt_judge:
            return {"genrm_score": 1, "genrm_pred": "wrong"}
        else:
            return {"genrm_score": 0, "genrm_pred": "wrong"}
