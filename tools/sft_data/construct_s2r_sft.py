#!/usr/bin/env python3
"""Step 3 — assemble S2R-style monologue SFT JSON from solutions + verifications.

Migrated from S2R ``tools/3_contruct_muliti_turn_data.py`` (typo kept in upstream name).

Correct retry tails are sampled from:
  - gold-correct **y0** (same as official S2R), and optionally
  - gold-correct **y1** from ``collect_y1.py`` (base model + GPT critique).

When a wrong y0 has a matching gold y1, that y1 is preferred as the stitch tail.

Output matches the release format of ``sft_qwen2.5_math_7B.json``.

Example:
  python tools/sft_data/construct_s2r_sft.py \\
    --solutions datasets/sft_collect/solutions.jsonl \\
    --verifications datasets/sft_collect/verifications.jsonl \\
    --y1 datasets/sft_collect/y1.jsonl \\
    --output datasets/sft_collect/sft_s2r_style.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from answer_extraction import (  # noqa: E402
    answer_corrected_match,
    extract_boxed_answers,
)
from sft_collect_utils import load_jsonl, parse_verdict  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
# y1 is not GPT-verified; convert() strips this Wait-tail. Needed for S2R format.
CANONICAL_CORRECT_VERIFY = (
    "The boxed answer is consistent with the problem constraints. "
    "Therefore, the answer is correct."
)


def get_soft_answer_correction(gold_answer: str, output_answer: str) -> bool:
    if "=" in gold_answer:
        gold_answer = gold_answer.strip().split("=")[-1].strip()
    if "=" in output_answer:
        output_answer = output_answer.strip().split("=")[-1].strip()
    if gold_answer == output_answer:
        return True
    if answer_corrected_match(gold_answer, output_answer) or answer_corrected_match(
        output_answer, gold_answer
    ):
        return True
    return False


def process_lines(
    lines: List[Dict[str, Any]],
    response_verification_dict: Dict[str, Dict[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for line in tqdm(lines, desc="match verify"):
        unique_id = line["unique_id"]
        gold = line["gold_extracted_answer"]
        result: Dict[str, Any] = {
            "unique_id": unique_id,
            "round_1_instruction": line.get("round_1_instruction", line["problem"]),
            "problem": line["problem"],
            "correct_response_veri": None,
            "incorrect_response_veri": None,
            "round_1_extracted_answer": line.get("round_1_extracted_answer"),
            "gold_extracted_answer": gold,
        }

        buckets = response_verification_dict.get(unique_id, {})
        res_veri_list: List[Dict[str, str]] = []
        for key in ("primary", "qwen1", "mistral", "qwen2"):
            res_veri_list.extend(buckets.get(key, []))

        # Prefer pairs whose boxed answer also appears in this problem's round_1
        reusable: List[Dict[str, str]] = []
        for pair in res_veri_list:
            other_boxed = extract_boxed_answers(pair["response"])
            for r1 in line.get("round_1_response") or []:
                if other_boxed == extract_boxed_answers(r1):
                    reusable.append(
                        {"response": r1, "verification": pair["verification"]}
                    )
                    break
        res_veri_list = reusable or res_veri_list
        if not res_veri_list:
            continue

        correct_prior: List[Dict[str, str]] = []
        correct_later: List[Dict[str, str]] = []
        incorrect_prior: List[Dict[str, str]] = []
        incorrect_later: List[Dict[str, str]] = []
        exist_answer_set = set()
        primary_resps = {
            p["response"]
            for p in (buckets.get("primary", []) + buckets.get("qwen1", []))
        }

        for res_veri in res_veri_list:
            response = res_veri["response"]
            verification = res_veri["verification"]
            boxed = extract_boxed_answers(response)
            if not boxed:
                continue
            extracted = boxed[-1].split("=")[-1].strip()
            is_ok = get_soft_answer_correction(gold, extracted)
            veri_answer, verification = parse_verdict(verification)
            if not veri_answer:
                continue

            in_primary = response in primary_resps
            if is_ok and veri_answer == "correct":
                item = {
                    "response": response,
                    "verification": verification,
                    "source": "y0",
                    "extracted": extracted,
                }
                (correct_prior if in_primary else correct_later).append(item)
            elif (not is_ok) and veri_answer == "incorrect":
                if extracted in exist_answer_set:
                    continue
                item = {
                    "response": response,
                    "verification": verification,
                    "source": "y0",
                    "extracted": extracted,
                }
                (incorrect_prior if in_primary else incorrect_later).append(item)
                exist_answer_set.add(extracted)

        result["correct_response_veri"] = correct_prior or correct_later
        result["incorrect_response_veri_prior"] = incorrect_prior
        result["incorrect_response_veri_later"] = incorrect_later
        result["y1_correct_by_wrong"] = {}
        processed.append(result)
    return processed


def merge_y1(
    processed: List[Dict[str, Any]], y1_path: Optional[Path]
) -> Dict[str, int]:
    """Add gold-correct y1 into the correct pool; index by the wrong y0 answer."""
    stats = {"n_y1": 0, "n_y1_gold": 0, "n_rescued": 0}
    if y1_path is None or not y1_path.exists():
        return stats
    by_uid: Dict[str, List[Dict[str, Any]]] = {}
    for rec in load_jsonl(y1_path):
        by_uid.setdefault(str(rec.get("unique_id", "")), []).append(rec)
        stats["n_y1"] += 1

    for result in processed:
        uid = str(result["unique_id"])
        had_y0_correct = bool(result.get("correct_response_veri"))
        y1_map: Dict[str, Dict[str, str]] = {}
        extra: List[Dict[str, str]] = []
        seen_resp = {c["response"] for c in (result.get("correct_response_veri") or [])}
        for rec in by_uid.get(uid, []):
            if not rec.get("y1_matches_gold"):
                continue
            y1 = rec.get("y1") or ""
            if not y1 or y1 in seen_resp:
                continue
            stats["n_y1_gold"] += 1
            item = {
                "response": y1,
                "verification": CANONICAL_CORRECT_VERIFY,
                "source": "y1",
                "extracted": rec.get("y1_extracted_answer") or "",
                "from_y0_answer": str(rec.get("y0_extracted_answer", "")),
            }
            extra.append(item)
            seen_resp.add(y1)
            y1_map[str(rec.get("y0_extracted_answer", ""))] = item
        if extra:
            result["correct_response_veri"] = (
                list(result.get("correct_response_veri") or []) + extra
            )
            if not had_y0_correct:
                stats["n_rescued"] += 1
        result["y1_correct_by_wrong"] = y1_map
    return stats


def construct_answer(res_veri_list: List[Dict[str, str]]) -> str:
    parts = []
    for idx, res_veri in enumerate(res_veri_list):
        parts.append(
            f'{res_veri["response"]}\n\nWait, let me recheck my solution.\n\n'
            f'{res_veri["verification"]}'
        )
        if idx != len(res_veri_list) - 1:
            parts.append("\n\nLet me try again.\n\n")
    return "".join(parts)


def _pick_correct_tail(
    incorrect_chosen: List[Dict[str, str]],
    correct: List[Dict[str, str]],
    y1_by_wrong: Dict[str, Dict[str, str]],
    prefer_y1: bool,
) -> Dict[str, str]:
    if prefer_y1 and incorrect_chosen:
        last_wrong = str(incorrect_chosen[-1].get("extracted", ""))
        if last_wrong and last_wrong in y1_by_wrong:
            return y1_by_wrong[last_wrong]
        y1s = [c for c in correct if c.get("source") == "y1"]
        if y1s:
            return random.choice(y1s)
    return random.choice(correct)


def select_train_data(
    processed_results: List[Dict[str, Any]],
    keep_prob_scale: float = 1.0,
    prefer_y1: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """S2R sampling heuristic over #wrong answers among n rollouts."""
    train_data: List[Dict[str, str]] = []
    data_length_count: Dict[str, int] = collections.defaultdict(int)

    def keep(p: float) -> bool:
        return random.random() <= min(1.0, p * keep_prob_scale)

    for result in processed_results:
        problem = result["problem"]
        correct = result.get("correct_response_veri") or []
        incorrect = (result.get("incorrect_response_veri_prior") or []) + (
            result.get("incorrect_response_veri_later") or []
        )
        y1_by_wrong = result.get("y1_correct_by_wrong") or {}
        if not correct:
            continue

        answers = result.get("round_1_extracted_answer") or []
        gold = result.get("gold_extracted_answer")
        correct_count = sum(1 for ans in answers if ans == gold)
        wrong_count = len(answers) - correct_count

        if wrong_count == 0:
            if keep(0.76 * 0.45 * 0.7):
                train_data.append(
                    {
                        "prompt": problem,
                        "answer": construct_answer([random.choice(correct)]),
                    }
                )
                data_length_count[1] += 1
        elif wrong_count == 1 and len(incorrect) >= 1:
            if keep(0.5):
                inc = random.sample(incorrect, 1)
                tail = _pick_correct_tail(inc, correct, y1_by_wrong, prefer_y1)
                train_data.append(
                    {"prompt": problem, "answer": construct_answer(inc + [tail])}
                )
                data_length_count[2] += 1
                data_length_count[f"tail_{tail.get('source', 'y0')}"] += 1
        elif 2 <= wrong_count <= 3:
            if len(incorrect) >= 2:
                if keep(0.5):
                    inc = random.sample(incorrect, 2)
                    tail = _pick_correct_tail(inc, correct, y1_by_wrong, prefer_y1)
                    train_data.append(
                        {"prompt": problem, "answer": construct_answer(inc + [tail])}
                    )
                    data_length_count[3] += 1
                    data_length_count[f"tail_{tail.get('source', 'y0')}"] += 1
            elif len(incorrect) == 1:
                inc = random.sample(incorrect, 1)
                tail = _pick_correct_tail(inc, correct, y1_by_wrong, prefer_y1)
                train_data.append(
                    {"prompt": problem, "answer": construct_answer(inc + [tail])}
                )
                data_length_count[2] += 1
                data_length_count[f"tail_{tail.get('source', 'y0')}"] += 1
        elif wrong_count >= 4:
            if len(incorrect) >= 3:
                if keep(0.5):
                    inc = random.sample(incorrect, 3)
                    tail = _pick_correct_tail(inc, correct, y1_by_wrong, prefer_y1)
                    train_data.append(
                        {"prompt": problem, "answer": construct_answer(inc + [tail])}
                    )
                    data_length_count[4] += 1
                    data_length_count[f"tail_{tail.get('source', 'y0')}"] += 1
            elif len(incorrect) == 2:
                inc = random.sample(incorrect, 2)
                tail = _pick_correct_tail(inc, correct, y1_by_wrong, prefer_y1)
                train_data.append(
                    {"prompt": problem, "answer": construct_answer(inc + [tail])}
                )
                data_length_count[3] += 1
                data_length_count[f"tail_{tail.get('source', 'y0')}"] += 1
            elif len(incorrect) == 1:
                inc = random.sample(incorrect, 1)
                tail = _pick_correct_tail(inc, correct, y1_by_wrong, prefer_y1)
                train_data.append(
                    {"prompt": problem, "answer": construct_answer(inc + [tail])}
                )
                data_length_count[2] += 1
                data_length_count[f"tail_{tail.get('source', 'y0')}"] += 1
    return train_data, dict(data_length_count)


def transform_data(
    dataset: List[Dict[str, str]], tokenizer: Any
) -> List[Dict[str, str]]:
    new_dataset = []
    for data in tqdm(dataset, desc="chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": data["prompt"]},
            {"role": "assistant", "content": data["answer"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=True, tokenize=False
        )
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        answer = text[len(prompt) :]
        new_dataset.append({"prompt": prompt, "answer": answer})
    return new_dataset


def load_verification_bucket(
    path: Path, key_name: str = "primary"
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    out: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = str(obj["unique_id"])
            out.setdefault(uid, {})
            out[uid].setdefault(key_name, [])
            responses = obj.get("round_1_response") or []
            veris = obj.get("verification") or []
            for resp, veri in zip(responses, veris):
                if not resp or not veri:
                    continue
                cleaned = (
                    str(veri)
                    .replace("original answer", "answer")
                    .replace(" without solving the problem step by step", "")
                )
                out[uid][key_name].append(
                    {"response": resp, "verification": cleaned}
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--verifications", type=Path, required=True)
    ap.add_argument(
        "--y1",
        type=Path,
        default=None,
        help="optional collect_y1.py jsonl; gold-correct y1 join the correct pool",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="tokenizer only (chat template)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--keep_prob_scale",
        type=float,
        default=1.0,
        help="multiply S2R keep probabilities ( >1 keeps more rows)",
    )
    ap.add_argument(
        "--prefer_y1",
        action="store_true",
        default=True,
        help="prefer a gold-correct y1 (paired to the wrong y0) as the stitch tail",
    )
    ap.add_argument("--no_prefer_y1", action="store_false", dest="prefer_y1")
    args = ap.parse_args()
    random.seed(args.seed)

    with args.solutions.open(encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    response_veri_dict = load_verification_bucket(args.verifications, "primary")
    processed = process_lines(lines, response_veri_dict)
    y1_stats = merge_y1(processed, args.y1)
    res_data, length_count = select_train_data(
        processed, args.keep_prob_scale, prefer_y1=args.prefer_y1
    )
    print("length_count:", length_count, "raw_selected:", len(res_data), "y1:", y1_stats)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    res_data = transform_data(res_data, tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(res_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    stats = {
        "n_solutions": len(lines),
        "n_processed": len(processed),
        "n_sft": len(res_data),
        "length_count": length_count,
        "seed": args.seed,
        "keep_prob_scale": args.keep_prob_scale,
        "model_tokenizer": args.model,
        "y1": y1_stats,
        "prefer_y1": args.prefer_y1,
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(res_data)} → {args.output}")
    print(f"stats → {stats_path}")


if __name__ == "__main__":
    main()
