"""Online GPT expert for infeasible tasks.

When V_F > B_F and no bootstrap expert exists for that sample, call an
OpenAI-compatible API to produce verify + correct rectify, then overwrite
the row's response with an expert trajectory for BC.

Requires OPENAI_API_KEY; optional OPENAI_BASE_URL.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from verl import DataProto

VERIFY_TEMPLATE = (
    "Verify the previous solution without re-solving the problem from scratch. "
    "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
    "explain why it is wrong, and end your response with 'The answer is wrong'. "
    "If all steps are correct, end your response with 'The answer is correct'."
)
REGEN_TEMPLATE = (
    "You indicated that your previous answer was wrong. "
    "Please provide the correct solution to the math problem."
)

CRITIQUE_SYSTEM = (
    "You are a math error analyst acting as a generative verifier. "
    "Given a problem and an incorrect solution, identify the SPECIFIC error "
    "(wrong step and why). "
    "Do NOT provide the full correct solution. "
    "Do NOT put a final answer in \\boxed{}. "
    "Write 1-4 sentences describing what went wrong and what should be fixed, "
    "and end your response with exactly: The answer is wrong."
)

SOLVE_SYSTEM = (
    "You are an expert math tutor. Given a problem and an incorrect student solution "
    "(plus a short error diagnosis), write a complete correct solution. "
    "Reason step by step and put the final answer within \\boxed{}."
)

MAX_WRONG_CHARS = 3500


def truncate(text: str, n: int = MAX_WRONG_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "\n[... truncated ...]"


def uses_completion_tokens(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in m


def _chat_create(client, model: str, messages: list, temperature: float, max_tokens: int):
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if uses_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e)
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            return client.chat.completions.create(**kwargs)
        if "temperature" in msg.lower() and "unsupported" in msg.lower():
            kwargs.pop("temperature", None)
            return client.chat.completions.create(**kwargs)
        raise


def get_template_token_ids(tokenizer, template: str) -> List[int]:
    """Match vLLMPAGRollout._get_template_tokens encoding."""
    messages = [{"role": "user", "content": template}]
    chat_template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if chat_template.startswith("<|im_start|>system"):
        system_end = chat_template.find("<|im_end|>") + len("<|im_end|>")
        chat_template = chat_template[system_end:].lstrip()
    if chat_template.startswith("<｜begin▁of▁sentence｜>"):
        chat_template = chat_template[len("<｜begin▁of▁sentence｜>"):]
    chat_template = "\n\n" + chat_template
    return tokenizer.encode(chat_template, add_special_tokens=False)


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if list(haystack[i:i + m]) == list(needle):
            return i
    return -1


def extract_problem_text(tokenizer, prompt_ids: torch.Tensor, pad_token_id: int) -> str:
    ids = prompt_ids.detach().cpu().tolist()
    ids = [t for t in ids if t != pad_token_id]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # Prefer last user turn content if chat markers present
    for marker in ("user\n", "User:", "user:"):
        if marker in text:
            text = text.split(marker)[-1]
            break
    for end in ("assistant\n", "Assistant:", "<|im_start|>assistant"):
        if end in text:
            text = text.split(end)[0]
            break
    return text.strip()


def extract_wrong_answer_ids(
    response_ids: Sequence[int],
    verify_tokens: Sequence[int],
    pad_token_id: int,
) -> List[int]:
    """Tokens before the verify user template (the latest wrong attempt)."""
    ids = [t for t in response_ids if t != pad_token_id]
    pos = find_subsequence(ids, verify_tokens)
    if pos <= 0:
        # fallback: whole non-pad response as wrong attempt
        return ids[: min(len(ids), 512)]
    return ids[:pos]


@dataclass
class GPTExpertResult:
    verify_text: str
    rectify_text: str


class OnlineGPTExpertClient:
    """OpenAI-compatible client; `solve_fn` can be injected for tests."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens_verify: int = 512,
        max_tokens_rectify: int = 1024,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        solve_fn: Optional[Callable[[str, str], GPTExpertResult]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens_verify = max_tokens_verify
        self.max_tokens_rectify = max_tokens_rectify
        self.max_retries = max_retries
        self.solve_fn = solve_fn
        self._client = None
        if solve_fn is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for online GPT expert")
            from openai import OpenAI
            kwargs = {"api_key": key}
            url = base_url or os.environ.get("OPENAI_BASE_URL")
            if url:
                kwargs["base_url"] = url
            self._client = OpenAI(**kwargs)

    def _call(self, messages: list, max_tokens: int) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = _chat_create(
                    self._client, self.model, messages, self.temperature, max_tokens
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"GPT call failed: {last_err}")

    def generate_expert(self, problem: str, wrong_attempt: str) -> GPTExpertResult:
        if self.solve_fn is not None:
            return self.solve_fn(problem, wrong_attempt)

        wrong = truncate(wrong_attempt)
        verify_messages = [
            {"role": "system", "content": CRITIQUE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Incorrect solution:\n{wrong}\n\n"
                    "Identify the specific error. Do not solve the problem."
                ),
            },
        ]
        verify_text = self._call(verify_messages, self.max_tokens_verify)
        verify_text = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", verify_text).strip()
        if "The answer is wrong" not in verify_text:
            verify_text = (verify_text + "\nThe answer is wrong.").strip()

        solve_messages = [
            {"role": "system", "content": SOLVE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Incorrect solution:\n{wrong}\n\n"
                    f"Error diagnosis:\n{verify_text}\n\n"
                    "Provide the correct solution with \\boxed{}."
                ),
            },
        ]
        rectify_text = self._call(solve_messages, self.max_tokens_rectify)
        if "\\boxed{" not in rectify_text:
            # still usable as BC target; keep as-is
            pass
        return GPTExpertResult(verify_text=verify_text, rectify_text=rectify_text)


def pack_expert_response(
    answer_ids: List[int],
    verify_tokens: List[int],
    verify_ids: List[int],
    regen_tokens: List[int],
    rectify_ids: List[int],
    max_resp: int,
    pad_token_id: int,
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build slide-window style response + masks.

    Returns responses, resp_attn, multiturn_mask, expert_token_mask (1D each).
    """
    pieces = [
        (answer_ids, False, False),       # context / failed attempt
        (verify_tokens, False, False),    # user template
        (verify_ids, True, True),         # GPT verify (BC)
        (regen_tokens, False, False),     # user template
        (rectify_ids, True, True),        # GPT rectify (BC)
    ]
    responses = torch.full((max_resp,), pad_token_id, device=device, dtype=dtype)
    resp_attn = torch.zeros((max_resp,), dtype=torch.bool, device=device)
    multiturn_mask = torch.zeros((max_resp,), dtype=torch.bool, device=device)
    expert_mask = torch.zeros((max_resp,), dtype=torch.bool, device=device)

    pos = 0
    for piece, is_model, is_expert in pieces:
        n = len(piece)
        if pos + n > max_resp:
            n = max_resp - pos
            piece = piece[:n]
        if n <= 0:
            break
        responses[pos:pos + n] = torch.tensor(piece, device=device, dtype=dtype)
        resp_attn[pos:pos + n] = True
        if is_model:
            multiturn_mask[pos:pos + n] = True
        if is_expert:
            expert_mask[pos:pos + n] = True
        pos += n
    return responses, resp_attn, multiturn_mask, expert_mask


def fill_infeasible_with_online_gpt(
    batch: DataProto,
    tokenizer,
    client: OnlineGPTExpertClient,
    max_per_step: int = 8,
    prefer_bootstrap: bool = True,
    num_workers: int = 4,
) -> Tuple[DataProto, Dict[str, float]]:
    """Overwrite infeasible rows lacking bootstrap experts with GPT trajectories.

    Call after critic update / before actor update. Keeps batch size unchanged.
    """
    metrics = {
        "feasibility/online_gpt_candidates": 0.0,
        "feasibility/online_gpt_filled": 0.0,
        "feasibility/online_gpt_failed": 0.0,
        "feasibility/online_gpt_skipped_bootstrap": 0.0,
    }
    if "feas_gate" not in batch.batch or "responses" not in batch.batch:
        return batch, metrics

    device = batch.batch["responses"].device
    dtype = batch.batch["responses"].dtype
    pad_id = tokenizer.pad_token_id
    responses = batch.batch["responses"]
    B, max_resp = responses.shape

    verify_tokens = get_template_token_ids(tokenizer, VERIFY_TEMPLATE)
    regen_tokens = get_template_token_ids(tokenizer, REGEN_TEMPLATE)

    feas_gate = batch.batch["feas_gate"]
    need = feas_gate < 0.5
    if "window_valid" in batch.non_tensor_batch:
        valid = torch.tensor(
            batch.non_tensor_batch["window_valid"].astype(np.bool_),
            device=need.device,
        )
        need = need & valid

    skipped_bootstrap = 0
    if prefer_bootstrap and "expert_token_mask" in batch.batch:
        has_expert = batch.batch["expert_token_mask"].reshape(B, -1).any(dim=-1)
        skipped_bootstrap = int((need & has_expert).sum().item())
        need = need & (~has_expert)
    metrics["feasibility/online_gpt_skipped_bootstrap"] = float(skipped_bootstrap)

    idxs = torch.where(need)[0].tolist()
    metrics["feasibility/online_gpt_candidates"] = float(len(idxs))
    if not idxs:
        return batch, metrics

    if max_per_step > 0 and len(idxs) > max_per_step:
        rng = np.random.default_rng()
        idxs = rng.choice(idxs, size=max_per_step, replace=False).tolist()

    # prompts may be under 'prompts' or left part of input_ids
    if "prompts" in batch.batch:
        prompts = batch.batch["prompts"]
    else:
        resp_len = responses.size(1)
        prompts = batch.batch["input_ids"][:, :-resp_len]

    jobs = []
    for i in idxs:
        problem = extract_problem_text(tokenizer, prompts[i], pad_id)
        wrong_ids = extract_wrong_answer_ids(
            responses[i].detach().cpu().tolist(), verify_tokens, pad_id
        )
        wrong_text = tokenizer.decode(wrong_ids, skip_special_tokens=True)
        jobs.append((i, problem, wrong_text, wrong_ids))

    results: Dict[int, Tuple[GPTExpertResult, List[int]]] = {}
    failed = 0

    def _one(job):
        i, problem, wrong_text, wrong_ids = job
        res = client.generate_expert(problem, wrong_text)
        return i, res, wrong_ids

    workers = max(1, min(num_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                i, res, wrong_ids = fut.result()
                if not res.verify_text or not res.rectify_text:
                    failed += 1
                    continue
                results[i] = (res, wrong_ids)
            except Exception as e:
                print(f"[online_gpt_expert] failed: {e}")
                failed += 1

    metrics["feasibility/online_gpt_failed"] = float(failed)
    if not results:
        return batch, metrics

    # Ensure expert_token_mask exists
    if "expert_token_mask" not in batch.batch:
        batch.batch["expert_token_mask"] = torch.zeros_like(responses, dtype=torch.bool)

    filled = 0
    for i, (res, wrong_ids) in results.items():
        verify_ids = tokenizer.encode(res.verify_text, add_special_tokens=False)
        rectify_ids = tokenizer.encode(res.rectify_text, add_special_tokens=False)
        new_resp, resp_attn, mt_mask, exp_mask = pack_expert_response(
            answer_ids=wrong_ids,
            verify_tokens=verify_tokens,
            verify_ids=verify_ids,
            regen_tokens=regen_tokens,
            rectify_ids=rectify_ids,
            max_resp=max_resp,
            pad_token_id=pad_id,
            device=device,
            dtype=dtype,
        )
        if not exp_mask.any():
            failed += 1
            continue

        batch.batch["responses"][i] = new_resp
        batch.batch["expert_token_mask"][i] = exp_mask
        batch.batch["multiturn_mask"][i] = mt_mask

        # rebuild full sequence tensors
        prompt_i = prompts[i]
        prompt_len = prompt_i.size(0)
        seq = torch.cat([prompt_i, new_resp], dim=-1)
        batch.batch["input_ids"][i] = seq

        # attention: keep prompt attn, replace response attn
        if "attention_mask" in batch.batch:
            attn = batch.batch["attention_mask"][i]
            prompt_attn = attn[:prompt_len]
            batch.batch["attention_mask"][i] = torch.cat([prompt_attn, resp_attn], dim=-1)

        if "position_ids" in batch.batch:
            pos = batch.batch["position_ids"][i]
            prompt_pos = pos[:prompt_len]
            # handle 1D or multi-dim rope
            if prompt_pos.dim() == 1:
                delta = torch.arange(1, max_resp + 1, device=device, dtype=prompt_pos.dtype)
                resp_pos = prompt_pos[-1:] + delta
                batch.batch["position_ids"][i] = torch.cat([prompt_pos, resp_pos], dim=-1)
            else:
                # (..., seq) — rare; leave as-is if shapes weird
                pass

        # Actor PPO path is gated off; keep advantages / old_log_probs safe
        if "advantages" in batch.batch:
            batch.batch["advantages"][i].zero_()
        if "old_log_probs" in batch.batch:
            batch.batch["old_log_probs"][i].zero_()
        if "ref_log_prob" in batch.batch:
            batch.batch["ref_log_prob"][i].zero_()
        batch.batch["feas_gate"][i] = 0.0
        if "feas_weight" in batch.batch:
            batch.batch["feas_weight"][i] = torch.clamp(
                batch.batch["feas_weight"][i], min=0.5
            )
        filled += 1

    metrics["feasibility/online_gpt_filled"] = float(filled)
    metrics["feasibility/online_gpt_failed"] = float(failed)
    return batch, metrics
