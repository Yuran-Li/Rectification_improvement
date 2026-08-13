"""Online GPT expert: state-conditioned recovery supervision.

Gate at decision state s → expert action with matching conditioning:
  F(s^V)>ε, s^V=(x,y_i)  → v^E ~ π_E(·|x,y_i)
  F(s^R)>ε, s^R=(x,y_i,v_i^A) → y_{i+1}^E ~ π_E(·|x,y_i,v_i^A)
  (freeze student prefix; do NOT regenerate verify when gating s^R)

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
    "(plus the student's own error diagnosis), write a complete correct solution. "
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
        return ids[: min(len(ids), 512)]
    return ids[:pos]


def split_response_roles(
    response_ids: Sequence[int],
    verify_tokens: Sequence[int],
    regen_tokens: Sequence[int],
    pad_token_id: int,
) -> Tuple[List[int], List[int], List[int], int, int]:
    """Split into (y_ids, v_ids, y_next_ids, verify_model_start, rectify_start)."""
    ids = [int(t) for t in response_ids if int(t) != pad_token_id]
    vpos = find_subsequence(ids, verify_tokens)
    if vpos < 0:
        return ids, [], [], -1, -1
    y_ids = ids[:vpos]
    after_v_tmpl = vpos + len(verify_tokens)
    rpos = find_subsequence(ids[after_v_tmpl:], regen_tokens)
    if rpos < 0:
        return y_ids, ids[after_v_tmpl:], [], after_v_tmpl, -1
    rpos = after_v_tmpl + rpos
    v_ids = ids[after_v_tmpl:rpos]
    rectify_start = rpos + len(regen_tokens)
    y_next = ids[rectify_start:]
    return y_ids, v_ids, y_next, after_v_tmpl, rectify_start


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

    def generate_verify(self, problem: str, wrong_attempt: str) -> str:
        """v^E ~ π_E(· | s^V=(x,y_i))."""
        if self.solve_fn is not None:
            return self.solve_fn(problem, wrong_attempt).verify_text
        wrong = truncate(wrong_attempt)
        messages = [
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
        verify_text = self._call(messages, self.max_tokens_verify)
        verify_text = re.sub(r"\\boxed\{[^{}]*\}", "[ANSWER REMOVED]", verify_text).strip()
        if "The answer is wrong" not in verify_text and "The answer is correct" not in verify_text:
            verify_text = (verify_text + "\nThe answer is wrong.").strip()
        return verify_text

    def generate_rectify(self, problem: str, wrong_attempt: str, student_verify: str) -> str:
        """y^E ~ π_E(· | s^R=(x,y_i,v_i^A)) — freeze student verify."""
        if self.solve_fn is not None:
            return self.solve_fn(problem, wrong_attempt).rectify_text
        wrong = truncate(wrong_attempt)
        diagnosis = truncate(student_verify, 2000)
        messages = [
            {"role": "system", "content": SOLVE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Incorrect solution:\n{wrong}\n\n"
                    f"Student error diagnosis:\n{diagnosis}\n\n"
                    "Provide the correct solution with \\boxed{}."
                ),
            },
        ]
        return self._call(messages, self.max_tokens_rectify)

    def generate_expert(self, problem: str, wrong_attempt: str) -> GPTExpertResult:
        """Legacy both-roles helper for tests."""
        if self.solve_fn is not None:
            return self.solve_fn(problem, wrong_attempt)
        verify_text = self.generate_verify(problem, wrong_attempt)
        rectify_text = self.generate_rectify(problem, wrong_attempt, verify_text)
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
    """Legacy full-traj pack (tests). Production fill uses role-specific rewrite."""
    pieces = [
        (answer_ids, False, False),
        (verify_tokens, False, False),
        (verify_ids, True, True),
        (regen_tokens, False, False),
        (rectify_ids, True, True),
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


def _rebuild_row_prefix(
    batch: DataProto,
    i: int,
    prompts: torch.Tensor,
    new_resp: torch.Tensor,
    resp_attn: torch.Tensor,
    max_resp: int,
    device,
) -> None:
    prompt_i = prompts[i]
    prompt_len = prompt_i.size(0)
    batch.batch["responses"][i] = new_resp
    batch.batch["input_ids"][i] = torch.cat([prompt_i, new_resp], dim=-1)
    if "attention_mask" in batch.batch:
        attn = batch.batch["attention_mask"][i]
        prompt_attn = attn[:prompt_len]
        batch.batch["attention_mask"][i] = torch.cat([prompt_attn, resp_attn], dim=-1)
    if "position_ids" in batch.batch:
        pos = batch.batch["position_ids"][i]
        prompt_pos = pos[:prompt_len]
        if prompt_pos.dim() == 1:
            delta = torch.arange(1, max_resp + 1, device=device, dtype=prompt_pos.dtype)
            resp_pos = prompt_pos[-1:] + delta
            batch.batch["position_ids"][i] = torch.cat([prompt_pos, resp_pos], dim=-1)
    if "advantages" in batch.batch:
        batch.batch["advantages"][i].zero_()
    if "old_log_probs" in batch.batch:
        batch.batch["old_log_probs"][i].zero_()
    if "ref_log_prob" in batch.batch:
        batch.batch["ref_log_prob"][i].zero_()


def fill_infeasible_with_online_gpt(
    batch: DataProto,
    tokenizer,
    client: OnlineGPTExpertClient,
    max_per_step: int = 8,
    prefer_bootstrap: bool = True,
    num_workers: int = 4,
) -> Tuple[DataProto, Dict[str, float]]:
    """State-conditioned GPT at infeasible s^V / s^R (matching student conditioning)."""
    metrics = {
        "feasibility/online_gpt_candidates": 0.0,
        "feasibility/online_gpt_filled": 0.0,
        "feasibility/online_gpt_filled_v": 0.0,
        "feasibility/online_gpt_filled_r": 0.0,
        "feasibility/online_gpt_failed": 0.0,
        "feasibility/online_gpt_skipped_bootstrap": 0.0,
    }
    if "responses" not in batch.batch:
        return batch, metrics

    device = batch.batch["responses"].device
    dtype = batch.batch["responses"].dtype
    pad_id = tokenizer.pad_token_id
    responses = batch.batch["responses"]
    B, max_resp = responses.shape

    verify_tokens = get_template_token_ids(tokenizer, VERIFY_TEMPLATE)
    regen_tokens = get_template_token_ids(tokenizer, REGEN_TEMPLATE)

    if "feas_gate_r" in batch.batch:
        need_r = batch.batch["feas_gate_r"] < 0.5
    elif "feas_gate" in batch.batch:
        need_r = batch.batch["feas_gate"] < 0.5
    else:
        need_r = torch.zeros(B, dtype=torch.bool, device=device)
    if "feas_gate_v" in batch.batch:
        need_v = batch.batch["feas_gate_v"] < 0.5
    elif "feas_gate" in batch.batch:
        need_v = batch.batch["feas_gate"] < 0.5
    else:
        need_v = torch.zeros(B, dtype=torch.bool, device=device)

    if "window_valid" in batch.non_tensor_batch:
        valid = torch.tensor(
            batch.non_tensor_batch["window_valid"].astype(np.bool_),
            device=device,
        )
        need_r = need_r & valid
        need_v = need_v & valid

    skipped = 0
    if prefer_bootstrap:
        if "expert_token_mask_r" in batch.batch:
            has_r = batch.batch["expert_token_mask_r"].reshape(B, -1).any(dim=-1)
            skipped += int((need_r & has_r).sum().item())
            need_r = need_r & (~has_r)
        elif "expert_token_mask" in batch.batch:
            has_e = batch.batch["expert_token_mask"].reshape(B, -1).any(dim=-1)
            skipped += int((need_r & has_e).sum().item())
            need_r = need_r & (~has_e)
        if "expert_token_mask_v" in batch.batch:
            has_v = batch.batch["expert_token_mask_v"].reshape(B, -1).any(dim=-1)
            skipped += int((need_v & has_v).sum().item())
            need_v = need_v & (~has_v)
        elif "expert_token_mask" in batch.batch:
            has_e = batch.batch["expert_token_mask"].reshape(B, -1).any(dim=-1)
            skipped += int((need_v & has_e).sum().item())
            need_v = need_v & (~has_e)
    metrics["feasibility/online_gpt_skipped_bootstrap"] = float(skipped)

    jobs: List[Tuple[str, int]] = []
    for i in torch.where(need_r)[0].tolist():
        jobs.append(("r", i))
    for i in torch.where(need_v & (~need_r))[0].tolist():
        jobs.append(("v", i))
    metrics["feasibility/online_gpt_candidates"] = float(len(jobs))
    if not jobs:
        return batch, metrics
    if max_per_step > 0 and len(jobs) > max_per_step:
        rng = np.random.default_rng()
        pick = rng.choice(len(jobs), size=max_per_step, replace=False)
        jobs = [jobs[j] for j in pick]

    if "prompts" in batch.batch:
        prompts = batch.batch["prompts"]
    else:
        prompts = batch.batch["input_ids"][:, :-max_resp]

    if "expert_token_mask" not in batch.batch:
        batch.batch["expert_token_mask"] = torch.zeros_like(responses, dtype=torch.bool)
    if "expert_token_mask_v" not in batch.batch:
        batch.batch["expert_token_mask_v"] = torch.zeros_like(responses, dtype=torch.bool)
    if "expert_token_mask_r" not in batch.batch:
        batch.batch["expert_token_mask_r"] = torch.zeros_like(responses, dtype=torch.bool)
    if "multiturn_mask" not in batch.batch:
        batch.batch["multiturn_mask"] = torch.zeros_like(responses, dtype=torch.bool)

    prepared = []
    for role, i in jobs:
        problem = extract_problem_text(tokenizer, prompts[i], pad_id)
        y_ids, v_ids, _, _, r_start = split_response_roles(
            responses[i].detach().cpu().tolist(), verify_tokens, regen_tokens, pad_id
        )
        wrong_text = tokenizer.decode(y_ids, skip_special_tokens=True)
        student_v = tokenizer.decode(v_ids, skip_special_tokens=True) if v_ids else ""
        prepared.append((role, i, problem, wrong_text, student_v, y_ids, v_ids, r_start))

    results: Dict[Tuple[str, int], str] = {}
    failed = 0

    def _one(item):
        role, i, problem, wrong_text, student_v, *_rest = item
        if role == "r":
            text = client.generate_rectify(
                problem, wrong_text, student_v or "The answer is wrong."
            )
        else:
            text = client.generate_verify(problem, wrong_text)
        return role, i, text

    workers = max(1, min(num_workers, len(prepared)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, p) for p in prepared]
        for fut in as_completed(futs):
            try:
                role, i, text = fut.result()
                if not text:
                    failed += 1
                    continue
                results[(role, i)] = text
            except Exception as e:
                print(f"[online_gpt_expert] failed: {e}")
                failed += 1

    filled = filled_v = filled_r = 0
    meta = {(p[0], p[1]): p for p in prepared}
    for (role, i), text in results.items():
        _, _, _, _, _, y_ids, v_ids, r_start = meta[(role, i)]
        new_resp = torch.full((max_resp,), pad_id, device=device, dtype=dtype)
        mt = torch.zeros(max_resp, dtype=torch.bool, device=device)
        exp = torch.zeros(max_resp, dtype=torch.bool, device=device)
        exp_v = torch.zeros(max_resp, dtype=torch.bool, device=device)
        exp_r = torch.zeros(max_resp, dtype=torch.bool, device=device)

        if role == "r":
            if r_start < 0:
                failed += 1
                continue
            rect_ids = tokenizer.encode(text, add_special_tokens=False)
            ids = [int(t) for t in responses[i].tolist() if int(t) != pad_id]
            full = ids[:r_start] + rect_ids
            if len(full) > max_resp:
                full = full[:max_resp]
            n = len(full)
            new_resp[:n] = torch.tensor(full, device=device, dtype=dtype)
            resp_attn = new_resp != pad_id
            y_len, v_tmpl, v_len, r_tmpl = (
                len(y_ids), len(verify_tokens), len(v_ids), len(regen_tokens)
            )
            vs, ve = y_len + v_tmpl, y_len + v_tmpl + v_len
            if ve <= n:
                mt[vs:ve] = True
            rs = ve + r_tmpl
            if rs < n:
                mt[rs:n] = True
                exp[rs:n] = True
                exp_r[rs:n] = True
            _rebuild_row_prefix(batch, i, prompts, new_resp, resp_attn, max_resp, device)
            batch.batch["multiturn_mask"][i] = mt
            batch.batch["expert_token_mask"][i] = exp
            batch.batch["expert_token_mask_v"][i] = exp_v
            batch.batch["expert_token_mask_r"][i] = exp_r
            if "feas_gate_r" in batch.batch:
                batch.batch["feas_gate_r"][i] = 0.0
            if "feas_weight_r" in batch.batch:
                batch.batch["feas_weight_r"][i] = torch.clamp(
                    batch.batch["feas_weight_r"][i], min=0.5
                )
            if "feas_gate" in batch.batch:
                batch.batch["feas_gate"][i] = 0.0
            filled += 1
            filled_r += 1
        else:
            v_ids_new = tokenizer.encode(text, add_special_tokens=False)
            full = y_ids + list(verify_tokens) + v_ids_new
            if len(full) > max_resp:
                full = full[:max_resp]
            n = len(full)
            new_resp[:n] = torch.tensor(full, device=device, dtype=dtype)
            resp_attn = new_resp != pad_id
            vs = len(y_ids) + len(verify_tokens)
            if vs < n:
                mt[vs:n] = True
                exp[vs:n] = True
                exp_v[vs:n] = True
            _rebuild_row_prefix(batch, i, prompts, new_resp, resp_attn, max_resp, device)
            batch.batch["multiturn_mask"][i] = mt
            batch.batch["expert_token_mask"][i] = exp
            batch.batch["expert_token_mask_v"][i] = exp_v
            batch.batch["expert_token_mask_r"][i] = exp_r
            if "feas_gate_v" in batch.batch:
                batch.batch["feas_gate_v"][i] = 0.0
            if "feas_weight_v" in batch.batch:
                batch.batch["feas_weight_v"][i] = torch.clamp(
                    batch.batch["feas_weight_v"][i], min=0.5
                )
            if "feas_gate" in batch.batch:
                batch.batch["feas_gate"][i] = 0.0
            filled += 1
            filled_v += 1

    metrics["feasibility/online_gpt_filled"] = float(filled)
    metrics["feasibility/online_gpt_filled_v"] = float(filled_v)
    metrics["feasibility/online_gpt_filled_r"] = float(filled_r)
    metrics["feasibility/online_gpt_failed"] = float(failed)
    return batch, metrics
