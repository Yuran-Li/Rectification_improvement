# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
vLLM Genrm Rollout - Multi-turn generation logic:
1. First turn: prompt -> answer
2. Verify previous answer -> correct/wrong judgment
3. If "wrong", next turn: request regeneration -> new answer
4. Repeat until specified turns or GenRM considers answer correct

When slide_window=True, generation context keeps only
(problem + latest answer [+ current verify/regen]), and each verify-round
is packed as its own training window so context stays O(1) in num_turns.
"""
import numpy as np
import re
from typing import List, Any, Union
from omegaconf import DictConfig
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.reward_score.math_verify import compute_score as get_policy_score
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from input token sequence"""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


def _sanitize_token_ids(
    token_ids: List[int], vocab_size: int, replace_id: int
) -> List[int]:
    """Replace ids outside tokenizer vocabulary (do not delete — avoids broken BPE).

    Qwen HF configs often set vocab_size (e.g. 152064) > len(tokenizer) (e.g. 151665).
    vLLM can sample those unused ids; feeding them back as the next-turn prompt
    raises ValueError: Token id ... is out of vocabulary.
    """
    if not token_ids:
        return token_ids
    max_id = vocab_size - 1
    out = []
    for t in token_ids:
        t = int(t)
        out.append(t if 0 <= t <= max_id else replace_id)
    return out


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMPAGRollout(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        if kwargs.get('train_tp', None) is not None:
            import os
            os.environ.update({
                'CUDA_TIMER_STREAM_KAFKA_ENABLE': '0',
                'MEGATRON_IMPORT_TIMERS': '0'
            })
            train_tp = kwargs.get('train_tp')
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp
            )

        sampling_kwargs = {
            'n': 1,
            'logprobs': 0,
            'max_tokens': config.response_length,
        }

        if vllm_version != '0.3.1':
            sampling_kwargs['detokenize'] = False

        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                sampling_kwargs[k] = config.get(k)

        print(f"Sampling kwargs: {sampling_kwargs}")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        # Prefer len(tokenizer): unused embedding rows above this are not valid prompt ids for vLLM
        self.tokenizer_vocab_size = len(tokenizer)
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        self.num_turns = config.get('num_turns', 2)
        self.is_only_genrm = config.get('is_only_genrm', False)
        self.end_with_verifer = config.get('end_with_verifer', False)
        # Keep only (problem + latest answer [+ current verify/regen]) in context.
        self.slide_window = bool(config.get('slide_window', False))

        self.prompt_templates = {
            "verify": (
                "Verify the previous solution without re-solving the problem from scratch. "
                "Check the given solution step-by-step: if you find a mistake, state the wrong step, "
                "explain why it is wrong, and end your response with 'The answer is wrong'. "
                "If all steps are correct, end your response with 'The answer is correct'."
            ),
            "regenerate": (
                "You indicated that your previous answer was wrong. "
                "Please provide the correct solution to the math problem."
            ),
        }

        if self.slide_window:
            # Bound by model context; real windows use problem+last answer only.
            max_model_len = model_hf_config.max_position_embeddings
        elif self.end_with_verifer:
            max_model_len = self.config.response_length * self.num_turns * 2 + config.prompt_length
            max_model_len = min(max_model_len, model_hf_config.max_position_embeddings)
        else:
            # Full-concat budget; clamp to HF context (e.g. Math-7B = 4096).
            max_model_len = (
                config.prompt_length
                + self.num_turns * self.config.response_length
                + (self.num_turns - 1) * (200 + self.config.response_length)
            )
            max_model_len = min(max_model_len, model_hf_config.max_position_embeddings)

        if config.get('max_model_len', None) is not None:
            max_model_len = config.get('max_model_len')
        assert model_hf_config.max_position_embeddings >= max_model_len, (
            f"model context length ({model_hf_config.max_position_embeddings}) "
            f"should be >= max_model_len ({max_model_len}); "
            f"reduce MAX_PROMPT/MAX_RESP/NUM_TURNS or set rollout.max_model_len"
        )
        print(f"[vLLMPAGRollout] slide_window={self.slide_window} max_model_len={max_model_len} "
              f"num_turns={self.num_turns} response_length={self.config.response_length}")

        max_num_batched_tokens = config.get('max_num_batched_tokens', 8192)
        if max_num_batched_tokens < max_model_len and config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # Prefix caching + sleep/free_cache_engine has caused
            # "Failed to reset prefix cache... blocks not freed" then
            # CUDA illegal memory access during sampling after ~10–20 steps.
            enable_prefix_caching=False,
            trust_remote_code=kwargs.get('trust_remote_code', False),
            seed=42,
        )

        self.inference_engine.sleep(level=1)

        self.correct_token_ids = self._find_token_ids_for_word("correct")
        self.wrong_token_ids = self._find_token_ids_for_word("wrong")
        print(f"Token IDs for 'correct': {self.correct_token_ids}")
        print(f"Token IDs for 'wrong': {self.wrong_token_ids}")

    def _find_token_ids_for_word(self, word):
        variants = [word, f" {word}"]
        token_ids = []
        for variant in variants:
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        return token_ids

    def _get_template_tokens(self, template_key):
        template = self.prompt_templates.get(template_key, "")
        messages = [{"role": "user", "content": template}]
        chat_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if chat_template.startswith("<|im_start|>system"):
            system_end = chat_template.find("<|im_end|>") + len("<|im_end|>")
            chat_template = chat_template[system_end:].lstrip()
        if chat_template.startswith("<｜begin▁of▁sentence｜>"):
            chat_template = chat_template[len(""):]
        chat_template = "\n\n" + chat_template
        return self.tokenizer.encode(chat_template, add_special_tokens=False)

    def _extract_judgment_and_probability(self, output, text):
        pattern = r"The answer is (correct|wrong)\.$"
        matches = []
        try:
            matches = [(match.group(1).lower(), match.group(0), match.start())
                       for match in re.finditer(pattern, text)]
        except Exception:
            pass
        if not matches:
            return None, None
        judgment, _, _ = matches[-1]
        target_token_ids = self.correct_token_ids if judgment == "correct" else self.wrong_token_ids
        token_ids = output.outputs[0].token_ids
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] in target_token_ids:
                token_logprobs = output.outputs[0].logprobs[i]
                prob = np.exp(token_logprobs[token_ids[i]].logprob)
                return judgment, prob
        print(f"Warning: prob is None for judgment: {judgment}, text: {text}")
        return judgment, None

    def _pack_slide_windows(
        self,
        idx,
        attention_mask,
        position_ids,
        traj_answers,
        traj_verifies,
        final_generation_turn,
        verify_probs,
        full_verify_probs,
        num_turns,
        batch_size,
    ) -> DataProto:
        """Pack each (answer, verify, optional rectify) round as one training window.

        Window w response layout:
          [y_w][verify_user][v_w][regen_user][y_{w+1}?]
        For w>0, y_w is context-only (multiturn_mask=False) to avoid double-counting
        the rectify tokens already supervised in window w-1.
        """
        verify_tokens = self._get_template_tokens("verify")
        regen_tokens = self._get_template_tokens("regenerate")
        W = int(num_turns)
        expand_bs = batch_size * W
        max_resp = (
            3 * self.config.response_length
            + len(verify_tokens)
            + len(regen_tokens)
        )
        device = idx.device

        new_idx = idx.repeat_interleave(W, dim=0)
        new_prompt_attn = attention_mask.repeat_interleave(W, dim=0)
        new_prompt_pos = position_ids.repeat_interleave(W, dim=0)

        responses = torch.full((expand_bs, max_resp), self.pad_token_id, device=device, dtype=idx.dtype)
        multiturn_mask = torch.zeros((expand_bs, max_resp), dtype=torch.bool, device=device)
        resp_attn = torch.zeros((expand_bs, max_resp), dtype=torch.bool, device=device)

        window_valid = np.zeros(expand_bs, dtype=np.bool_)
        window_index = np.zeros(expand_bs, dtype=np.int32)
        answer_is_context = np.zeros(expand_bs, dtype=np.bool_)
        context_answer_len = np.zeros(expand_bs, dtype=np.int32)
        exp_final_turn = np.zeros(expand_bs, dtype=np.int32)
        exp_verify_probs = np.zeros(expand_bs, dtype=np.float32)
        exp_full_verify_probs = np.empty(expand_bs, dtype=object)

        for i in range(batch_size):
            answers = traj_answers[i]
            verifies = traj_verifies[i]
            n_verify = len(verifies)
            for w in range(W):
                row = i * W + w
                window_index[row] = w
                exp_final_turn[row] = final_generation_turn[i]
                exp_verify_probs[row] = verify_probs[i] if w == 0 else 0.0
                exp_full_verify_probs[row] = full_verify_probs[i]
                if w >= n_verify or w >= len(answers):
                    continue
                window_valid[row] = True
                answer_is_context[row] = (w > 0)

                pieces = []
                mask_flags = []  # per-piece: whether model tokens get multiturn loss

                # y_w
                pieces.append(answers[w])
                mask_flags.append(w == 0)  # only y0 is a generator action in its window

                # verify user + verify response
                pieces.append(verify_tokens)
                mask_flags.append(False)
                pieces.append(verifies[w])
                mask_flags.append(True)

                # optional rectify
                if len(answers) > w + 1:
                    pieces.append(regen_tokens)
                    mask_flags.append(False)
                    pieces.append(answers[w + 1])
                    mask_flags.append(True)

                pos = 0
                for pi, (piece, do_mask) in enumerate(zip(pieces, mask_flags)):
                    n = len(piece)
                    if pos + n > max_resp:
                        n = max_resp - pos
                        piece = piece[:n]
                    if n <= 0:
                        break
                    responses[row, pos:pos + n] = torch.tensor(piece, device=device, dtype=idx.dtype)
                    resp_attn[row, pos:pos + n] = True
                    if do_mask:
                        multiturn_mask[row, pos:pos + n] = True
                    # y_w is always the first piece; record its packed length for scoring
                    if pi == 0:
                        context_answer_len[row] = n
                    pos += n

        seq = torch.cat([new_idx, responses], dim=-1)
        delta_position_id = torch.arange(1, max_resp + 1, device=device).unsqueeze(0).expand(expand_bs, -1)
        response_position_ids = new_prompt_pos[:, -1:] + delta_position_id
        new_position_ids = torch.cat([new_prompt_pos, response_position_ids], dim=-1)
        new_attention_mask = torch.cat([new_prompt_attn, resp_attn], dim=-1)

        batch = TensorDict({
            'prompts': new_idx,
            'responses': responses,
            'input_ids': seq,
            'attention_mask': new_attention_mask,
            'position_ids': new_position_ids,
            'multiturn_mask': multiturn_mask,
        }, batch_size=expand_bs)

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "final_generation_turn": exp_final_turn,
                "verify_probs": exp_verify_probs,
                "full_verify_probs": exp_full_verify_probs,
                "window_valid": window_valid,
                "window_index": window_index,
                "answer_is_context": answer_is_context,
                "context_answer_len": context_answer_len,
            },
            meta_info={"window_expand": W, "slide_window": True},
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate multi-turn dialogue sequences"""
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)
        num_turns = prompts.meta_info.get('num_turns', self.num_turns)
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        revise_gate = prompts.meta_info.get('revise_gate', 'pag')
        ground_truths = prompts.non_tensor_batch.get('ground_truth', None)
        if revise_gate == 'oracle' and ground_truths is None:
            raise ValueError("revise_gate=oracle requires non_tensor_batch['ground_truth']")

        if not do_sample:
            kwargs = {'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1}
        elif is_validate:
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }

        if do_sample and self.config.n > 1 and not is_validate:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            kwargs['n'] = 1

        current_inputs = [
            _sanitize_token_ids(
                _pre_process_inputs(self.pad_token_id, idx[i]),
                self.tokenizer_vocab_size,
                self.eos_token_id,
            )
            for i in range(batch_size)
        ]

        final_generation_turn = [0] * batch_size
        verify_probs = [0] * batch_size
        full_verify_probs = [[] for _ in range(batch_size)]
        verify_tokens = self._get_template_tokens("verify")
        regenerate_tokens = self._get_template_tokens("regenerate")
        traj_answers = [[] for _ in range(batch_size)]
        traj_verifies = [[] for _ in range(batch_size)]
        last_answer_ids = [[] for _ in range(batch_size)]
        last_verify_ids = [[] for _ in range(batch_size)]

        if self.end_with_verifer:
            max_total_length = (
                2 * num_turns * self.config.response_length
                + num_turns * (len(verify_tokens) + len(regenerate_tokens))
            )
        else:
            max_total_length = (
                num_turns * self.config.response_length
                + (num_turns - 1) * (len(verify_tokens) + self.config.response_length + len(regenerate_tokens))
            )

        combined_response = torch.full((batch_size, max_total_length), self.pad_token_id, device=idx.device)
        multiturn_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        response_attention_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        current_positions = [0] * batch_size
        turns_positions = [[0] for _ in range(batch_size)]

        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=current_inputs,
                use_tqdm=False
            )

        response = [output.outputs[sample_id].token_ids
                    for output in outputs for sample_id in range(len(output.outputs))]
        current_response = pad_2d_list_to_length(
            response, self.pad_token_id, max_length=self.config.response_length
        ).to(idx.device)
        current_response_mask = get_response_mask(current_response, eos_token_id)

        for i in range(batch_size):
            pos = current_positions[i]
            response_length = current_response_mask[i].sum().item()
            ans_ids = _sanitize_token_ids(
                current_response[i, :response_length].tolist(),
                self.tokenizer_vocab_size,
                self.eos_token_id,
            )
            response_length = len(ans_ids)
            if response_length > 0:
                combined_response[i, pos:pos + response_length] = torch.tensor(
                    ans_ids, device=idx.device, dtype=combined_response.dtype
                )
                multiturn_mask[i, pos:pos + response_length] = True
                response_attention_mask[i, pos:pos + response_length] = True
            current_positions[i] = pos + response_length
            turns_positions[i].append(pos + response_length)
            traj_answers[i].append(ans_ids)
            last_answer_ids[i] = ans_ids

        active_samples = list(range(batch_size))
        kwargs_for_verification = kwargs.copy()
        kwargs_for_verification["logprobs"] = 1

        # PAG-compatible loop: range(1, max_turns). Last regenerated answer may be
        # unverified when the loop ends after generation (same as upstream PAG).
        max_turns = num_turns + 1 if self.end_with_verifer else num_turns
        for answer_turn in range(1, max_turns):
            next_inputs = []
            for i, original_idx in enumerate(active_samples):
                pos = current_positions[original_idx]
                verify_tensor = torch.tensor(verify_tokens, device=idx.device)
                combined_response[original_idx, pos:pos + len(verify_tokens)] = verify_tensor
                response_attention_mask[original_idx, pos:pos + len(verify_tokens)] = True
                current_positions[original_idx] = pos + len(verify_tokens)
                turns_positions[original_idx].append(pos + len(verify_tokens))

                if self.slide_window:
                    history = current_inputs[original_idx] + last_answer_ids[original_idx] + verify_tokens
                else:
                    response_tokens = combined_response[original_idx, :current_positions[original_idx]].tolist()
                    history = current_inputs[original_idx] + response_tokens
                next_inputs.append(
                    _sanitize_token_ids(history, self.tokenizer_vocab_size, self.eos_token_id)
                )

            with self.update_sampling_params(**kwargs_for_verification):
                outputs = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=next_inputs,
                    use_tqdm=False
                )

            verification_response = [output.outputs[sample_id].token_ids
                                     for output in outputs for sample_id in range(len(output.outputs))]
            active_verification = pad_2d_list_to_length(
                verification_response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            active_verification_mask = get_response_mask(active_verification, eos_token_id)

            for i, original_idx in enumerate(active_samples):
                pos = current_positions[original_idx]
                verification_length = active_verification_mask[i].sum().item()
                v_ids = _sanitize_token_ids(
                    active_verification[i, :verification_length].tolist(),
                    self.tokenizer_vocab_size,
                    self.eos_token_id,
                )
                verification_length = len(v_ids)
                if verification_length > 0:
                    combined_response[original_idx, pos:pos + verification_length] = torch.tensor(
                        v_ids, device=idx.device, dtype=combined_response.dtype
                    )
                    multiturn_mask[original_idx, pos:pos + verification_length] = True
                    response_attention_mask[original_idx, pos:pos + verification_length] = True
                current_positions[original_idx] = pos + verification_length
                turns_positions[original_idx].append(pos + verification_length)
                traj_verifies[original_idx].append(v_ids)
                last_verify_ids[original_idx] = v_ids

            if self.is_only_genrm and not is_validate:
                break

            new_active_samples = []
            next_inputs = []
            for i, original_idx in enumerate(active_samples):
                verification_length = active_verification_mask[i].sum().item()
                verification_tokens_i = active_verification[i][:verification_length].tolist()
                verification_text = self.tokenizer.decode(verification_tokens_i, skip_special_tokens=True)

                judgment, prob = self._extract_judgment_and_probability(outputs[i], verification_text)
                if answer_turn == 1:
                    verify_probs[original_idx] = prob
                full_verify_probs[original_idx].append(prob)

                if revise_gate == 'always':
                    should_revise = True
                elif revise_gate == 'oracle':
                    t1_text = self.tokenizer.decode(last_answer_ids[original_idx], skip_special_tokens=True)
                    gt = ground_truths[original_idx]
                    t1_acc = get_policy_score(solution_str=t1_text, ground_truth=gt)["acc"]
                    should_revise = t1_acc < 0.5
                else:
                    should_revise = judgment == "wrong" or (is_validate and judgment != "correct")

                if should_revise:
                    new_active_samples.append(original_idx)
                    pos = current_positions[original_idx]
                    regenerate_tensor = torch.tensor(regenerate_tokens, device=idx.device)
                    combined_response[original_idx, pos:pos + len(regenerate_tokens)] = regenerate_tensor
                    response_attention_mask[original_idx, pos:pos + len(regenerate_tokens)] = True
                    current_positions[original_idx] = pos + len(regenerate_tokens)
                    turns_positions[original_idx].append(pos + len(regenerate_tokens))

                    if self.slide_window:
                        history = (
                            current_inputs[original_idx]
                            + last_answer_ids[original_idx]
                            + verify_tokens
                            + last_verify_ids[original_idx]
                            + regenerate_tokens
                        )
                    else:
                        response_tokens = combined_response[original_idx, :current_positions[original_idx]].tolist()
                        history = current_inputs[original_idx] + response_tokens
                    next_inputs.append(
                        _sanitize_token_ids(history, self.tokenizer_vocab_size, self.eos_token_id)
                    )
                else:
                    final_generation_turn[original_idx] = answer_turn - 1

            active_samples = new_active_samples
            if not active_samples or answer_turn == num_turns:
                break

            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=next_inputs,
                    use_tqdm=False
                )

            regenerated_response = [output.outputs[sample_id].token_ids
                                    for output in outputs for sample_id in range(len(output.outputs))]
            active_response = pad_2d_list_to_length(
                regenerated_response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            active_response_mask = get_response_mask(active_response, eos_token_id)

            for i, original_idx in enumerate(active_samples):
                pos = current_positions[original_idx]
                response_length = active_response_mask[i].sum().item()
                ans_ids = _sanitize_token_ids(
                    active_response[i, :response_length].tolist(),
                    self.tokenizer_vocab_size,
                    self.eos_token_id,
                )
                response_length = len(ans_ids)
                if response_length > 0:
                    combined_response[original_idx, pos:pos + response_length] = torch.tensor(
                        ans_ids, device=idx.device, dtype=combined_response.dtype
                    )
                    multiturn_mask[original_idx, pos:pos + response_length] = True
                    response_attention_mask[original_idx, pos:pos + response_length] = True
                current_positions[original_idx] = pos + response_length
                turns_positions[original_idx].append(pos + response_length)
                final_generation_turn[original_idx] = answer_turn
                traj_answers[original_idx].append(ans_ids)
                last_answer_ids[original_idx] = ans_ids

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        if self.slide_window:
            return self._pack_slide_windows(
                idx=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                traj_answers=traj_answers,
                traj_verifies=traj_verifies,
                final_generation_turn=final_generation_turn,
                verify_probs=verify_probs,
                full_verify_probs=full_verify_probs,
                num_turns=num_turns,
                batch_size=batch_size,
            )

        seq = torch.cat([idx, combined_response], dim=-1)
        delta_position_id = torch.arange(1, combined_response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict({
            'prompts': idx,
            'responses': combined_response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'multiturn_mask': multiturn_mask
        }, batch_size=batch_size)

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "final_generation_turn": np.array(final_generation_turn, dtype=np.int32),
                "verify_probs": np.array(verify_probs, dtype=np.float32),
                "full_verify_probs": np.array(full_verify_probs, dtype=object),
            },
            meta_info={"window_expand": 1},
        )
