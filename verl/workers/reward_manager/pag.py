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

from verl import DataProto
from verl.utils.reward_score.math_verify import compute_score as get_policy_score
from verl.utils.reward_score.genrm_verify import get_verification_score
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class PAGRewardManager:
    """Multi-turn dialogue reward manager with GenRM verification.

    Flow: User question -> Model answer -> Verification -> Regeneration (if needed)

    When utility_aware=True (feasibility-guided design):
      R_gen = 1[y0 correct]
      R_rect = hat_R(y_i) + alpha * (hat_R(y_i) - hat_R(y_{i-1}))
      R_ver  = hat_R_v + beta * (hat_R(y_i) - hat_R(y_{i-1}))   # utility after rectify
      Also logs c^ver / c^rect / c^F feasibility costs.
      Token costs are role-aligned (event markers only): c^ver @ verify end, c^rect @
      rectify end. Per correction turn: c_i^F = 1[c_i^ver ∨ c_i^rect].
      Turn-level cost-to-go G_{F,k} = Σ_{j≥k} γ_F^{j-k} c_{j+1}^F is written at answer
      routing states (after each y_k); VF trains only there — not on every token.
    """

    def __init__(self, tokenizer, num_examine, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        config = config or {}
        self.max_turns = config.get('num_turns', 2)
        self.is_only_genrm = config.get('is_only_genrm', False)
        self.policy_rs = config.get('policy_rs', False)
        self.rs_coef = config.get('rs_coef', 10.0)
        self.end_with_verifer = config.get('end_with_verifer', False)
        # Feasibility-guided rewards
        self.utility_aware = bool(config.get('utility_aware', False))
        self.alpha = float(config.get('alpha', config.get('rs_coef', 1.0)))
        self.beta = float(config.get('beta', 0.5))
        # Turn-level γ_F for C_F(τ)=Σ_i γ_F^{i-1} c_i^F (correction turns, not tokens)
        self.gamma_f = float(config.get('gamma_f', 0.9))

    def _turn_boundaries(self, multiturn_mask: torch.Tensor) -> List[int]:
        """Return end indices (exclusive) of each contiguous True run in multiturn_mask.

        Important for slide_window w>0: mask starts False (context answer + verify user
        template). The old leading-True pad invented a spurious boundary at 0 and broke
        verify/rectify slicing.
        """
        if multiturn_mask.numel() == 0:
            return []
        m = multiturn_mask.bool()
        # append False so a trailing True run is closed
        ext = torch.cat([m, m.new_zeros(1, dtype=torch.bool)])
        ends = torch.where(ext[:-1] & ~ext[1:])[0] + 1
        return ends.tolist()

    @staticmethod
    def _fill_routing_targets(
        feasibility_mask: torch.Tensor,
        feasibility_returns: torch.Tensor,
        row: int,
        answer_ends: List[int],
        turn_costs: List[float],
        gamma_f: float,
    ) -> None:
        """Write G_{F,k} at answer routing token positions.

        answer_ends[k] = token index of y_k end (state s_k).
        turn_costs[j] = c_{j+1}^F for correction round j+1.
        G_{F,k} = Σ_{j≥k} γ_F^{j-k} turn_costs[j].
        """
        L = feasibility_mask.shape[-1]
        for k, aend in enumerate(answer_ends):
            if aend < 0 or aend >= L:
                continue
            g = 0.0
            for j in range(k, len(turn_costs)):
                g += (gamma_f ** (j - k)) * float(turn_costs[j])
            feasibility_mask[row, aend] = True
            feasibility_returns[row, aend] = g

    def __call__(self, data: DataProto, return_dict: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute rewards for multi-turn dialogue."""
        if 'rm_scores' in data.batch:
            return data.batch['rm_scores'], {}

        batch_size = data.batch['responses'].shape[0]
        device = data.batch['responses'].device
        if 'num_turns' in data.meta_info:
            self.max_turns = data.meta_info['num_turns']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # Event markers only (c^ver / c^rect at role ends); not VF regression targets
        cost_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # Answer routing states: VF learns turn-level cost-to-go G_F here
        feasibility_mask = torch.zeros_like(data.batch['responses'], dtype=torch.bool)
        feasibility_returns = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # tokens to imitate when using expert BC (successful verify+rectify spans)
        expert_token_mask = torch.zeros_like(data.batch['responses'], dtype=torch.bool)
        metrics_tensors = {
            'turn_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
            'verify_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
            'turn_counts': torch.zeros(batch_size, dtype=torch.long, device=device),
            'c_ver': torch.zeros(batch_size, dtype=torch.float32, device=device),
            'c_rect': torch.zeros(batch_size, dtype=torch.float32, device=device),
            'c_f': torch.zeros(batch_size, dtype=torch.float32, device=device),
            'c_f_discounted': torch.zeros(batch_size, dtype=torch.float32, device=device),
        }

        answer_logs = []
        reward_extra_info = defaultdict(list)
        printed_sources = {}
        ntb = data.non_tensor_batch

        for i in range(batch_size):
            if self.end_with_verifer:
                for key in ["all_pred", "all_acc", "all_genrm_pred", "all_genrm_score", "all_genrm_probs"]:
                    reward_extra_info[key].append([])

            # Skip padded slide-windows
            window_valid = True
            if 'window_valid' in ntb:
                window_valid = bool(ntb['window_valid'][i])
            if not window_valid:
                ft = int(ntb['final_generation_turn'][i]) if 'final_generation_turn' in ntb else 0
                reward_extra_info["pred"].append("")
                reward_extra_info["acc"].append(0.0)
                reward_extra_info["length"].append(0)
                reward_extra_info["ground_truth"].append(
                    data[i].non_tensor_batch['reward_model']['ground_truth']
                )
                reward_extra_info["data_source"].append(
                    data[i].non_tensor_batch.get('data_source', 'unknown')
                )
                reward_extra_info["acc_t1"].append(0.0)
                reward_extra_info["pred_t1"].append("")
                reward_extra_info["acc_t2"].append(-1.0)
                reward_extra_info["pred_t2"].append("")
                reward_extra_info["revised"].append(False)
                reward_extra_info["final_turn"].append(ft)
                reward_extra_info["acc_final"].append(0.0)
                reward_extra_info["genrm_pred"].append("none")
                reward_extra_info["genrm_score"].append(0.0)
                reward_extra_info["genrm_probs"].append(None)
                reward_extra_info["c_ver"].append(0.0)
                reward_extra_info["c_rect"].append(0.0)
                reward_extra_info["c_f"].append(0.0)
                reward_extra_info["c_f_discounted"].append(0.0)
                reward_extra_info["n_routing_states"].append(0)
                reward_extra_info["expert_bootstrap"].append(False)
                reward_extra_info["window_index"].append(
                    int(ntb['window_index'][i]) if 'window_index' in ntb else 0
                )
                reward_extra_info["window_valid"].append(False)
                continue

            data_item = data[i]
            response_ids = data_item.batch['responses']
            multiturn_mask = data_item.batch['multiturn_mask']
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            answer_is_context = bool(ntb['answer_is_context'][i]) if 'answer_is_context' in ntb else False
            window_index = int(ntb['window_index'][i]) if 'window_index' in ntb else 0
            context_answer_len = (
                int(ntb['context_answer_len'][i]) if 'context_answer_len' in ntb else 0
            )

            turn_boundaries = self._turn_boundaries(multiturn_mask)
            if not turn_boundaries:
                # no model tokens
                ft = int(data_item.non_tensor_batch.get("final_generation_turn", 0))
                reward_extra_info["pred"].append("")
                reward_extra_info["acc"].append(0.0)
                reward_extra_info["length"].append(0)
                reward_extra_info["ground_truth"].append(ground_truth)
                reward_extra_info["data_source"].append(data_source)
                reward_extra_info["acc_t1"].append(0.0)
                reward_extra_info["pred_t1"].append("")
                reward_extra_info["acc_t2"].append(-1.0)
                reward_extra_info["pred_t2"].append("")
                reward_extra_info["revised"].append(False)
                reward_extra_info["final_turn"].append(ft)
                reward_extra_info["acc_final"].append(0.0)
                reward_extra_info["genrm_pred"].append("none")
                reward_extra_info["genrm_score"].append(0.0)
                reward_extra_info["genrm_probs"].append(None)
                reward_extra_info["c_ver"].append(0.0)
                reward_extra_info["c_rect"].append(0.0)
                reward_extra_info["c_f"].append(0.0)
                reward_extra_info["c_f_discounted"].append(0.0)
                reward_extra_info["n_routing_states"].append(0)
                reward_extra_info["expert_bootstrap"].append(False)
                reward_extra_info["window_index"].append(window_index)
                reward_extra_info["window_valid"].append(False)
                continue

            metrics_tensors['turn_counts'][i] = max(len(turn_boundaries) - 1, 0)
            sample_answers = []
            final_turn = int(data_item.non_tensor_batch.get("final_generation_turn", 0))
            acc_t2 = None
            pred_t2 = None
            revised = False
            genrm_pred = None
            genrm_score = None
            genrm_probs = None
            c_ver_i = 0.0
            c_rect_i = 0.0
            c_f_i = 0.0
            expert_bootstrap = False
            # Correction-turn costs c_1^F, c_2^F, ... and answer-end token indices for G_F
            answer_ends: List[int] = []
            turn_costs: List[float] = []

            # --- Score first model segment (y or verify if context-only answer) ---
            # With slide_window packing:
            #   w==0: boundaries cover [y0, v, y1?]
            #   w>0:  y_w has multiturn_mask=False, so first boundary segment is verify
            # Only decode model tokens of the first segment (exclude verify/regen user templates)
            _b0 = turn_boundaries[0]
            first_response = self.tokenizer.decode(
                response_ids[:_b0][multiturn_mask[:_b0]],
                skip_special_tokens=True,
            )
            # If answer is context-only, decode ONLY y_w (not verify user template).
            if answer_is_context:
                if context_answer_len > 0:
                    ctx_ans = self.tokenizer.decode(
                        response_ids[:context_answer_len], skip_special_tokens=True
                    )
                else:
                    # fallback: tokens before first masked (verify) run — may include
                    # verify-user template if packing omitted context_answer_len
                    first_true = (
                        int(multiturn_mask.nonzero(as_tuple=False)[0].item())
                        if multiturn_mask.any() else 0
                    )
                    ctx_ans = self.tokenizer.decode(
                        response_ids[:first_true], skip_special_tokens=True
                    )
                prev_result = get_policy_score(solution_str=ctx_ans, ground_truth=ground_truth)
                prev_acc = float(prev_result["acc"])
                gt_judge = prev_acc >= 0.5
                # first masked segment is verify
                verify_end = turn_boundaries[0]
                verify_response = self.tokenizer.decode(
                    response_ids[:verify_end][multiturn_mask[:verify_end]],
                    skip_special_tokens=True,
                )
                verify_result = get_verification_score(verify_response, gt_judge)
                # defer verify reward until we know rectify utility
                deferred_verify_end = verify_end
                deferred_verify_result = verify_result
                genrm_pred = verify_result["genrm_pred"]
                genrm_score = verify_result["genrm_score"]
                genrm_probs = data_item.non_tensor_batch.get("verify_probs", None)

                reward_extra_info["pred"].append(prev_result["pred"])
                reward_extra_info["acc"].append(prev_acc)
                reward_extra_info["length"].append(verify_end)
                metrics_tensors['turn_accuracies'][i, 0] = prev_acc >= 0.5
                metrics_tensors['verify_accuracies'][i, 0] = verify_result["genrm_score"]
                sample_answers.append(prev_result["pred"])

                # feasibility: verification failure
                z_i = 1 if gt_judge else 0
                d_i = 1 if verify_result["genrm_pred"] == "correct" else 0
                c_ver_i = 1.0 if d_i != z_i else 0.0

                # optional rectify
                if len(turn_boundaries) >= 2:
                    policy_end = turn_boundaries[1]
                    policy_response = self.tokenizer.decode(
                        response_ids[verify_end:policy_end][multiturn_mask[verify_end:policy_end]]
                    )
                    policy_result = get_policy_score(solution_str=policy_response, ground_truth=ground_truth)
                    cur_acc = float(policy_result["acc"])
                    if self.utility_aware:
                        b_y = cur_acc - prev_acc
                        reward_value = cur_acc + self.alpha * b_y
                    elif self.policy_rs:
                        reward_value = cur_acc + self.rs_coef * (cur_acc - prev_acc)
                    else:
                        reward_value = cur_acc
                    reward_tensor[i, policy_end - 1] = reward_value
                    metrics_tensors['turn_accuracies'][i, min(1, self.max_turns - 1)] = cur_acc
                    revised = True
                    acc_t2 = cur_acc
                    pred_t2 = policy_result["pred"]
                    sample_answers.append(policy_result["pred"])
                    answer_logs.append(sample_answers)

                    # rectify failure: said wrong (D=0) but still incorrect
                    if d_i == 0 and cur_acc < 0.5:
                        c_rect_i = 1.0
                    if verify_result["genrm_score"] >= 0.5 and d_i == 0 and cur_acc >= 0.5:
                        expert_bootstrap = True
                        expert_token_mask[i, verify_end:policy_end] = multiturn_mask[verify_end:policy_end]

                    u_v = cur_acc - prev_acc
                else:
                    u_v = 0.0
                    cur_acc = prev_acc

                if self.utility_aware:
                    r_v = float(verify_result["genrm_score"]) + self.beta * u_v
                else:
                    r_v = float(verify_result["genrm_score"])
                reward_tensor[i, deferred_verify_end - 1] = r_v
                # Role-aligned event markers: c^ver @ verify end, c^rect @ rectify end
                cost_tensor[i, deferred_verify_end - 1] = c_ver_i
                if revised and c_rect_i > 0:
                    cost_tensor[i, policy_end - 1] = c_rect_i
                c_f_i = 1.0 if (c_ver_i > 0 or c_rect_i > 0) else 0.0
                # Routing state at context answer end (may be outside multiturn_mask)
                if context_answer_len > 0:
                    answer_ends.append(context_answer_len - 1)
                else:
                    answer_ends.append(max(deferred_verify_end - 1, 0))
                turn_costs.append(c_f_i)
                if revised:
                    answer_ends.append(policy_end - 1)
                metrics_tensors['c_f_discounted'][i] += (self.gamma_f ** window_index) * c_f_i

            else:
                # Standard path: first segment is y0 (or y_w with loss)
                first_result = get_policy_score(solution_str=first_response, ground_truth=ground_truth)
                reward_extra_info["pred"].append(first_result["pred"])
                reward_extra_info["acc"].append(first_result["acc"])
                reward_extra_info["length"].append(turn_boundaries[0])
                if self.end_with_verifer:
                    reward_extra_info["all_pred"][-1].append(first_result["pred"])
                    reward_extra_info["all_acc"][-1].append(first_result["acc"])

                reward_tensor[i, turn_boundaries[0] - 1] = first_result["acc"]
                metrics_tensors['turn_accuracies'][i, 0] = first_result["acc"] >= 0.5
                sample_answers.append(first_result['pred'])
                answer_ends.append(turn_boundaries[0] - 1)  # routing state s_0 after y0

                gt_judge = first_result["acc"] >= 0.5
                prev_acc = float(first_result["acc"])
                max_turns = self.max_turns + 1 if self.end_with_verifer else self.max_turns

                for turn in range(1, max_turns):
                    verify_start = turn_boundaries[2 * turn - 2] if turn > 1 else turn_boundaries[0]
                    if 2 * turn - 1 >= len(turn_boundaries):
                        break
                    verify_end = turn_boundaries[2 * turn - 1]
                    verify_response = self.tokenizer.decode(
                        response_ids[verify_start:verify_end][multiturn_mask[verify_start:verify_end]],
                        skip_special_tokens=True,
                    )
                    verify_result = get_verification_score(verify_response, gt_judge)
                    metrics_tensors['verify_accuracies'][i, turn - 1] = verify_result["genrm_score"]

                    if turn == 1:
                        genrm_pred = verify_result["genrm_pred"]
                        genrm_score = verify_result["genrm_score"]
                        genrm_probs = data_item.non_tensor_batch.get("verify_probs", None)
                        reward_extra_info["genrm_pred"].append(genrm_pred)
                        reward_extra_info["genrm_score"].append(genrm_score)
                        reward_extra_info["genrm_probs"].append(genrm_probs)
                    if self.end_with_verifer:
                        reward_extra_info["all_genrm_pred"][-1].append(verify_result["genrm_pred"])
                        reward_extra_info["all_genrm_score"][-1].append(verify_result["genrm_score"])
                        reward_extra_info['all_genrm_probs'][-1].append(
                            data_item.non_tensor_batch.get("verify_probs", None)
                        )

                    z_i = 1 if gt_judge else 0
                    d_i = 1 if verify_result["genrm_pred"] == "correct" else 0
                    step_c_ver = 1.0 if d_i != z_i else 0.0
                    c_ver_i = max(c_ver_i, step_c_ver)

                    if 2 * turn >= len(turn_boundaries) or (self.end_with_verifer and turn == max_turns - 1):
                        # no rectify; verify utility = 0
                        if self.utility_aware:
                            reward_tensor[i, verify_end - 1] = float(verify_result["genrm_score"])
                        else:
                            reward_tensor[i, verify_end - 1] = verify_result["genrm_score"]
                        cost_tensor[i, verify_end - 1] = step_c_ver
                        step_c_f = 1.0 if step_c_ver > 0 else 0.0
                        c_f_i = max(c_f_i, step_c_f)
                        turn_costs.append(step_c_f)
                        metrics_tensors['c_f_discounted'][i] += (self.gamma_f ** (turn - 1)) * step_c_f
                        break

                    if self.is_only_genrm:
                        multiturn_mask[:verify_end] = False

                    policy_start = verify_end
                    policy_end = turn_boundaries[2 * turn]
                    policy_response = self.tokenizer.decode(
                        response_ids[policy_start:policy_end][multiturn_mask[policy_start:policy_end]]
                    )
                    policy_result = get_policy_score(solution_str=policy_response, ground_truth=ground_truth)
                    cur_acc = float(policy_result["acc"])

                    if self.utility_aware:
                        b_y = cur_acc - prev_acc
                        reward_value = cur_acc + self.alpha * b_y
                        u_v = b_y
                        reward_tensor[i, verify_end - 1] = float(verify_result["genrm_score"]) + self.beta * u_v
                    else:
                        reward_value = policy_result["acc"]
                        if self.policy_rs:
                            reward_value += self.rs_coef * (policy_result["acc"] - prev_acc)
                        reward_tensor[i, verify_end - 1] = verify_result["genrm_score"]

                    reward_tensor[i, policy_end - 1] = reward_value
                    metrics_tensors['turn_accuracies'][i, turn] = policy_result["acc"]

                    step_c_rect = 1.0 if (d_i == 0 and cur_acc < 0.5) else 0.0
                    c_rect_i = max(c_rect_i, step_c_rect)
                    # Event markers only — VF targets are turn-level G at answer ends
                    cost_tensor[i, verify_end - 1] = step_c_ver
                    if step_c_rect > 0:
                        cost_tensor[i, policy_end - 1] = step_c_rect
                    step_c_f = 1.0 if (step_c_ver > 0 or step_c_rect > 0) else 0.0
                    c_f_i = max(c_f_i, step_c_f)
                    turn_costs.append(step_c_f)
                    answer_ends.append(policy_end - 1)  # routing state after y_turn
                    metrics_tensors['c_f_discounted'][i] += (self.gamma_f ** (turn - 1)) * step_c_f
                    if verify_result["genrm_score"] >= 0.5 and d_i == 0 and cur_acc >= 0.5:
                        expert_bootstrap = True
                        expert_token_mask[i, policy_start:policy_end] = multiturn_mask[policy_start:policy_end]

                    prev_acc = cur_acc
                    gt_judge = cur_acc >= 0.5

                    if turn == 1:
                        revised = True
                        acc_t2 = policy_result["acc"]
                        pred_t2 = policy_result["pred"]
                        sample_answers.append(policy_result['pred'])
                        answer_logs.append(sample_answers)
                    if self.end_with_verifer:
                        reward_extra_info["all_pred"][-1].append(policy_result["pred"])
                        reward_extra_info["all_acc"][-1].append(policy_result["acc"])

            # keep genrm_* aligned
            if genrm_pred is None:
                reward_extra_info["genrm_pred"].append("none")
                reward_extra_info["genrm_score"].append(0.0)
                reward_extra_info["genrm_probs"].append(None)
            elif "genrm_pred" not in reward_extra_info or len(reward_extra_info["genrm_pred"]) < len(reward_extra_info["acc"]):
                # context path already may have skipped appending in loop
                if len(reward_extra_info["genrm_pred"]) < len(reward_extra_info["acc"]):
                    reward_extra_info["genrm_pred"].append(genrm_pred)
                    reward_extra_info["genrm_score"].append(genrm_score)
                    reward_extra_info["genrm_probs"].append(genrm_probs)

            c_f_i = 1.0 if (c_ver_i > 0 or c_rect_i > 0) else 0.0
            metrics_tensors['c_ver'][i] = c_ver_i
            metrics_tensors['c_rect'][i] = c_rect_i
            metrics_tensors['c_f'][i] = c_f_i
            self._fill_routing_targets(
                feasibility_mask, feasibility_returns, i, answer_ends, turn_costs, self.gamma_f
            )

            acc_t1 = float(reward_extra_info["acc"][-1])
            acc_final = acc_t2 if revised else acc_t1
            reward_extra_info["ground_truth"].append(ground_truth)
            reward_extra_info["data_source"].append(data_source)
            reward_extra_info["acc_t1"].append(float(acc_t1))
            reward_extra_info["pred_t1"].append(reward_extra_info["pred"][-1])
            reward_extra_info["acc_t2"].append(float(acc_t2) if acc_t2 is not None else -1.0)
            reward_extra_info["pred_t2"].append(pred_t2 if pred_t2 is not None else "")
            reward_extra_info["revised"].append(bool(revised))
            reward_extra_info["final_turn"].append(int(final_turn))
            reward_extra_info["acc_final"].append(float(acc_final))
            reward_extra_info["c_ver"].append(float(c_ver_i))
            reward_extra_info["c_rect"].append(float(c_rect_i))
            reward_extra_info["c_f"].append(float(c_f_i))
            reward_extra_info["c_f_discounted"].append(float(metrics_tensors['c_f_discounted'][i].item()))
            reward_extra_info["n_routing_states"].append(int(len(answer_ends)))
            reward_extra_info["expert_bootstrap"].append(bool(expert_bootstrap))
            reward_extra_info["window_index"].append(int(window_index))
            reward_extra_info["window_valid"].append(True)

            if self.num_examine > 0 and printed_sources.get(data_source, 0) < self.num_examine:
                printed_sources[data_source] = printed_sources.get(data_source, 0) + 1
                full_sequence = torch.cat((data_item.batch['prompts'], response_ids))
                print(self.tokenizer.decode(full_sequence, skip_special_tokens=True))
            if self.end_with_verifer:
                reward_extra_info["response"].append(
                    self.tokenizer.decode(response_ids, skip_special_tokens=True)
                )

        data_sources = None
        if data.meta_info.get('validate', False):
            data_sources = [data[i].non_tensor_batch.get('data_source', 'unknown') for i in range(len(data))]

        metrics = self._compute_metrics(
            metrics_tensors, data_sources, answer_logs, data.non_tensor_batch["final_generation_turn"]
        )
        # feasibility aggregates (ignore invalid windows if present)
        if 'window_valid' in ntb:
            valid = torch.tensor(ntb['window_valid'].astype(np.float32), device=device)
            denom = valid.sum().clamp(min=1.0)
            metrics['feasibility/c_ver_rate'] = (metrics_tensors['c_ver'] * valid).sum().item() / denom.item()
            metrics['feasibility/c_rect_rate'] = (metrics_tensors['c_rect'] * valid).sum().item() / denom.item()
            metrics['feasibility/c_f_rate'] = (metrics_tensors['c_f'] * valid).sum().item() / denom.item()
            metrics['feasibility/c_f_discounted_mean'] = (
                (metrics_tensors['c_f_discounted'] * valid).sum().item() / denom.item()
            )
            metrics['feasibility/window_valid_rate'] = valid.mean().item()
        else:
            metrics['feasibility/c_ver_rate'] = metrics_tensors['c_ver'].mean().item()
            metrics['feasibility/c_rect_rate'] = metrics_tensors['c_rect'].mean().item()
            metrics['feasibility/c_f_rate'] = metrics_tensors['c_f'].mean().item()
            metrics['feasibility/c_f_discounted_mean'] = metrics_tensors['c_f_discounted'].mean().item()
        if reward_extra_info.get("expert_bootstrap"):
            metrics['feasibility/expert_bootstrap_rate'] = float(np.mean(reward_extra_info["expert_bootstrap"]))
        n_route = feasibility_mask.float().sum().item()
        metrics['feasibility/n_routing_states'] = n_route
        if n_route > 0:
            metrics['feasibility/G_F_mean'] = (
                feasibility_returns[feasibility_mask].mean().item()
            )
            metrics['feasibility/G_F_pos_rate'] = (
                (feasibility_returns[feasibility_mask] > 0).float().mean().item()
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "cost_tensor": cost_tensor,
                "feasibility_mask": feasibility_mask,
                "feasibility_returns": feasibility_returns,
                "c_f_discounted": metrics_tensors['c_f_discounted'],
                "expert_token_mask": expert_token_mask,
                "reward_extra_info": reward_extra_info,
                "metrics": metrics,
            }
        return reward_tensor, metrics

    def _compute_single_group_metrics(self, turn_acc, verify_acc, turn_counts, final_turn, prefix=""):
        metrics = {}
        final_acc = turn_acc.gather(dim=-1, index=final_turn.unsqueeze(-1))
        metrics[f'{prefix}final_acc'] = final_acc.mean().item()

        for i in range(self.max_turns):
            clamped_turn = final_turn.clone().clamp(max=i)
            turn_policy_acc = turn_acc.gather(dim=-1, index=clamped_turn.unsqueeze(-1))
            metrics[f'{prefix}turn_{i+1}_accuracy'] = turn_policy_acc.mean().item()

        for i in range(2 * self.max_turns - 1):
            count = (turn_counts == i).sum().item()
            if count > 0:
                metrics[f'{prefix}turn_count_{i}'] = count
                metrics[f'{prefix}turn_count_{i}_ratio'] = count / len(turn_counts)

        for i in range(1, self.max_turns):
            policy_mask = turn_counts >= i * 2
            if policy_mask.any():
                metrics[f'{prefix}turn_{i+1}_accuracy_selection'] = turn_acc[policy_mask, i].mean().item()
            verify_mask = turn_counts >= i * 2 - 1
            if verify_mask.any():
                metrics[f'{prefix}verify_{i}_accuracy'] = verify_acc[:, i - 1].mean().item()

        if len(turn_acc) > 0:
            turn1_policy = turn_acc[:, 0]
            turn1_verify = verify_acc[:, 0]
            TP = ((turn1_verify > 0.5) & (turn1_policy > 0.5)).sum().item()
            FP = ((turn1_verify <= 0.5) & (turn1_policy <= 0.5)).sum().item()
            FN = ((turn1_verify <= 0.5) & (turn1_policy > 0.5)).sum().item()
            TN = ((turn1_verify > 0.5) & (turn1_policy <= 0.5)).sum().item()
            metrics[f'{prefix}verify_TP'] = TP
            metrics[f'{prefix}verify_FP'] = FP
            metrics[f'{prefix}verify_FN'] = FN
            metrics[f'{prefix}verify_TN'] = TN
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics.update({
                f'{prefix}verify_precision': precision,
                f'{prefix}verify_recall': recall,
                f'{prefix}verify_recall_negative': recall_neg,
                f'{prefix}verify_f1': f1,
            })
            if turn_acc.shape[1] > 1:
                turn2_mask = (turn_counts == 2)
                if turn2_mask.any():
                    turn2_policy = turn_acc[:, 1][turn2_mask]
                    turn1_policy_masked = turn1_policy[turn2_mask]
                    i_to_c = ((turn2_policy > 0.5) & (turn1_policy_masked <= 0.5)).sum().item()
                    c_to_i = ((turn2_policy <= 0.5) & (turn1_policy_masked > 0.5)).sum().item()
                    total_incorrect = (turn1_policy_masked <= 0.5).sum().item()
                    total_correct = (turn1_policy_masked > 0.5).sum().item()
                    if total_incorrect > 0:
                        metrics.update({
                            f'{prefix}i_to_c_rate': i_to_c / total_incorrect,
                            f'{prefix}i_to_c_rate_gt': i_to_c / len(turn1_policy),
                            f'{prefix}i_to_c_count': i_to_c,
                        })
                    if total_correct > 0:
                        metrics.update({
                            f'{prefix}c_to_i_rate': c_to_i / total_correct,
                            f'{prefix}c_to_i_rate_gt': c_to_i / len(turn1_policy),
                            f'{prefix}c_to_i_count': c_to_i,
                        })
        return metrics

    def _compute_metrics(self, metrics_tensors, data_sources=None, answer_logs=None, final_generation_turn=None):
        final_turn = torch.tensor(
            final_generation_turn, device=metrics_tensors['turn_accuracies'].device, dtype=torch.long
        )
        print("final_answer_turn", final_turn)
        metrics = self._compute_single_group_metrics(
            metrics_tensors['turn_accuracies'], metrics_tensors['verify_accuracies'],
            metrics_tensors['turn_counts'], final_turn
        )
        if answer_logs:
            regen_samples = answer_changed = 0
            for answers in answer_logs:
                if len(answers) >= 2 and all(a is not None for a in answers[:2]):
                    regen_samples += 1
                    if answers[0] != answers[1]:
                        answer_changed += 1
            if regen_samples > 0:
                metrics.update({
                    'answer_change_ratio': answer_changed / regen_samples,
                    'answer_changed_samples': answer_changed,
                    'regeneration_samples': regen_samples,
                })
        if data_sources is not None:
            data_sources = np.array(data_sources) if isinstance(data_sources, list) else data_sources
            for source in np.unique(data_sources):
                indices = torch.tensor(np.where(data_sources == source)[0], device=final_turn.device)
                if len(indices) > 0:
                    source_metrics = self._compute_single_group_metrics(
                        metrics_tensors['turn_accuracies'][indices],
                        metrics_tensors['verify_accuracies'][indices],
                        metrics_tensors['turn_counts'][indices],
                        final_turn[indices],
                        prefix=f'{source}/'
                    )
                    metrics.update(source_metrics)
        return metrics
