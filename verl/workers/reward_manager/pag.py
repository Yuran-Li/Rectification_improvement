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
from verl.utils.reward_score.genrm_verify import (
    find_feedback_last_token_index,
    get_verification_score,
)
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


def critique_advantage(acc_self: float, acc_generic: float) -> float:
    """R_critique = R_y(y_self) - R_y(y_generic). With 0/1 acc this is {+1, 0, -1}."""
    return float(acc_self) - float(acc_generic)


FEEDBACK_MODES = ("generic", "regen", "delta", "acc", "disc")
_FEEDBACK_MODE_ALIASES = {
    "base": "delta",
    "delta_self": "delta",
    "critique": "generic",
    "r_critique": "generic",
    "acc_t2": "acc",
    "acc_cw": "acc",
    "none": "disc",
    "pag": "disc",
    "disc_only": "disc",
}


def _as_feedback_mode(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = _FEEDBACK_MODE_ALIASES.get(s, s)
    return s if s in FEEDBACK_MODES else None


def resolve_feedback_mode(config: Optional[Dict[str, Any]] = None) -> str:
    """generic | regen | delta | acc | disc.

    Priority: ``feedback_mode``, then ``lambda_regen`` as a mode name, then
    numeric ``lambda_regen>0`` → delta (compat with LAMBDA_REGEN=1.0), else
    generic if the counterfactual fork is on, else policy_rs shaping.
    """
    config = config or {}
    mode = _as_feedback_mode(config.get("feedback_mode"))
    if mode:
        return mode
    mode = _as_feedback_mode(config.get("lambda_regen"))
    if mode:
        return mode
    try:
        lam = float(config.get("lambda_regen", 0.0) or 0.0)
    except (TypeError, ValueError):
        lam = 0.0
    if lam > 0.0:
        return "delta"
    if bool(config.get("generic_counterfactual", False)):
        return "generic"
    return "policy_rs"


def _lambda_regen_scale(config: Optional[Dict[str, Any]] = None) -> float:
    config = config or {}
    raw = config.get("lambda_regen", 1.0)
    if _as_feedback_mode(raw):
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def regen_aware_feedback_reward(delta_self: float, p_regen: float = 0.0, lambda_regen: float = 1.0) -> float:
    """R_feedback for feedback_mode=regen: Δ_self * [1 + λ (1 - p_regen)]."""
    return float(delta_self) * (1.0 + float(lambda_regen) * (1.0 - float(p_regen)))


def delta_feedback_reward(delta_self: float) -> float:
    """R_feedback for feedback_mode=delta: Δ_self = acc_t2 - acc_t1."""
    return float(delta_self)


def acc_t2_cw_feedback_reward(
    acc_t2: float, acc_t1: float, lambda_cw: float = 0.2
) -> float:
    """R_use = acc_t2 - λ · 1[C→W]. λ only applies when t1 is correct and t2 is wrong."""
    acc_t2_f = float(acc_t2)
    c_to_w = 1.0 if (float(acc_t1) >= 0.5 and acc_t2_f < 0.5) else 0.0
    return acc_t2_f - float(lambda_cw) * c_to_w


def _lambda_cw_scale(config: Optional[Dict[str, Any]] = None) -> float:
    config = config or {}
    raw = config.get("lambda_cw", 0.2)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.2


def prompt_group_ids(data: DataProto) -> List[str]:
    """Original-prompt ids for grouping K independent y0 rollouts.

    Training assigns `uid` once per prompt, then `repeat(n, interleave=True)`,
    so K siblings share `uid`. Do not use traj_id / verify / rectify forks.
    Fallback: dataset `index`, then a hash of `raw_prompt_ids` / prompt tokens.
    """
    ntb = data.non_tensor_batch
    n = len(data)
    if "uid" in ntb:
        return [str(x) for x in ntb["uid"]]
    if "index" in ntb:
        ids = [str(x) for x in ntb["index"]]
        if n <= 1 or len(set(ids)) > 1:
            return ids
    if "raw_prompt_ids" in ntb:
        out = []
        for i in range(n):
            ids = ntb["raw_prompt_ids"][i]
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            out.append("rp:" + ",".join(map(str, ids)))
        return out
    prompts = data.batch["prompts"]
    return ["p:" + ",".join(map(str, prompts[i].tolist())) for i in range(n)]


def compute_loo_p_regen(y0_correct: np.ndarray, group_ids: List[str]) -> np.ndarray:
    """Leave-one-out turn-1 pass rate of the current policy on the same prompt.

    y0_correct: (B,) in {0, 1} — correctness of each trajectory's own y0.
    group_ids:  (B,) original-prompt id (typically `uid`).
    returns:    (B,) p_regen_i = sum_{j != i} c_j / (K - 1).

    K == 1 fallback: p_regen_i = c_i (single-sample MLE; LOO is undefined).
    """
    y0_correct = np.asarray(y0_correct, dtype=np.float64).reshape(-1)
    n = int(y0_correct.size)
    assert len(group_ids) == n, f"group_ids len {len(group_ids)} != y0_correct {n}"
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[str(gid)].append(i)
    p_regen = np.zeros(n, dtype=np.float64)
    for idxs in groups.values():
        k = len(idxs)
        c = y0_correct[idxs]
        if k == 1:
            p_regen[idxs[0]] = float(c[0])
            continue
        total = float(c.sum())
        for i in idxs:
            p_i = (total - float(y0_correct[i])) / float(k - 1)
            assert 0.0 - 1e-9 <= p_i <= 1.0 + 1e-9, f"p_regen out of range: {p_i}"
            p_regen[i] = min(1.0, max(0.0, p_i))
    return p_regen


# LOO p_regen bins (K=4 → {0, 1/3, 2/3, 1}): low = p < 0.5, high = p >= 0.5.
P_REGEN_LOW_THRESH = 0.5


def regen_feedback_metrics(
    p_regen: np.ndarray,
    regen_weight: np.ndarray,
    r_self: np.ndarray,
    r_feedback: np.ndarray,
    group_ids: List[str],
    revised=None,
    acc_t2=None,
) -> Dict[str, float]:
    """Lightweight train logs for the regeneration-aware feedback reward."""
    p_regen = np.asarray(p_regen, dtype=np.float64)
    regen_weight = np.asarray(regen_weight, dtype=np.float64)
    r_self = np.asarray(r_self, dtype=np.float64)
    r_feedback = np.asarray(r_feedback, dtype=np.float64)
    hard = p_regen <= 1e-12
    low = p_regen < P_REGEN_LOW_THRESH
    high = p_regen >= P_REGEN_LOW_THRESH
    metrics = {
        "mean_p_regen": float(p_regen.mean()) if p_regen.size else 0.0,
        "frac_p_regen_0": float(hard.mean()) if p_regen.size else 0.0,
        "mean_regen_weight": float(regen_weight.mean()) if regen_weight.size else 0.0,
        "mean_R_self": float(r_self.mean()) if r_self.size else 0.0,
        "mean_R_feedback": float(r_feedback.mean()) if r_feedback.size else 0.0,
        "frac_rows_p_regen_zero": float(hard.mean()) if p_regen.size else 0.0,
    }
    if hard.any():
        metrics["mean_R_feedback_p_regen_zero"] = float(r_feedback[hard].mean())
    # unique prompts whose group y0 pass-rate is 0 (all siblings wrong → LOO p=0)
    by_g: Dict[str, List[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        by_g[str(gid)].append(i)
    n_g = max(len(by_g), 1)
    n_hard_g = sum(1 for idxs in by_g.values() if float(np.mean(p_regen[idxs])) <= 1e-12)
    metrics["frac_prompts_p_regen_zero"] = float(n_hard_g) / float(n_g)
    self_ok = r_self >= 0.5
    if self_ok.any():
        metrics["frac_success_rectify_p_regen_zero"] = float(hard[self_ok].mean())

    if revised is None:
        revised_mask = np.ones(p_regen.shape[0], dtype=bool) if acc_t2 is not None else self_ok
    else:
        revised_mask = np.asarray(revised, dtype=bool)
    if acc_t2 is None:
        rect_correct = r_self >= 0.5
    else:
        rect_correct = np.asarray(acc_t2, dtype=np.float64) >= 0.5

    def _rect_acc(bin_mask: np.ndarray):
        m = bin_mask & revised_mask
        if not m.any():
            return None
        return float(rect_correct[m].mean())

    acc0 = _rect_acc(hard)
    acc_low = _rect_acc(low)
    acc_high = _rect_acc(high)
    if acc0 is not None:
        metrics["rect_acc_p_regen_0"] = acc0
    if acc_low is not None:
        metrics["rect_acc_p_regen_low"] = acc_low
    if acc_high is not None:
        metrics["rect_acc_p_regen_high"] = acc_high
    return metrics


def pair_cell_label(acc_self: float, acc_generic: float) -> str:
    """Four-way (self, generic) cell: CW, CC, WW, WC. C := acc >= 0.5."""
    self_c = float(acc_self) >= 0.5
    gen_c = float(acc_generic) >= 0.5
    if self_c and not gen_c:
        return "CW"
    if self_c and gen_c:
        return "CC"
    if (not self_c) and (not gen_c):
        return "WW"
    return "WC"


def paired_outcome_metrics(cells) -> Dict[str, float]:
    """P(CW/CC/WW/WC), E[R_cf]=P(CW)-P(WC), CritiqueExclusiveRate=CW/(CW+CC).

    `cells` are forked samples only. CritiqueExclusiveRate is defined only when
    self succeeded at least once (CW+CC > 0).
    """
    counts = {"CW": 0, "CC": 0, "WW": 0, "WC": 0}
    for cell in cells:
        if cell in counts:
            counts[cell] += 1
    n = int(sum(counts.values()))
    metrics: Dict[str, float] = {
        "pair_n": float(n),
        "pair_n_cw": float(counts["CW"]),
        "pair_n_cc": float(counts["CC"]),
        "pair_n_ww": float(counts["WW"]),
        "pair_n_wc": float(counts["WC"]),
    }
    if n == 0:
        return metrics
    p_cw = counts["CW"] / n
    p_cc = counts["CC"] / n
    p_ww = counts["WW"] / n
    p_wc = counts["WC"] / n
    metrics.update({
        "p_cw": p_cw,
        "p_cc": p_cc,
        "p_ww": p_ww,
        "p_wc": p_wc,
        "e_r_cf": p_cw - p_wc,
    })
    self_ok = counts["CW"] + counts["CC"]
    if self_ok > 0:
        metrics["critique_exclusive_rate"] = counts["CW"] / self_ok
    return metrics


class PAGRewardManager:
    """Multi-turn dialogue reward manager with GenRM verification.

    Flow: User question -> Model answer -> Verification -> Regeneration (if needed)

    split_verify_reward=True:
      R_disc = genrm_score at last verdict token (GAE γ=1 also credits feedback)
      R_use  = usefulness at last feedback token, selected by feedback_mode:
        generic: R_critique = R_y(y_self) - R_y(y_generic)
        regen:   R_feedback = Δ_self * [1 + λ (1 - p_regen)]
        delta:   R_feedback = Δ_self = acc_t2 - acc_t1 ∈ {+1, 0, -1}
        acc:     R_use = acc_t2 - λ_cw · 1[C→W]  (λ_cw = lambda_cw)
        disc:    R_use = 0 (original PAG: only R_disc / genrm_score on verify)
      rectifier last token is unchanged: acc + optional policy_rs shaping
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
        # Split R_disc (verdict last token) vs R_use (feedback last token).
        self.split_verify_reward = bool(config.get('split_verify_reward', False))
        requested_cf = bool(config.get('generic_counterfactual', False))
        self.generic_counterfactual = requested_cf
        self.feedback_mode = resolve_feedback_mode(config)
        self.lambda_regen = _lambda_regen_scale(config)
        self.lambda_cw = _lambda_cw_scale(config)
        if self.feedback_mode == "generic":
            if not requested_cf:
                print(
                    "Warning: feedback_mode=generic needs "
                    "actor_rollout_ref.rollout.generic_counterfactual=True to sample "
                    "y_generic; R_critique is 0 if generic_response is missing."
                )
            self.generic_counterfactual = True
        if self.generic_counterfactual and not self.split_verify_reward:
            print(
                "Warning: generic_counterfactual=True requires split_verify_reward=True "
                "to place R_critique on the self-feedback span; R_critique will not be used."
            )
        if self.feedback_mode in ("regen", "delta", "acc") and not self.split_verify_reward:
            print(
                "Warning: feedback_mode="
                f"{self.feedback_mode} is intended for the verifier-feedback token "
                "(split_verify_reward=True); otherwise R_feedback is added on the verdict token."
            )

    def _feedback_last_global(
        self,
        response_ids: torch.Tensor,
        multiturn_mask: torch.Tensor,
        verify_start: int,
        verify_end: int,
    ) -> Optional[int]:
        """Global index of last feedback token inside the verify assistant span."""
        if verify_end <= verify_start:
            return None
        seg_mask = multiturn_mask[verify_start:verify_end]
        if seg_mask.numel() == 0 or not bool(seg_mask.any()):
            return None
        local_ids = response_ids[verify_start:verify_end][seg_mask]
        local_i = find_feedback_last_token_index(self.tokenizer, local_ids)
        if local_i is None:
            return None
        rel = torch.where(seg_mask)[0]
        if local_i < 0 or local_i >= rel.numel():
            return None
        return int(verify_start + rel[local_i].item())

    def _place_verify_rewards(
        self,
        reward_tensor: torch.Tensor,
        row: int,
        response_ids: torch.Tensor,
        multiturn_mask: torch.Tensor,
        verify_start: int,
        verify_end: int,
        r_disc: float,
        r_use: float,
    ) -> Tuple[int, Optional[int]]:
        """Place discrimination vs feedback-usefulness rewards.

        GAE with γ=1:
          R_disc at last verdict token → credits feedback + verdict
          R_use  at last feedback token → credits feedback only (does not flow forward)
        """
        last = verify_end - 1
        if last < 0:
            return last, None
        r_disc = float(r_disc)
        r_use = float(r_use)
        if (not self.split_verify_reward) or r_use == 0.0:
            reward_tensor[row, last] = r_disc + r_use
            return last, None
        fb = self._feedback_last_global(response_ids, multiturn_mask, verify_start, verify_end)
        if fb is None or fb == last:
            reward_tensor[row, last] = r_disc + r_use
            return last, None
        reward_tensor[row, last] = r_disc
        reward_tensor[row, fb] = r_use
        return last, fb

    def __call__(self, data: DataProto, return_dict: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute rewards for multi-turn dialogue."""
        if 'rm_scores' in data.batch:
            return data.batch['rm_scores'], {}

        batch_size = data.batch['responses'].shape[0]
        device = data.batch['responses'].device
        if 'num_turns' in data.meta_info:
            self.max_turns = data.meta_info['num_turns']
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metrics_tensors = {
            'turn_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
            'verify_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
            'turn_counts': torch.zeros(batch_size, dtype=torch.long, device=device)
        }
        
        answer_logs = []
        reward_extra_info = defaultdict(list)
        printed_sources = {}

        group_ids = prompt_group_ids(data)
        y0_correct = np.zeros(batch_size, dtype=np.float64)
        y0_cache: List[Optional[Dict[str, Any]]] = [None] * batch_size
        y0_boundaries: List[List[int]] = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            item = data[i]
            mask = item.batch['multiturn_mask']
            tbs: List[int] = []
            if mask.numel() > 0:
                padded = torch.cat([torch.tensor([True], device=mask.device), mask])
                diff = padded[1:].long() - padded[:-1].long()
                tbs = torch.where(diff == -1)[0].tolist()
                if mask[-1]:
                    tbs.append(mask.size(0))
            y0_boundaries[i] = tbs
            if not tbs:
                continue
            y0_text = self.tokenizer.decode(item.batch['responses'][:tbs[0]])
            y0_res = get_policy_score(
                solution_str=y0_text,
                ground_truth=item.non_tensor_batch['reward_model']['ground_truth'],
            )
            y0_cache[i] = y0_res
            y0_correct[i] = 1.0 if float(y0_res["acc"]) >= 0.5 else 0.0
        p_regen = compute_loo_p_regen(y0_correct, group_ids)
        
        for i in range(batch_size):
            # Initialize extra info for verifier mode
            if self.end_with_verifer:
                for key in ["all_pred", "all_acc", "all_genrm_pred", "all_genrm_score", "all_genrm_probs"]:
                    reward_extra_info[key].append([])

            data_item = data[i]
            response_ids = data_item.batch['responses']
            multiturn_mask = data_item.batch['multiturn_mask']
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            
            # Find turn boundaries from mask (cached from the y0 pre-pass)
            turn_boundaries = y0_boundaries[i]
            if not turn_boundaries and multiturn_mask.numel() > 0:
                padded_mask = torch.cat([torch.tensor([True], device=multiturn_mask.device), multiturn_mask])
                diff = padded_mask[1:].long() - padded_mask[:-1].long()
                turn_boundaries = torch.where(diff == -1)[0].tolist()
                if multiturn_mask[-1]:
                    turn_boundaries.append(multiturn_mask.size(0))
            
            metrics_tensors['turn_counts'][i] = len(turn_boundaries) - 1
            sample_answers = []
            final_turn = int(data_item.non_tensor_batch.get("final_generation_turn", 0))
            acc_t2 = None
            pred_t2 = None
            revised = False
            genrm_pred = None
            genrm_score = None
            genrm_probs = None
            acc_generic = None
            r_critique = 0.0
            r_self = 0.0
            r_feedback = 0.0
            delta_self = 0.0
            regen_weight = 1.0
            verify_text = ""
            generic_text_export = ""
                
            # Process first turn (reuse cached math-verify)
            first_result = y0_cache[i]
            if first_result is None:
                first_response = self.tokenizer.decode(response_ids[:turn_boundaries[0]])
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
            
            gt_judge = first_result["acc"] >= 0.5
            prev_acc = first_result["acc"]
            max_turns = self.max_turns + 1 if self.end_with_verifer else self.max_turns
                
            # Process subsequent turns
            for turn in range(1, max_turns):
                # GenRM verification
                verify_start = turn_boundaries[2*turn - 2] if turn > 1 else turn_boundaries[0]
                verify_end = turn_boundaries[2*turn - 1]
                verify_response = self.tokenizer.decode(
                    response_ids[verify_start:verify_end][multiturn_mask[verify_start:verify_end]], 
                    skip_special_tokens=True
                )
                
                verify_result = get_verification_score(verify_response, gt_judge)
                metrics_tensors['verify_accuracies'][i, turn-1] = verify_result["genrm_score"]
                r_disc = float(verify_result["genrm_score"])
                r_use = 0.0
                
                # Store verification info
                if turn == 1:
                    verify_text = verify_response
                    genrm_pred = verify_result["genrm_pred"]
                    genrm_score = verify_result["genrm_score"]
                    genrm_probs = data_item.non_tensor_batch.get("verify_probs", None)
                    reward_extra_info["genrm_pred"].append(genrm_pred)
                    reward_extra_info["genrm_score"].append(genrm_score)
                    reward_extra_info["genrm_probs"].append(genrm_probs)
                if self.end_with_verifer:
                    reward_extra_info["all_genrm_pred"][-1].append(verify_result["genrm_pred"])
                    reward_extra_info["all_genrm_score"][-1].append(verify_result["genrm_score"])
                    reward_extra_info['all_genrm_probs'][-1].append(data_item.non_tensor_batch["verify_probs"])
                
                # Policy response (if exists)
                if 2*turn >= len(turn_boundaries) or (self.end_with_verifer and turn == max_turns - 1):
                    self._place_verify_rewards(
                        reward_tensor, i, response_ids, multiturn_mask,
                        verify_start, verify_end, r_disc, 0.0,
                    )
                    if turn == 1:
                        reward_extra_info["r_disc"].append(r_disc)
                        reward_extra_info["r_use"].append(0.0)
                    break
                
                if self.is_only_genrm:
                    multiturn_mask[:verify_end] = False
                
                policy_start = verify_end
                policy_end = turn_boundaries[2*turn]
                policy_response = self.tokenizer.decode(
                    response_ids[policy_start:policy_end][multiturn_mask[policy_start:policy_end]]
                )
                
                policy_result = get_policy_score(solution_str=policy_response, ground_truth=ground_truth)
                
                # Set reward with optional reward shaping (rectifier unchanged)
                reward_value = policy_result["acc"]
                delta = float(policy_result["acc"]) - float(prev_acc)
                delta_self = delta
                if self.policy_rs:
                    reward_value += self.rs_coef * delta
                # Generic scoring: used by feedback_mode=generic; otherwise logging only.
                if self.generic_counterfactual:
                    generic_text = data_item.non_tensor_batch.get("generic_response", "") or ""
                    if isinstance(generic_text, bytes):
                        generic_text = generic_text.decode("utf-8", errors="ignore")
                    generic_text = str(generic_text).strip()
                    generic_text_export = generic_text
                    if generic_text:
                        generic_result = get_policy_score(
                            solution_str=generic_text, ground_truth=ground_truth
                        )
                        acc_generic = float(generic_result["acc"])
                r_self = 1.0 if float(policy_result["acc"]) >= 0.5 else 0.0
                if self.feedback_mode == "regen":
                    regen_weight = 1.0 + self.lambda_regen * (1.0 - float(p_regen[i]))
                    r_feedback = regen_aware_feedback_reward(
                        delta, float(p_regen[i]), self.lambda_regen
                    )
                    r_critique = r_feedback
                    r_use = r_feedback
                elif self.feedback_mode == "delta":
                    r_feedback = delta_feedback_reward(delta)
                    r_critique = r_feedback
                    r_use = r_feedback
                elif self.feedback_mode == "acc":
                    r_feedback = acc_t2_cw_feedback_reward(
                        float(policy_result["acc"]),
                        float(prev_acc),
                        self.lambda_cw,
                    )
                    r_critique = r_feedback
                    r_use = r_feedback
                elif self.feedback_mode == "disc":
                    r_feedback = 0.0
                    r_critique = 0.0
                    r_use = 0.0
                elif self.feedback_mode == "generic":
                    if acc_generic is not None:
                        r_critique = critique_advantage(
                            float(policy_result["acc"]), acc_generic
                        )
                    if self.split_verify_reward:
                        r_use = r_critique
                elif self.policy_rs and self.split_verify_reward:
                    r_use = self.rs_coef * delta
                self._place_verify_rewards(
                    reward_tensor, i, response_ids, multiturn_mask,
                    verify_start, verify_end, r_disc, r_use,
                )
                if turn == 1:
                    reward_extra_info["r_disc"].append(r_disc)
                    reward_extra_info["r_use"].append(float(r_use))
                reward_tensor[i, policy_end - 1] = reward_value
                
                metrics_tensors['turn_accuracies'][i, turn] = policy_result["acc"]
                prev_acc = policy_result["acc"]
                gt_judge = policy_result["acc"] >= 0.5
                
                if turn == 1:
                    revised = True
                    acc_t2 = policy_result["acc"]
                    pred_t2 = policy_result["pred"]
                    sample_answers.append(policy_result['pred'])
                    answer_logs.append(sample_answers)
                if self.end_with_verifer:
                    reward_extra_info["all_pred"][-1].append(policy_result["pred"])
                    reward_extra_info["all_acc"][-1].append(policy_result["acc"])
            
            # Per-sample rectify-analysis exports (always aligned with batch index)
            acc_t1 = first_result["acc"]
            acc_final = acc_t2 if revised else acc_t1
            reward_extra_info["ground_truth"].append(ground_truth)
            reward_extra_info["data_source"].append(data_source)
            reward_extra_info["acc_t1"].append(float(acc_t1))
            reward_extra_info["pred_t1"].append(first_result["pred"])
            reward_extra_info["acc_t2"].append(float(acc_t2) if acc_t2 is not None else -1.0)
            reward_extra_info["pred_t2"].append(pred_t2 if pred_t2 is not None else "")
            reward_extra_info["revised"].append(bool(revised))
            reward_extra_info["final_turn"].append(int(final_turn))
            reward_extra_info["acc_final"].append(float(acc_final))
            reward_extra_info["acc_generic"].append(
                float(acc_generic) if acc_generic is not None else -1.0
            )
            reward_extra_info["r_critique"].append(float(r_critique))
            reward_extra_info["p_regen"].append(float(p_regen[i]))
            reward_extra_info["regen_weight"].append(float(regen_weight))
            reward_extra_info["r_self"].append(float(r_self))
            reward_extra_info["delta_self"].append(float(delta_self))
            reward_extra_info["r_feedback"].append(float(r_feedback))
            reward_extra_info["verify_text"].append(verify_text)
            reward_extra_info["generic_response"].append(generic_text_export)
            if acc_generic is not None and acc_t2 is not None:
                reward_extra_info["pair_cell"].append(pair_cell_label(acc_t2, acc_generic))
            else:
                reward_extra_info["pair_cell"].append("")
            # Keep genrm_* lists aligned even if verify segment missing
            if genrm_pred is None:
                reward_extra_info["genrm_pred"].append("none")
                reward_extra_info["genrm_score"].append(0.0)
                reward_extra_info["genrm_probs"].append(None)
            if len(reward_extra_info["r_disc"]) < len(reward_extra_info["acc"]):
                reward_extra_info["r_disc"].append(float(genrm_score) if genrm_score is not None else 0.0)
                reward_extra_info["r_use"].append(0.0)

            # Debug output
            if self.num_examine > 0 and printed_sources.get(data_source, 0) < self.num_examine:
                printed_sources[data_source] = printed_sources.get(data_source, 0) + 1
                full_sequence = torch.cat((data_item.batch['prompts'], response_ids))
                print(self.tokenizer.decode(full_sequence, skip_special_tokens=True))
            if self.end_with_verifer:
                reward_extra_info["response"].append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        
        # Compute metrics
        data_sources = None
        if data.meta_info.get('validate', False):
            data_sources = [data[i].non_tensor_batch.get('data_source', 'unknown') for i in range(len(data))]
        
        metrics = self._compute_metrics(metrics_tensors, data_sources, answer_logs, data.non_tensor_batch["final_generation_turn"])
        metrics.update(regen_feedback_metrics(
            p_regen=p_regen,
            regen_weight=np.asarray(reward_extra_info["regen_weight"], dtype=np.float64),
            r_self=np.asarray(reward_extra_info["r_self"], dtype=np.float64),
            r_feedback=np.asarray(reward_extra_info["r_feedback"], dtype=np.float64),
            group_ids=group_ids,
            revised=reward_extra_info["revised"],
            acc_t2=reward_extra_info["acc_t2"],
        ))
        metrics["lambda_regen"] = float(self.lambda_regen)
        metrics["lambda_cw"] = float(self.lambda_cw)
        metrics["feedback_mode_id"] = float(
            {"generic": 0.0, "regen": 1.0, "delta": 2.0, "acc": 3.0, "disc": 4.0}.get(
                self.feedback_mode, -1.0
            )
        )
        if reward_extra_info.get("delta_self"):
            ds = np.asarray(reward_extra_info["delta_self"], dtype=np.float64)
            rev = np.asarray(reward_extra_info.get("revised", []), dtype=bool)
            if rev.size == ds.size and rev.any():
                metrics["mean_delta_self"] = float(ds[rev].mean())
            elif ds.size:
                metrics["mean_delta_self"] = float(ds.mean())
        if self.generic_counterfactual and reward_extra_info.get("acc_generic"):
            acc_g = np.asarray(reward_extra_info["acc_generic"], dtype=np.float64)
            n_all = max(int(acc_g.size), 1)
            forked = acc_g >= 0.0
            metrics["generic_fork_rate"] = float(forked.mean()) if acc_g.size else 0.0
            cells = [c for c in reward_extra_info.get("pair_cell", []) if c in ("CW", "CC", "WW", "WC")]
            metrics.update(paired_outcome_metrics(cells))
            if forked.any():
                metrics["generic_acc"] = float(acc_g[forked].mean())
                metrics["r_critique"] = float(metrics.get("e_r_cf", 0.0))
            if not data.meta_info.get("validate", False):
                cer = metrics.get("critique_exclusive_rate", float("nan"))
                print(
                    "pair CW/CC/WW/WC = "
                    f"{metrics.get('p_cw', 0):.3f}/{metrics.get('p_cc', 0):.3f}/"
                    f"{metrics.get('p_ww', 0):.3f}/{metrics.get('p_wc', 0):.3f}  "
                    f"E[R_cf]={metrics.get('e_r_cf', 0):.3f}  "
                    f"CER={cer:.3f}  "
                    f"n_fork={int(metrics.get('pair_n', 0))}/{n_all}"
                )
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info, "metrics": metrics}
        return reward_tensor, metrics 
        
    def _compute_single_group_metrics(self, turn_acc, verify_acc, turn_counts, final_turn, prefix=""):
        """Compute metrics for a group (global or source-specific)."""
        metrics = {}
        
        # Basic accuracy
        final_acc = turn_acc.gather(dim=-1, index=final_turn.unsqueeze(-1))
        metrics[f'{prefix}final_acc'] = final_acc.mean().item()
        
        for i in range(self.max_turns):
            clamped_turn = final_turn.clone().clamp(max=i)
            turn_policy_acc = turn_acc.gather(dim=-1, index=clamped_turn.unsqueeze(-1))
            metrics[f'{prefix}turn_{i+1}_accuracy'] = turn_policy_acc.mean().item()
        
        # Turn distribution
        for i in range(2*self.max_turns-1):
            count = (turn_counts == i).sum().item()
            if count > 0:
                metrics[f'{prefix}turn_count_{i}'] = count
                metrics[f'{prefix}turn_count_{i}_ratio'] = count / len(turn_counts)
        
        # Turn-specific accuracy
        for i in range(1, self.max_turns):
            policy_mask = turn_counts >= i*2
            if policy_mask.any():
                metrics[f'{prefix}turn_{i+1}_accuracy_selection'] = turn_acc[policy_mask, i].mean().item()
            
            verify_mask = turn_counts >= i*2-1
            if verify_mask.any():
                metrics[f'{prefix}verify_{i}_accuracy'] = verify_acc[:, i-1].mean().item()
        
        # Confusion matrix for verification
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
                f'{prefix}verify_f1': f1
            })
            
            # Transition metrics
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
                            f'{prefix}i_to_c_count': i_to_c
                        })
                    
                    if total_correct > 0:
                        metrics.update({
                            f'{prefix}c_to_i_rate': c_to_i / total_correct,
                            f'{prefix}c_to_i_rate_gt': c_to_i / len(turn1_policy),
                            f'{prefix}c_to_i_count': c_to_i
                        })
        
        return metrics

    def _compute_metrics(self, metrics_tensors, data_sources=None, answer_logs=None, final_generation_turn=None):
        """Compute all metrics."""
        final_turn = torch.tensor(final_generation_turn, device=metrics_tensors['turn_accuracies'].device, dtype=torch.long)
        print("final_answer_turn", final_turn)
        
        # Global metrics
        metrics = self._compute_single_group_metrics(
            metrics_tensors['turn_accuracies'], metrics_tensors['verify_accuracies'], 
            metrics_tensors['turn_counts'], final_turn
        )
        
        # Answer change analysis
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
                    'regeneration_samples': regen_samples
                })
        
        # Source-specific metrics
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