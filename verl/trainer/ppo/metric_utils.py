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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable, Optional
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from functools import partial


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True, max_singleturn_resp_length: int = None) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    multiturn_mask = batch.batch['multiturn_mask'].bool()
    response_mask = response_mask & multiturn_mask

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # 找到每一轮对话的开始位置
    turn_starts = multiturn_mask & (~torch.roll(multiturn_mask, shifts=1, dims=1))
    turn_starts[:, 0] = multiturn_mask[:, 0]  # 第一个位置特殊处理
    
    # 计算每个token属于哪一轮
    turn_indices = torch.cumsum(turn_starts.long(), dim=1)
    
    # 动态推断对话最大轮次
    max_turns = turn_indices.max().item()
    
    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    for turn in range(1, max_turns + 1):
        turn_mask = (turn_indices == turn) & multiturn_mask
        turn_lengths = turn_mask.sum(dim=1).float()  # (batch_size,)
        pos_indices = torch.arange(multiturn_mask.size(1), device=multiturn_mask.device).unsqueeze(0).expand_as(multiturn_mask)
        last_pos = torch.max(torch.where(turn_mask, pos_indices, torch.zeros_like(pos_indices)), dim=1)[0]
        eos_value_indices = torch.clamp(last_pos + 1, 0, values.size(1) - 1)
        turn_values = torch.masked_select(values, turn_mask)
        turn_eos_value = torch.gather(values, dim=1, index=eos_value_indices.unsqueeze(1))

        valid_mask = turn_lengths > 0
        turn_lengths = turn_lengths[valid_mask]
        turn_eos_value = turn_eos_value[valid_mask]
        
        # 只计算至少有一个样本存在此轮对话的情况
        if turn_lengths.sum() > 0:
            turn_metrics = {
                f'response_length_turn{turn}/mean': torch.mean(turn_lengths).detach().item(),
                f'response_length_turn{turn}/max': torch.max(turn_lengths).detach().item(),
                f'response_length_turn{turn}/min': torch.min(turn_lengths[turn_lengths > 0]).detach().item(),
                
                f'response_length_turn{turn}/samples': (turn_lengths > 0).sum().item(),  # 有多少样本包含此轮对话
                f'response_length_turn{turn}/clip_ratio': torch.mean(
                    torch.eq(turn_lengths, max_singleturn_resp_length).float()
                ).detach().item(),
            }
            values_metrics = {
                f'critic/turn{turn}_eos_value/mean': torch.mean(turn_eos_value).detach().item(),
                f'critic/turn{turn}_eos_value/max': torch.max(turn_eos_value).detach().item(),
                f'critic/turn{turn}_eos_value/min': torch.min(turn_eos_value).detach().item(),
                f'critic/turn{turn}_values/min': torch.min(turn_values).detach().item(),
                f'critic/turn{turn}_values/max': torch.max(turn_values).detach().item(),
                f'critic/turn{turn}_values/mean': torch.mean(turn_values).detach().item(),
            }
            
            metrics.update(turn_metrics)
            metrics.update(values_metrics)
    
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(data: list[Any],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

def calc_maj_all_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    assert len(data[0][vote_key]) == len(data[0][val_key])
    for d in data:
        for i in range(len(d[vote_key])):
            vote2vals[d[vote_key][i]].append(d[val_key][i])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

def calc_maj_final_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key][-1]].append(d[val_key][-1])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def calc_genrm_val(data: list[dict[str, Any]], val_key: str, pred_key: str) -> float:
    correct_samples = [d for d in data if d[pred_key] == "correct"]
    if not correct_samples:
        return 0
    return np.mean([d[val_key] for d in correct_samples])


def calc_genrm_bo1_val(data: list[dict[str, Any]], val_key: str, pred_key: str, prob_key: str) -> float:
    correct_samples = [d for d in data if d[pred_key] == "correct" and d[prob_key] is not None]
    wrong_samples = [d for d in data if d[pred_key] == "wrong" and d[prob_key] is not None]
    if correct_samples:
        bo1_index = np.argmax([d[prob_key] for d in correct_samples])
        return correct_samples[bo1_index][val_key]
    elif wrong_samples:
        bo1_index = np.argmin([d[prob_key] for d in wrong_samples])
        return wrong_samples[bo1_index][val_key]
    else:
        return np.mean([d[val_key] for d in data])
    
def calc_genrm_all_bo1_val(data: list[dict[str, Any]], val_key: str, pred_key: str, prob_key: str) -> float:
    total = []
    for d in data:
        assert len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        for i in range(len(d[pred_key])):
            total.append((d[pred_key][i], d[prob_key][i], d[val_key][i]))
    correct_samples = [d for d in total if d[0] == "correct" and d[1] is not None]
    wrong_samples = [d for d in total if d[0] == "wrong" and d[1] is not None]
    if correct_samples:
        bo1_index = np.argmax([d[1] for d in correct_samples])
        return correct_samples[bo1_index][2]
    elif wrong_samples:
        bo1_index = np.argmin([d[1] for d in wrong_samples])
        return wrong_samples[bo1_index][2]
    else:
        return np.mean([d[2] for d in data])


def calc_genrm_final_bo1_val(data: list[dict[str, Any]], val_key: str, pred_key: str, prob_key: str) -> float:
    total = []
    for d in data:
        assert len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        total.append((d[pred_key][-1], d[prob_key][-1], d[val_key][-1]))
    correct_samples = [d for d in total if d[0] == "correct" and d[1] is not None]
    wrong_samples = [d for d in total if d[0] == "wrong" and d[1] is not None]
    if correct_samples:
        bo1_index = np.argmax([d[1] for d in correct_samples])
        return correct_samples[bo1_index][2]
    elif wrong_samples:
        bo1_index = np.argmin([d[1] for d in wrong_samples])
        return wrong_samples[bo1_index][2]
    else:
        return np.mean([d[2] for d in data])


def calc_genrm_weighted_val(data: list[dict[str, Any]], val_key: str, vote_key: str, pred_key: str, prob_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    vote2rm = defaultdict(int)

    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])
        if d[pred_key] == "correct":
            vote2rm[d[vote_key]] += d[prob_key]
        elif d[pred_key] == "wrong":
            vote2rm[d[vote_key]] += - d[prob_key]

    max_vote = max(vote2rm, key=vote2rm.get)
    maj_val = vote2vals[max_vote][0]
    return maj_val


def calc_genrm_weighted_all_val(data: list[dict[str, Any]], val_key: str, vote_key: str, pred_key: str, prob_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    vote2rm = defaultdict(int)

    for d in data:
        assert len(d[vote_key]) == len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[vote_key])={len(d[vote_key])}, len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        for i in range(len(d[vote_key])):
            vote2vals[d[vote_key][i]].append(d[val_key][i])
            if d[pred_key][i] == "correct":
                vote2rm[d[vote_key][i]] += d[prob_key][i]
            elif d[pred_key][i] == "wrong":
                vote2rm[d[vote_key][i]] += - d[prob_key][i]
    max_vote = max(vote2rm, key=vote2rm.get)
    maj_val = vote2vals[max_vote][0]
    return maj_val


def calc_genrm_weighted_final_val(data: list[dict[str, Any]], val_key: str, vote_key: str, pred_key: str, prob_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    # 保存data 
    np.save("data_debug/calc_genrm_all_data.npy", data)

    vote2vals = defaultdict(list)
    vote2rm = defaultdict(int)
    for d in data:
        assert len(d[vote_key]) == len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[vote_key])={len(d[vote_key])}, len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        vote2vals[d[vote_key][-1]].append(d[val_key][-1])
        if d[pred_key][-1] == "correct":
            vote2rm[d[vote_key][-1]] += d[prob_key][-1]
        elif d[pred_key][-1] == "wrong":
            vote2rm[d[vote_key][-1]] += - d[prob_key][-1]

    try:
        max_vote = max(vote2rm, key=vote2rm.get)
        maj_val = vote2vals[max_vote][0]
    except:
        breakpoint()
    return maj_val


def process_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        seed: Random seed for bootstrapping
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        
        for var_name, var_vals in infos_dict.items():
            if sample_idx < len(var_vals):
                var2vals[var_name].append(var_vals[sample_idx])
    
    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if var_name not in ["acc", "all_acc"]:
                    continue
                metric = {}
                n_resps = len(var_vals)
                if var_name == "acc":
                    metric[f"mean@{n_resps}"] = np.mean(var_vals)
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    if var_name == "acc":
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                                                                                    subset_size=n,
                                                                                    reduce_fns=[np.max, np.min],
                                                                                    seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None and var_name == "acc":
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std
                    
                    if var2vals.get("all_pred", None) is not None and var_name == "all_acc":
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["all_pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_final_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj_final@{n}/mean"], metric[f"maj_final@{n}/std"] = maj_n_mean, maj_n_std
                    
                    if var2vals.get("all_pred", None) is not None and var_name == "all_acc":
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["all_pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_all_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj_all@{n}/mean"], metric[f"maj_all@{n}/std"] = maj_n_mean, maj_n_std

                    if var2vals.get("genrm_probs", None) is not None and var_name == "acc":
                        genrm_data = [{"val": val, "pred": pred, "prob": prob} for val, pred, prob in zip(var_vals, var2vals["genrm_pred"], var2vals["genrm_probs"])]
                        [(genrm_n_mean, genrm_n_std)
                        ] = bootstrap_metric(data=genrm_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_genrm_bo1_val, val_key="val", pred_key="pred", prob_key="prob")],
                                             seed=seed)
                        metric[f"genrm_prob_bo1@{n}/mean"], metric[f"genrm_prob_bo1@{n}/std"] = genrm_n_mean, genrm_n_std

                    if var2vals.get("genrm_probs", None) is not None and var_name == "all_acc":
                        genrm_data = [{"val": val, "pred": pred, "prob": prob} for val, pred, prob in zip(var_vals, var2vals["all_genrm_pred"], var2vals["all_genrm_probs"])]
                        [(genrm_n_mean, genrm_n_std)
                        ] = bootstrap_metric(data=genrm_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_genrm_final_bo1_val, val_key="val", pred_key="pred", prob_key="prob")],
                                             seed=seed)
                        metric[f"genrm_prob_final_bo1@{n}/mean"], metric[f"genrm_prob_final_bo1@{n}/std"] = genrm_n_mean, genrm_n_std
                    
                    if var2vals.get("genrm_probs", None) is not None and var_name == "all_acc":
                        genrm_data = [{"val": val, "pred": pred, "prob": prob} for val, pred, prob in zip(var_vals, var2vals["all_genrm_pred"], var2vals["all_genrm_probs"])]
                        [(genrm_n_mean, genrm_n_std)
                        ] = bootstrap_metric(data=genrm_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_genrm_all_bo1_val, val_key="val", pred_key="pred", prob_key="prob")],
                                             seed=seed)
                        metric[f"genrm_prob_all_bo1@{n}/mean"], metric[f"genrm_prob_all_bo1@{n}/std"] = genrm_n_mean, genrm_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val


def collapse_slide_window_trajectories(
    sample_inputs: list[str],
    data_sources: np.ndarray,
    infos_dict: dict[str, list[Any]],
    window_expand: int,
) -> tuple[list[str], np.ndarray, dict[str, list[Any]]]:
    """Collapse slide_window packed rows into one record per trajectory sample.

    Packed layout (interleave expand): for each traj, W consecutive rows with
    window_index = 0..W-1. Invalid (unreached) windows are ignored.
    Trajectory-level acc/pred come from the window at final_generation_turn.
    """
    n = len(sample_inputs)
    if window_expand is None or window_expand <= 1 or n == 0:
        out = {k: list(v) for k, v in infos_dict.items()}
        if 'num_turns' not in out and 'final_turn' in out:
            out['num_turns'] = [int(ft) + 1 for ft in out['final_turn']]
        return sample_inputs, np.asarray(data_sources), out

    if n % window_expand != 0:
        raise ValueError(
            f'slide_window val length {n} not divisible by window_expand={window_expand}'
        )

    n_traj = n // window_expand
    window_index = infos_dict.get('window_index', [0] * n)
    window_valid = infos_dict.get('window_valid', [True] * n)
    final_turns = infos_dict.get('final_turn', [0] * n)
    acc_t1 = infos_dict.get('acc_t1', infos_dict.get('acc', [0.0] * n))
    pred_t1 = infos_dict.get('pred_t1', infos_dict.get('pred', [''] * n))
    acc_t2 = infos_dict.get('acc_t2', [-1.0] * n)
    pred_t2 = infos_dict.get('pred_t2', [''] * n)
    genrm_pred = infos_dict.get('genrm_pred', ['none'] * n)
    genrm_score = infos_dict.get('genrm_score', [0.0] * n)
    genrm_probs = infos_dict.get('genrm_probs', [None] * n)
    c_ver = infos_dict.get('c_ver', [0.0] * n)
    c_rect = infos_dict.get('c_rect', [0.0] * n)
    c_f = infos_dict.get('c_f', [0.0] * n)
    ground_truth = infos_dict.get('ground_truth', [None] * n)
    data_source_info = infos_dict.get('data_source', list(data_sources))

    new_inputs: list[str] = []
    new_sources: list[Any] = []
    new_infos: dict[str, list[Any]] = defaultdict(list)

    for t in range(n_traj):
        base = t * window_expand
        idxs = list(range(base, base + window_expand))
        valid_idxs = [j for j in idxs if bool(window_valid[j])]
        if not valid_idxs:
            # should be rare; keep a zeroed traj row
            chosen = base
            ft = int(final_turns[base])
            traj_acc = 0.0
            traj_pred = ""
            revised = False
            y0_acc = 0.0
            y1_acc = -1.0
        else:
            ft = int(final_turns[valid_idxs[0]])
            # prefer the window that holds y_final (requires last answer was verified)
            chosen = None
            for j in valid_idxs:
                if int(window_index[j]) == ft:
                    chosen = j
                    break
            if chosen is None:
                # Legacy mismatch: last answer only as rectify on previous window
                last = max(valid_idxs, key=lambda j: int(window_index[j]))
                if float(acc_t2[last]) >= 0.0:
                    chosen = last
                    ft = int(window_index[last]) + 1
                    traj_acc = float(acc_t2[last])
                    traj_pred = pred_t2[last]
                else:
                    chosen = last
                    ft = int(window_index[chosen])
                    traj_acc = float(acc_t1[chosen])
                    traj_pred = pred_t1[chosen]
            else:
                traj_acc = float(acc_t1[chosen])
                traj_pred = pred_t1[chosen]
            # y0 / optional y1 for ECR from window 0
            w0 = base  # window_index 0 row
            y0_acc = float(acc_t1[w0]) if bool(window_valid[w0]) else 0.0
            if ft >= 1:
                # y1 from window0.acc_t2 or window1.acc_t1
                if float(acc_t2[w0]) >= 0:
                    y1_acc = float(acc_t2[w0])
                elif base + 1 < base + window_expand and bool(window_valid[base + 1]):
                    y1_acc = float(acc_t1[base + 1])
                else:
                    y1_acc = -1.0
                revised = True
            else:
                y1_acc = -1.0
                revised = False

        new_inputs.append(sample_inputs[base])
        new_sources.append(data_sources[base])
        new_infos['acc'].append(float(traj_acc))
        new_infos['pred'].append(traj_pred)
        new_infos['acc_final'].append(float(traj_acc))
        new_infos['acc_t1'].append(y0_acc)
        new_infos['pred_t1'].append(pred_t1[base] if bool(window_valid[base]) else "")
        new_infos['acc_t2'].append(y1_acc)
        new_infos['pred_t2'].append(pred_t2[base] if revised else "")
        new_infos['revised'].append(bool(revised))
        new_infos['final_turn'].append(int(ft))
        new_infos['num_turns'].append(int(ft) + 1)
        new_infos['genrm_pred'].append(genrm_pred[base] if bool(window_valid[base]) else "none")
        new_infos['genrm_score'].append(float(genrm_score[base]) if bool(window_valid[base]) else 0.0)
        new_infos['genrm_probs'].append(genrm_probs[base] if bool(window_valid[base]) else None)
        # feasibility: OR over valid windows (traj failed if any round failed)
        new_infos['c_ver'].append(float(max(float(c_ver[j]) for j in valid_idxs)) if valid_idxs else 0.0)
        new_infos['c_rect'].append(float(max(float(c_rect[j]) for j in valid_idxs)) if valid_idxs else 0.0)
        new_infos['c_f'].append(float(max(float(c_f[j]) for j in valid_idxs)) if valid_idxs else 0.0)
        new_infos['ground_truth'].append(ground_truth[chosen])
        new_infos['data_source'].append(data_source_info[base])
        new_infos['reward'].append(float(traj_acc))
        new_infos['window_valid'].append(True)
        new_infos['window_index'].append(0)

    return new_inputs, np.asarray(new_sources), dict(new_infos)


def compute_all_turn_event_metrics(infos_dict: dict[str, list[Any]]) -> dict[str, float]:
    """Pool verify / rectify events over all valid turn-windows (not first-only).

    For each window_valid row:
      - one verify event on y_w (acc_t1 vs genrm)
      - if acc_t2 >= 0, one rectify event y_w -> y_{w+1}
    """
    n = len(infos_dict.get('acc_t1', infos_dict.get('acc', [])))
    if n == 0:
        return {}

    window_valid = infos_dict.get('window_valid', [True] * n)
    acc_t1 = infos_dict.get('acc_t1', infos_dict.get('acc', [0.0] * n))
    acc_t2 = infos_dict.get('acc_t2', [-1.0] * n)
    genrm_score = infos_dict.get('genrm_score', [0.0] * n)
    genrm_pred = infos_dict.get('genrm_pred', ['none'] * n)

    # --- all verifies ---
    # Standard confusion (verify predicts "correct"):
    #   TP: said correct, was correct
    #   FN: said wrong,   was correct
    #   TN: said wrong,   was wrong
    #   FP: said correct, was wrong
    TP = FP = FN = TN = 0
    n_verify = 0
    for i in range(n):
        if not bool(window_valid[i]):
            continue
        gp = genrm_pred[i]
        if gp in (None, 'none'):
            continue
        n_verify += 1
        pol = float(acc_t1[i]) >= 0.5
        # prefer explicit verdict; fall back to score
        if gp in ('correct', 'wrong'):
            ver = gp == 'correct'
        else:
            ver = float(genrm_score[i]) > 0.5
        if ver and pol:
            TP += 1
        elif (not ver) and (not pol):
            TN += 1
        elif ver and (not pol):
            FP += 1
        else:
            FN += 1

    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    tnr = TN / (TN + FP) if (TN + FP) else 0.0
    # legacy pag-style aliases (note: old FP/TN naming was swapped vs standard)
    legacy_FP = TN  # said wrong & was wrong
    legacy_TN = FP  # said correct & was wrong
    metrics = {
        'n_verify': float(n_verify),
        'verify_TP': TP,
        'verify_FP': legacy_FP,
        'verify_FN': FN,
        'verify_TN': legacy_TN,
        'TPR': tpr,
        'TNR': tnr,
        'verify_recall': tpr,
        'verify_recall_negative': tnr,
        'verify_precision': TP / (TP + FP) if (TP + FP) else 0.0,
        'verify_f1': (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) else 0.0,
    }

    # --- all rectifies ---
    # ECR_TP: I→C count; EIR_FP: C→I count
    ecr_tp = eir_fp = 0
    n_rect = n_prev_wrong = n_prev_correct = 0
    i_to_c = c_to_i = c_to_c = i_to_i = 0
    for i in range(n):
        if not bool(window_valid[i]):
            continue
        prev = float(acc_t1[i])
        cur = float(acc_t2[i])
        if cur < 0:
            continue
        n_rect += 1
        prev_ok = prev >= 0.5
        cur_ok = cur >= 0.5
        if not prev_ok:
            n_prev_wrong += 1
            if cur_ok:
                ecr_tp += 1
                i_to_c += 1
            else:
                i_to_i += 1
        else:
            n_prev_correct += 1
            if not cur_ok:
                eir_fp += 1
                c_to_i += 1
            else:
                c_to_c += 1

    ecr = ecr_tp / n_prev_wrong if n_prev_wrong else 0.0
    eir = eir_fp / n_prev_correct if n_prev_correct else 0.0
    metrics.update({
        'n_rectify': float(n_rect),
        'ECR_TP': ecr_tp,
        'EIR_FP': eir_fp,
        'ECR': ecr,
        'EIR': eir,
        'i_to_c_rate': ecr,
        'c_to_i_rate': eir,
        'i_to_c_count': i_to_c,
        'c_to_i_count': c_to_i,
        'i_to_i_count': i_to_i,
        'c_to_c_count': c_to_c,
    })
    return metrics


def compute_vf_target_audit(
    costs: torch.Tensor,
    returns_f: torch.Tensor,
    values_f: torch.Tensor,
    multiturn_mask: torch.Tensor,
    window_valid: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Audit what the failure critic actually regresses onto (token-level).

    Compares sample-level P(c^F=1) with the distribution of ``returns_f`` under the
    same ``multiturn_mask`` used by ``compute_value_loss``. If failure rate is ~30%
    but E[returns_f|mask] and P(returns_f>0|mask) are near 0, the VF target is being
    diluted / misaligned — not a "sparse failure" data problem.
    """
    mt = multiturn_mask.bool()
    c = costs.float()
    tgt = returns_f.float()
    vf = values_f.float()
    bsz = c.shape[0]

    if window_valid is not None:
        valid = torch.as_tensor(window_valid.astype(np.bool_), device=c.device)
    else:
        valid = torch.ones(bsz, dtype=torch.bool, device=c.device)
    if not valid.any():
        return {
            'vf_audit/n_valid': 0.0,
            'vf_audit/P_c_f': 0.0,
            'vf_audit/P_target_pos': 0.0,
            'vf_audit/E_target': 0.0,
        }

    # per-sample episode cost (same as J_F construction)
    ep_cost = c.sum(dim=-1)
    sample_fail = ep_cost > eps
    p_c_f = sample_fail[valid].float().mean().item()

    # token mask used by VF loss
    row_mt = mt & valid.unsqueeze(-1)
    n_mt = row_mt.sum().clamp(min=1).float()
    tgt_mt = tgt[row_mt]
    vf_mt = vf[row_mt]
    c_mt = c[row_mt]

    p_tgt_pos = (tgt_mt > eps).float().mean().item() if tgt_mt.numel() else 0.0
    e_tgt = tgt_mt.mean().item() if tgt_mt.numel() else 0.0
    e_tgt_max = tgt_mt.max().item() if tgt_mt.numel() else 0.0
    e_vf = vf_mt.mean().item() if vf_mt.numel() else 0.0

    cost_tok = row_mt & (c > eps)
    n_cost = cost_tok.sum()
    if n_cost > 0:
        e_tgt_at_c = tgt[cost_tok].mean().item()
        e_vf_at_c = vf[cost_tok].mean().item()
        # among sample-fail rows, fraction of cost>0 tokens that sit inside multiturn_mask
        # (should be ~1; <1 ⇒ cost written off-mask / indexing bug)
        cost_all = (c > eps) & valid.unsqueeze(-1)
        cov = ((c > eps) & mt & valid.unsqueeze(-1)).sum().float() / cost_all.sum().clamp(min=1).float()
        mask_cov_on_cf = cov.item()
    else:
        e_tgt_at_c = 0.0
        e_vf_at_c = 0.0
        mask_cov_on_cf = 1.0

    zero_tok = row_mt & (c <= eps)
    if zero_tok.any():
        e_tgt_at_zero_c = tgt[zero_tok].mean().item()
        e_vf_at_zero_c = vf[zero_tok].mean().item()
    else:
        e_tgt_at_zero_c = 0.0
        e_vf_at_zero_c = 0.0

    # Dilution proxy: on failing samples, mean token target under multiturn vs at cost tokens
    fail_rows = sample_fail & valid
    if fail_rows.any():
        dil_num = 0.0
        dil_den = 0.0
        for i in torch.where(fail_rows)[0].tolist():
            m = mt[i]
            if not m.any():
                continue
            dil_num += tgt[i][m].mean().item()
            dil_den += 1.0
        e_tgt_fail_seqmean = dil_num / max(dil_den, 1.0)
        # expected if one unit cost smeared over L multiturn tokens
        L = mt[fail_rows].float().sum(dim=-1).mean().item()
    else:
        e_tgt_fail_seqmean = 0.0
        L = 0.0

    # Theoretical MSE if critic predicts constant E[target|mask] (sanity vs logged vf_loss_f)
    if tgt_mt.numel():
        const = e_tgt
        mse_const = 0.5 * ((tgt_mt - const) ** 2).mean().item()
    else:
        mse_const = 0.0

    # Segment-end diagnostic: binary suffix G at each multiturn True-run end
    # G_end = max_{t'>=end} c_{t'}  (token-wise max from that position)
    g_ends = []
    tgt_ends = []
    c_ends = []
    for i in torch.where(valid)[0].tolist():
        m = mt[i]
        if not m.any():
            continue
        ext = torch.cat([m, m.new_zeros(1, dtype=torch.bool)])
        ends = (torch.where(ext[:-1] & ~ext[1:])[0]).tolist()  # inclusive end indices
        # suffix max of costs (inclusive)
        # reverse cummax then flip
        c_i = c[i]
        # for position t, max(c[t:])
        rev = torch.flip(c_i, dims=[0])
        suf = torch.flip(torch.cummax(rev, dim=0).values, dims=[0])
        for e in ends:
            g_ends.append(float(suf[e].item() > eps))
            tgt_ends.append(float(tgt[i, e].item()))
            c_ends.append(float(c_i[e].item()))

    if g_ends:
        g_arr = np.asarray(g_ends, dtype=np.float64)
        t_arr = np.asarray(tgt_ends, dtype=np.float64)
        c_arr = np.asarray(c_ends, dtype=np.float64)
        p_g = float(g_arr.mean())
        e_tgt_at_ends = float(t_arr.mean())
        # where G=1, target should be high if cost-to-go is correct
        if (g_arr > 0.5).any():
            e_tgt_g1 = float(t_arr[g_arr > 0.5].mean())
        else:
            e_tgt_g1 = 0.0
        if (g_arr <= 0.5).any():
            e_tgt_g0 = float(t_arr[g_arr <= 0.5].mean())
        else:
            e_tgt_g0 = 0.0
        p_c_at_ends = float((c_arr > eps).mean())
    else:
        p_g = 0.0
        e_tgt_at_ends = 0.0
        e_tgt_g1 = 0.0
        e_tgt_g0 = 0.0
        p_c_at_ends = 0.0

    return {
        'vf_audit/n_valid': float(valid.sum().item()),
        'vf_audit/P_c_f': float(p_c_f),
        'vf_audit/P_target_pos': float(p_tgt_pos),
        'vf_audit/E_target': float(e_tgt),
        'vf_audit/max_target': float(e_tgt_max),
        'vf_audit/E_target_at_c_pos': float(e_tgt_at_c),
        'vf_audit/E_target_at_zero_c': float(e_tgt_at_zero_c),
        'vf_audit/E_vf': float(e_vf),
        'vf_audit/E_vf_at_c_pos': float(e_vf_at_c),
        'vf_audit/E_vf_at_zero_c': float(e_vf_at_zero_c),
        'vf_audit/mask_coverage_on_c_pos': float(mask_cov_on_cf),
        'vf_audit/E_target_fail_seqmean': float(e_tgt_fail_seqmean),
        'vf_audit/mean_multiturn_len_fail': float(L),
        'vf_audit/mse_const_at_E_target': float(mse_const),
        # answer/segment-end diagnostic (not a methodology change)
        'vf_audit/P_G_at_seg_end': float(p_g),
        'vf_audit/E_target_at_seg_end': float(e_tgt_at_ends),
        'vf_audit/E_target_at_seg_end_G1': float(e_tgt_g1),
        'vf_audit/E_target_at_seg_end_G0': float(e_tgt_g0),
        'vf_audit/P_c_at_seg_end': float(p_c_at_ends),
    }


def compute_trajectory_val_metrics(infos_dict: dict[str, list[Any]]) -> dict[str, float]:
    """Per-trajectory validation metrics (equal weight per sample, not per window).

    Verify/rectify event rates are NOT computed here — use
    ``compute_all_turn_event_metrics`` on pre-collapse window rows.
    """
    acc = np.asarray(infos_dict.get('acc_final', infos_dict.get('acc', [])), dtype=np.float64)
    if acc.size == 0:
        return {}
    turns = np.asarray(
        infos_dict.get('num_turns', [int(t) + 1 for t in infos_dict.get('final_turn', [0] * len(acc))]),
        dtype=np.float64,
    )
    y0 = np.asarray(infos_dict.get('acc_t1', acc), dtype=np.float64)
    revised = np.asarray(infos_dict.get('revised', [False] * len(acc)), dtype=bool)

    correct = acc >= 0.5
    metrics = {
        'final_acc': float(acc.mean()),
        'turn1_acc': float(y0.mean()),
        'mean_turns': float(turns.mean()),
        'mean_turns_correct': float(turns[correct].mean()) if correct.any() else 0.0,
        'mean_turns_incorrect': float(turns[~correct].mean()) if (~correct).any() else 0.0,
        'early_stop_rate': float((turns <= 1).mean()),
        'revised_rate': float(revised.mean()),
        'n_traj': float(len(acc)),
    }
    if 'c_ver' in infos_dict:
        metrics['c_ver_rate'] = float(np.mean(infos_dict['c_ver']))
        metrics['c_rect_rate'] = float(np.mean(infos_dict['c_rect']))
        metrics['c_f_rate'] = float(np.mean(infos_dict['c_f']))
    return metrics
