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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import ray
import numpy as np
import verl.utils.hdfs_io as hdfs_io
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.validation_utils import save_validation_results_to_json
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.curriculum_sampler import GenerationFrontierSampler
from verl.utils.dataset.sec_sampler import (
    SECSampler,
    aggregate_prompt_refresh_stats,
    load_initial_category_stats,
    prompt_ids_from_rlhf_dataset,
    refresh_protocol_from_mapping,
)

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    metrics = {}
    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    if kl_ctrl.multiturn_kl:
        multiturn_mask = data.batch['multiturn_mask']
        
        beta_mask = torch.ones_like(response_mask, dtype=torch.float32) * beta
        
        beta1 = kl_ctrl.kl_coef_turn1
        beta2 = kl_ctrl.kl_coef_turn2
        
        turn_starts = multiturn_mask & (~torch.roll(multiturn_mask, shifts=1, dims=1))
        turn_starts[:, 0] = multiturn_mask[:, 0]
        
        turn_indices = torch.cumsum(turn_starts.long(), dim=1)
        
        turn1_mask = (turn_indices == 1) & multiturn_mask
        turn2_mask = (turn_indices == 2) & multiturn_mask
        
        # 应用不同的beta值
        beta_mask = torch.where(turn1_mask, beta1, beta_mask)
        beta_mask = torch.where(turn2_mask, beta2, beta_mask)
        metrics['critic/kl_coeff_turn1'] = beta1
        metrics['critic/kl_coeff_turn2'] = beta2
        
        token_level_rewards = token_level_scores - beta_mask * kld
    else:
        token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics.update({'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta})

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, norm_type="whole"):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        if 'multiturn_mask' in data.batch.keys():
            multiturn_mask = data.batch['multiturn_mask']
        else:
            multiturn_mask = None
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=data.batch['response_mask'],
                                                                      gamma=gamma,
                                                                      lam=lam,
                                                                      multiturn_mask=multiturn_mask,
                                                                      norm_type=norm_type)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            multiturn_mask=data.batch['multiturn_mask'],
            norm_type=norm_type,
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
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


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "token-mean-unbiased"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                         num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        curriculum_cfg = self.config.get("curriculum", {})
        sec_cfg        = self.config.get("sec", {})

        # SEC and frontier curriculum are mutually exclusive
        assert not (sec_cfg.get("enabled", False) and curriculum_cfg.get("enabled", False)), (
            "sec.enabled and curriculum.enabled cannot both be True. "
            "Choose one curriculum strategy."
        )

        self.sec_sampler: Optional[SECSampler] = None

        if sec_cfg.get("enabled", False):
            prompt_ids = prompt_ids_from_rlhf_dataset(self.train_dataset)
            print(f"[SEC] {len(prompt_ids)} prompts keyed by extra_info.index "
                  f"(min={min(prompt_ids)}, max={max(prompt_ids)})")

            sec_log = sec_cfg.get("log_path") or os.path.join(
                self.config.trainer.default_local_dir, "sec_q_log.jsonl"
            )
            self.sec_sampler = SECSampler(
                prompt_ids=prompt_ids,
                q_alpha=float(sec_cfg.get("q_alpha", 0.1)),
                temperature=float(sec_cfg.get("temperature", 1.0)),
                seed=int(self.config.data.get("seed", 42)),
                log_path=sec_log,
            )
            self._sec_actor_updated_at = None
            # SEC bypasses StatefulDataLoader; use RandomSampler as placeholder
            sampler = RandomSampler(data_source=self.train_dataset)
            self.curriculum_sampler = None
            print(f"[SEC] SECSampler enabled (dynamic C1–C5): α={self.sec_sampler.q_alpha}, "
                  f"τ={self.sec_sampler.temperature}, "
                  f"refresh_interval={sec_cfg.get('refresh_interval', 50)}, log={sec_log}")

        elif curriculum_cfg.get("enabled", False):
            # curriculum_cfg.get("log_path", default) returns None when key exists but is null.
            # Use explicit None check so null in YAML falls back to the default path.
            log_path = curriculum_cfg.get("log_path") or os.path.join(
                self.config.trainer.default_local_dir, "curriculum_competence.jsonl"
            )
            self.curriculum_sampler = GenerationFrontierSampler(
                n_total=len(self.train_dataset),
                epsilon=float(curriculum_cfg.get("epsilon", 0.3)),
                seed=int(self.config.data.get("seed", 42)),
                log_path=log_path,
            )
            sampler = self.curriculum_sampler
            print(
                f"[curriculum] GenerationFrontierSampler enabled: "
                f"N={len(self.train_dataset)}, epsilon={self.curriculum_sampler.epsilon}, "
                f"log={log_path}"
            )
        elif self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
            self.curriculum_sampler = None
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
            self.curriculum_sampler = None

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                       num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        metric_dict = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Keep GT for oracle/always revise gates before popping tensors.
            revise_gate = self.config.actor_rollout_ref.rollout.val_kwargs.get('revise_gate', 'pag')
            ground_truths = None
            if 'reward_model' in test_batch.non_tensor_batch:
                ground_truths = np.array(
                    [rm.get('ground_truth', None) for rm in test_batch.non_tensor_batch['reward_model']],
                    dtype=object,
                )

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
            if ground_truths is not None:
                test_gen_batch.non_tensor_batch['ground_truth'] = ground_truths
            if self.config.actor_rollout_ref.rollout.val_kwargs.num_turns is not None:
                num_turns = self.config.actor_rollout_ref.rollout.val_kwargs.num_turns
            else:
                num_turns = self.config.actor_rollout_ref.rollout.num_turns
            val_kwargs = {
                # 'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
                'num_turns': num_turns,
                'revise_gate': revise_gate,
            }
            test_gen_batch.meta_info.update(val_kwargs)
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # TODO: multi-turn evaluation
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            test_batch.meta_info = {'validate': True, 'num_turns': num_turns}
            result = self.val_reward_fn(test_batch, return_dict=True)
            
            if isinstance(result, dict) and "reward_tensor" in result:
                reward_tensor = result["reward_tensor"]
                reward_metrics = result.get("metrics", {})
                
                # 添加多轮验证指标
                if reward_metrics:
                    val_multiturn_metrics = {f'val/multiturn/{k}': v for k, v in reward_metrics.items()}
                    metric_dict.update(val_multiturn_metrics)
            else:
                reward_tensor = result

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        
        save_validation_results = self.config.trainer.get('save_validation_results', False)
        json_path = self.config.trainer.get('validation_results_path', 'validation_results.json')
        
        if save_validation_results:
            saved_file = save_validation_results_to_json(
                data_sources=data_sources,
                sample_inputs=sample_inputs,
                infos_dict=reward_extra_infos_dict,
                json_path=json_path
            )

            import pickle
            import os
            import time
            
            save_dir = os.path.join("validation_data", time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(save_dir, exist_ok=True)
            
            with open(os.path.join(save_dir, "data_sources.pkl"), "wb") as f:
                pickle.dump(data_sources, f)
            
            with open(os.path.join(save_dir, "sample_inputs.pkl"), "wb") as f:
                pickle.dump(sample_inputs, f)
                
            with open(os.path.join(save_dir, "reward_extra_infos.pkl"), "wb") as f:
                pickle.dump(reward_extra_infos_dict, f)
                
            print(f"Validation data saved to {save_dir}")

        data_src2var2metric2val = process_validation_metrics(
            data_sources=data_sources, 
            sample_inputs=sample_inputs, 
            infos_dict=reward_extra_infos_dict
        )
        
        # 将其它指标也添加到metric_dict中
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if var_name == core_var and f"@{n_max}" in metric_name:
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Full validation window (all val batches). Per-batch val/multiturn/* would
        # otherwise keep only the last dataloader shard.
        pair_cells = [
            c for c in reward_extra_infos_dict.get("pair_cell", [])
            if c in ("CW", "CC", "WW", "WC")
        ]
        if pair_cells:
            from verl.workers.reward_manager.pag import paired_outcome_metrics
            for key, val in paired_outcome_metrics(pair_cells).items():
                metric_dict[f"val/multiturn/{key}"] = val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _sec_refresh_protocol(self) -> dict:
        roll = self.config.actor_rollout_ref.rollout
        sec_cfg = self.config.get("sec", {})
        k_cfg = sec_cfg.get("refresh_rollouts")
        k = int(roll.n) if k_cfg in (None, "null") else int(k_cfg)
        if k != int(roll.n):
            raise ValueError(
                f"sec.refresh_rollouts={k} must equal actor_rollout_ref.rollout.n="
                f"{int(roll.n)} so refresh uses the same PAG n as training."
            )
        return refresh_protocol_from_mapping({
            "refresh_rollouts": k,
            "model_path": str(self.config.actor_rollout_ref.model.path),
            "rollout_type": str(roll.get("rollout_type", "pag")),
            "num_turns": int(roll.num_turns),
            "temperature": float(roll.get("temperature", 1.0)),
            "top_k": int(roll.get("top_k", -1)),
            "top_p": float(roll.get("top_p", 1.0)),
            "revise_gate": str(roll.get("revise_gate", "pag")),
            "do_sample": True,
        })

    def _ensure_sec_categories(self) -> dict:
        """Before the first SEC batch: resume mapping, load protocol-matched stats, or refresh."""
        if self.sec_sampler is None:
            return {}
        if self.sec_sampler.has_membership():
            print(f"[SEC] categories already present "
                  f"(last_refresh_step={self.sec_sampler.last_refresh_step})")
            return {}
        stats_path = self.config.sec.get("initial_category_stats_path")
        if stats_path not in (None, "null", ""):
            protocol = self._sec_refresh_protocol()
            g_map, nwc_map = load_initial_category_stats(str(stats_path), protocol)
            expected = set(int(x) for x in self.sec_sampler.prompt_ids.tolist())
            missing = expected - set(g_map)
            if missing:
                raise ValueError(
                    f"{stats_path} is missing {len(missing)} prompt ids "
                    f"(e.g. {sorted(missing)[:5]})"
                )
            metrics = self.sec_sampler.replace_membership(g_map, nwc_map, step=0)
            print(f"[SEC] loaded initial C1–C5 from {stats_path}")
            return metrics
        print("[SEC] no membership yet; running synchronized full-train refresh at step 0")
        return self._sec_refresh(step=0)

    def _sec_refresh(self, step: int) -> dict:
        """Synchronized full-train PAG measurement of the current policy snapshot."""
        if self.sec_sampler is None:
            return {}
        if step > 0 and self._sec_actor_updated_at != step:
            raise RuntimeError(
                f"SEC refresh at step={step} but actor was last updated at "
                f"{self._sec_actor_updated_at}. Refresh must run after the PPO "
                "update that produced the current snapshot."
            )
        protocol = self._sec_refresh_protocol()
        k = int(protocol["refresh_rollouts"])
        n_prompts = len(self.train_dataset)
        chunk = int(self.config.data.get("gen_batch_size") or self.config.data.train_batch_size)
        world = int(self.actor_rollout_wg.world_size)
        t0 = time.perf_counter()

        all_pids: list = []
        all_acc1: list = []
        all_acc2: list = []
        all_rev: list = []
        n_traj = 0
        n_rect = 0
        synced = False
        fingerprint = None

        roll = self.config.actor_rollout_ref.rollout
        for start in range(0, n_prompts, chunk):
            end = min(start + chunk, n_prompts)
            items = [self.train_dataset[i] for i in range(start, end)]
            batch = DataProto.from_single_dict(collate_fn(items))
            n_unique = len(batch)
            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
            gen_batch.meta_info.update({
                'sec_category_refresh': True,
                'do_sample': True,
                'validate': False,
                'num_turns': int(roll.num_turns),
                'revise_gate': str(roll.get('revise_gate', 'pag')),
            })
            padded, pad_size = pad_dataproto_to_divisor(gen_batch, world)
            out_padded = self.actor_rollout_wg.generate_sequences(padded)
            if out_padded.meta_info.get('rollout_weights_synced'):
                synced = True
                fingerprint = out_padded.meta_info.get('rollout_weight_fingerprint')
            # pad_size is extra *unique* prompts; generate_sequences then
            # repeat_interleaves K, so drop pad_size * K trajectories at the end.
            keep = n_unique * k
            out = unpad_dataproto(out_padded, pad_size * k)
            if len(out) != keep:
                raise RuntimeError(
                    f"SEC refresh expected {keep} trajectories, got {len(out)} "
                    f"(n_unique={n_unique}, K={k}, pad_size={pad_size})"
                )
            batch = batch.repeat(repeat_times=k, interleave=True)
            batch = batch.union(out)
            result = self.reward_fn(batch, return_dict=True)
            extra = result['reward_extra_info']
            pids = np.asarray(batch.non_tensor_batch['index']).astype(np.int64).ravel()
            acc1 = np.asarray(extra['acc_t1'], dtype=np.float64).ravel()
            acc2 = np.asarray(extra['acc_t2'], dtype=np.float64).ravel()
            rev = np.asarray(extra['revised'], dtype=bool).ravel()
            all_pids.extend(int(x) for x in pids.tolist())
            all_acc1.extend(float(x) for x in acc1.tolist())
            all_acc2.extend(float(x) for x in acc2.tolist())
            all_rev.extend(bool(x) for x in rev.tolist())
            n_traj += int(pids.size)
            n_rect += int(rev.sum())

        if not synced:
            raise RuntimeError(
                "SEC refresh did not observe rollout_weights_synced after "
                "generate_sequences. Refresh must run on the current actor "
                "snapshot (FSDP→vLLM sync inside the sharding manager), not "
                "stale rollout weights."
            )

        g_map, nwc_map = aggregate_prompt_refresh_stats(
            all_pids, all_acc1, all_acc2, all_rev, k_refresh=k,
        )
        expected = set(int(x) for x in self.sec_sampler.prompt_ids.tolist())
        got = set(g_map)
        if got != expected:
            raise RuntimeError(
                f"SEC refresh coverage mismatch: missing={len(expected - got)} "
                f"extra={len(got - expected)}"
            )
        mem_metrics = self.sec_sampler.replace_membership(g_map, nwc_map, step=step)
        elapsed = time.perf_counter() - t0
        mem_metrics['dynamic/refresh_wall_time_s'] = float(elapsed)
        mem_metrics['dynamic/refresh_n_prompts'] = float(n_prompts)
        mem_metrics['dynamic/refresh_n_trajectories'] = float(n_traj)
        mem_metrics['dynamic/refresh_n_rectified'] = float(n_rect)
        mem_metrics['dynamic/refresh_rollouts_K'] = float(k)
        mem_metrics['dynamic/rollout_weights_synced'] = 1.0 if synced else 0.0
        if fingerprint is not None:
            mem_metrics['dynamic/rollout_weight_fingerprint'] = float(fingerprint)
        print(
            f"[SEC] refresh step={step} wall={elapsed:.1f}s "
            f"prompts={n_prompts} traj={n_traj} rectified={n_rect} K={k} "
            f"(wall-clock, not the ~rollout-workload fraction)"
        )
        return mem_metrics

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save curriculum sampler state (g[], n_seen[], epoch, mode)
        if self.curriculum_sampler is not None:
            sampler_path = os.path.join(local_global_step_folder, 'curriculum_sampler.pt')
            torch.save(self.curriculum_sampler.state_dict(), sampler_path)

        # save SEC sampler state (Q[], cumulative_counts[], step)
        if self.sec_sampler is not None:
            sec_path = os.path.join(local_global_step_folder, 'sec_sampler.pt')
            torch.save(self.sec_sampler.state_dict(), sec_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load curriculum sampler state
        if self.curriculum_sampler is not None:
            sampler_path = os.path.join(global_step_folder, 'curriculum_sampler.pt')
            if os.path.exists(sampler_path):
                sampler_state = torch.load(sampler_path, weights_only=False)
                self.curriculum_sampler.load_state_dict(sampler_state)
                print(f"[curriculum] Resumed sampler state: epoch={self.curriculum_sampler.epoch}, "
                      f"n_seen={self.curriculum_sampler.n_seen.sum()}, mode={self.curriculum_sampler.mode}")
            else:
                print(f"Warning: No curriculum sampler state found at {sampler_path}, g[] starts from scratch")

        # load SEC sampler state
        if self.sec_sampler is not None:
            sec_path = os.path.join(global_step_folder, 'sec_sampler.pt')
            if os.path.exists(sec_path):
                sec_state = torch.load(sec_path, weights_only=False)
                self.sec_sampler.load_state_dict(sec_state)
                print(f"[SEC] Resumed Q={self.sec_sampler.Q.tolist()}, step={self.sec_sampler.step}")
            else:
                print(f"[SEC] No sampler state found at {sec_path}, Q starts from zero")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        if self.sec_sampler is not None:
            _init_sec = self._ensure_sec_categories()
            if _init_sec:
                logger.log(data=_init_sec, step=self.global_steps)

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # SEC: number of steps per epoch matches the DataLoader length (same compute budget).
        _sec_steps_per_epoch = len(self.train_dataloader)
        _sec_batch_size = self.config.data.get('gen_batch_size', self.config.data.train_batch_size)

        for epoch in range(self.config.trainer.total_epochs):
            if self.curriculum_sampler is not None:
                # tell the sampler which epoch we're in so it can switch mode
                # even epoch → uniform refresh (update g); odd → curriculum freeze (g frozen)
                self.curriculum_sampler.set_epoch(epoch)
                print(
                    f"[curriculum] epoch={epoch} mode={self.curriculum_sampler.mode} "
                    f"({self.curriculum_sampler.stats()['n_seen']}/{self.curriculum_sampler.N} seen)"
                )

            # SEC: build an on-demand iterator that yields batch_dicts constructed
            # using the current Q-values (updated at the END of each step below).
            # This completely replaces StatefulDataLoader iteration to avoid prefetch.
            if self.sec_sampler is not None:
                def _sec_batch_iter(sec_sampler, dataset, batch_size, steps):
                    from verl.utils.dataset.rl_dataset import collate_fn as _cfn
                    for _ in range(steps):
                        indices = sec_sampler.sample_batch(batch_size)
                        items = [dataset[int(i)] for i in indices]
                        for item, idx in zip(items, indices):
                            item['math_level'] = sec_sampler.category_of_row(int(idx))
                        yield _cfn(items)
                _batch_source = _sec_batch_iter(
                    self.sec_sampler, self.train_dataset, _sec_batch_size, _sec_steps_per_epoch
                )
            else:
                _batch_source = self.train_dataloader

            for batch_dict in _batch_source:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                            reward_metrics = reward_result['metrics']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}
                            reward_metrics = {}
                        multiturn_metrics = {f'multiturn/{k}': v for k, v in reward_metrics.items()}
                        metrics.update(multiturn_metrics)
                        batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # curriculum sampler update.
                        # update() is a no-op in curriculum (odd) epochs — g is frozen.
                        # In uniform (even) epochs it refreshes g[i] = mean(acc_t1 over n rollouts).
                        if self.curriculum_sampler is not None:
                            _idx = batch.non_tensor_batch.get('index', None)
                            _acc = batch.non_tensor_batch.get('acc_t1', None)
                            if _idx is not None and _acc is not None:
                                _idx = np.asarray(_idx, dtype=np.int64).ravel()
                                _acc = np.asarray(_acc, dtype=np.float32).ravel()
                                # aggregate n rollouts → one g_i per unique prompt
                                _uniq, _inv = np.unique(_idx, return_inverse=True)
                                _g_per_prompt = np.array(
                                    [_acc[_inv == k].mean() for k in range(len(_uniq))],
                                    dtype=np.float32,
                                )
                                # mode check lives inside update() — no-op when mode=="curriculum"
                                self.curriculum_sampler.update(_uniq, _g_per_prompt, self.global_steps)
                            if self.global_steps % 10 == 0:
                                _stats = self.curriculum_sampler.stats()
                                metrics['curriculum/mode']         = 0 if _stats['mode'] == 'uniform' else 1
                                metrics['curriculum/n_seen']       = _stats['n_seen']
                                metrics['curriculum/g_mean_seen']  = _stats['g_mean_seen'] if not np.isnan(_stats['g_mean_seen']) else 0.0
                                metrics['curriculum/frontier_frac']= _stats['frontier_frac']
                                # per-bin histogram: N prompts in each g bucket (K=4 → {0,.25,.5,.75,1})
                                metrics['curriculum/n_g0']   = _stats['n_g0']
                                metrics['curriculum/n_g025'] = _stats['n_g025']
                                metrics['curriculum/n_g050'] = _stats['n_g050']
                                metrics['curriculum/n_g075'] = _stats['n_g075']
                                metrics['curriculum/n_g100'] = _stats['n_g100']

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        assert self.config.algorithm.adv_estimator == 'gae', "Currently, only PPO is supported for multi-turn"
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  norm_type=self.config.algorithm.norm_type)

                        # ── SEC: cache r_t(c) AFTER advantages are ready ──────────
                        # advantages = post-GAE-norm tensor actually used by the actor.
                        # r_t(c) is computed here (BEFORE optimizer step) and applied
                        # to update Q AFTER the optimizer step, implementing the
                        # correct semantic: P_t→batch t→advantages→PAG update→Q_{t+1}.
                        _sec_r_per_level: dict = {}
                        if self.sec_sampler is not None:
                            _adv     = batch.batch['advantages']         # (B*K, seq_len)
                            _mt_mask = batch.batch['multiturn_mask']     # (B*K, seq_len)
                            # Build turn-1 mask (y0 tokens only)
                            _turn_starts = _mt_mask & (~torch.roll(_mt_mask, 1, dims=1))
                            _turn_starts[:, 0] = _mt_mask[:, 0]
                            _turn_idx = torch.cumsum(_turn_starts.long(), dim=1)
                            _turn1_mask = (_turn_idx == 1) & _mt_mask   # (B*K, seq_len)

                            _levels_BK = batch.non_tensor_batch.get('math_level', None)
                            if _levels_BK is not None:
                                _levels_BK = np.asarray(_levels_BK, dtype=np.int32)
                                _sec_r_per_level = SECSampler.compute_r_per_level(
                                    advantages=_adv,
                                    turn1_mask=_turn1_mask,
                                    levels=_levels_BK,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                )

                            # Diagnostic: log |A| per level for all three turns
                            for _turn_num, _turn_name in [(1, 'generate'), (2, 'verify'), (3, 'rectify')]:
                                _tmask = (_turn_idx == _turn_num) & _mt_mask
                                _t_len = _tmask.float().sum(dim=1).clamp(min=1.)
                                _u_t   = (_adv.abs() * _tmask.float()).sum(dim=1) / _t_len  # (B*K,)
                                _u_p   = _u_t.view(-1, self.config.actor_rollout_ref.rollout.n).mean(1)  # (B,)
                                _lvls_B = _levels_BK[::self.config.actor_rollout_ref.rollout.n] if _levels_BK is not None else None
                                if _lvls_B is not None:
                                    for _lvl in range(1, 6):
                                        _m = (_lvls_B == _lvl)
                                        if _m.any():
                                            metrics[f'sec/A_{_turn_name}_C{_lvl}'] = float(
                                                _u_p[torch.tensor(_m)].mean().item())

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        if self.sec_sampler is not None:
                            self._sec_actor_updated_at = self.global_steps

                    # ── SEC: update Q_{t+1} AFTER the PAG optimizer step ────────
                    # Semantic: P_t → batch t → advantages → PAG update → Q_{t+1}
                    # _sec_r_per_level was cached BEFORE the optimizer step above.
                    if self.sec_sampler is not None and _sec_r_per_level:
                        self.sec_sampler.update(_sec_r_per_level, step=self.global_steps)
                        _sec_stats = self.sec_sampler.stats()
                        total_counts = max(1, int(_sec_stats['cumulative_counts'].sum()))
                        sampled_lvls = (batch.non_tensor_batch.get('math_level', np.array([])) if
                                        batch.non_tensor_batch is not None else np.array([]))
                        sampled_lvls = sampled_lvls[::self.config.actor_rollout_ref.rollout.n]  # original prompts
                        for _lvl in range(1, 6):
                            metrics[f'sec/Q_C{_lvl}'] = float(_sec_stats['Q'][_lvl - 1])
                            metrics[f'sec/P_C{_lvl}'] = float(_sec_stats['P'][_lvl - 1])
                            metrics[f'sec/reward_C{_lvl}'] = float(_sec_r_per_level.get(_lvl, float('nan')))
                            metrics[f'sec/batch_count_C{_lvl}'] = int(
                                (sampled_lvls == _lvl).sum() if len(sampled_lvls) > 0 else 0)
                            metrics[f'sec/cumulative_frac_C{_lvl}'] = float(
                                _sec_stats['cumulative_counts'][_lvl - 1] / total_counts)
                        if len(sampled_lvls) > 0:
                            metrics['sec/mean_sampled_category'] = float(sampled_lvls.mean())

                    # Synchronized C1–C5 refresh (measurement, not a uniform-training phase).
                    if self.sec_sampler is not None:
                        _interval = int(self.config.sec.get("refresh_interval", 50) or 50)
                        if _interval > 0 and (
                            is_last_step or
                            (self.global_steps - int(self.sec_sampler.last_refresh_step)) >= _interval
                        ):
                            with _timer('sec_refresh', timing_raw):
                                metrics.update(self._sec_refresh(step=self.global_steps))

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, max_singleturn_resp_length=self.config.data.max_response_length))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
