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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)

                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'multiturn_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # Feasibility-gated expert BC (optional tensors from trainer)
        for k in (
            'expert_token_mask',
            'expert_token_mask_y',
            'expert_token_mask_v',
            'expert_token_mask_r',
            'feas_gate',
            'feas_weight',
            'feas_gate_v',
            'feas_gate_r',
            'feas_weight_v',
            'feas_weight_r',
        ):
            if k in data.batch.keys():
                select_keys.append(k)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()
        expert_bc_coef = float(self.config.get('expert_bc_coef', 0.0))
        # Light BC weight on successful expert spans even when feasible (no-API warmup)
        expert_bc_light = float(self.config.get('expert_bc_light', 0.0))

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    multiturn_mask = data['multiturn_mask']
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                    
                    # Always keep PG (multiturn). Feasibility only up-weights expert BC below.
                    ppo_mask = multiturn_mask
                    if ppo_mask.float().sum() < 1:
                        pg_loss = (log_prob * 0).sum()
                        pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
                        ppo_kl = torch.tensor(0.0, device=log_prob.device)
                        pg_clipfrac_lower = torch.tensor(0.0, device=log_prob.device)
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=ppo_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c)
                    # compute entropy loss from entropy
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=multiturn_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob,
                                         ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld,
                                           loss_mask=multiturn_mask,
                                           loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    # Expert BC (role-aware, type-specific spans on τ_B+):
                    #   first-shot: generator y + true-accept v (no rectifier)
                    #   I→C: true-reject v + successful rectifier
                    #   F(s^V)>0 → weight verify expert tokens
                    #   F(s^R)>0 → weight rectify expert tokens
                    #   F(s_A)>0 → weight generator expert tokens (row gate)
                    #   F≤0 → PPO only unless expert_bc_light>0
                    bc_loss_val = 0.0
                    bc_scale_val = 0.0
                    bc_weighted_val = 0.0
                    bc_n_tokens = 0.0
                    bc_n_samples = 0.0
                    bc_n_high = 0.0
                    bc_n_light = 0.0
                    if expert_bc_coef > 0 and 'expert_token_mask' in data:
                        expert_mask_raw = data['expert_token_mask'].float()
                        has_e = (
                            expert_mask_raw.reshape(expert_mask_raw.size(0), -1).sum(dim=-1, keepdim=True) > 0
                        ).float()
                        # Token weights: start from zeros, add role-specific high BC
                        expert_w = torch.zeros_like(expert_mask_raw)
                        role_aware = (
                            'expert_token_mask_v' in data
                            and 'expert_token_mask_r' in data
                            and 'feas_gate_v' in data
                            and 'feas_gate_r' in data
                        )
                        if role_aware:
                            ev = data['expert_token_mask_v'].float()
                            er = data['expert_token_mask_r'].float()
                            gv = data['feas_gate_v'].float().view(-1, 1)
                            gr = data['feas_gate_r'].float().view(-1, 1)
                            wv = data['feas_weight_v'].float().view(-1, 1) if 'feas_weight_v' in data else torch.ones_like(gv)
                            wr = data['feas_weight_r'].float().view(-1, 1) if 'feas_weight_r' in data else torch.ones_like(gr)
                            expert_w = expert_w + ev * (1.0 - gv) * torch.clamp(wv, min=1.0)
                            expert_w = expert_w + er * (1.0 - gr) * torch.clamp(wr, min=1.0)
                            if 'expert_token_mask_y' in data:
                                ey = data['expert_token_mask_y'].float()
                                if 'feas_gate' in data:
                                    gy = data['feas_gate'].float().view(-1, 1)
                                else:
                                    gy = torch.minimum(gv, gr)
                                wy = data['feas_weight'].float().view(-1, 1) if 'feas_weight' in data else torch.ones_like(gy)
                                expert_w = expert_w + ey * (1.0 - gy) * torch.clamp(wy, min=1.0)
                            high_rows = (expert_w.reshape(expert_w.size(0), -1).sum(dim=-1) > 0) & (
                                has_e.squeeze(-1) > 0
                            )
                            row_w_mean = (1.0 - torch.minimum(gv, gr)).squeeze(-1)
                        else:
                            row_w = torch.zeros(
                                expert_mask_raw.size(0), 1, device=expert_mask_raw.device, dtype=expert_mask_raw.dtype
                            )
                            if 'feas_gate' in data:
                                g = data['feas_gate'].float()
                                if g.dim() == 1:
                                    g = g.unsqueeze(-1)
                                row_w = (1.0 - g)
                            if 'feas_weight' in data:
                                w = data['feas_weight'].float()
                                if w.dim() == 1:
                                    w = w.unsqueeze(-1)
                                row_w = row_w * torch.clamp(w, min=1.0)
                            high_rows = (row_w.squeeze(-1) > 0) & (has_e.squeeze(-1) > 0)
                            expert_w = expert_mask_raw * row_w
                            row_w_mean = row_w.squeeze(-1)
                        if expert_bc_light > 0:
                            expert_w = torch.maximum(expert_w, expert_bc_light * expert_mask_raw * has_e)
                        light_only = (
                            (has_e.squeeze(-1) > 0)
                            & (~high_rows)
                            & (expert_w.reshape(expert_w.size(0), -1).sum(dim=-1) > 0)
                        )
                        # unweighted expert tokens (before row β); for coverage sanity
                        bc_n_tokens = float(expert_mask_raw.sum().item())
                        bc_n_samples = float(has_e.sum().item())
                        bc_n_high = float(high_rows.sum().item())
                        bc_n_light = float(light_only.sum().item())
                        if expert_w.sum() > 0:
                            bc_loss = agg_loss(
                                loss_mat=-log_prob,
                                loss_mask=expert_w,
                                loss_agg_mode=loss_agg_mode,
                            )
                            weighted_bc = expert_bc_coef * bc_loss
                            policy_loss = policy_loss + weighted_bc
                            bc_loss_val = float(bc_loss.detach().item())
                            bc_weighted_val = float(weighted_bc.detach().item())
                            bc_scale_val = float(row_w_mean.mean().item())

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                        'actor/expert_bc_loss': bc_loss_val,
                        'actor/expert_bc_weighted': bc_weighted_val,
                        'actor/expert_bc_row_w_mean': bc_scale_val,
                        'actor/expert_bc_n_tokens': bc_n_tokens,
                        'actor/expert_bc_n_samples': bc_n_samples,
                        'actor/expert_bc_n_high': bc_n_high,
                        'actor/expert_bc_n_light': bc_n_light,
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
