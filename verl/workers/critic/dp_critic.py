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
Implement a multiprocess PPOCritic

Supports:
  - num_labels=1: standard value head VR
  - num_labels=2: dual heads VR (reward return) + VF (failure prob; BCE-with-logits)
"""
import itertools
from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOCritic']


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        self.num_labels = int(self.config.model.get('num_labels', 1))
        self.vf_loss_coef = float(self.config.get('vf_loss_coef', 1.0))
        print(f'Critic use_remove_padding={self.use_remove_padding} num_labels={self.num_labels}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    @property
    def dual_head(self) -> bool:
        return self.num_labels >= 2

    def _forward_micro_batch(self, micro_batch):
        """Return values shaped (bs, response_len) or (bs, response_len, C)."""
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            **multi_modal_inputs,
                                            use_cache=False)
                values_rmpad = output.logits
                if values_rmpad.dim() == 3:
                    # (1, total_nnz, C) -> (total_nnz, C)
                    values_rmpad = values_rmpad.squeeze(0)
                elif values_rmpad.dim() == 2 and values_rmpad.size(0) == 1:
                    values_rmpad = values_rmpad.squeeze(0)

                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                # values: (bs, seq, C) or (bs, seq, 1)
                if values.dim() == 3 and values.size(-1) == 1 and not self.dual_head:
                    values = values.squeeze(-1)
                    values = values[:, -response_length - 1:-1]
                elif values.dim() == 3:
                    values = values[:, -response_length - 1:-1, :]
                else:
                    values = values[:, -response_length - 1:-1]
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)
                values = output.logits  # (bs, seq, C)
                values = values[:, -response_length - 1:-1]
                if values.size(-1) == 1 and not self.dual_head:
                    values = values.squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'multiturn_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)
        multiturn_mask = data.batch['multiturn_mask']

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        if values.dim() == 3:
            values = values * multiturn_mask.unsqueeze(-1)
        else:
            values = values * multiturn_mask

        return values

    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {}

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns', 'multiturn_mask']
        dual = self.dual_head and ('values_f' in data.batch.keys()) and ('returns_f' in data.batch.keys())
        if dual:
            select_keys = select_keys + ['values_f', 'returns_f']
            if 'feasibility_mask' in data.batch.keys():
                select_keys = select_keys + ['feasibility_mask']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())
                    multiturn_mask = data['multiturn_mask']
                    values = data['values']
                    returns = data['returns']
                    response_mask = multiturn_mask

                    vpreds = self._forward_micro_batch(data)
                    if vpreds.dim() == 3:
                        vpred_r = vpreds[..., 0]
                        vpred_f = vpreds[..., 1]
                    else:
                        vpred_r = vpreds
                        vpred_f = None

                    vf_loss_r, vf_clipfrac_r = core_algos.compute_value_loss(
                        vpreds=vpred_r,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                    )
                    vf_loss = vf_loss_r
                    log_data = {
                        'critic/vf_loss': vf_loss_r.detach().item(),
                        'critic/vf_clipfrac': vf_clipfrac_r.detach().item(),
                        'critic/vpred_mean': masked_mean(vpred_r, response_mask).detach().item(),
                    }

                    if dual and vpred_f is not None:
                        # VF: BCE-with-logits on G_F ∈ {0,1} at s^V / s^R only
                        if 'feasibility_mask' in data:
                            vf_f_mask = data['feasibility_mask']
                        else:
                            vf_f_mask = response_mask
                        if vf_f_mask.float().sum() < 1:
                            vf_loss_f = torch.zeros((), device=vpred_f.device)
                            vpred_f_mean = 0.0
                        else:
                            vf_loss_f, vpred_f_mean_t = core_algos.compute_feasibility_bce_loss(
                                logits=vpred_f,
                                targets=data['returns_f'],
                                response_mask=vf_f_mask,
                            )
                            vpred_f_mean = float(vpred_f_mean_t.detach().item())
                        vf_loss = vf_loss_r + self.vf_loss_coef * vf_loss_f
                        log_data.update({
                            'critic/vf_loss_r': vf_loss_r.detach().item(),
                            'critic/vf_loss_f': vf_loss_f.detach().item(),
                            'critic/vpred_f_mean': vpred_f_mean,
                            'critic/vf_f_n_routing': vf_f_mask.float().sum().detach().item(),
                        })

                    if self.config.use_dynamic_bsz:
                        loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()
                    append_to_dict(metrics, log_data)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'critic/grad_norm': grad_norm.detach().item()})
        self.critic_optimizer.zero_grad()
        return metrics
