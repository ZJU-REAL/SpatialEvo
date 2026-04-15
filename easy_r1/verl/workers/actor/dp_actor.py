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
"""Dp Actor."""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig

try:
    # flash_attn padding-free
    # padding
    # - unpad_input: padding
    # - pad_input: padding
    # - index_first_axis:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    # flash_attn
    pass

__all__ = ["DataParallelPPOActor"]

class DataParallelPPOActor(BasePPOActor):
    """Data Parallel P P O Actor."""
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Init."""
        super().__init__(config)
        
        # rank rank
        # rank GPU/
        self.rank = int(os.getenv("RANK", "0"))
        
        # world_size
        # world_size GPU/
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        
        # Actor
        # Transformer GPT Qwen
        self.actor_module = actor_module
        
        # Actor
        # None
        self.actor_optimizer = actor_optimizer
        
        # log_probs_from_logits: logits
        # torch.compile
        if config.use_torch_compile:
            # torch.compile: PyTorch 2.0+
            # dynamic=True:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """Forward Micro Batch."""
        input_ids = micro_batch["input_ids"]  # (batch_size, seq_length)
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]  # (batch_size, seq_length)
        position_ids = micro_batch["position_ids"] # (batch_size, seq_length) (batch_size, 4, seq_length) for mRoPE
        responses = micro_batch["responses"]  # (batch_size, response_length)
        response_length = responses.size(-1)
        
        # Qwen2-VL mRoPE
        # mRoPE (batch_size, 4, seq_length)
        # (4, batch_size, seq_length)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            # batch_collate:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
            # position_ids position_ids
            # position_ids multi_modal_inputs position_ids
            # FlashAttention
            multi_modal_inputs.pop("position_ids", None)
        else:
            multi_modal_inputs = {}

        # ========== Padding-free ==========
        # Padding-free
        # padding
        # 1: [1, 2, 3, pad, pad] (3)
        # 2: [4, 5, pad, pad, pad] (2)
        # 
        # Padding-free padding token
        # : [1, 2, 3, 4, 5] (token = 5)
        # 
        # Padding-free
        # 1. padding token
        # 2. GPU
        # 3.
        if self.config.padding_free:
            # ========== 1: Padding ==========
            # unpad_input: padding
            # input_ids.unsqueeze(-1): (batch_size, seq_length, 1)
            # attention_mask: token 1 padding 0
            # 
            # - input_ids_rmpad: padding token IDs (total_nnz, 1)
            # total_nnz = token
            # batch_size=2, seq_length=5, token=[3, 2]
            #        → total_nnz = 5, input_ids_rmpad = [token1, token2, token3, token4, token5]
            # - indices:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # ========== 2: ==========
            # padding Rotary Position Embedding
            # position_ids 187-188 (4, batch_size, seq_length) mRoPE
            # dim() == 3 (4, batch_size, seq_length)
            if position_ids.dim() == 3:
                # mRoPE (4, batch_size, seq_length)
                # 1. rearrange: (4, bsz, seqlen) → (bsz * seqlen, 4)
                # 2. index_first_axis: indices
                # 3. transpose: (4, 1, bsz * seqlen)
                # position_ids 4 mRoPE 4
                num_dims = position_ids.size(0) # 4 Qwen2-VL/Qwen3-VL
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (num_dims, bsz, seqlen) -> (num_dims, 1, bsz * seqlen)
                # num_dims 4
                assert position_ids_rmpad.size(0) == num_dims, (
                    f"The first dimension of position_ids_rmpad should be {num_dims}, but got {position_ids_rmpad.size(0)}"
                )
            else:
                # (batch_size, seq_length)
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # ========== 3: ==========
            # " token"
            # torch.roll(..., shifts=-1): 1
            # [1, 2, 3, 4] → [2, 3, 4, 1]
            # input_ids_rmpad_rolled[i] input_ids_rmpad[i] token
            # P(token[i+1] | token[0:i])
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # ========== 4: Ulysses ==========
            # Ulysses
            # Ulysses GPU
            # 1000 ulysses_size=4
            #   GPU 0: token[0:250]
            #   GPU 1: token[250:500]
            #   GPU 2: token[500:750]
            #   GPU 3: token[750:1000]
            # 
            # 1. GPU
            # 2. GPU
            if self.config.ulysses_size > 1:
                # ulysses_pad_and_slice_inputs: padding
                # - input_ids_rmpad: GPU
                # - pad_size: padding
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                # rolled
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            # batch
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # ========== 5: ==========
            # flash_attn_varlen
            # attention_mask=None: padding-free attention_mask
            # flash_attn
            # use_cache=False: KV cache
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None, # padding-free
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            
            # logits token
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            
            # temperature
            logits_rmpad.div_(temperature)
            
            # ========== 6: ==========
            # log_probs_from_logits: logits
            # logits_rmpad: logits (total_nnz, vocab_size)
            # input_ids_rmpad_rolled: token IDs (total_nnz,)
            # token (total_nnz,)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # ========== 7: Ulysses ==========
            if self.config.ulysses_size > 1:
                # Ulysses GPU
                # gather_outputs_and_unpad: 
                # 1. gather GPU
                # 2. padding unpad
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # ========== 8: ==========
            # pad_input:
            # log_probs.unsqueeze(-1): (total_nnz, 1)
            # indices:
            # full_log_probs (batch_size, seq_length)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            
            # response
            # [:, -response_length - 1 : -1]: response_length+1 2
            # -response_length - 1 : -1
            # - [prompt tokens..., response tokens...]
            # - response
            # - response response_length+1 EOS
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            # ========== padding-free ==========
            # padding
            # flash_attn
            
            output = self.actor_module(
                input_ids=input_ids,  # (batch_size, seq_length)
                attention_mask=attention_mask,  # (batch_size, seq_length)
                position_ids=position_ids, # (batch_size, seq_length) (batch_size, 4, seq_length)
                **multi_modal_inputs,
                use_cache=False,
            )
            
            # logits token
            logits: torch.Tensor = output.logits  # (batch_size, seq_length, vocab_size)
            
            # temperature
            logits.div_(temperature)
            
            # response logits
            # [:, -response_length - 1 : -1, :]: response_length+1 2
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            
            # logits: (batch_size, response_length, vocab_size)
            # responses: (batch_size, response_length)
            # log_probs (batch_size, response_length)
            # log_probs[i][j] = log P(response[i][j] | prompt[i] + response[i][:j])
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        """Optimizer Step."""
        if isinstance(self.actor_module, FSDP):
            # FSDP Fully Sharded Data Parallel
            # clip_grad_norm_ FSDP
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            # nn.utils.clip_grad_norm_ PyTorch
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        # finite
        # NaN Inf
        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            # optimizer.step()
            self.actor_optimizer.step()

        # backward() step()
        self.actor_optimizer.zero_grad()
        
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute Log Prob."""
        # eval mode
        # eval()
        # 1. dropout dropout
        # 2. batch normalization
        # 3.
        self.actor_module.eval()

        # meta_info
        temperature = data.meta_info["temperature"]
        
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        # data
        data = data.select(select_keys, non_tensor_select_keys)
        
        # ========== vs ==========
        # dynamic batching
        # - 50 = 32
        # - 200 = 8
        # 
        # 1. GPU GPU
        # 2.
        # 
        if self.config.dynamic_batching:
            # token
            # max_token_len: token
            # micro_batch_size=16, max_seq_len=100 → max_token_len=1600
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            
            # prepare_dynamic_batch:
            # - micro_batches:
            # - batch_idx_list:
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            # split():
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        
        # rank 0
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            # batch non_tensor_batch
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            
            log_probs_lst.append(log_probs)

        # torch.concat(): dim=0
        # micro_batch1 (8, 50) + micro_batch2 (8, 50) → (16, 50)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            # restore_dynamic_batch: batch_idx_list
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        """Update Policy."""
        # train mode
        # train()
        # 1. dropout dropout
        # 2. batch normalization
        # 3.
        self.actor_module.train()

        # temperature data.meta_info
        temperature = data.meta_info["temperature"]
        
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        # PPO
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # ========== Mini-batches ==========
        # Mini-batch
        # Mini-batch PPO
        # 
        # Mini-batch
        # 1.
        # 2. epoch
        # 3.
        # 
        # PPO https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        
        # ========== PPO Epochs ==========
        # 1-4
        for _ in range(self.config.ppo_epochs):
            # rank 0
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                # ========== response token ==========
                # loss GPU loss
                # response_mask: 1 token 0 padding
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                
                # all_reduce: GPU
                # token
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                # ========== Micro-batches ==========
                # OOM
                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                # rank 0
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    # batch non_tensor_batch
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    response_mask = model_inputs["response_mask"]  # (batch_size, response_length)
                    old_log_probs = model_inputs["old_log_probs"]  # (batch_size, response_length)
                    advantages = model_inputs["advantages"]  # (batch_size, response_length)

                    # update_policy
                    # 
                    # 1. mini-batch
                    # 2. ""
                    # 3. ratio = exp(new_log_prob - old_log_prob)
                    # 
                    # - compute_log_prob old_log_probs
                    # - new_log_probs
                    # - ratio = exp(new_log_prob - old_log_prob)
                    # 
                    # old_log_probs[0][0] = -0.5 (1token)
                    # new_log_probs[0][0] = -0.4 (1token)
                    # ratio = exp(-0.4 - (-0.5)) = exp(0.1) ≈ 1.105
                    # → token 10.5%
                    # 
                    # (batch_size, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    # ========== PPO Loss ==========
                    # compute_policy_loss
                    # 1. ratio = exp(new_log_prob - old_log_prob)
                    # 2. PPO loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
                    # 3. loss clip_fraction ratio_mean
                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask, # response
                        clip_ratio_low=self.config.clip_ratio_low, # PPO clip 0.2
                        clip_ratio_high=self.config.clip_ratio_high, # PPO clip 0.3
                        clip_ratio_dual=self.config.clip_ratio_dual, # clip C 3.0
                        loss_type=self.config.loss_type, # loss default gspo cispo
                        loss_avg_mode=self.config.loss_avg_mode, # loss token seq
                    )
                    
                    # ========== KL Loss ==========
                    # KL Loss
                    # KL Loss KL
                    # KL
                    # 
                    # KL Loss
                    # use_kl_loss=True loss KL
                    # KL compute_advantage
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        
                        # KL
                        # compute_kl KL
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty, # KL kl abs mse
                        )
                        
                        # KL response_mask
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        
                        # loss = PPO loss + KL loss * KL
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        
                        # KL
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        # KL loss PPO loss
                        loss = pg_loss

                    # ========== Loss ==========
                    # GPU loss
                    # loss = loss * (GPUtoken * GPU) / token
                    # GPU loss
                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    
                    # backward()
                    # 1. gradient
                    # 2. .grad
                    loss.backward()

                    # PPO
                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
