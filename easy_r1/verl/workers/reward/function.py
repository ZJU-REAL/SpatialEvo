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
"""Reward function helpers."""

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig

class RewardInput(TypedDict, total=False):
    """Input passed to a reward function."""
    response: str
    response_length: int
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str] # Used for GRPO-style grouping.
    round: Optional[int] # Round index for two-round training.

class RewardScore(TypedDict):
    """Structured reward output."""
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

# Sequential reward function signature.
SequentialRewardFunction = Callable[[RewardInput], RewardScore]

# Batched reward function signature.
BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]

class SequentialFunctionRewardManagerMixin:
    """Reward manager for per-sample functions."""
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute rewards by calling the function once per sample."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        reward_metrics = defaultdict(list)
        
        response_ids = data.batch["responses"]  # (batch_size, max_response_length)
        response_length = torch.sum(data.batch["response_mask"], dim=-1)  # (batch_size,)
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            
            # Strip padding before decoding.
            valid_response_ids = response_ids[i][:cur_response_length]
            
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length, # tokens
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            
            # Pass through optional metadata when available.
            if "multi_modal_data" in data.non_tensor_batch:
                reward_input["multi_modal_data"] = data.non_tensor_batch["multi_modal_data"][i]
            
            if "uid" in data.non_tensor_batch:
                reward_input["uid"] = data.non_tensor_batch["uid"][i]
            
            score = self.reward_fn(reward_input)
            
            # Store the scalar reward on the last valid response token.
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics

class BatchFunctionRewardManagerMixin:
    """Reward manager for batched functions."""
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute rewards by calling the function on the full batch."""
        reward_inputs = []
        
        response_ids = data.batch["responses"]  # (batch_size, max_response_length)
        response_length = torch.sum(data.batch["response_mask"], dim=-1)  # (batch_size,)
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            
            # Strip padding before decoding.
            valid_response_ids = response_ids[i][:cur_response_length]
            
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length, # tokens
            }
            
            # Copy remaining metadata into the reward input.
            for key in data.non_tensor_batch.keys():
                if key not in reward_input:
                    reward_input[key] = data.non_tensor_batch[key][i]
                    
            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        reward_metrics = defaultdict(list)
        
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())
            
            # Store the scalar reward on the last valid response token.
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics

class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Load and dispatch a user-defined reward function."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        """Load the configured reward function."""
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        # Import the reward module from a file path.
        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        
        try:
            # Register the module so relative imports work as expected.
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        
        # Default to batched execution when not specified.
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        
        # Bind configured keyword arguments once.
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        
        self.reward_type = reward_type # "batch" or "sequential"
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute Reward."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
