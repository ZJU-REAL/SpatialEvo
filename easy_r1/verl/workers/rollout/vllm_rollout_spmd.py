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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    """Repeat entries along the batch axis."""
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    """Block image tokens during text-only decoding."""
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None

def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    """Convert raw image/video paths into vLLM-ready inputs."""
    images, videos = [], []
    
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None

class vLLMRollout(BaseRollout):
    """vLLM-backed rollout worker."""
    
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """Initialize the rollout engine."""
        super().__init__()
        
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config

        self.tokenizer = tokenizer
        self.processor = processor
        
        self.pad_token_id = tokenizer.pad_token_id
        # Only rank 0 renders progress bars.
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        
        # Tensor parallelism cannot exceed world size.
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        # A batch must fit at least one prompt-response pair.
        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None: # VLM processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            # Cap images per prompt to reduce OOM risk.
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        # Configure the shared vLLM engine.
        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False, # Keep tokenizer support inside vLLM.
            trust_remote_code=config.trust_remote_code,
            load_format="dummy", # Required for FSDP-managed weights.
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)), # float16 or bfloat16
            seed=config.seed,
            # Default to prompt + response length when max_model_len is unset.
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher", # External launcher for FSDP.
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization, # Fraction in [0, 1].
            max_num_batched_tokens=config.max_num_batched_tokens, # Token budget per batch.
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager, # Skip CUDA graphs when needed.
            disable_custom_all_reduce=True, # Use PyTorch all-reduce.
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True, # Allow GPU memory to be released between phases.
            **engine_kwargs,
        )

        # Put vLLM into sleep mode until generation starts.
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False, # Keep token ids to avoid re-tokenization.
            "logit_bias": _get_logit_bias(processor),
        }
        
        # Copy any sampling fields exposed by `SamplingParams`.
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Temporarily override sampling parameters."""
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate responses with vLLM and pack them into a `DataProto`."""
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (batch_size, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]  # (batch_size, prompt_length)
        position_ids: torch.Tensor = prompts.batch["position_ids"] # (batch_size, prompt_length) or (batch_size, 4, prompt_length) for mRoPE
        eos_token_id: int = prompts.meta_info["eos_token_id"] # EOS token id
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        # vLLM consumes raw prompt token ids plus optional multimodal payloads.
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # Allow per-call sampling overrides from `prompts.meta_info`.
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, 
                sampling_params=self.sampling_params, 
                use_tqdm=self.use_tqdm
            )
            
            # Flatten all generated completions into a single response tensor.
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            
            response_ids = VF.pad_2d_list_to_length(
                
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            # Repeat prompt-side tensors when sampling multiple responses per prompt.
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        # Concatenate prompt and response tokens.
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)  # (batch_size, prompt_length + response_length)
        response_length = response_ids.size(1)

        # Extend position ids across the generated response span.
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)  # (batch_size, response_length)
        
        # Qwen2-VL mRoPE uses an extra position-id axis.
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # Build a mask that drops padding and tokens after EOS.
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # Package tensors back into the expected `DataProto` format.
        batch = TensorDict(
            {
                "prompts": input_ids,  # prompt token IDs
                "responses": response_ids,  # response token IDs
                "input_ids": sequence_ids, # prompt + response
                "attention_mask": attention_mask,
                "response_mask": response_mask, # response loss mask
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
