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

"""Merge sharded model checkpoints."""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)

def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    """Merge tensors according to a DTensor placement."""
    if placement.is_replicate():
        # Replicated tensors are identical on every rank.
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        # shard
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement type: {placement}")

def upload_model_to_huggingface(local_path: str, remote_path: str):
    """Upload a merged checkpoint directory to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    # Upload the entire merged checkpoint directory.
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FSDP shards into a HuggingFace-compatible checkpoint")
    parser.add_argument(
        "--local_dir", 
        required=True, 
        type=str, 
        help="Directory that stores model_world_size_*_rank_*.pt shard files"
    )
    parser.add_argument(
        "--hf_upload_path", 
        default=False, 
        type=str, 
        help="HuggingFace Hub repo path in username/repo-name format; uploads if provided"
    )
    args = parser.parse_args()
    local_dir: str = args.local_dir

    # Avoid pointing at the output directory itself.
    assert not local_dir.endswith("huggingface"), "local_dir must not end with 'huggingface'"

    # Infer world size from the rank-0 shard filename.
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        # Expected shard name: model_world_size_{world_size}_rank_0.pt
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = int(match.group(1))
            break

    assert world_size > 0, "No valid model_world_size_*_rank_0.pt shard was found"

    # Load the rank-0 shard first.
    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    
    # Use one parameter to recover mesh metadata.
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    
    if isinstance(weight, DTensor):
        # DTensor checkpoints store mesh metadata directly.
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names # ("fsdp",) or ("ddp", "fsdp")
    else:
        # Fall back to a plain FSDP layout.
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Detected device mesh shape: {mesh}, dimension names: {mesh_dim_names}")

    # Only pure FSDP and DDP+FSDP are supported here.
    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names: {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # Tensor-parallel shards span both mesh axes.
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # Pure FSDP uses the last mesh axis only.
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Need to process {total_shards} model shards in total.")
    
    # Preallocate the shard list.
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict) # rank 0
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        """Process One Shard."""
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    # Collect merged tensors and their placements.
    state_dict: dict[str, list[torch.Tensor]] = {}
    param_placements: dict[str, list[Placement]] = {}
    
    # All shards share the same parameter keys.
    keys = set(model_state_dict_lst[0].keys())
    
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Missing key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                # Save the local shard in bf16 for the merged checkpoint.
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                
                # Drop the replicated DDP axis when present.
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    # Every shard for a key must agree on placement.
                    assert param_placements[key] == placements, f"Inconsistent placement for parameter {key}."
            else:
                # Non-DTensor checkpoints already store local slices.
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"Key {key} does not need merging")
            continue

        if key in param_placements:
            # Merge shards using the recorded placement metadata.
            placements: tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # Single-axis FSDP expects exactly one placement entry.
                assert len(placements) == 1, f"A 1-D mesh should have 1 placement, but got {len(placements)}."
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D FSDP + TP is not implemented yet.
                raise NotImplementedError("FSDP + TP mode is not supported yet.")
        else:
            # Plain tensors are concatenated along the first axis.
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge complete.")
    
    hf_path = os.path.join(local_dir, "huggingface")
    # Recreate the Hugging Face model class from config.
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    # Pick the matching auto model class.
    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForImageTextToText
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown model architecture: {architectures}.")

    # Build the model on the meta device before loading weights.
    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)
    # Materialize empty parameters on CPU before saving.
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}...")
    # Save the merged checkpoint.
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict, model

    # Optionally upload to the Hub.
    if args.hf_upload_path:
        print(f"Uploading model to HuggingFace Hub: {args.hf_upload_path}...")
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
        print("Upload complete.")
