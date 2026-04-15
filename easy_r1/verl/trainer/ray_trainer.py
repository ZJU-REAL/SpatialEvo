
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

import json
import itertools
import hashlib
import math
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Optional, Type
from openai import OpenAI
import numpy as np
import ray
import torch
from jinja2 import Template
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
import re

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.dataset import process_image
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulator.task_support import NON_ANCHOR_LABELS
from src.simulator.world_simulator import WorldSimulator

DATASET_SPLIT_DIR_NAMES = {"train", "val", "validation", "test"}
PROMPT_NON_ANCHOR_LABELS = {str(label).strip().lower() for label in NON_ANCHOR_LABELS}
TASK_ANSWER_HINTS_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "task_answer_format_hints.json"
)
ROUND2_ANSWER_PROMPT_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "answer_question_round2.jinja"
)
ROUND2_INVALID_EXPLANATION_PROMPT_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "answer_invalid_question_round2.jinja"
)
ROUND1_SCENE_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_scene_task_conditioned.jinja"
)
ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_single_image_task_conditioned.jinja"
)
ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_image_pair_task_conditioned.jinja"
)
ROUND1_TASK_PROMPT_SPECS_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_task_conditioned_specs.json"
)

ROUND1_TASKS_BY_MODALITY = {
    "scene": [
        "object_counting",
        "absolute_distance",
        "object_size",
        "room_size",
        "relative_distance",
        "relative_direction_hard",
    ],
    "single_image": [
        "single_image_relative_direction",
        "distance_cam_obj",
        "depth_order_obj_obj",
    ],
    "image_pair": [
        "position_cam_cam",
        "elevation_cam_cam",
        "visibility_compare",
        "position_cam_obj",
        "position_cam_reg",
        "motion_camera",
        "attribute_measurement",
    ],
}
ROUND1_TASK_SAMPLING_PRIOR_ACC = 0.35
ROUND1_TASK_SAMPLING_PRIOR_COUNT = 2.0
ROUND1_TASK_SAMPLING_MIN_WEIGHT = 0.05
ROUND1_TASK_SCHEDULER_ACC_TARGETS = {
    "absolute_distance": 0.50,
    "distance_cam_obj": 0.50,
    "room_size": 0.50,
}

def _load_required_text_file(path: str, label: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to load required {label} from `{path}`: {exc}") from exc

    if not text:
        raise RuntimeError(f"Required {label} file is empty: `{path}`")
    return text

def _load_required_json_file(path: str, label: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load required {label} from `{path}`: {exc}") from exc

TASK_ANSWER_HINTS = _load_required_json_file(
    TASK_ANSWER_HINTS_PATH,
    "task answer hints",
).get("tasks") or {}

TASK_ANSWER_HINT_TASKS = set(TASK_ANSWER_HINTS.keys())
TASK_ANSWER_FORMAT_SUFFIXES = {
    task_name: str(task_config.get("format_prompt_suffix", "") or "")
    for task_name, task_config in TASK_ANSWER_HINTS.items()
}
ROUND2_DIRECTION_EXAMPLE_COUNT = 4
ROUND2_DYNAMIC_DIRECTION_HINT_TASKS = {
    "relative_direction_hard",
    "single_image_relative_direction",
    "position_cam_cam",
    "position_cam_obj",
    "position_cam_reg",
    "motion_camera",
}
ROUND1_GENERIC_DIRECTION_SUFFIXES = {
    "relative_direction_hard": " A concise answer is usually a short direction phrase.",
    "single_image_relative_direction": " A concise answer is usually a short relative direction phrase.",
    "position_cam_cam": " A concise answer is usually a short direction phrase.",
    "position_cam_obj": " A concise answer is usually a short direction phrase.",
    "position_cam_reg": " A concise answer is usually a short direction phrase.",
    "motion_camera": " A concise answer is usually a short camera-motion label.",
}
ROUND2_DIRECTION_LABEL_POOLS = {
    "relative_direction_hard": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "position_cam_cam": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "position_cam_obj": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "position_cam_reg": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "motion_camera": [
        "left", "right", "front", "back",
        "up", "down",
        "left-front", "right-front", "left-back", "right-back",
        "left-up", "right-up", "front-up", "back-up",
        "left-down", "right-down", "front-down", "back-down",
    ],
    "single_image_relative_direction": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
        "up", "down",
        "up-front", "up-back", "up-left", "up-right",
        "down-front", "down-back", "down-left", "down-right",
    ],
}
DYNAMIC_HINT_STRIP_PATTERNS = (
    re.compile(r"\s*A concise answer is usually one of:\s*[^.]+\.?\s*$", re.IGNORECASE),
    re.compile(r"\s*Answer with exactly one of:\s*[^.]+\.?\s*$", re.IGNORECASE),
)
ROUND_DYNAMIC_FIXED_CHOICE_POOLS = {
    "visibility_compare": ["image1", "image2", "same", "neither"],
    "elevation_cam_cam": ["higher", "lower"],
}

def _normalize_suffix_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())

def _strip_known_answer_suffix(question: str) -> str:
    base = str(question or "").rstrip()
    candidate_suffixes = set(TASK_ANSWER_FORMAT_SUFFIXES.values()) | set(ROUND1_GENERIC_DIRECTION_SUFFIXES.values())
    normalized_base = _normalize_suffix_spacing(base)
    for suffix in sorted(candidate_suffixes, key=len, reverse=True):
        normalized_suffix = _normalize_suffix_spacing(suffix)
        if not normalized_suffix:
            continue
        if normalized_base.endswith(normalized_suffix):
            suffix_index = base.rfind(suffix.strip())
            if suffix_index >= 0:
                base = base[:suffix_index].rstrip()
                break
    for pattern in DYNAMIC_HINT_STRIP_PATTERNS:
        base = pattern.sub("", base).rstrip()
    return base

def _normalize_direction_label_for_hint(label: Any) -> str:
    text = str(label or "").strip().lower()
    text = text.replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    return text

def _build_round2_direction_hint(question: str, task_type: str, gt: str) -> str:
    if task_type not in ROUND2_DYNAMIC_DIRECTION_HINT_TASKS:
        return ""

    normalized_gt = _normalize_direction_label_for_hint(gt)
    if not normalized_gt or normalized_gt == "invalid":
        return ""

    label_pool = list(ROUND2_DIRECTION_LABEL_POOLS.get(task_type, []))
    if normalized_gt not in label_pool:
        label_pool.append(normalized_gt)

    distractors = [label for label in label_pool if label != normalized_gt]
    seed_source = f"{task_type}||{question}||{normalized_gt}"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(distractors)

    examples = [normalized_gt] + distractors[: max(0, ROUND2_DIRECTION_EXAMPLE_COUNT - 1)]
    rng.shuffle(examples)

    if task_type == "motion_camera":
        prefix = " A concise answer is usually a short camera-motion label, for example: "
    elif task_type == "single_image_relative_direction":
        prefix = " A concise answer is usually a short relative direction phrase, for example: "
    else:
        prefix = " A concise answer is usually a short direction phrase, for example: "
    return prefix + ", ".join(examples[:ROUND2_DIRECTION_EXAMPLE_COUNT]) + "."

def _seeded_shuffle_labels(task_type: str, question: str, labels: list[str]) -> list[str]:
    seed_source = f"{task_type}||{question}||{'|'.join(labels)}"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return shuffled

def _load_optional_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

def _normalize_signature_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"null", "none"}:
        return ""
    text = re.sub(r"\bimage\s+one\b", "image1", text)
    text = re.sub(r"\bimage\s+two\b", "image2", text)
    text = re.sub(r"\bimg\s*1\b", "image1", text)
    text = re.sub(r"\bimg\s*2\b", "image2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _normalize_image_ref_token(value: Any) -> str:
    text = _normalize_signature_text(value)
    if text in {"image1", "1", "one"}:
        return "image1"
    if text in {"image2", "2", "two"}:
        return "image2"
    return text

def _extract_ordered_image_refs(question: Any) -> list[str]:
    text = str(question or "")
    matches = re.finditer(r"\bimage(?:\s+|_)?(1|2|one|two)\b", text, flags=re.IGNORECASE)
    refs: list[str] = []
    for match in matches:
        ref = _normalize_image_ref_token(match.group(1))
        if ref:
            refs.append(ref)
    return refs

def _extract_depth_order_answer_candidates_from_sim(
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> tuple[str, str] | None:
    key_pairs = (
        ("object1_label", "object2_label"),
        ("object_a", "object_b"),
    )

    def _extract_from_mapping(mapping: Any) -> tuple[str, str] | None:
        if not isinstance(mapping, dict):
            return None

        for left_key, right_key in key_pairs:
            left = str(mapping.get(left_key, "") or "").strip()
            right = str(mapping.get(right_key, "") or "").strip()
            if left and right and left.lower() != "null" and right.lower() != "null":
                return left, right

        for nested_key in ("parsed_params", "context", "result", "validation_result"):
            nested = mapping.get(nested_key)
            found = _extract_from_mapping(nested)
            if found is not None:
                return found
        return None

    for payload in (sim_parsed_params, sim_validation_result, sim_result):
        found = _extract_from_mapping(_load_optional_json_payload(payload))
        if found is not None:
            return found
    return None

def _build_dynamic_choice_hint(
    question: str,
    task_type: str,
    *,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> str:
    if task_type == "depth_order_obj_obj":
        candidates = _extract_depth_order_answer_candidates_from_sim(
            sim_parsed_params=sim_parsed_params,
            sim_validation_result=sim_validation_result,
            sim_result=sim_result,
        )
        if not candidates:
            return ""
        labels = _seeded_shuffle_labels(task_type, question, [candidates[0], candidates[1], "same"])
        return " Answer with exactly one of: " + ", ".join(labels) + "."

    fixed_choices = ROUND_DYNAMIC_FIXED_CHOICE_POOLS.get(task_type)
    if not fixed_choices:
        return ""
    labels = _seeded_shuffle_labels(task_type, question, fixed_choices)
    return " A concise answer is usually one of: " + ", ".join(labels) + "."

def _build_question_with_answer_hint(
    question: str,
    task_type: str,
    gt: str | None = None,
    *,
    round_stage: int,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> str:
    base_question = _strip_known_answer_suffix(question)
    choice_suffix = _build_dynamic_choice_hint(
        base_question,
        task_type,
        sim_parsed_params=sim_parsed_params,
        sim_validation_result=sim_validation_result,
        sim_result=sim_result,
    )
    if round_stage == 1:
        suffix = choice_suffix or ROUND1_GENERIC_DIRECTION_SUFFIXES.get(
            task_type,
            TASK_ANSWER_FORMAT_SUFFIXES.get(task_type, ""),
        )
    else:
        suffix = _build_round2_direction_hint(base_question, task_type, str(gt or ""))
        if not suffix:
            suffix = choice_suffix or ROUND1_GENERIC_DIRECTION_SUFFIXES.get(
                task_type,
                TASK_ANSWER_FORMAT_SUFFIXES.get(task_type, ""),
            )

    if suffix and not _normalize_suffix_spacing(base_question).endswith(_normalize_suffix_spacing(suffix)):
        return base_question + suffix
    return base_question

ROUND2_ANSWER_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND2_ANSWER_PROMPT_PATH,
    "Round2 answer prompt",
)
ROUND2_INVALID_EXPLANATION_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND2_INVALID_EXPLANATION_PROMPT_PATH,
    "Round2 invalid explanation prompt",
)
ROUND1_SCENE_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_SCENE_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 scene task prompt",
)
ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 single-image task prompt",
)
ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 image-pair task prompt",
)
ROUND1_TASK_PROMPT_SPECS = _load_required_json_file(
    ROUND1_TASK_PROMPT_SPECS_PATH,
    "Round1 task prompt specs",
).get("tasks") or {}

class Role(IntEnum):

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto() # Actor + Rollout
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()  # Actor + Rollout + RefPolicy

@dataclass
class ResourcePoolManager:

    resource_pool_spec: dict[str, list[int]]  # Pool name -> GPU counts per node.
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,  # GPU count per node.
                use_gpu=True,  # Allocate GPU resources.
                max_colocate_count=1,  # One worker group per pool slot.
                name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Available GPU count {gpus_available} is smaller than required GPU count {gpus_required}.")

def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]  # Token-level reward scores.
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]  # Valid response tokens.

    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # Token-level KL over valid response tokens.

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics

def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],  # Token-level rewards.
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,  # GAE lambda.
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

class RayPPOTrainer:

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self._world_simulator_cache: dict[str, WorldSimulator] = {}
        self._world_simulator_thread_local = threading.local()
        self._round1_summary_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._round1_task_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "acc_sum": 0.0, "scheduler_acc_sum": 0.0}
        )
        self._round1_task_rng = np.random.default_rng(int(getattr(config.data, "seed", 1) or 1))
        for task_names in ROUND1_TASKS_BY_MODALITY.values():
            for task_name in task_names:
                _ = self._round1_task_stats[task_name]

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping  # Role -> worker class map.
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, so KL metrics will not be logged. Set `kl_coef=0` instead if you still want KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by the actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by the actor experience micro-batch size."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by the critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by the critic experience micro-batch size."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO require `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    @staticmethod
    def _extract_media_paths_from_item(item: Any) -> list[str]:
        if not isinstance(item, dict):
            return []

        for key in ("images", "videos"):
            raw_paths = item.get(key)
            if not isinstance(raw_paths, list):
                continue
            return [path for path in raw_paths if isinstance(path, str) and path.strip()]
        return []

    @staticmethod
    def _infer_scene_id_from_paths(paths: list[str]) -> str:
        # Infer the dataset scene id from media paths.
        for raw_path in paths:
            path = Path(raw_path)
            candidates = [path.stem] + [parent.name for parent in path.parents]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.lower().startswith("scene"):
                    return candidate
        return "unknown"

    @staticmethod
    def _infer_scene_root_and_path(scene_id: str, paths: list[str]) -> tuple[Optional[str], Optional[str]]:
        # Infer the normalized scene directory.
        if not isinstance(scene_id, str) or not scene_id.strip():
            return None, None

        for raw_path in paths:
            path = Path(raw_path)
            for current in [path.parent] + list(path.parents):
                if current.name == scene_id:
                    scene_root = str(current.parent)
                    scene_data_path = str(current)
                    return scene_root, scene_data_path
        return None, None

    @staticmethod
    def _infer_metadata_dir_from_paths(paths: list[str]) -> Optional[str]:
        # Infer the metadata directory.
        for raw_path in paths:
            path = Path(raw_path)
            for current in [path.parent] + list(path.parents):
                if current.name == "metadata":
                    return str(current)
                if current.name.lower() in DATASET_SPLIT_DIR_NAMES:
                    candidate = current.parent / "metadata"
                    if candidate.exists():
                        return str(candidate)
        return None

    def _build_simulator_media_context(self, item: Any) -> dict[str, Any]:
        # Build simulator context from media paths.
        image_paths = self._extract_media_paths_from_item(item)
        scene_id = self._infer_scene_id_from_paths(image_paths)
        scene_root, scene_data_path = self._infer_scene_root_and_path(scene_id, image_paths)
        metadata_dir = self._infer_metadata_dir_from_paths(image_paths)

        context: dict[str, Any] = {
            "scene_id": scene_id,
            "image_paths": image_paths,
            "_scene_root": scene_root,
        }
        if len(image_paths) == 1:
            context["image_path"] = image_paths[0]
        if len(image_paths) >= 2:
            context["image_path_1"] = image_paths[0]
            context["image_path_2"] = image_paths[1]
        if metadata_dir:
            context["metadata_dir"] = metadata_dir
        if scene_data_path:
            context["scene_data_path"] = scene_data_path
        if isinstance(item, dict):
            summary_context = item.get("_round1_summary_context")
            if isinstance(summary_context, dict):
                context["_round1_summary_context"] = deepcopy(summary_context)
        return context

    def _get_world_simulator_for_root(self, scene_root: Optional[str]) -> WorldSimulator:
        cache_key = scene_root or "__default__"
        simulator = self._world_simulator_cache.get(cache_key)
        if simulator is not None:
            return simulator

        simulator = self._create_world_simulator(scene_root)
        self._world_simulator_cache[cache_key] = simulator
        return simulator

    def _create_world_simulator(self, scene_root: Optional[str]) -> WorldSimulator:
        init_kwargs = {"enable_vlm": True}
        if isinstance(scene_root, str) and scene_root.strip():
            init_kwargs["scannet_root"] = scene_root
        return WorldSimulator(**init_kwargs)

    def _get_thread_local_world_simulator_for_root(self, scene_root: Optional[str]) -> WorldSimulator:
        cache = getattr(self._world_simulator_thread_local, "cache", None)
        if cache is None:
            cache = {}
            self._world_simulator_thread_local.cache = cache

        cache_key = scene_root or "__default__"
        simulator = cache.get(cache_key)
        if simulator is None:
            simulator = self._create_world_simulator(scene_root)
            cache[cache_key] = simulator
        return simulator

    def _get_world_simulator_max_workers(self, batch_size: int) -> int:
        configured = getattr(self.config.trainer, "world_simulator_max_workers", None)
        try:
            configured_workers = int(configured) if configured is not None else 0
        except (TypeError, ValueError):
            configured_workers = 0

        if configured_workers <= 0:
            configured_workers = 4
        return max(1, min(batch_size, configured_workers))

    @staticmethod
    def _build_simulator_query_task(
        media_context: dict[str, Any],
        canonical_task_type: str,
        question_text: str,
    ) -> dict[str, Any]:
        summary_context = media_context.get("_round1_summary_context")
        prompt_mode = "scene"
        if isinstance(summary_context, dict):
            prompt_mode = str(summary_context.get("prompt_mode", "scene") or "scene")

        query_task = {
            "task_type": canonical_task_type,
            "scene_id": media_context.get("scene_id", "unknown"),
            "question": question_text,
        }
        allowed_media_keys = ["metadata_dir", "scene_data_path"]
        if prompt_mode == "single_image":
            allowed_media_keys.extend(["image_path", "image_paths"])
        elif prompt_mode == "image_pair":
            allowed_media_keys.extend(["image_paths", "image_path_1", "image_path_2"])
        else:
            allowed_media_keys.append("image_paths")

        for key in allowed_media_keys:
            value = media_context.get(key)
            if isinstance(value, str) and value.strip():
                query_task[key] = value
            elif isinstance(value, list) and value:
                query_task[key] = value
        if isinstance(summary_context, dict):
            for key in (
                "prompt_mode",
                "scene_label_counts",
                "frame_visible_label_counts",
                "image1_visible_label_counts",
                "image2_visible_label_counts",
                "pair_visibility_contrast_labels",
                "pair_camera_elevation_answer",
            ):
                value = summary_context.get(key)
                if isinstance(value, dict) and value:
                    query_task[key] = deepcopy(value)
                elif isinstance(value, list) and value:
                    query_task[key] = list(value)
                elif isinstance(value, str) and value.strip():
                    query_task[key] = value.strip()
        return query_task

    def _run_simulator_validation_job(
        self,
        *,
        media_context: dict[str, Any],
        canonical_task_type: str,
        question_text: str,
        use_thread_local_simulator: bool,
    ) -> dict[str, Any]:
        if use_thread_local_simulator:
            simulator = self._get_thread_local_world_simulator_for_root(media_context.get("_scene_root"))
        else:
            simulator = self._get_world_simulator_for_root(media_context.get("_scene_root"))

        query_task = self._build_simulator_query_task(
            media_context=media_context,
            canonical_task_type=canonical_task_type,
            question_text=question_text,
        )
        res = simulator.validate_and_answer(query_task)
        resolved_task_type = res.get("task_type")
        if not (isinstance(resolved_task_type, str) and resolved_task_type.strip()):
            resolved_task_type = canonical_task_type

        answer = res.get("answer")
        if answer == "":
            answer = "invalid"
        is_valid = res.get("is_valid", False)
        failure_stage = res.get("failure_stage")
        if not isinstance(failure_stage, str) or not failure_stage.strip():
            failure_stage = "ok" if bool(is_valid) else "unknown"

        return {
            "resolved_task_type": resolved_task_type,
            "answer": answer,
            "is_valid": bool(is_valid),
            "error": str(res.get("error", "") or ""),
            "error_code": str(res.get("error_code", "") or ""),
            "failure_stage": failure_stage,
            "parsed_params": res.get("parsed_params", {}) or {},
            "validation_result": res.get("validation_result"),
            "judge_reference": res.get("judge_reference"),
            "sim_result": res,
        }

    @staticmethod
    def _resolve_canonical_task_type(simulator: WorldSimulator, raw_question_type: Any) -> str:
        resolution = simulator.resolve_task_type_name(raw_question_type)
        task_type = resolution.get("task_type")
        if isinstance(task_type, str) and task_type.strip():
            return task_type
        return "unknown"

    @staticmethod
    def _format_label_counts_text(label_counts: Optional[dict[str, Any]]) -> str:
        if not isinstance(label_counts, dict) or not label_counts:
            return ""

        formatted = []
        for label, count in sorted(label_counts.items()):
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            suffix = "(Unique)" if numeric_count == 1 else "(Non-Unique)"
            formatted.append(f"{label}{suffix}")
        return ", ".join(formatted)

    @staticmethod
    def _format_label_list_text(labels: Optional[list[str]]) -> str:
        if not isinstance(labels, list) or not labels:
            return ""
        return ", ".join(sorted(str(label) for label in labels if str(label).strip()))

    @staticmethod
    def _split_label_counts(label_counts: Optional[dict[str, Any]]) -> tuple[list[str], list[str]]:
        if not isinstance(label_counts, dict):
            return [], []

        unique_labels = []
        non_unique_labels = []
        for label, count in sorted(label_counts.items()):
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            if numeric_count == 1:
                unique_labels.append(str(label))
            elif numeric_count > 1:
                non_unique_labels.append(str(label))
        return unique_labels, non_unique_labels

    @staticmethod
    def _get_label_count(label_counts: Optional[dict[str, Any]], label: str) -> int:
        if not isinstance(label_counts, dict):
            return 0
        try:
            return int(label_counts.get(label, 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _is_usable_prompt_label(label: Any) -> bool:
        text = str(label or "").strip().lower()
        if not text:
            return False
        return text not in PROMPT_NON_ANCHOR_LABELS

    @classmethod
    def _filter_prompt_label_counts(
        cls,
        label_counts: Optional[dict[str, Any]],
    ) -> dict[str, int]:
        if not isinstance(label_counts, dict):
            return {}

        filtered: dict[str, int] = {}
        for label, count in label_counts.items():
            text = str(label or "").strip()
            if not cls._is_usable_prompt_label(text):
                continue
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            if numeric_count > 0:
                filtered[text] = numeric_count
        return dict(sorted(filtered.items()))

    @staticmethod
    def _visible_object_instance_key(
        obj: dict[str, Any],
        frame_index: int,
        object_index: int,
    ) -> str:
        object_id = obj.get("object_id", obj.get("id"))
        if object_id not in (None, ""):
            return f"id::{object_id}"

        location = obj.get("3d_location")
        if isinstance(location, (list, tuple)) and len(location) >= 3:
            try:
                xyz = tuple(round(float(location[idx]), 4) for idx in range(3))
                return f"loc::{xyz[0]}::{xyz[1]}::{xyz[2]}"
            except (TypeError, ValueError):
                pass

        label = str(obj.get("label", "")).strip().lower()
        return f"fallback::{frame_index}::{object_index}::{label}"

    @classmethod
    def _build_observed_scene_label_counts(
        cls,
        frame_summaries: Optional[list[dict[str, Any]]],
    ) -> dict[str, int]:
        if not isinstance(frame_summaries, list):
            return {}

        label_to_instances: dict[str, set[str]] = defaultdict(set)
        for frame_index, frame_summary in enumerate(frame_summaries, 1):
            visible_objects = frame_summary.get("visible_objects", []) if isinstance(frame_summary, dict) else []
            if not isinstance(visible_objects, list):
                continue
            for object_index, obj in enumerate(visible_objects):
                if not isinstance(obj, dict):
                    continue
                label = str(obj.get("label", "")).strip()
                if not cls._is_usable_prompt_label(label):
                    continue
                instance_key = cls._visible_object_instance_key(obj, frame_index, object_index)
                label_to_instances[label].add(instance_key)

        return {
            label: len(instance_keys)
            for label, instance_keys in sorted(label_to_instances.items())
            if instance_keys
        }

    def _get_pair_non_ambiguous_labels(
        self,
        image1_counts: Optional[dict[str, Any]],
        image2_counts: Optional[dict[str, Any]],
    ) -> list[str]:
        labels = set()
        if isinstance(image1_counts, dict):
            labels.update(str(label) for label in image1_counts.keys())
        if isinstance(image2_counts, dict):
            labels.update(str(label) for label in image2_counts.keys())

        valid_labels = []
        for label in sorted(labels):
            count1 = self._get_label_count(image1_counts, label)
            count2 = self._get_label_count(image2_counts, label)
            if count1 + count2 >= 1 and count1 <= 1 and count2 <= 1:
                valid_labels.append(label)
        return valid_labels

    def _get_shared_unique_pair_labels(
        self,
        image1_counts: Optional[dict[str, Any]],
        image2_counts: Optional[dict[str, Any]],
    ) -> list[str]:
        shared_unique = []
        labels = set()
        if isinstance(image1_counts, dict):
            labels.update(str(label) for label in image1_counts.keys())
        if isinstance(image2_counts, dict):
            labels.update(str(label) for label in image2_counts.keys())

        for label in sorted(labels):
            if self._get_label_count(image1_counts, label) == 1 and self._get_label_count(image2_counts, label) == 1:
                shared_unique.append(label)
        return shared_unique

    @staticmethod
    def _build_label_visibility_map(visible_objects: Optional[list[dict[str, Any]]]) -> dict[str, float]:
        if not isinstance(visible_objects, list):
            return {}

        label_to_visibility: dict[str, float] = {}
        for obj in visible_objects:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label") or obj.get("name") or "").strip()
            if not RayPPOTrainer._is_usable_prompt_label(label):
                continue
            try:
                visibility = float(obj.get("visibility", 0.0))
            except (TypeError, ValueError):
                visibility = 0.0
            prev_visibility = label_to_visibility.get(label)
            if prev_visibility is None or visibility > prev_visibility:
                label_to_visibility[label] = visibility
        return dict(sorted(label_to_visibility.items()))

    def _get_pair_visibility_contrast_candidates(
        self,
        image1_counts: Optional[dict[str, Any]],
        image2_counts: Optional[dict[str, Any]],
        image1_visibility_map: Optional[dict[str, Any]],
        image2_visibility_map: Optional[dict[str, Any]],
        *,
        min_delta: float = 0.15,
    ) -> list[dict[str, Any]]:
        candidate_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        candidates: list[dict[str, Any]] = []

        for label in candidate_labels:
            try:
                image1_visibility = float((image1_visibility_map or {}).get(label, 0.0))
            except (TypeError, ValueError):
                image1_visibility = 0.0
            try:
                image2_visibility = float((image2_visibility_map or {}).get(label, 0.0))
            except (TypeError, ValueError):
                image2_visibility = 0.0

            delta = abs(image1_visibility - image2_visibility)
            if delta < min_delta:
                continue

            if image1_visibility > image2_visibility:
                preferred_answer = "image1"
            elif image2_visibility > image1_visibility:
                preferred_answer = "image2"
            else:
                preferred_answer = "same"

            candidates.append(
                {
                    "label": label,
                    "image1_visibility": image1_visibility,
                    "image2_visibility": image2_visibility,
                    "delta": delta,
                    "preferred_answer": preferred_answer,
                }
            )

        candidates.sort(
            key=lambda item: (
                -float(item.get("delta", 0.0)),
                -max(float(item.get("image1_visibility", 0.0)), float(item.get("image2_visibility", 0.0))),
                str(item.get("label", "")),
            )
        )
        return candidates

    @staticmethod
    def _format_visibility_contrast_text(candidates: Optional[list[dict[str, Any]]], max_items: int = 12) -> str:
        if not isinstance(candidates, list) or not candidates:
            return ""

        formatted: list[str] = []
        for item in candidates[:max_items]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            try:
                delta = float(item.get("delta", 0.0))
            except (TypeError, ValueError):
                delta = 0.0
            try:
                image1_visibility = float(item.get("image1_visibility", 0.0))
            except (TypeError, ValueError):
                image1_visibility = 0.0
            try:
                image2_visibility = float(item.get("image2_visibility", 0.0))
            except (TypeError, ValueError):
                image2_visibility = 0.0
            preferred_answer = str(item.get("preferred_answer", "")).strip()
            formatted.append(
                f"{label}(Δvis={delta:.2f}, I1={image1_visibility:.2f}, I2={image2_visibility:.2f}, likely={preferred_answer})"
            )
        return ", ".join(formatted)

    @staticmethod
    def _dedupe_string_list(values: Optional[list[str]]) -> list[str]:
        if not isinstance(values, list):
            return []
        deduped = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _shuffle_string_list(self, values: Optional[list[str]]) -> list[str]:
        items = self._dedupe_string_list(values)
        if len(items) <= 1:
            return items
        indices = self._round1_task_rng.permutation(len(items)).tolist()
        return [items[idx] for idx in indices]

    @staticmethod
    def _region_name_from_anchor(anchor_label: str) -> str:
        anchor = str(anchor_label or "").strip()
        return f"{anchor} area" if anchor else ""

    @staticmethod
    def _format_prompt_question(template_text: str, **kwargs: str) -> str:
        try:
            question = str(template_text).format(**kwargs).strip()
        except Exception:
            question = ""
        return question

    @staticmethod
    def _replace_prompt_example_tokens(template_text: str, replacements: dict[str, str]) -> str:
        text = str(template_text or "").strip()
        if not text:
            return ""
        rendered = text
        expanded_replacements: dict[str, str] = {}
        for key, value in replacements.items():
            cleaned_value = str(value or "").strip()
            if not cleaned_value:
                continue
            expanded_replacements[str(key)] = cleaned_value
            capitalized_key = str(key)[:1].upper() + str(key)[1:]
            expanded_replacements.setdefault(capitalized_key, cleaned_value)

        for key in sorted(expanded_replacements.keys(), key=len, reverse=True):
            value = expanded_replacements[key]
            if not value:
                continue
            rendered = re.sub(rf"\b{re.escape(key)}\b", value, rendered)
        return " ".join(rendered.split())

    @staticmethod
    def _build_round1_example_replacements(format_kwargs: dict[str, str]) -> dict[str, str]:
        replacements = {
            str(key): str(value).strip()
            for key, value in (format_kwargs or {}).items()
            if str(value).strip()
        }
        if "object_label" in replacements:
            replacements.setdefault("object1", replacements["object_label"])
        if "object_a" in replacements:
            replacements.setdefault("object1", replacements["object_a"])
        if "object_b" in replacements:
            replacements.setdefault("object2", replacements["object_b"])
        if "object_c" in replacements:
            replacements.setdefault("object3", replacements["object_c"])
        if "target_object" in replacements:
            replacements.setdefault("object1", replacements["target_object"])
        if "reference_object" in replacements:
            replacements.setdefault("object2", replacements["reference_object"])
        if "stand_object" in replacements:
            replacements.setdefault("object1", replacements["stand_object"])
            replacements.setdefault("stand_object", replacements["stand_object"])
        if "face_object" in replacements:
            replacements.setdefault("object2", replacements["face_object"])
            replacements.setdefault("face_object", replacements["face_object"])
        if "query_object" in replacements:
            replacements.setdefault("object3", replacements["query_object"])
            replacements.setdefault("query_object", replacements["query_object"])
        if "anchor_object" in replacements:
            replacements.setdefault("anchor_object", replacements["anchor_object"])
        if "region_name" in replacements:
            replacements.setdefault("region1", replacements["region_name"])
            anchor_object = re.sub(r"\s+area$", "", replacements["region_name"]).strip()
            if anchor_object:
                replacements.setdefault("object1", anchor_object)
        return replacements

    def _build_round1_valid_example_specs(
        self,
        task_type: str,
        summary_context: dict[str, Any],
        max_examples: int = 4,
    ) -> list[dict[str, Any]]:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        templates = [
            str(template).strip()
            for template in (prompt_spec.get("question_templates", []) or [])
            if str(template).strip()
        ]
        if not templates:
            return []

        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        image1_unique, _ = self._split_label_counts(image1_counts)
        image2_unique, _ = self._split_label_counts(image2_counts)
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        pair_shared_unique_labels = self._get_shared_unique_pair_labels(image1_counts, image2_counts)
        pair_visibility_contrast_labels = summary_context.get("pair_visibility_contrast_labels", []) or []

        specs: list[dict[str, Any]] = []
        seen_questions: set[str] = set()

        def add_spec(template_index: int = 0, **kwargs: str) -> None:
            if len(specs) >= max_examples:
                return
            template_text = templates[template_index % len(templates)]
            question = self._format_prompt_question(template_text, **kwargs)
            if question and question not in seen_questions:
                seen_questions.add(question)
                specs.append(
                    {
                        "template_index": template_index % len(templates),
                        "format_kwargs": dict(kwargs),
                        "question": question,
                    }
                )

        if task_type == "object_counting":
            labels = self._shuffle_string_list(scene_non_unique)
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "absolute_distance":
            labels = self._shuffle_string_list(scene_unique)
            for idx, (object_a, object_b) in enumerate(itertools.combinations(labels, 2)):
                add_spec(idx, object_a=object_a, object_b=object_b)
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "object_size":
            labels = self._shuffle_string_list(scene_unique)
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "room_size":
            for idx in range(min(len(templates), max_examples)):
                add_spec(idx)
            return specs

        if task_type == "relative_distance":
            labels = self._shuffle_string_list(scene_unique)
            if len(labels) < 4:
                return specs
            for idx, anchor_object in enumerate(labels):
                remaining = [label for label in labels if label != anchor_object]
                if len(remaining) < 3:
                    continue
                add_spec(
                    idx,
                    anchor_object=anchor_object,
                    object_a=remaining[0],
                    object_b=remaining[1],
                    object_c=remaining[2],
                )
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "relative_direction_hard":
            labels = self._shuffle_string_list(scene_unique)
            if len(labels) < 3:
                return specs
            for idx, triple in enumerate(itertools.permutations(labels, 3)):
                stand_object, face_object, query_object = triple
                add_spec(
                    idx,
                    stand_object=stand_object,
                    face_object=face_object,
                    query_object=query_object,
                )
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "single_image_relative_direction":
            labels = self._shuffle_string_list(frame_unique)
            for idx, (reference_object, target_object) in enumerate(itertools.permutations(labels, 2)):
                add_spec(idx, reference_object=reference_object, target_object=target_object)
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "distance_cam_obj":
            labels = self._shuffle_string_list(frame_unique)
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "depth_order_obj_obj":
            labels = self._shuffle_string_list(frame_unique)
            for idx, (object_a, object_b) in enumerate(itertools.combinations(labels, 2)):
                add_spec(idx, object_a=object_a, object_b=object_b)
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "elevation_cam_cam":
            pair_elevation_answer = self._normalize_answer_token(
                summary_context.get("pair_camera_elevation_answer", "")
            )
            if pair_elevation_answer == "same_level":
                return specs
            for idx in range(min(len(templates), max_examples)):
                add_spec(idx)
            return specs

        if task_type in {"position_cam_cam", "motion_camera"}:
            for idx in range(min(len(templates), max_examples)):
                add_spec(idx)
            return specs

        if task_type == "visibility_compare":
            labels = self._shuffle_string_list(
                pair_visibility_contrast_labels or pair_shared_unique_labels or pair_non_ambiguous_labels
            )
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "position_cam_obj":
            image1_labels = self._shuffle_string_list(image1_unique)
            image2_labels = self._shuffle_string_list(image2_unique)
            balanced_candidates: list[tuple[int, str]] = []
            max_branch = max(len(image1_labels), len(image2_labels))
            for idx in range(max_branch):
                if idx < len(image1_labels):
                    balanced_candidates.append((0, image1_labels[idx]))
                if len(templates) > 1 and idx < len(image2_labels):
                    balanced_candidates.append((1, image2_labels[idx]))
            for template_index, label in balanced_candidates[:max_examples]:
                add_spec(template_index, object_label=label)
            return specs

        if task_type == "position_cam_reg":
            image1_regions = [self._region_name_from_anchor(label) for label in self._shuffle_string_list(image1_unique)]
            image2_regions = [self._region_name_from_anchor(label) for label in self._shuffle_string_list(image2_unique)]
            balanced_candidates: list[tuple[int, str]] = []
            max_branch = max(len(image1_regions), len(image2_regions))
            for idx in range(max_branch):
                if idx < len(image1_regions):
                    balanced_candidates.append((0, image1_regions[idx]))
                if len(templates) > 1 and idx < len(image2_regions):
                    balanced_candidates.append((1, image2_regions[idx]))
            for template_index, region_name in balanced_candidates[:max_examples]:
                add_spec(template_index, region_name=region_name)
            return specs

        if task_type == "attribute_measurement":
            image1_labels = self._shuffle_string_list(image1_unique)
            image2_labels = self._shuffle_string_list(image2_unique)
            cross_pairs = []
            for object_a in image1_labels:
                for object_b in image2_labels:
                    if object_a != object_b:
                        cross_pairs.append((object_a, object_b))
            candidate_pairs = cross_pairs
            if not candidate_pairs:
                pair_labels = self._shuffle_string_list(pair_non_ambiguous_labels)
                candidate_pairs = list(itertools.combinations(pair_labels, 2))
            for idx, (object_a, object_b) in enumerate(candidate_pairs):
                add_spec(idx, object_a=object_a, object_b=object_b)
                if len(specs) >= max_examples:
                    break
            return specs

        return specs

    def _get_round1_valid_example_budget(self, task_type: str) -> int:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        return 6 if str(prompt_spec.get("task_group", "") or "") == "image_pair" else 4

    def _build_round1_valid_question_examples(
        self,
        task_type: str,
        summary_context: dict[str, Any],
        max_examples: int | None = None,
    ) -> list[str]:
        budget = max_examples if max_examples is not None else self._get_round1_valid_example_budget(task_type)
        specs = self._build_round1_valid_example_specs(
            task_type=task_type,
            summary_context=summary_context,
            max_examples=budget,
        )
        return [str(item.get("question", "")).strip() for item in specs if str(item.get("question", "")).strip()]

    @staticmethod
    def _infer_prompt_mode_from_media_context(media_context: dict[str, Any]) -> str:
        image_paths = media_context.get("image_paths", [])
        if len(image_paths) == 1:
            return "single_image"
        if len(image_paths) == 2:
            return "image_pair"
        return "scene"

    @staticmethod
    def _normalize_answer_token(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        return re.sub(r"_+", "_", text)

    def _get_image_pair_camera_relation_summary(
        self,
        *,
        simulator: WorldSimulator,
        media_context: dict[str, Any],
    ) -> dict[str, Any]:
        scene_data_path = media_context.get("scene_data_path")
        image_paths = media_context.get("image_paths", []) or []
        if not isinstance(scene_data_path, str) or not scene_data_path.strip() or len(image_paths) < 2:
            return {}

        camera_pair_tool = simulator.tools.get("camera_pair_tool")
        camera_elevation_tool = simulator.tools.get("camera_elevation_tool")
        if camera_pair_tool is None or camera_elevation_tool is None:
            return {}

        try:
            pair_result = camera_pair_tool.execute(
                scene_data_path=scene_data_path,
                image_paths=image_paths[:2],
                answer_mode="position",
            )
            if not pair_result.get("success", False):
                return {}

            elevation_result = camera_elevation_tool.execute(
                camera_entity_1=pair_result.get("camera_entity_1"),
                camera_entity_2=pair_result.get("camera_entity_2"),
                camera_reference_image_idx=1,
                camera_target_image_idx=2,
            )
            if not elevation_result.get("success", False):
                return {}

            return {
                "pair_camera_elevation_answer": str(elevation_result.get("answer", "") or ""),
                "pair_camera_height_delta_m": elevation_result.get("height_delta_m"),
            }
        except Exception:
            return {}

    def _get_round1_summary_context(self, media_context: dict[str, Any]) -> dict[str, Any]:
        prompt_mode = self._infer_prompt_mode_from_media_context(media_context)
        scene_id = media_context.get("scene_id", "unknown")
        metadata_dir = media_context.get("metadata_dir")
        image_paths = media_context.get("image_paths", [])
        scene_root = media_context.get("_scene_root")

        context = {
            "prompt_mode": prompt_mode,
            "scene_id": scene_id,
            "metadata_dir": metadata_dir,
            "scene_label_counts": {},
            "scene_labels_text": "",
            "scene_unique_labels_text": "",
            "scene_non_unique_labels_text": "",
            "frame_visible_label_counts": {},
            "frame_visible_labels_text": "",
            "frame_unique_labels_text": "",
            "frame_non_unique_labels_text": "",
            "image1_visible_label_counts": {},
            "image1_visible_labels_text": "",
            "image1_unique_labels_text": "",
            "image1_label_visibility_map": {},
            "image2_visible_label_counts": {},
            "image2_visible_labels_text": "",
            "image2_unique_labels_text": "",
            "image2_label_visibility_map": {},
            "pair_union_visible_label_counts": {},
            "pair_union_visible_labels_text": "",
            "shared_visible_labels": [],
            "shared_visible_labels_text": "",
            "pair_visibility_contrast_candidates": [],
            "pair_visibility_contrast_labels": [],
            "pair_visibility_contrast_text": "",
            "pair_camera_elevation_answer": "",
            "pair_camera_height_delta_m": None,
        }
        if not isinstance(scene_id, str) or not scene_id.strip() or scene_id == "unknown":
            return context

        cache_key = (prompt_mode, scene_id, metadata_dir or "", tuple(image_paths))
        cached = self._round1_summary_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        simulator = self._get_world_simulator_for_root(scene_root)
        try:
            if prompt_mode == "single_image":
                summary = simulator.get_environment_summary(
                    {
                        "summary_type": "single_image",
                        "scene_id": scene_id,
                        "image_path": image_paths[0] if image_paths else None,
                        "metadata_dir": metadata_dir,
                    }
                )
                label_counts = self._filter_prompt_label_counts(summary.get("visible_label_counts", {}) or {})
                unique_labels, non_unique_labels = self._split_label_counts(label_counts)
                context["frame_visible_label_counts"] = label_counts
                context["frame_visible_labels_text"] = self._format_label_counts_text(label_counts)
                context["frame_unique_labels_text"] = self._format_label_list_text(unique_labels)
                context["frame_non_unique_labels_text"] = self._format_label_list_text(non_unique_labels)
            elif prompt_mode == "image_pair":
                summary = simulator.get_environment_summary(
                    {
                        "summary_type": "image_pair",
                        "scene_id": scene_id,
                        "image_paths": image_paths[:2],
                        "metadata_dir": metadata_dir,
                        "include_objects": True,
                    }
                )
                frame_summaries = summary.get("frame_summaries", []) or []
                image1_counts = self._filter_prompt_label_counts(
                    frame_summaries[0].get("visible_label_counts", {}) if len(frame_summaries) >= 1 else {}
                )
                image2_counts = self._filter_prompt_label_counts(
                    frame_summaries[1].get("visible_label_counts", {}) if len(frame_summaries) >= 2 else {}
                )
                image1_visibility_map = self._build_label_visibility_map(
                    frame_summaries[0].get("visible_objects", []) if len(frame_summaries) >= 1 else []
                )
                image2_visibility_map = self._build_label_visibility_map(
                    frame_summaries[1].get("visible_objects", []) if len(frame_summaries) >= 2 else []
                )
                image1_unique, _ = self._split_label_counts(image1_counts)
                image2_unique, _ = self._split_label_counts(image2_counts)
                visibility_contrast_candidates = self._get_pair_visibility_contrast_candidates(
                    image1_counts=image1_counts,
                    image2_counts=image2_counts,
                    image1_visibility_map=image1_visibility_map,
                    image2_visibility_map=image2_visibility_map,
                )
                aggregate = summary.get("aggregate", {}) or {}
                shared_labels = [
                    str(label).strip()
                    for label in (aggregate.get("intersection_visible_labels", []) or [])
                    if self._is_usable_prompt_label(label)
                ]
                union_counts = self._filter_prompt_label_counts(
                    aggregate.get("sum_visible_label_counts", {}) or {}
                )
                context["image1_visible_label_counts"] = image1_counts
                context["image1_visible_labels_text"] = self._format_label_counts_text(image1_counts)
                context["image1_unique_labels_text"] = self._format_label_list_text(image1_unique)
                context["image1_label_visibility_map"] = image1_visibility_map
                context["image2_visible_label_counts"] = image2_counts
                context["image2_visible_labels_text"] = self._format_label_counts_text(image2_counts)
                context["image2_unique_labels_text"] = self._format_label_list_text(image2_unique)
                context["image2_label_visibility_map"] = image2_visibility_map
                context["pair_union_visible_label_counts"] = union_counts
                context["pair_union_visible_labels_text"] = self._format_label_counts_text(union_counts)
                context["shared_visible_labels"] = shared_labels
                context["shared_visible_labels_text"] = self._format_label_list_text(shared_labels)
                context["pair_visibility_contrast_candidates"] = visibility_contrast_candidates
                context["pair_visibility_contrast_labels"] = [
                    str(item.get("label", "")).strip()
                    for item in visibility_contrast_candidates
                    if str(item.get("label", "")).strip()
                ]
                context["pair_visibility_contrast_text"] = self._format_visibility_contrast_text(
                    visibility_contrast_candidates
                )
                context.update(
                    self._get_image_pair_camera_relation_summary(
                        simulator=simulator,
                        media_context=media_context,
                    )
                )
            else:
                summary = simulator.get_environment_summary(
                    {
                        "summary_type": "multi_image",
                        "scene_id": scene_id,
                        "image_paths": image_paths,
                        "metadata_dir": metadata_dir,
                        "include_objects": True,
                    }
                )
                label_counts = self._build_observed_scene_label_counts(
                    summary.get("frame_summaries", []) or []
                )
                if not label_counts:
                    fallback_summary = simulator.get_environment_summary(
                        {
                            "summary_type": "scene",
                            "scene_id": scene_id,
                            "metadata_dir": metadata_dir,
                        }
                    )
                    label_counts = self._filter_prompt_label_counts(
                        fallback_summary.get("label_counts", {}) or {}
                    )
                unique_labels, non_unique_labels = self._split_label_counts(label_counts)
                context["scene_label_counts"] = label_counts
                context["scene_labels_text"] = self._format_label_counts_text(label_counts)
                context["scene_unique_labels_text"] = self._format_label_list_text(unique_labels)
                context["scene_non_unique_labels_text"] = self._format_label_list_text(non_unique_labels)
        except Exception:
            return context

        self._round1_summary_cache[cache_key] = dict(context)
        return dict(context)

    def _infer_feasible_round1_tasks(self, prompt_mode: str, summary_context: dict[str, Any]) -> list[str]:
        default_tasks = list(ROUND1_TASKS_BY_MODALITY.get(prompt_mode, []))
        if prompt_mode == "scene":
            label_counts = summary_context.get("scene_label_counts", {}) or {}
            unique_labels, non_unique_labels = self._split_label_counts(label_counts)
            num_labels = len(label_counts)
            feasible = []
            if len(non_unique_labels) >= 1:
                feasible.append("object_counting")
            if len(unique_labels) >= 2:
                feasible.append("absolute_distance")
            if len(unique_labels) >= 1:
                feasible.append("object_size")
            feasible.append("room_size")
            if len(unique_labels) >= 4:
                feasible.append("relative_distance")
            if len(unique_labels) >= 3:
                feasible.append("relative_direction_hard")
            return feasible or default_tasks

        if prompt_mode == "single_image":
            label_counts = summary_context.get("frame_visible_label_counts", {}) or {}
            if not label_counts:
                return default_tasks
            unique_labels, _ = self._split_label_counts(label_counts)
            feasible = []
            if len(unique_labels) >= 2:
                feasible.append("single_image_relative_direction")
            if len(unique_labels) >= 1:
                feasible.append("distance_cam_obj")
            if len(unique_labels) >= 2:
                feasible.append("depth_order_obj_obj")
            return feasible

        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        image1_unique, _ = self._split_label_counts(image1_counts)
        image2_unique, _ = self._split_label_counts(image2_counts)
        pair_elevation_answer = self._normalize_answer_token(
            summary_context.get("pair_camera_elevation_answer", "")
        )

        feasible = ["position_cam_cam", "motion_camera"]
        if pair_elevation_answer and pair_elevation_answer != "same_level":
            feasible.append("elevation_cam_cam")
        if len(pair_non_ambiguous_labels) >= 1:
            feasible.append("visibility_compare")
        if len(image1_unique) >= 1 or len(image2_unique) >= 1:
            feasible.append("position_cam_obj")
        if len(image1_counts) >= 1 or len(image2_counts) >= 1:
            feasible.append("position_cam_reg")
        if len(pair_non_ambiguous_labels) >= 2:
            feasible.append("attribute_measurement")
        return feasible or default_tasks

    def _build_round1_task_context_sections(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        image1_unique, _ = self._split_label_counts(image1_counts)
        image2_unique, _ = self._split_label_counts(image2_counts)
        pair_union_counts = summary_context.get("pair_union_visible_label_counts", {}) or {}
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        pair_shared_unique_labels = self._get_shared_unique_pair_labels(image1_counts, image2_counts)
        pair_visibility_contrast_labels = summary_context.get("pair_visibility_contrast_labels", []) or []
        pair_visibility_contrast_text = str(summary_context.get("pair_visibility_contrast_text", "") or "").strip()

        def make_section(title: str, text: str) -> dict[str, str] | None:
            cleaned = str(text or "").strip()
            if not cleaned:
                return None
            return {"title": title, "text": cleaned}

        def format_list(labels: list[str]) -> str:
            return self._format_label_list_text(labels)

        def format_counts(label_counts: dict[str, Any]) -> str:
            return self._format_label_counts_text(label_counts)

        sections: list[dict[str, str]] = []

        if task_type == "object_counting":
            repeated_text = format_list(scene_non_unique)
            section = make_section("Observed Repeated Labels For Counting (count >= 2)", repeated_text)
            return [section] if section is not None else []

        if task_type in {"absolute_distance", "object_size", "relative_distance", "relative_direction_hard"}:
            section = make_section("Scene Unique Labels", format_list(scene_unique))
            return [section] if section is not None else []

        if task_type == "room_size":
            section = make_section(
                "Task-Specific Focus",
                "No object-name pool is provided for this task. Focus on room boundary, occupied edges, and open floor layout.",
            )
            return [section] if section is not None else []

        if task_type in {"single_image_relative_direction", "distance_cam_obj"}:
            primary_text = format_list(frame_unique)
            title = "Unique Visible Labels In This Image"
            if not primary_text:
                primary_text = format_counts(frame_counts)
                title = "Visible Labels In This Image"
            section = make_section(title, primary_text)
            return [section] if section is not None else []

        if task_type == "depth_order_obj_obj":
            primary_text = format_list(frame_unique)
            title = "Unique Visible Labels For Depth Comparison"
            section = make_section(title, primary_text)
            return [section] if section is not None else []

        if task_type == "elevation_cam_cam":
            elevation_answer = self._normalize_answer_token(
                summary_context.get("pair_camera_elevation_answer", "")
            )
            if elevation_answer in {"higher", "lower"}:
                focus_text = (
                    f"Current pair already shows a non-zero vertical relation: relative to Image 1, "
                    f"Image 2 is {elevation_answer}. Focus on camera height change and shared scene structure."
                )
            else:
                focus_text = (
                    "This task is only appropriate when the two images have a noticeable vertical difference. "
                    "same_level pairs are invalid."
                )
            section = make_section("Task-Specific Focus", focus_text)
            return [section] if section is not None else []

        if task_type in {"position_cam_cam", "motion_camera"}:
            section = make_section(
                "Task-Specific Focus",
                "No object-name pool is provided for this task. Focus on camera relation, framing change, and shared scene structure.",
            )
            return [section] if section is not None else []

        if task_type == "visibility_compare":
            contrast_section = make_section(
                "Strong Visibility-Contrast Labels Across The Pair",
                pair_visibility_contrast_text or format_list(pair_visibility_contrast_labels),
            )
            shared_section = make_section(
                "Shared Non-Ambiguous Labels Across Both Images",
                format_list(pair_shared_unique_labels),
            )
            pair_section = make_section(
                "Non-Ambiguous Visibility Labels Across The Pair",
                format_list(pair_non_ambiguous_labels),
            )
            sections = [section for section in (contrast_section, shared_section, pair_section) if section is not None]
            return sections

        if task_type == "position_cam_obj":
            image1_section = make_section("Unique Visible Labels In Image 1", format_list(image1_unique))
            image2_section = make_section("Unique Visible Labels In Image 2", format_list(image2_unique))
            sections = [section for section in (image1_section, image2_section) if section is not None]
            return sections

        if task_type == "position_cam_reg":
            image1_section = make_section(
                "Unique Object Anchors In Image 1",
                format_list(image1_unique),
            )
            image2_section = make_section(
                "Unique Object Anchors In Image 2",
                format_list(image2_unique),
            )
            sections = [section for section in (image1_section, image2_section) if section is not None]
            if sections:
                return sections
            fallback_section = make_section(
                "Visible Object Anchors Across The Pair",
                format_counts(pair_union_counts),
            )
            return [fallback_section] if fallback_section is not None else []

        if task_type == "attribute_measurement":
            image1_section = make_section(
                "Non-Ambiguous Labels In Image 1",
                format_list(image1_unique),
            )
            image2_section = make_section(
                "Non-Ambiguous Labels In Image 2",
                format_list(image2_unique),
            )
            pair_section = make_section(
                "Pair-Grounded Non-Ambiguous Object Labels",
                format_list(pair_non_ambiguous_labels),
            )
            sections = [section for section in (image1_section, image2_section, pair_section) if section is not None]
            return sections

        default_section = make_section("Grounded Context", format_counts(scene_counts) or format_counts(frame_counts) or format_counts(pair_union_counts))
        return [default_section] if default_section is not None else []

    def _build_round1_task_candidate_guidance(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> list[str]:
        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        pair_shared_unique_labels = self._get_shared_unique_pair_labels(image1_counts, image2_counts)
        pair_visibility_contrast_labels = summary_context.get("pair_visibility_contrast_labels", []) or []

        def format_list(labels: list[str]) -> str:
            return self._format_label_list_text(labels) or "None"

        default_guidance = [
            "Only use labels or regions grounded by the provided context.",
            "If a candidate looks weak, ambiguous, or awkward, replace it with a clearer grounded candidate before writing the question.",
        ]

        guidance_by_task = {
            "object_counting": [
                "Use only the counting label pool shown above for this task.",
                "Each chosen category must have observed count >= 2 in the provided scene views.",
                "Copy the category name exactly from that counting pool; do not paraphrase it, pluralize it differently, or wrap it in extra attributes.",
                "Prefer a category that is clearly distributed across the scene instead of a tiny or doubtful label.",
                "Do not ask about an absent category or any category whose answer would be 0 or 1.",
            ],
            "absolute_distance": [
                f"Allowed object pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Pick two different isolated landmarks from that Unique pool; do not use repeated labels or room regions.",
                "If one chosen label feels ambiguous, replace it with another Unique label before asking the distance question.",
                "Copy the object labels exactly from the provided pool; do not add modifiers such as left, right, on the table, partially visible, or attached descriptions.",
            ],
            "object_size": [
                f"Allowed object pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Prefer one prominent, well-bounded object rather than clutter or a weakly grounded label.",
                "Keep the target on one object only.",
            ],
            "room_size": [
                "Do not choose an object as the measurement target; the target must stay on the room or combined visible room area.",
                "Do not convert this task into an object-centric question just because objects are visible in the images.",
                "Avoid drifting into object size, object distance, or counting just because objects are listed in the context.",
            ],
            "relative_distance": [
                f"Use only the scene Unique-label pool shown above: {format_list(scene_unique)}.",
                "Use three distinct candidates that are spatially separable around the anchor, and avoid duplicated clutter when a clearer label exists.",
            ],
            "relative_direction_hard": [
                f"Allowed standing, facing, and query pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Use exactly three different labels from that Unique pool and keep the standing point, facing object, and query object all distinct.",
                "Write one single-sentence question in the canonical form: standing at/by X and facing Y, where is Z located relative to me/you.",
                "Do not use alternate forms such as look toward, look up toward, will I see, which object, or right side of the room.",
                "Copy each object label exactly from the provided Unique pool; do not add location words, attributes, or descriptive suffixes.",
                "Prefer landmarks with a stable orientation frame instead of nearly overlapping, repeated-category, or awkward triples.",
            ],
            "single_image_relative_direction": [
                "Use only the visible-label pool shown above for this task, preferring Unique labels when available.",
                "Choose two different objects with a clear directional relation and avoid heavily overlapping or ambiguous pairs.",
            ],
            "distance_cam_obj": [
                "Use only the visible-label pool shown above for this task, preferring Unique labels when available.",
                "Prefer one prominent, fully grounded object instead of a tiny, truncated, or heavily occluded target.",
            ],
            "depth_order_obj_obj": [
                f"Use only the frame Unique-label pool shown above for this task: {format_list(frame_unique)}.",
                "Choose two objects that occupy clearly different depth layers and avoid pairs that appear side-by-side at nearly the same depth.",
                "Copy the labels exactly as shown; do not rewrite them into phrases such as partially visible chair or chair on the left.",
            ],
            "position_cam_cam": [
                "Do not choose an object label as the question target; the target is the camera-to-camera direction between Image 1 and Image 2.",
                "Use shared scene structure only as evidence for the camera relation.",
                "Both question directions are valid: you may ask from Image 1 to Image 2, or from Image 2 to Image 1, as long as the wording stays explicit.",
                "Keep the final question on the two camera positions, not on object direction.",
            ],
            "elevation_cam_cam": [
                "Do not choose an object label as the question target; the target is the vertical relation between the two cameras.",
                "Use shared scene structure only as evidence for height change between the two viewpoints.",
                "Only ask this task when the pair has a noticeable vertical difference; same_level pairs are invalid.",
                "Both question directions are valid: you may compare Image 2 against Image 1, or compare Image 1 against Image 2, as long as the wording and answer space stay consistent.",
                "Keep the answer space aligned with higher or lower only.",
            ],
            "visibility_compare": [
                f"Use the strong visibility-contrast label pool shown above when it is available: {format_list(pair_visibility_contrast_labels)}.",
                "If the strong visibility-contrast pool is non-empty, choose only from that pool.",
                f"Otherwise, fall back to the pair non-ambiguous label pool shown above: {format_list(pair_non_ambiguous_labels)}.",
                "Prefer a target whose visibility clearly changes between Image 1 and Image 2 instead of a label that looks equally weak in both.",
                f"If possible, prefer the shared non-ambiguous labels visible in both images: {format_list(pair_shared_unique_labels)}.",
                "Copy the label exactly as shown; do not output placeholders such as object1 or generic words such as object.",
                "Do not always write the answer candidates in one fixed order inside the question; the answer hint can express the candidate order later.",
            ],
            "position_cam_obj": [
                "Choose the image first, then use only that image's Unique-label pool shown above.",
                "Both Image 1 and Image 2 question forms are valid; do not always default to Image 1 when Image 2 has a cleaner grounded target.",
                "Choose the image first, then choose one prominent label visible in that image; avoid tiny or weakly grounded targets.",
            ],
            "position_cam_reg": [
                "Choose the image first, then use only that image's anchor-object pool shown above.",
                "Both Image 1 and Image 2 question forms are valid; prefer the image whose anchor gives the clearest grounded region instead of always defaulting to Image 1.",
                "Convert one suitable anchor object into a short grounded region phrase such as bed area, desk area, sink area, or sofa area.",
                "Avoid vague or overly broad region names that are not visually localized in the selected image.",
            ],
            "motion_camera": [
                "Do not choose an object label as the question target; the target is the camera motion from Image 1 to Image 2.",
                "Use framing change and shared scene structure only as evidence for the camera motion.",
                "Keep the final question on camera movement, not on object visibility or object direction.",
            ],
            "attribute_measurement": [
                f"Use only the pair-grounded non-ambiguous label pool shown above: {format_list(pair_non_ambiguous_labels)}.",
                "Each object may come from Image 1 or Image 2; the two objects do not need to come from the same image.",
                "Choose two rigid, clearly identifiable objects with a meaningful size contrast; avoid tiny clutter, room regions, or nearly indistinguishable pairs.",
                "Copy the labels exactly as shown; do not invent object names that are outside the provided pool.",
            ],
        }

        return guidance_by_task.get(task_type, default_guidance)

    def _get_round1_task_sampling_weight(self, task_type: str) -> float:
        stats = self._round1_task_stats.get(task_type, {"count": 0.0, "acc_sum": 0.0, "scheduler_acc_sum": 0.0})
        count = float(stats.get("count", 0.0))
        acc_sum = float(stats.get("scheduler_acc_sum", stats.get("acc_sum", 0.0)))
        if count > 0:
            smoothed_acc = (
                acc_sum + ROUND1_TASK_SAMPLING_PRIOR_ACC * ROUND1_TASK_SAMPLING_PRIOR_COUNT
            ) / (count + ROUND1_TASK_SAMPLING_PRIOR_COUNT)
        else:
            smoothed_acc = ROUND1_TASK_SAMPLING_PRIOR_ACC
        return max(ROUND1_TASK_SAMPLING_MIN_WEIGHT, 1.0 - smoothed_acc)

    def _remap_round1_scheduler_acc(self, task_type: str, mean_acc: float) -> float:
        target = ROUND1_TASK_SCHEDULER_ACC_TARGETS.get(str(task_type or "").strip())
        clipped_acc = float(np.clip(mean_acc, 0.0, 1.0))
        if target is None or target <= 0:
            return clipped_acc
        return float(np.clip(clipped_acc / float(target), 0.0, 1.0))

    def _sample_round1_task_type(
        self,
        prompt_mode: str,
        summary_context: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, float]]:
        feasible_tasks = self._infer_feasible_round1_tasks(prompt_mode, summary_context)
        if not feasible_tasks:
            fallback_task = {
                "scene": "room_size",
                "image_pair": "motion_camera",
                "single_image": "distance_cam_obj",
            }.get(prompt_mode, "unknown")
            return fallback_task, [fallback_task], {fallback_task: 1.0}

        weights = np.array([self._get_round1_task_sampling_weight(task) for task in feasible_tasks], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones(len(feasible_tasks), dtype=np.float64)
        probs = weights / weights.sum()
        sampled_idx = int(self._round1_task_rng.choice(len(feasible_tasks), p=probs))
        weight_map = {task: float(prob) for task, prob in zip(feasible_tasks, probs.tolist())}
        return feasible_tasks[sampled_idx], feasible_tasks, weight_map

    def _build_round1_observation_style_example(
        self,
        task_type: str,
        prompt_mode: str,
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        text = str(prompt_spec.get("example_observation", "") or "").strip()
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        filtered_sentences = []
        skip_markers = (
            "naturally leads to asking",
            "this naturally leads to asking",
            "this naturally supports",
            "supports either phrasing",
        )
        for sentence in sentences:
            normalized = sentence.strip().lower()
            if not normalized:
                continue
            if any(marker in normalized for marker in skip_markers):
                continue
            filtered_sentences.append(sentence.strip())

        example = " ".join(filtered_sentences).strip() or text
        if prompt_mode == "scene":
            tail = (
                " End by narrowing from the full-scene caption to the specific scene-level target and why it is valid "
                "for this task, then stop before writing the concrete question (...)."
            )
        elif prompt_mode == "image_pair":
            tail = (
                " End by narrowing from the pair-level caption to the task-relevant relation or target and why it is "
                "valid for this task, then stop before writing the concrete question (...)."
            )
        else:
            tail = (
                " End by narrowing from the whole-image caption to the task-relevant target and why it is valid for "
                "this task, then stop before writing the concrete question (...)."
            )
        return (example + tail).strip()

    def _render_round1_task_prompt(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        prompt_mode = str(summary_context.get("prompt_mode", "scene") or "scene")
        if prompt_mode == "single_image":
            template_text = ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE
        elif prompt_mode == "image_pair":
            template_text = ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE
        else:
            template_text = ROUND1_SCENE_TASK_PROMPT_TEMPLATE
        prompt_template = Template(template_text)
        task_context_sections = self._build_round1_task_context_sections(task_type, summary_context)
        task_candidate_guidance = self._build_round1_task_candidate_guidance(task_type, summary_context)
        valid_question_examples = self._build_round1_valid_question_examples(task_type, summary_context)
        example_observation_style = self._build_round1_observation_style_example(task_type, prompt_mode)
        prompt_text = prompt_template.render(
            **summary_context,
            task_type=task_type,
            task_display_name=prompt_spec.get("display_name", task_type),
            task_goal=prompt_spec.get("task_goal", f"Generate one grounded question for `{task_type}`."),
            task_requirements=prompt_spec.get(
                "task_requirements",
                ["Keep the observation grounded.", "Generate exactly one question for the assigned task."],
            ),
            question_templates=prompt_spec.get("question_templates", []),
            example_observation_style=example_observation_style,
            task_context_sections=task_context_sections,
            task_candidate_guidance=task_candidate_guidance,
            valid_question_examples=valid_question_examples,
        )
        return prompt_text.strip()

    @staticmethod
    def _json_dumps_non_tensor(value: Any) -> str:
        if value is None:
            return ""
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    @staticmethod
    def _compact_invalid_reference_line(text: Any) -> str:
        line = str(text or "").strip()
        if not line:
            return ""

        line = re.sub(r"\bshown above\b", "for this task", line, flags=re.IGNORECASE)
        line = re.sub(r"\bprovided pool\b", "task-allowed pool", line, flags=re.IGNORECASE)

        if any(token in line.lower() for token in ["pool", "labels visible", "label pool", "object pool"]):
            if ":" in line:
                line = line.split(":", 1)[0].rstrip(" .") + "."
            elif "=" in line:
                line = line.split("=", 1)[0].rstrip(" .") + "."

        return line.strip()

    def _build_task_reference_text(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        guidance = self._build_round1_task_candidate_guidance(task_type, summary_context)
        requirements = prompt_spec.get("task_requirements", [])

        lines: list[str] = []
        merged_rules: list[str] = []
        seen_rules: set[str] = set()

        for source_items in (guidance, requirements):
            for item in source_items or []:
                cleaned = self._compact_invalid_reference_line(item)
                if not cleaned:
                    continue
                dedupe_key = re.sub(r"\s+", " ", cleaned).strip().lower()
                if dedupe_key in seen_rules:
                    continue
                seen_rules.add(dedupe_key)
                merged_rules.append(cleaned)

        if merged_rules:
            lines.append("# TASK VALIDITY RULES:")
            for item in merged_rules:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines).strip()

    def _render_round2_invalid_explanation_prompt(
        self,
        *,
        question: str,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> tuple[str, str]:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        prompt_template = Template(ROUND2_INVALID_EXPLANATION_PROMPT_TEMPLATE)
        task_reference_text = self._build_task_reference_text(task_type, summary_context)
        prompt_text = prompt_template.render(
            task_type=task_type,
            task_display_name=prompt_spec.get("display_name", task_type),
            question=question,
            task_reference_text=task_reference_text,
        )
        return prompt_text.strip(), task_reference_text

    @staticmethod
    def _build_round1_signature_payload(
        task_type: str,
        question: str,
        parsed_params_json: Any,
    ) -> dict[str, Any]:
        parsed_params = _load_optional_json_payload(parsed_params_json)
        if not isinstance(parsed_params, dict):
            parsed_params = {}

        def first_non_empty(*keys: str) -> str:
            for key in keys:
                value = _normalize_signature_text(parsed_params.get(key))
                if value:
                    return value
            return ""

        def sorted_labels(*keys: str) -> list[str]:
            labels: list[str] = []
            for key in keys:
                value = parsed_params.get(key)
                if isinstance(value, (list, tuple)):
                    labels.extend(_normalize_signature_text(item) for item in value)
                else:
                    normalized = _normalize_signature_text(value)
                    if normalized:
                        labels.append(normalized)
            return sorted(label for label in labels if label)

        refs = _extract_ordered_image_refs(question)
        payload: dict[str, Any] = {"task_type": str(task_type or "unknown").strip() or "unknown"}

        if task_type == "room_size":
            payload["scene_scope"] = "room"
        elif task_type == "object_counting":
            payload["target_category"] = first_non_empty("target_category", "object_label")
        elif task_type == "object_size":
            payload["object_label"] = first_non_empty("object_label", "target_label")
        elif task_type in {"absolute_distance", "attribute_measurement", "depth_order_obj_obj"}:
            payload["object_pair"] = sorted_labels("object1_label", "object2_label")
        elif task_type == "relative_distance":
            payload["anchor"] = first_non_empty("target_label", "anchor_object", "anchor_label")
            payload["candidate_labels"] = sorted_labels("candidate_labels")
        elif task_type == "relative_direction_hard":
            payload["positioning_label"] = first_non_empty("positioning_label")
            payload["orienting_label"] = first_non_empty("orienting_label")
            payload["querying_label"] = first_non_empty("querying_label")
        elif task_type == "single_image_relative_direction":
            payload["reference_label"] = first_non_empty("reference_label")
            payload["target_label"] = first_non_empty("target_label")
        elif task_type == "distance_cam_obj":
            payload["target_label"] = first_non_empty("target_label", "object_label")
        elif task_type == "position_cam_obj":
            payload["camera_image"] = refs[0] if refs else ""
            payload["target_label"] = first_non_empty("target_label", "object_label")
        elif task_type == "position_cam_reg":
            payload["camera_image"] = refs[0] if refs else ""
            payload["region_name"] = first_non_empty("region_name")
        elif task_type == "visibility_compare":
            payload["target_label"] = first_non_empty("target_label", "object_label")
        elif task_type in {"position_cam_cam", "elevation_cam_cam"}:
            payload["reference_image"] = refs[0] if len(refs) >= 1 else ""
            payload["target_image"] = refs[1] if len(refs) >= 2 else ""
        elif task_type == "motion_camera":
            payload["pair_scope"] = "camera_motion"
        else:
            cleaned_params = {
                _normalize_signature_text(key): _normalize_signature_text(value)
                for key, value in parsed_params.items()
                if _normalize_signature_text(key) and _normalize_signature_text(value)
            }
            if cleaned_params:
                payload["parsed_params"] = cleaned_params

        compact_payload = {k: v for k, v in payload.items() if v not in ("", [], {}, None)}
        if len(compact_payload) == 1:
            compact_payload["question_fallback"] = _normalize_signature_text(question)
        return compact_payload

    def _build_round1_semantic_signature(
        self,
        *,
        task_type: str,
        question: Any,
        parsed_params_json: Any,
    ) -> str:
        payload = self._build_round1_signature_payload(
            task_type=str(task_type or "").strip() or "unknown",
            question=str(question or ""),
            parsed_params_json=parsed_params_json,
        )
        signature_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(signature_text.encode("utf-8")).hexdigest()

    def _deduplicate_round1_batch_for_round2(
        self,
        round1_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        if len(round1_batch) == 0:
            return round1_batch

        source_uids = round1_batch.non_tensor_batch.get("uid")
        if source_uids is None:
            round1_batch.non_tensor_batch["duplicate_weight"] = np.ones(len(round1_batch), dtype=np.int32)
            return round1_batch

        question_types = round1_batch.non_tensor_batch.get("question_type")
        scheduled_task_types = round1_batch.non_tensor_batch.get("scheduled_task_type")
        questions = round1_batch.non_tensor_batch.get("question")
        parsed_params_list = round1_batch.non_tensor_batch.get("sim_parsed_params_json")

        grouped_indices: dict[str, list[int]] = defaultdict(list)
        ordered_group_ids: list[str] = []
        for idx, uid in enumerate(source_uids.tolist()):
            uid_key = str(uid)
            if uid_key not in grouped_indices:
                ordered_group_ids.append(uid_key)
            grouped_indices[uid_key].append(idx)

        kept_indices: list[int] = []
        kept_weights: list[int] = []
        kept_signatures: list[str] = []

        for uid_key in ordered_group_ids:
            seen_signatures: dict[str, int] = {}
            for idx in grouped_indices[uid_key]:
                task_type = ""
                if question_types is not None:
                    task_type = str(question_types[idx] or "").strip()
                if not task_type and scheduled_task_types is not None:
                    task_type = str(scheduled_task_types[idx] or "").strip()
                if not task_type:
                    task_type = "unknown"

                signature = self._build_round1_semantic_signature(
                    task_type=task_type,
                    question=questions[idx] if questions is not None else "",
                    parsed_params_json=parsed_params_list[idx] if parsed_params_list is not None else None,
                )
                if signature in seen_signatures:
                    kept_weights[seen_signatures[signature]] += 1
                    continue

                seen_signatures[signature] = len(kept_indices)
                kept_indices.append(idx)
                kept_weights.append(1)
                kept_signatures.append(signature)

        if len(kept_indices) == len(round1_batch):
            round1_batch.non_tensor_batch["duplicate_weight"] = np.ones(len(round1_batch), dtype=np.int32)
            round1_batch.non_tensor_batch["semantic_signature"] = np.array(kept_signatures, dtype=object)
            metrics["round2/dedup_removed_question_count"] = 0.0
            metrics["round2/dedup_removed_question_ratio"] = 0.0
            metrics["round2/unique_question_count_before_answer"] = float(len(round1_batch))
            metrics["round2/raw_question_count_before_answer"] = float(len(round1_batch))
            return round1_batch

        deduped_batch = round1_batch.index_select(np.array(kept_indices, dtype=np.int64))
        deduped_batch.non_tensor_batch["duplicate_weight"] = np.array(kept_weights, dtype=np.int32)
        deduped_batch.non_tensor_batch["semantic_signature"] = np.array(kept_signatures, dtype=object)

        removed_count = len(round1_batch) - len(deduped_batch)
        metrics["round2/dedup_removed_question_count"] = float(removed_count)
        metrics["round2/dedup_removed_question_ratio"] = float(removed_count / max(len(round1_batch), 1))
        metrics["round2/unique_question_count_before_answer"] = float(len(deduped_batch))
        metrics["round2/raw_question_count_before_answer"] = float(len(round1_batch))
        return deduped_batch

    def _get_round2_padding_requirements(self) -> tuple[int, int]:
        divisors: list[int] = []
        update_batch_sizes: list[int] = []

        world_size = int(getattr(self.actor_rollout_ref_wg, "world_size", 1) or 1)
        if world_size > 1:
            divisors.append(world_size)

        rollout_repeat = int(getattr(self.config.worker.rollout, "n", 1) or 1)

        actor_batch_size = int(getattr(self.config.worker.actor, "global_batch_size", 0) or 0)
        if actor_batch_size > 0:
            effective_actor_batch_size = actor_batch_size * max(1, rollout_repeat)
            divisors.append(effective_actor_batch_size)
            update_batch_sizes.append(effective_actor_batch_size)

        if self.use_critic:
            critic_batch_size = int(getattr(self.config.worker.critic, "global_batch_size", 0) or 0)
            if critic_batch_size > 0:
                effective_critic_batch_size = critic_batch_size * max(1, rollout_repeat)
                divisors.append(effective_critic_batch_size)
                update_batch_sizes.append(effective_critic_batch_size)

        if not divisors:
            return 1, max(1, self.config.worker.rollout.n)

        target_divisor = divisors[0]
        for value in divisors[1:]:
            target_divisor = math.lcm(target_divisor, value)

        if update_batch_sizes:
            padding_chunk = update_batch_sizes[0]
            for value in update_batch_sizes[1:]:
                padding_chunk = math.gcd(padding_chunk, value)
        else:
            padding_chunk = target_divisor

        return max(1, target_divisor), max(1, padding_chunk)

    def _pad_round2_batch_for_balance(
        self,
        round2_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        world_size = int(getattr(self.actor_rollout_ref_wg, "world_size", 1) or 1)
        batch_size = len(round2_batch)
        if world_size <= 1 or batch_size == 0 or batch_size % world_size == 0:
            metrics["round2/balance_pad_group_count"] = 0.0
            metrics["round2/balance_pad_sample_count"] = 0.0
            return round2_batch

        if "is_padding" not in round2_batch.non_tensor_batch:
            round2_batch.non_tensor_batch["is_padding"] = np.zeros(batch_size, dtype=bool)

        group_size = int(self.config.worker.rollout.n)
        if group_size <= 0 or batch_size % group_size != 0:
            return round2_batch

        pad_group_count = 0
        while ((batch_size + pad_group_count * group_size) % world_size) != 0:
            pad_group_count += 1
            if pad_group_count > world_size:
                return round2_batch

        if pad_group_count == 0:
            metrics["round2/balance_pad_group_count"] = 0.0
            metrics["round2/balance_pad_sample_count"] = 0.0
            return round2_batch

        uid_values = round2_batch.non_tensor_batch.get("uid")
        if uid_values is None:
            return round2_batch

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        ordered_uids: list[str] = []
        for idx, uid in enumerate(uid_values.tolist()):
            uid_key = str(uid)
            if uid_key not in uid_to_indices:
                ordered_uids.append(uid_key)
            uid_to_indices[uid_key].append(idx)

        source_groups = [uid_to_indices[uid_key] for uid_key in ordered_uids if len(uid_to_indices[uid_key]) == group_size]
        if not source_groups:
            return round2_batch

        pad_groups: list[DataProto] = []
        for pad_idx in range(pad_group_count):
            source_indices = source_groups[pad_idx % len(source_groups)]
            pad_group = round2_batch.index_select(np.array(source_indices, dtype=np.int64))
            pad_size = len(pad_group)
            pad_group.non_tensor_batch["uid"] = np.array(
                [f"round2-balance-pad-{pad_idx}"] * pad_size,
                dtype=object,
            )
            pad_group.non_tensor_batch["is_padding"] = np.ones(pad_size, dtype=bool)
            pad_group.non_tensor_batch["duplicate_weight"] = np.zeros(pad_size, dtype=np.int32)
            pad_groups.append(pad_group)

        pad_batch = DataProto.concat(pad_groups)
        metrics["round2/balance_pad_group_count"] = float(pad_group_count)
        metrics["round2/balance_pad_sample_count"] = float(len(pad_batch))
        return DataProto.concat([round2_batch, pad_batch])

    @staticmethod
    def _interleave_round2_padding(
        real_batch: DataProto,
        pad_batch: DataProto,
        chunk_size: int,
    ) -> DataProto:
        combined = DataProto.concat([real_batch, pad_batch])
        if len(pad_batch) == 0 or chunk_size <= 0:
            return combined

        total_size = len(combined)
        if total_size % chunk_size != 0:
            return combined

        num_real = len(real_batch)
        num_chunks = total_size // chunk_size
        if num_real < num_chunks:
            return combined

        real_indices = list(range(num_real))
        pad_indices = list(range(num_real, total_size))
        ordered_indices: list[int] = []

        base_real = num_real // num_chunks
        extra_real = num_real % num_chunks
        real_counts = [base_real + (1 if chunk_idx < extra_real else 0) for chunk_idx in range(num_chunks)]
        if any(count <= 0 or count > chunk_size for count in real_counts):
            return combined

        for real_count in real_counts:
            pad_count = chunk_size - real_count
            if real_count > len(real_indices) or pad_count > len(pad_indices):
                return combined
            current_chunk = [real_indices.pop(0) for _ in range(real_count)]
            current_chunk.extend(pad_indices.pop(0) for _ in range(pad_count))
            ordered_indices.extend(current_chunk)

        if real_indices or pad_indices:
            return combined
        return combined.index_select(np.array(ordered_indices, dtype=np.int64))

    def _pad_round2_batch_for_updates(
        self,
        round2_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        batch_size = len(round2_batch)
        if batch_size == 0:
            return round2_batch

        if "is_padding" not in round2_batch.non_tensor_batch:
            round2_batch.non_tensor_batch["is_padding"] = np.zeros(batch_size, dtype=bool)

        existing_padding_mask = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool)
        existing_pad_batch = None
        if existing_padding_mask.any():
            existing_pad_indices = np.flatnonzero(existing_padding_mask).astype(np.int64)
            real_indices = np.flatnonzero(~existing_padding_mask).astype(np.int64)
            padding_tensor_mask = torch.from_numpy(existing_padding_mask)
            for field_name in ("token_level_scores", "token_level_rewards", "old_log_probs", "ref_log_probs", "values"):
                if field_name in round2_batch.batch:
                    field = round2_batch.batch[field_name]
                    round2_batch.batch[field_name][padding_tensor_mask.to(field.device)] = 0
            if "response_mask" in round2_batch.batch:
                response_mask = round2_batch.batch["response_mask"]
                round2_batch.batch["response_mask"][padding_tensor_mask.to(response_mask.device)] = 0
            existing_pad_batch = round2_batch.index_select(existing_pad_indices)
            real_batch = round2_batch.index_select(real_indices)
        else:
            real_batch = round2_batch

        target_divisor, padding_chunk = self._get_round2_padding_requirements()
        group_size = int(self.config.worker.rollout.n)
        if group_size <= 0:
            return round2_batch
        if len(real_batch) == 0 or batch_size % group_size != 0:
            return round2_batch

        pad_group_count = 0
        while ((batch_size + pad_group_count * group_size) % target_divisor) != 0:
            pad_group_count += 1
            if pad_group_count > target_divisor:
                return round2_batch

        if pad_group_count == 0:
            if existing_pad_batch is not None and len(existing_pad_batch) > 0:
                padded_batch = self._interleave_round2_padding(real_batch, existing_pad_batch, chunk_size=padding_chunk)
                metrics["round2/pad_group_count"] = 0.0
                metrics["round2/pad_sample_count"] = float(len(existing_pad_batch))
                metrics["round2/padded_batch_size"] = float(len(padded_batch))
                return padded_batch
            metrics["round2/pad_group_count"] = 0.0
            metrics["round2/pad_sample_count"] = 0.0
            return round2_batch

        uid_values = real_batch.non_tensor_batch.get("uid")
        if uid_values is None:
            return round2_batch

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        ordered_uids: list[str] = []
        for idx, uid in enumerate(uid_values.tolist()):
            uid_key = str(uid)
            if uid_key not in uid_to_indices:
                ordered_uids.append(uid_key)
            uid_to_indices[uid_key].append(idx)

        source_groups = [uid_to_indices[uid_key] for uid_key in ordered_uids if len(uid_to_indices[uid_key]) == group_size]
        if not source_groups:
            return round2_batch

        pad_groups: list[DataProto] = []
        for pad_idx in range(pad_group_count):
            source_indices = source_groups[pad_idx % len(source_groups)]
            pad_group = real_batch.index_select(np.array(source_indices, dtype=np.int64))
            pad_size = len(pad_group)

            pad_group.non_tensor_batch["uid"] = np.array(
                [f"round2-pad-{pad_idx}"] * pad_size,
                dtype=object,
            )
            pad_group.non_tensor_batch["is_padding"] = np.ones(pad_size, dtype=bool)
            pad_group.non_tensor_batch["duplicate_weight"] = np.zeros(pad_size, dtype=np.int32)

            for field_name in ("token_level_scores", "token_level_rewards", "old_log_probs", "ref_log_probs", "values"):
                if field_name in pad_group.batch:
                    pad_group.batch[field_name] = torch.zeros_like(pad_group.batch[field_name])
            if "response_mask" in pad_group.batch:
                pad_group.batch["response_mask"] = torch.zeros_like(pad_group.batch["response_mask"])

            pad_groups.append(pad_group)

        pad_batches: list[DataProto] = []
        if existing_pad_batch is not None and len(existing_pad_batch) > 0:
            pad_batches.append(existing_pad_batch)
        if pad_groups:
            pad_batches.append(DataProto.concat(pad_groups))
        if not pad_batches:
            metrics["round2/pad_group_count"] = 0.0
            metrics["round2/pad_sample_count"] = float(existing_padding_mask.sum())
            return round2_batch

        pad_batch = pad_batches[0] if len(pad_batches) == 1 else DataProto.concat(pad_batches)
        padded_batch = self._interleave_round2_padding(real_batch, pad_batch, chunk_size=padding_chunk)
        metrics["round2/pad_group_count"] = float(pad_group_count)
        metrics["round2/pad_sample_count"] = float(len(pad_batch))
        metrics["round2/padded_batch_size"] = float(len(padded_batch))
        return padded_batch

    @staticmethod
    def _filter_reward_metrics_for_padding(
        reward_metrics: dict[str, Any],
        padding_mask: np.ndarray,
    ) -> dict[str, Any]:
        if padding_mask.size == 0 or not bool(padding_mask.any()):
            return reward_metrics

        keep_mask = ~padding_mask
        filtered: dict[str, Any] = {}
        for key, value in reward_metrics.items():
            if isinstance(value, np.ndarray) and len(value) == len(padding_mask):
                filtered[key] = value[keep_mask].tolist()
            elif isinstance(value, (list, tuple)) and len(value) == len(padding_mask):
                filtered[key] = [item for item, keep in zip(value, keep_mask.tolist()) if keep]
            else:
                filtered[key] = value
        return filtered

    @staticmethod
    def _strip_padding_for_metrics(batch: DataProto) -> DataProto:
        if "is_padding" not in batch.non_tensor_batch:
            return batch

        padding_mask = np.asarray(batch.non_tensor_batch["is_padding"], dtype=bool)
        if padding_mask.size == 0 or not bool(padding_mask.any()):
            return batch

        keep_indices = np.flatnonzero(~padding_mask).astype(np.int64)
        if keep_indices.size == 0:
            return batch
        return batch.index_select(keep_indices)

    def _build_round1_task_conditioned_batch(self, original_batch: DataProto) -> Optional[DataProto]:
        multi_modal_data = original_batch.non_tensor_batch.get("multi_modal_data", None)
        uid_array = original_batch.non_tensor_batch.get("uid")

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids_list = []
        new_multi_modal_data = []
        scheduled_task_types = []
        prompt_modes = []
        scene_ids = []
        feasible_task_lists = []
        sampling_prob_lists = []
        kept_uids = []

        for i in range(len(original_batch)):
            image_data = multi_modal_data[i] if multi_modal_data is not None else None
            media_context = self._build_simulator_media_context(image_data)
            summary_context = self._get_round1_summary_context(media_context)
            prompt_mode = summary_context.get("prompt_mode", "scene")
            feasible_tasks = self._infer_feasible_round1_tasks(prompt_mode, summary_context)

            if (
                prompt_mode == "single_image"
                and summary_context.get("frame_visible_label_counts")
                and not feasible_tasks
            ):
                continue

            scheduled_task_type, feasible_tasks, sampling_probs = self._sample_round1_task_type(
                prompt_mode=prompt_mode,
                summary_context=summary_context,
            )
            prompt_text = self._render_round1_task_prompt(
                task_type=scheduled_task_type,
                summary_context=summary_context,
            )

            if image_data is not None and isinstance(image_data, dict) and "images" in image_data:
                images = image_data["images"]
                messages = self._build_answer_messages_with_images(prompt_text, len(images))
                prompt_formatted = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                processed_images = [
                    process_image(img, self.config.data.min_pixels, self.config.data.max_pixels)
                    for img in images
                ]
                model_inputs = self.processor(
                    processed_images,
                    [prompt_formatted],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                new_multi_modal_data.append(
                    {
                        "images": images,
                        "_round1_summary_context": dict(summary_context),
                    }
                )
            else:
                messages = [{"role": "user", "content": prompt_text}]
                prompt_formatted = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                model_inputs = self.tokenizer(
                    [prompt_formatted], add_special_tokens=False, return_tensors="pt"
                )
                new_multi_modal_data.append(
                    {"_round1_summary_context": dict(summary_context)}
                )

            input_ids = model_inputs["input_ids"][0]
            attention_mask = model_inputs["attention_mask"][0]

            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )
                text_position_ids = torch.arange(len(input_ids), device=input_ids.device).unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            raw_prompt_ids_list.append(self.tokenizer.encode(prompt_formatted, add_special_tokens=False))
            scheduled_task_types.append(scheduled_task_type)
            prompt_modes.append(prompt_mode)
            scene_ids.append(summary_context.get("scene_id", "unknown"))
            feasible_task_lists.append(",".join(feasible_tasks))
            sampling_prob_lists.append(
                json.dumps(sampling_probs, ensure_ascii=False, sort_keys=True)
            )
            kept_uids.append(uid_array[i])

        if not input_ids_list:
            return None

        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "position_ids": torch.stack(position_ids_list),
            },
            non_tensors={
                "raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object),
                "multi_modal_data": np.array(new_multi_modal_data, dtype=object),
                "uid": np.array(kept_uids, dtype=object),
                "scheduled_task_type": np.array(scheduled_task_types, dtype=object),
                "prompt_mode": np.array(prompt_modes, dtype=object),
                "scene_id": np.array(scene_ids, dtype=object),
                "feasible_task_types": np.array(feasible_task_lists, dtype=object),
                "task_sampling_probs": np.array(sampling_prob_lists, dtype=object),
            },
            meta_info={
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            },
        )
        return batch

    def _update_round1_task_stats(self, round2_batch: DataProto, reward_metrics_round2: dict[str, Any]) -> None:
        accuracies = reward_metrics_round2.get("accuracy")
        if not isinstance(accuracies, list) or not accuracies:
            return
        if "question_type" not in round2_batch.non_tensor_batch or "uid" not in round2_batch.non_tensor_batch:
            return

        question_types = round2_batch.non_tensor_batch["question_type"].tolist()
        uids = round2_batch.non_tensor_batch["uid"].tolist()
        if "is_valid" in round2_batch.non_tensor_batch:
            valid_flags = np.asarray(round2_batch.non_tensor_batch["is_valid"], dtype=bool).tolist()
        else:
            valid_flags = [True] * len(question_types)
        if "duplicate_weight" in round2_batch.non_tensor_batch:
            duplicate_weights = np.asarray(round2_batch.non_tensor_batch["duplicate_weight"], dtype=np.float32).tolist()
        else:
            duplicate_weights = [1.0] * len(question_types)
        if "is_padding" in round2_batch.non_tensor_batch:
            padding_flags = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool).tolist()
        else:
            padding_flags = [False] * len(question_types)

        if not (
            len(question_types)
            == len(uids)
            == len(valid_flags)
            == len(accuracies)
            == len(duplicate_weights)
            == len(padding_flags)
        ):
            return

        uid_to_task = {}
        uid_to_accs = defaultdict(list)
        uid_to_weight = {}

        for uid, task_type, is_valid, accuracy, duplicate_weight, is_padding in zip(
            uids, question_types, valid_flags, accuracies, duplicate_weights, padding_flags
        ):
            if bool(is_padding):
                continue
            if uid not in uid_to_task:
                uid_to_task[uid] = str(task_type or "").strip()
                uid_to_weight[uid] = max(1.0, float(duplicate_weight))
            if not bool(is_valid):
                uid_to_accs[uid].append(0.0)
                continue
            if accuracy is None:
                continue
            uid_to_accs[uid].append(float(accuracy))

        for uid, task_type in uid_to_task.items():
            if not task_type or task_type == "unknown":
                continue
            acc_values = uid_to_accs.get(uid, [])
            if not acc_values:
                continue
            mean_acc = float(np.mean(acc_values))
            scheduler_acc = self._remap_round1_scheduler_acc(task_type, mean_acc)
            weight = float(uid_to_weight.get(uid, 1.0))
            self._round1_task_stats[task_type]["count"] += weight
            self._round1_task_stats[task_type]["acc_sum"] += mean_acc * weight
            self._round1_task_stats[task_type]["scheduler_acc_sum"] += scheduler_acc * weight

    def _add_round1_task_scheduler_metrics(self, metrics: dict[str, Any]) -> None:
        for task_names in ROUND1_TASKS_BY_MODALITY.values():
            for task_name in task_names:
                stats = self._round1_task_stats.get(
                    task_name,
                    {"count": 0.0, "acc_sum": 0.0, "scheduler_acc_sum": 0.0},
                )
                count = float(stats.get("count", 0.0))
                mean_acc = float(stats.get("acc_sum", 0.0) / count) if count > 0 else 0.0
                scheduler_mean_acc = float(stats.get("scheduler_acc_sum", 0.0) / count) if count > 0 else 0.0
                weight = self._get_round1_task_sampling_weight(task_name)
                metrics[f"round1_scheduler/history_count/{task_name}"] = count
                metrics[f"round1_scheduler/history_acc/{task_name}"] = mean_acc
                metrics[f"round1_scheduler/history_direct_acc/{task_name}"] = mean_acc
                metrics[f"round1_scheduler/history_scheduler_acc/{task_name}"] = scheduler_mean_acc
                metrics[f"round1_scheduler/sampling_weight/{task_name}"] = weight

    def _log_generation_io(self, inputs: DataProto, outputs: DataProto, scores: torch.Tensor, step: int, tag: str = "train"):
        try:
            def _to_python_value(value):
                if isinstance(value, np.generic):
                    return value.item()
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        return value.item()
                    return value.detach().cpu().tolist()
                return value

            def _extract_optional_field(field_name: str, default=None):
                source = None
                if field_name in outputs.non_tensor_batch:
                    source = outputs.non_tensor_batch[field_name]
                elif field_name in inputs.non_tensor_batch:
                    source = inputs.non_tensor_batch[field_name]
                else:
                    return [default] * len(prompts)

                if isinstance(source, np.ndarray):
                    values = source.tolist()
                elif torch.is_tensor(source):
                    values = source.detach().cpu().tolist()
                elif isinstance(source, (list, tuple)):
                    values = list(source)
                else:
                    values = [source] * len(prompts)

                values = [_to_python_value(v) for v in values]
                if len(values) < len(prompts):
                    values.extend([default] * (len(prompts) - len(values)))
                elif len(values) > len(prompts):
                    values = values[: len(prompts)]
                return values

            log_dir = os.path.join(self.config.trainer.save_checkpoint_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "io_log.jsonl")

            if torch.is_tensor(scores):
                if scores.dim() > 1:
                    sample_scores = scores.sum(dim=-1).cpu().tolist()
                else:
                    sample_scores = scores.cpu().tolist()
            else:
                sample_scores = scores

            input_ids = inputs.batch["input_ids"]
            attention_mask = inputs.batch["attention_mask"]
            prompts = []
            for i in range(len(input_ids)):
                valid_ids = input_ids[i][attention_mask[i] == 1]
                p_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                prompts.append(p_text)

            responses = self._decode_responses_from_batch(outputs)
            
            images = []
            if "multi_modal_data" in inputs.non_tensor_batch:
                mm_data = inputs.non_tensor_batch["multi_modal_data"]
                images = [item.get("images") if isinstance(item, dict) else None for item in mm_data]
            else:
                images = [None] * len(prompts)

            question_types = _extract_optional_field("question_type", "")
            question_types_raw = _extract_optional_field("question_type_raw", "")
            questions = _extract_optional_field("question", "")
            ground_truths = _extract_optional_field("gt", "")
            validity_flags = _extract_optional_field("is_valid", None)
            question_uids = _extract_optional_field("question_uid", "")
            sim_errors = _extract_optional_field("sim_error", "")
            sim_error_codes = _extract_optional_field("sim_error_code", "")
            sim_failure_stages = _extract_optional_field("sim_failure_stage", "")
            round2_modes = _extract_optional_field("round2_mode", "")
            padding_flags = _extract_optional_field("is_padding", False)

            with open(log_file, "a", encoding="utf-8") as f:
                for p, r, img, s, q, q_type, q_type_raw, gt, is_valid, q_uid, sim_error, sim_error_code, sim_failure_stage, round2_mode, is_padding in zip(
                    prompts,
                    responses,
                    images,
                    sample_scores,
                    questions,
                    question_types,
                    question_types_raw,
                    ground_truths,
                    validity_flags,
                    question_uids,
                    sim_errors,
                    sim_error_codes,
                    sim_failure_stages,
                    round2_modes,
                    padding_flags,
                ):
                    if bool(is_padding):
                        continue
                    log_entry = {
                        "step": step,
                        "tag": tag,
                        "reward": round(float(s), 4),  # Keep logs compact.
                        "image": img,
                        "prompt": p,
                        "response": r,
                        "question": q,
                        "question_type": q_type,
                        "question_type_raw": q_type_raw,
                        "gt": gt,
                        "is_valid": is_valid,
                        "sim_error": sim_error,
                        "sim_error_code": sim_error_code,
                        "sim_failure_stage": sim_failure_stage,
                        "round2_mode": round2_mode,
                        "question_uid": q_uid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Warning] Failed to log IO with rewards: {e}")

    def _log_round2_valid_qa(
        self,
        round2_batch: DataProto,
        reward_metrics_round2: dict[str, Any],
        *,
        step: int,
    ) -> None:
        try:
            if len(round2_batch) == 0:
                return
            if "uid" not in round2_batch.non_tensor_batch or "is_valid" not in round2_batch.non_tensor_batch:
                return

            log_dir = os.path.join(self.config.trainer.save_checkpoint_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "round2_valid_qa.jsonl")

            responses = self._decode_responses_from_batch(round2_batch)
            accuracies = reward_metrics_round2.get("accuracy", [])
            if not isinstance(accuracies, list) or len(accuracies) != len(round2_batch):
                accuracies = [None] * len(round2_batch)

            question_types = round2_batch.non_tensor_batch.get("question_type")
            questions = round2_batch.non_tensor_batch.get("question")
            ground_truths = round2_batch.non_tensor_batch.get("gt")
            validity_flags = round2_batch.non_tensor_batch.get("is_valid")
            duplicate_weights = round2_batch.non_tensor_batch.get("duplicate_weight")
            padding_flags = round2_batch.non_tensor_batch.get("is_padding")
            prompt_modes = round2_batch.non_tensor_batch.get("prompt_mode")
            scene_ids = round2_batch.non_tensor_batch.get("scene_id")
            multi_modal_data = round2_batch.non_tensor_batch.get("multi_modal_data")

            if any(field is None for field in (question_types, questions, ground_truths, validity_flags)):
                return

            if duplicate_weights is None:
                duplicate_weights = np.ones(len(round2_batch), dtype=np.int32)
            if padding_flags is None:
                padding_flags = np.zeros(len(round2_batch), dtype=bool)
            if prompt_modes is None:
                prompt_modes = np.array(["unknown"] * len(round2_batch), dtype=object)
            if scene_ids is None:
                scene_ids = np.array(["unknown"] * len(round2_batch), dtype=object)

            grouped: dict[str, dict[str, Any]] = {}
            ordered_uids: list[str] = []

            for idx in range(len(round2_batch)):
                if bool(padding_flags[idx]):
                    continue
                if not bool(validity_flags[idx]):
                    continue

                uid = str(round2_batch.non_tensor_batch["uid"][idx] or "")
                if not uid:
                    continue

                if uid not in grouped:
                    ordered_uids.append(uid)
                    image_paths: list[str] = []
                    if multi_modal_data is not None:
                        mm_item = multi_modal_data[idx]
                        if isinstance(mm_item, dict):
                            raw_images = mm_item.get("images")
                            if isinstance(raw_images, (list, tuple)):
                                image_paths = [str(item) for item in raw_images]
                            elif raw_images:
                                image_paths = [str(raw_images)]

                    grouped[uid] = {
                        "step": int(step),
                        "question_uid": uid,
                        "question_type": str(question_types[idx] or ""),
                        "question": str(questions[idx] or ""),
                        "gt": str(ground_truths[idx] or ""),
                        "images": image_paths,
                        "prompt_mode": str(prompt_modes[idx] or ""),
                        "scene_id": str(scene_ids[idx] or ""),
                        "duplicate_weight": int(duplicate_weights[idx]),
                        "responses": [],
                        "accuracies": [],
                    }

                grouped[uid]["responses"].append(str(responses[idx] or ""))
                grouped[uid]["accuracies"].append(
                    None if accuracies[idx] is None else float(accuracies[idx])
                )

            if not ordered_uids:
                return

            with open(log_file, "a", encoding="utf-8") as f:
                for uid in ordered_uids:
                    f.write(json.dumps(grouped[uid], ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[Warning] Failed to log Round2 valid QA: {exc}")

    def _add_per_type_reward_metrics(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        metrics: dict[str, Any],
        metric_prefix: str,
    ) -> None:
        try:
            if reward_tensor is None:
                return
            if "question_type" not in batch.non_tensor_batch:
                return

            if torch.is_tensor(reward_tensor):
                if reward_tensor.dim() > 1:
                    sample_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
                else:
                    sample_rewards = reward_tensor.detach().cpu().tolist()
            else:
                sample_rewards = list(reward_tensor)

            q_types = batch.non_tensor_batch["question_type"].tolist()
            if "is_padding" in batch.non_tensor_batch:
                is_padding = np.asarray(batch.non_tensor_batch["is_padding"], dtype=bool).tolist()
            else:
                is_padding = [False] * len(q_types)
            if len(q_types) != len(sample_rewards) or len(is_padding) != len(sample_rewards):
                return

            supported_task_names = TASK_ANSWER_HINT_TASKS or {
                "object_counting",
                "relative_distance",
                "relative_direction_hard",
                "object_size",
                "absolute_distance",
                "room_size",
                "single_image_relative_direction",
                "distance_cam_obj",
                "depth_order_obj_obj",
                "position_cam_cam",
                "elevation_cam_cam",
                "visibility_compare",
                "position_cam_obj",
                "position_cam_reg",
                "motion_camera",
                "attribute_measurement",
            }

            def _normalize_type_name(s: str) -> str:
                if not isinstance(s, str):
                    return ""
                return re.sub(r"[\s_\[\]【】]+", "", s).lower()

            def _canonicalize_type_name(raw_type: str) -> str:
                norm_input = _normalize_type_name(raw_type)
                if not norm_input:
                    return "unknown"

                for registry_type in supported_task_names:
                    if norm_input == _normalize_type_name(registry_type):
                        return registry_type
                for registry_type in supported_task_names:
                    norm_registry = _normalize_type_name(registry_type)
                    if norm_input in norm_registry or norm_registry in norm_input:
                        return registry_type
                return "unknown"

            type_to_rewards = defaultdict(list)
            for t, r, pad in zip(q_types, sample_rewards, is_padding):
                if bool(pad):
                    continue
                t_key = _canonicalize_type_name(str(t))
                type_to_rewards[t_key].append(float(r))

            for t_key, vals in type_to_rewards.items():
                metrics[f"{metric_prefix}/type_reward_mean/{t_key}"] = float(np.mean(vals))
                metrics[f"{metric_prefix}/type_count/{t_key}"] = float(len(vals))
        except Exception:
            pass

    def init_workers(self) -> None:
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], 
                config=self.config.worker, 
                role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError("Only hybrid-engine mode is supported at the moment.")

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.worker, 
                role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], 
                config=self.config.worker, 
                role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Loading checkpoint from: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Dataloader state not found at {dataloader_path}; starting from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        if self.config.trainer.val_generations_to_log <= 0:
            return

        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []  # Per-batch token rewards.
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)  # Scalar reward metrics.
        length_metrics_lst = defaultdict(list)  # Sequence-length metrics.
        
        print("Starting validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()  # Collapse token rewards to sequence scores.
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        
        print("Validation finished.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None  # Release the batch reference early.
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Starting batch generation...")

        while True:
            num_try_make_batch += 1

            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            gen_batch, gen_pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_batch_output = unpad_dataproto(
                gen_batch_output,
                pad_size=gen_pad_size * int(self.config.worker.rollout.n),
            )

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1  # One sample for baseline scoring.
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
                gen_baseline_output = unpad_dataproto(gen_baseline_output, pad_size=gen_pad_size)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No samples remained after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size  # Prompt batch size.
            
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continuing generation...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Too many generation attempts. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Generation finished.")
                
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def _make_round1_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Starting Round1 batch generation...")

        while True:
            num_try_make_batch += 1

            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            source_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            source_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(source_batch.batch))], dtype=object
            )
            new_batch = self._build_round1_task_conditioned_batch(source_batch)
            if new_batch is None or len(new_batch) == 0:
                metrics["round1/skipped_no_feasible_prompt_count"] = (
                    metrics.get("round1/skipped_no_feasible_prompt_count", 0.0) + float(len(source_batch))
                )
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch > 0 and num_try_make_batch >= max_try_make_batch:
                    raise RuntimeError(
                        f"Too many consecutively skipped samples: {num_try_make_batch=} >= {max_try_make_batch=}. "
                        "No feasible Round1 tasks remain in the current batch; check your data or feasibility filter."
                    )
                continue

            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            gen_batch, gen_pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_batch_output = unpad_dataproto(
                gen_batch_output,
                pad_size=gen_pad_size * int(self.config.worker.rollout.n),
            )

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
                gen_baseline_output = unpad_dataproto(gen_baseline_output, pad_size=gen_pad_size)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No samples remained after filtering. Please check your data.")
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch

            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continuing generation...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Too many generation attempts. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Generation finished.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                batch = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                break

        def parse_vlm_output(vlm_response, fallback_model_client=True):
            def _contains_invalid_inner_tags(text: str) -> bool:
                return bool(re.search(r"<\s*/?\s*[A-Za-z][^>]*>", str(text or "")))

            observation_open = len(re.findall(r"<\s*observation\s*>", vlm_response, flags=re.IGNORECASE))
            observation_close = len(re.findall(r"<\s*/\s*observation\s*>", vlm_response, flags=re.IGNORECASE))
            question_open = len(re.findall(r"<\s*question\s*>", vlm_response, flags=re.IGNORECASE))
            question_close = len(re.findall(r"<\s*/\s*question\s*>", vlm_response, flags=re.IGNORECASE))
            observation_count = observation_open
            question_count = question_open
            is_multi_question = observation_count > 1 or question_count > 1

            result = {
                "observation": "",
                "question": "Unknown",
                "format_score": 0.0,
                "observation_count": observation_count,
                "question_count": question_count,
                "is_multi_question": is_multi_question,
                "has_invalid_inner_tags": False,
            }

            obs_match = re.search(
                r"<\s*observation\s*>(.*?)<\s*/\s*observation\s*>",
                vlm_response,
                flags=re.IGNORECASE | re.DOTALL,
            )
            ques_match = re.search(
                r"<\s*question\s*>(.*?)<\s*/\s*question\s*>",
                vlm_response,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if ques_match:
                extracted_obs = obs_match.group(1).strip() if obs_match else ""
                extracted_question = ques_match.group(1).strip()
                stripped = re.sub(
                    r"<\s*observation\s*>.*?<\s*/\s*observation\s*>",
                    "",
                    vlm_response,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                stripped = re.sub(
                    r"<\s*question\s*>.*?<\s*/\s*question\s*>",
                    "",
                    stripped,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                is_clean_format = (
                    question_open == 1
                    and question_close == 1
                    and observation_open == observation_close
                    and observation_open <= 1
                    and not stripped.strip()
                )
                has_invalid_inner_tags = _contains_invalid_inner_tags(extracted_obs) or _contains_invalid_inner_tags(
                    extracted_question
                )
                result.update(
                    {
                        "observation": extracted_obs,
                        "question": extracted_question,
                        "format_score": 1.0 if (is_clean_format and extracted_obs) else 0.5 if is_clean_format else 0.0,
                        "has_invalid_inner_tags": has_invalid_inner_tags,
                    }
                )
                return result

            if fallback_model_client:
                fallback_res = call_fallback_agent(vlm_response)
                fallback_res.update(
                    {
                        "observation_count": observation_count,
                        "question_count": question_count,
                        "is_multi_question": is_multi_question,
                    }
                )
                return fallback_res

            return result

        def call_fallback_agent(raw_text, ak="", model="gpt-oss-120b-ldm", url="https://example.com/v1"):
            repair_prompt = f"""You are a data recovery assistant. Your task is to extract one grounded observation and one spatial question from the raw text below.

                ### RAW TEXT TO REPAIR:
                {raw_text}

                ### CONSTRAINTS:
                1. If the raw text contains an <observation> block or grounded scene description, preserve it inside <observation>. If none exists, leave <observation> empty.
                2. Extract exactly one final question and place it inside <question>.
                3. Output EXACTLY ONE <observation> tag and EXACTLY ONE <question> tag.
                4. No extra text outside the tags.

                ### OUTPUT FORMAT:
                <observation>[Extracted grounded observation, or empty if not found]</observation>
                <question>[Extracted Question]</question>"""

            messages = [{"role": "user", "content": repair_prompt}]
            client = OpenAI(api_key=ak, base_url=url)

            for attempt in range(10):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.1,
                        timeout=30,
                    )
                    repaired_response = chat_completion.choices[0].message.content.strip()
                    if repaired_response:
                        repaired = parse_vlm_output(repaired_response, fallback_model_client=False)
                        repaired["format_score"] = 0.0
                        repaired["is_fallback"] = True
                        return repaired
                except Exception as exc:
                    if attempt < 9:
                        time.sleep((attempt + 1) * 2)
                    else:
                        print(f"Fallback repair failed: {exc}")

            return {
                "observation": "",
                "question": "Unknown",
                "format_score": 0.0,
                "is_fallback": True,
            }

        def process_question_content(q_item, canonical_task_type: str):
            original_text = q_item["question"]
            return _build_question_with_answer_hint(
                original_text,
                canonical_task_type,
                round_stage=1,
            )

        questions_raw = self._decode_responses_from_batch(batch)
        questions = [parse_vlm_output(question) for question in questions_raw]
        scheduled_task_types = batch.non_tensor_batch["scheduled_task_type"].tolist()
        multi_modal_data = batch.non_tensor_batch.get("multi_modal_data", [])
        media_contexts = [
            self._build_simulator_media_context(multi_modal_data[i] if i < len(multi_modal_data) else None)
            for i in range(len(questions))
        ]

        batch.non_tensor_batch["question"] = np.array(
            [process_question_content(q, scheduled_task_types[i]) for i, q in enumerate(questions)],
            dtype=object,
        )
        batch.non_tensor_batch["question_type_raw"] = np.array(scheduled_task_types, dtype=object)
        batch.non_tensor_batch["observation"] = np.array(
            [question.get("observation", "") for question in questions], dtype=object
        )
        batch.non_tensor_batch["format_score"] = np.array(
            [question["format_score"] for question in questions], dtype=np.float32
        )
        batch.non_tensor_batch["question_count"] = np.array(
            [q["question_count"] for q in questions], dtype=np.int32
        )
        batch.non_tensor_batch["has_invalid_inner_tags"] = np.array(
            [bool(q.get("has_invalid_inner_tags", False)) for q in questions], dtype=bool
        )

        sim_jobs = [
            {
                "media_context": media_contexts[i],
                "canonical_task_type": scheduled_task_types[i],
                "question_text": batch.non_tensor_batch["question"][i],
            }
            for i in range(len(questions))
        ]

        simulator_max_workers = self._get_world_simulator_max_workers(len(sim_jobs))
        if simulator_max_workers > 1 and len(sim_jobs) > 1:
            try:
                with ThreadPoolExecutor(max_workers=simulator_max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_simulator_validation_job,
                            media_context=job["media_context"],
                            canonical_task_type=job["canonical_task_type"],
                            question_text=job["question_text"],
                            use_thread_local_simulator=True,
                        )
                        for job in sim_jobs
                    ]
                    sim_results = [future.result() for future in futures]
            except Exception as exc:
                print(f"WorldSimulator parallel validation failed; falling back to serial execution: {exc}")
                sim_results = [
                    self._run_simulator_validation_job(
                        media_context=job["media_context"],
                        canonical_task_type=job["canonical_task_type"],
                        question_text=job["question_text"],
                        use_thread_local_simulator=False,
                    )
                    for job in sim_jobs
                ]
        else:
            sim_results = [
                self._run_simulator_validation_job(
                    media_context=job["media_context"],
                    canonical_task_type=job["canonical_task_type"],
                    question_text=job["question_text"],
                    use_thread_local_simulator=False,
                )
                for job in sim_jobs
            ]

        resolved_question_types = [result.get("resolved_task_type", "unknown") for result in sim_results]
        sim_answers = [result.get("answer", "invalid") for result in sim_results]
        is_valid_list = [bool(result.get("is_valid", False)) for result in sim_results]
        sim_error_list = [str(result.get("error", "") or "") for result in sim_results]
        sim_error_code_list = [str(result.get("error_code", "") or "") for result in sim_results]
        sim_failure_stage_list = [str(result.get("failure_stage", "") or "") for result in sim_results]
        sim_parsed_params_list = [
            self._json_dumps_non_tensor(result.get("parsed_params", {}) or {})
            for result in sim_results
        ]
        sim_validation_result_list = [
            self._json_dumps_non_tensor(result.get("validation_result"))
            for result in sim_results
        ]
        sim_judge_reference_list = [
            self._json_dumps_non_tensor(result.get("judge_reference"))
            for result in sim_results
        ]
        sim_result_json_list = [
            self._json_dumps_non_tensor(result.get("sim_result"))
            for result in sim_results
        ]

        batch.non_tensor_batch["question_type"] = np.array(resolved_question_types, dtype=object)
        batch.non_tensor_batch["gt"] = np.array(sim_answers, dtype=object)
        batch.non_tensor_batch["is_valid"] = np.array(is_valid_list, dtype=bool)
        batch.non_tensor_batch["sim_error"] = np.array(sim_error_list, dtype=object)
        batch.non_tensor_batch["sim_error_code"] = np.array(sim_error_code_list, dtype=object)
        batch.non_tensor_batch["sim_failure_stage"] = np.array(sim_failure_stage_list, dtype=object)
        batch.non_tensor_batch["sim_parsed_params_json"] = np.array(sim_parsed_params_list, dtype=object)
        batch.non_tensor_batch["sim_validation_result_json"] = np.array(sim_validation_result_list, dtype=object)
        batch.non_tensor_batch["sim_judge_reference_json"] = np.array(sim_judge_reference_list, dtype=object)
        batch.non_tensor_batch["sim_result_json"] = np.array(sim_result_json_list, dtype=object)

        selected_uid_to_type = {}
        for uid, task_type in zip(
            batch.non_tensor_batch["uid"].tolist(),
            batch.non_tensor_batch["scheduled_task_type"].tolist(),
        ):
            selected_uid_to_type.setdefault(uid, task_type)
        for task_type in selected_uid_to_type.values():
            metrics[f"round1_scheduler/selected_count/{task_type}"] = (
                metrics.get(f"round1_scheduler/selected_count/{task_type}", 0.0) + 1.0
            )

        batch_size = len(batch)
        question_uids = np.array([f"question-{i}" for i in range(batch_size)], dtype=object)
        batch.non_tensor_batch["question_uid"] = question_uids
        batch.non_tensor_batch["round"] = np.full(batch_size, 1, dtype=np.int32)
        return batch

    def _extract_questions_from_batch(self, batch: DataProto) -> list[str]:
        return batch.non_tensor_batch["question"].tolist()
    def _extract_ground_truth_from_batch(self, batch: DataProto) -> list[str]:
        return batch.non_tensor_batch["gt"].tolist()

    def _extract_question_type_from_batch(self, batch: DataProto) -> list[str]:
        return batch.non_tensor_batch["question_type"].tolist()

    def _process_round1_to_round2_data(self, round1_batch: DataProto, questions: list[str]) -> DataProto:
        answer_batch = self._build_answer_batch(round1_batch, questions)
        
        return answer_batch

    def _make_round2_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        if not hasattr(self, 'round1_batch') or self.round1_batch is None:
            raise RuntimeError("Round1 batch data does not exist. Call `_make_round1_batch_data` first.")
        
        round1_batch = self._deduplicate_round1_batch_for_round2(self.round1_batch, metrics=metrics)

        ground_truths = self._extract_ground_truth_from_batch(round1_batch)
        question_types = self._extract_question_type_from_batch(round1_batch)
        questions = self._extract_questions_from_batch(round1_batch)
        sim_parsed_params_list = round1_batch.non_tensor_batch.get("sim_parsed_params_json", None)
        sim_validation_result_list = round1_batch.non_tensor_batch.get("sim_validation_result_json", None)
        sim_result_json_list = round1_batch.non_tensor_batch.get("sim_result_json", None)
        round2_questions = [
            _build_question_with_answer_hint(
                questions[i],
                question_types[i],
                gt=ground_truths[i],
                round_stage=2,
                sim_parsed_params=sim_parsed_params_list[i] if sim_parsed_params_list is not None else None,
                sim_validation_result=sim_validation_result_list[i] if sim_validation_result_list is not None else None,
                sim_result=sim_result_json_list[i] if sim_result_json_list is not None else None,
            )
            for i in range(len(questions))
        ]
        processed_batch = self._process_round1_to_round2_data(round1_batch, round2_questions)

        processed_batch.non_tensor_batch["question"] = np.array(round2_questions, dtype=object)
        processed_batch.non_tensor_batch["gt"] = np.array(ground_truths, dtype=object)
        processed_batch.non_tensor_batch["question_type"] = np.array(question_types, dtype=object)
        if "duplicate_weight" in round1_batch.non_tensor_batch:
            processed_batch.non_tensor_batch["duplicate_weight"] = np.array(
                round1_batch.non_tensor_batch["duplicate_weight"], dtype=np.int32
            )
        else:
            processed_batch.non_tensor_batch["duplicate_weight"] = np.ones(len(processed_batch), dtype=np.int32)
        if "is_valid" in round1_batch.non_tensor_batch:
            processed_batch.non_tensor_batch["is_valid"] = np.array(
                round1_batch.non_tensor_batch["is_valid"], dtype=bool
            )
        processed_batch.non_tensor_batch["is_padding"] = np.zeros(len(processed_batch), dtype=bool)
        for field_name in (
            "sim_error",
            "sim_error_code",
            "sim_failure_stage",
            "sim_parsed_params_json",
            "sim_validation_result_json",
            "sim_judge_reference_json",
            "sim_result_json",
            "question_type_raw",
        ):
            if field_name in round1_batch.non_tensor_batch:
                processed_batch.non_tensor_batch[field_name] = np.array(
                    round1_batch.non_tensor_batch[field_name], dtype=object
                )

        if "question_uid" in round1_batch.non_tensor_batch:
            question_uids = round1_batch.non_tensor_batch["question_uid"]
        else:
            question_uids = round1_batch.non_tensor_batch["uid"]
        
        processed_batch.non_tensor_batch["uid"] = question_uids.copy()

        new_batch = processed_batch.select()
        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        
        gen_batch, gen_pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        gen_batch_output = unpad_dataproto(
            gen_batch_output,
            pad_size=gen_pad_size * int(self.config.worker.rollout.n),
        )

        if self.config.algorithm.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
            gen_baseline_output = unpad_dataproto(gen_baseline_output, pad_size=gen_pad_size)
            
            new_batch = new_batch.union(gen_baseline_output)
            reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
            
            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output
        
        new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_batch_output)
        
        all_metrics = defaultdict(list)
        if self.config.algorithm.online_filtering:
            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            new_batch.batch["token_level_scores"] = reward_tensor
            for k, v in reward_metrics.items():
                all_metrics[k].extend(v)
            
            filter_scores = reward_metrics[self.config.algorithm.filter_key]
            uids = new_batch.non_tensor_batch["uid"]
            
            uid2scores = defaultdict(list)
            for uid, score in zip(uids, filter_scores):
                uid2scores[uid].append(score)
            
            uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
            
            kept_uids = [
                uid
                for uid, avg_score in uid2mean.items()
                if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
            ]
            kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
            
            if len(kept_sample_idxs) == 0:
                raise RuntimeError("No samples remained after filtering. Please check your data.")
            
            new_batch = new_batch[kept_sample_idxs]
            
            metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
        
        new_batch = self._pad_round2_batch_for_balance(new_batch, metrics=metrics)
        batch_size = len(new_batch)
        new_batch.non_tensor_batch["round"] = np.full(batch_size, 2, dtype=np.int32)
        
        return new_batch

    def _decode_responses_from_batch(self, gen_output: DataProto) -> list[str]:
        responses = []
        response_ids = gen_output.batch["responses"]
        response_mask = gen_output.batch["response_mask"]
        
        for i in range(len(gen_output)):
            valid_length = int(response_mask[i].sum().item())
            valid_ids = response_ids[i][:valid_length]
            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            responses.append(text)
        
        return responses

    def _build_answer_batch(
        self, 
        original_batch: DataProto,
        questions: list[str],
    ) -> DataProto:
        answer_template = Template(ROUND2_ANSWER_PROMPT_TEMPLATE)
        
        multi_modal_data = original_batch.non_tensor_batch.get("multi_modal_data", None)
        question_types = original_batch.non_tensor_batch.get("question_type", None)
        validity_flags = original_batch.non_tensor_batch.get("is_valid", None)
        
        input_ids_list = []  # Token ids.
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids_list = []  # Tokenized prompt ids.
        new_multi_modal_data = []  # vLLM-ready multimodal payloads.
        round2_modes = []
        round2_task_references = []
        
        for i, question in enumerate(questions):
            task_type = ""
            if question_types is not None and i < len(question_types):
                task_type = str(question_types[i] or "")

            is_valid_question = True
            if validity_flags is not None and i < len(validity_flags):
                is_valid_question = bool(validity_flags[i])

            image_data = multi_modal_data[i] if multi_modal_data is not None and i < len(multi_modal_data) else None
            media_context = self._build_simulator_media_context(image_data)
            summary_context = self._get_round1_summary_context(media_context)
            num_imgs = len(media_context.get("image_paths", []) or [])

            if is_valid_question:
                prompt_text = answer_template.render(question=question).strip()
                round2_modes.append("answer")
                round2_task_references.append("")
            else:
                prompt_text, task_reference_text = self._render_round2_invalid_explanation_prompt(
                    question=question,
                    task_type=task_type,
                    summary_context=summary_context,
                )
                round2_modes.append("invalid_explanation")
                round2_task_references.append(task_reference_text)
            
            if multi_modal_data is not None and multi_modal_data[i] is not None:
                if isinstance(image_data, dict) and "images" in image_data:
                    num_imgs = len(image_data["images"])

                messages = self._build_answer_messages_with_images(prompt_text,num_imgs)
                
                prompt_formatted = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )

                if isinstance(image_data, dict) and "images" in image_data:
                    images = image_data["images"]
                    
                    processed_images = [
                        process_image(img, self.config.data.min_pixels, self.config.data.max_pixels)
                        for img in images
                    ]
                    
                    model_inputs = self.processor(
                        processed_images, [prompt_formatted], 
                        add_special_tokens=False, return_tensors="pt"
                    )
                    
                    new_multi_modal_data.append({"images": images})
                else:
                    model_inputs = self.tokenizer(
                        [prompt_formatted], add_special_tokens=False, return_tensors="pt"
                    )
                    new_multi_modal_data.append(None)
            else:
                messages = [{"role": "user", "content": prompt_text}]
                
                prompt_formatted = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                
                model_inputs = self.tokenizer(
                    [prompt_formatted], add_special_tokens=False, return_tensors="pt"
                )
                new_multi_modal_data.append(None)
            
            input_ids = model_inputs["input_ids"][0]  # Token ids, shape `(seq_length,)`.
            attention_mask = model_inputs["attention_mask"][0]  # Attention mask, shape `(seq_length,)`.
            
            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index
                
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )  # (3, seq_length)
                
                text_position_ids = torch.arange(len(input_ids), device=input_ids.device).unsqueeze(0)
                
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)
            
            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",  # Keep the prompt tail when truncating.
            )
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            
            raw_prompt_ids_list.append(
                self.tokenizer.encode(prompt_formatted, add_special_tokens=False)
            )
        
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),        # (batch_size, max_prompt_length)
                "attention_mask": torch.stack(attention_mask_list),  # (batch_size, max_prompt_length)
                "position_ids": torch.stack(position_ids_list),  # mRoPE may use `(batch_size, 4, max_prompt_length)`.
            },
            non_tensors={
                "raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object),  # Prompt token ids.
                "multi_modal_data": np.array(new_multi_modal_data, dtype=object),
                "round2_mode": np.array(round2_modes, dtype=object),
                "round2_task_reference": np.array(round2_task_references, dtype=object),
            },
            meta_info={
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            },
        )
        
        return batch

    def _build_answer_messages_with_images(self, prompt_text: str, num_images: int) -> list[dict]:
        content = []
        for _ in range(num_images):
            content.append({"type": "image"})
        
        content.append({"type": "text", "text": prompt_text})
        
        return [{"role": "user", "content": content}]

    def fit(self):
        
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        
        self.global_step = 0
        
        main_tqdm = tqdm(range(self.training_steps), desc="Training steps", position=0)
        
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            
            self.logger.log(data=val_metrics, step=self.global_step)
            
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            
            with timer("step", timing_raw):
                
                use_two_round_training = getattr(self.config.algorithm, 'two_round_training', False) 
                skip_generic_ppo_update = False

                if use_two_round_training:
                    if getattr(self.config.algorithm, "online_filtering", False):
                        raise ValueError("`two_round_training` currently requires `algorithm.online_filtering=false`.")
                    
                    print(f"========== Round 1 training: question-generation stage (step {self.global_step}) ==========")
                    
                    with timer("gen_round1", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        round1_batch = self._make_round1_batch_data(metrics=metrics)
                        self.actor_rollout_ref_wg.release_rollout_engine()
                    
                    self.round1_batch = round1_batch
                    
                    self._balance_batch(round1_batch, metrics=metrics)
                    
                    round1_batch.meta_info["global_token_num"] = torch.sum(round1_batch.batch["attention_mask"], dim=-1).tolist()
                    
                    if "token_level_scores" not in round1_batch.batch:
                        with timer("reward_round1", timing_raw):
                            assert "round" in round1_batch.non_tensor_batch, "Round1 batch must include the `round` field"
                            assert np.all(round1_batch.non_tensor_batch["round"] == 1), "The `round` field in Round1 batch must be 1"
                            quality_reward_ref = self.reward_fn.compute_reward.remote(round1_batch)
                            
                            quality_reward_tensor, quality_reward_metrics = ray.get(quality_reward_ref)

                            quality_reward_metrics_reduced = reduce_metrics(quality_reward_metrics)
                            self._log_generation_io(
                                inputs=round1_batch, # DataProto
                                outputs=round1_batch, # union batch
                                scores=quality_reward_tensor,
                                step=self.global_step, 
                                tag="round1_quality_check"
                            )
                            round1_batch.batch["token_level_scores"] = quality_reward_tensor
                            self._add_per_type_reward_metrics(
                                batch=round1_batch,
                                reward_tensor=quality_reward_tensor,
                                metrics=metrics,
                                metric_prefix="reward_round1",
                            )
                            
                            quality_overall = quality_reward_metrics_reduced.get("overall", 0.0)

                            quality_reward_per_sample = quality_reward_tensor.sum(dim=-1).cpu().tolist()
                            quality_reward_stats = {
                                "max": np.max(quality_reward_per_sample),
                                "min": np.min(quality_reward_per_sample),
                            }
                            
                            reward_metrics_round1 = {
                                f"reward_round1/quality": quality_overall,
                                f"reward_round1/quality_max": quality_reward_stats["max"],
                                f"reward_round1/quality_min": quality_reward_stats["min"],
                            }
                            
                            for k, v in quality_reward_metrics_reduced.items():
                                if k != "overall":
                                    reward_metrics_round1[f"reward_round1/quality_{k}"] = v

                            metrics.update(reward_metrics_round1)
                    
                    with timer("old_round1", timing_raw):
                        old_log_probs_round1 = self.actor_rollout_ref_wg.compute_log_probs(round1_batch)
                        round1_batch = round1_batch.union(old_log_probs_round1)
                    
                    if self.use_reference_policy:
                        with timer("ref_round1", timing_raw):
                            ref_log_probs_round1 = self.actor_rollout_ref_wg.compute_ref_log_probs(round1_batch)
                            round1_batch = round1_batch.union(ref_log_probs_round1)
                    
                    if self.use_critic:
                        with timer("values_round1", timing_raw):
                            values_round1 = self.critic_wg.compute_values(round1_batch)
                            round1_batch = round1_batch.union(values_round1)
                    
                    with timer("adv_round1", timing_raw):
                        if "token_level_scores" not in round1_batch.batch:
                            reward_ref_round1 = self.reward_fn.compute_reward.remote(round1_batch)
                            reward_tensor_round1, reward_metrics_round1 = ray.get(reward_ref_round1)
                            round1_batch.batch["token_level_scores"] = reward_tensor_round1
                            reward_metrics_round1 = {f"reward_round1/{k}": v for k, v in reduce_metrics(reward_metrics_round1).items()}
                            metrics.update(reward_metrics_round1)
                        
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            round1_batch, kl_metrics_round1 = apply_kl_penalty(round1_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            kl_metrics_round1 = {f"{k}_round1": v for k, v in kl_metrics_round1.items()}
                            metrics.update(kl_metrics_round1)
                        else:
                            round1_batch.batch["token_level_rewards"] = round1_batch.batch["token_level_scores"]
                        
                        round1_batch = compute_advantage(
                            round1_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                    
                    if self.use_critic:
                        with timer("update_critic_round1", timing_raw):
                            critic_output_round1 = self.critic_wg.update_critic(round1_batch)
                        critic_metrics_round1 = {f"{k}_round1": v for k, v in reduce_metrics(critic_output_round1.non_tensor_batch).items()}
                        metrics.update(critic_metrics_round1)
                    
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor_round1", timing_raw):
                            actor_output_round1 = self.actor_rollout_ref_wg.update_actor(round1_batch)
                        actor_metrics_round1 = {f"{k}_round1": v for k, v in reduce_metrics(actor_output_round1.non_tensor_batch).items()}
                        metrics.update(actor_metrics_round1)
                    
                    print(f"========== Round 2 training: answering stage (step {self.global_step}) ==========")
                    
                    with timer("gen_round2", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        round2_batch = self._make_round2_batch_data(metrics=metrics)
                        self.actor_rollout_ref_wg.release_rollout_engine()
                    
                    del self.round1_batch
                    
                    self._balance_batch(round2_batch, metrics=metrics)
                        
                    round2_batch.meta_info["global_token_num"] = torch.sum(round2_batch.batch["attention_mask"], dim=-1).tolist()
                        
                    if "token_level_scores" not in round2_batch.batch:
                        with timer("reward_round2", timing_raw):
                            assert "round" in round2_batch.non_tensor_batch, "Round2 batch must include the `round` field"
                            assert np.all(round2_batch.non_tensor_batch["round"] == 2), "The `round` field in Round2 batch must be 2"
                            assert "uid" in round2_batch.non_tensor_batch, "Round2 batch must include the `uid` field (for GRPO grouping and per-question stats)"
                            reward_ref_round2 = self.reward_fn.compute_reward.remote(round2_batch)
                        
                    with timer("old_round2", timing_raw):
                        old_log_probs_round2 = self.actor_rollout_ref_wg.compute_log_probs(round2_batch)
                        round2_batch = round2_batch.union(old_log_probs_round2)
                        
                    if self.use_reference_policy:
                        with timer("ref_round2", timing_raw):
                            ref_log_probs_round2 = self.actor_rollout_ref_wg.compute_ref_log_probs(round2_batch)
                            if "ref_log_probs" in round2_batch.batch:
                                round2_batch.batch["ref_log_probs"] = ref_log_probs_round2.batch["ref_log_probs"]
                            else:
                                round2_batch = round2_batch.union(ref_log_probs_round2)
                        
                    if self.use_critic:
                        with timer("values_round2", timing_raw):
                            values_round2 = self.critic_wg.compute_values(round2_batch)
                            round2_batch = round2_batch.union(values_round2)
                    
                    with timer("adv_round2", timing_raw):
                        reward_tensor_round2 = round2_batch.batch.get("token_level_scores")
                        if "token_level_scores" not in round2_batch.batch:
                            reward_tensor_round2, reward_metrics_round2 = ray.get(reward_ref_round2)
                            filtered_reward_metrics_round2 = reward_metrics_round2
                            if "is_padding" in round2_batch.non_tensor_batch:
                                padding_mask = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool)
                                if padding_mask.any():
                                    reward_tensor_round2 = reward_tensor_round2.clone()
                                    reward_tensor_round2[torch.from_numpy(padding_mask).to(reward_tensor_round2.device)] = 0
                                    filtered_reward_metrics_round2 = self._filter_reward_metrics_for_padding(
                                        reward_metrics_round2,
                                        padding_mask=padding_mask,
                                    )
                            self._update_round1_task_stats(round2_batch, reward_metrics_round2)
                            self._add_round1_task_scheduler_metrics(metrics)
                            self._log_round2_valid_qa(
                                round2_batch,
                                reward_metrics_round2,
                                step=self.global_step,
                            )
                            round2_batch.batch["token_level_scores"] = reward_tensor_round2
                            reward_metrics_round2 = {
                                f"reward_round2/{k}": v for k, v in reduce_metrics(filtered_reward_metrics_round2).items()
                            }
                            metrics.update(reward_metrics_round2)
                        
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            round2_batch, kl_metrics_round2 = apply_kl_penalty(round2_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            kl_metrics_round2 = {f"{k}_round2": v for k, v in kl_metrics_round2.items()}
                            metrics.update(kl_metrics_round2)
                        else:
                            round2_batch.batch["token_level_rewards"] = round2_batch.batch["token_level_scores"]
                        
                        self._log_generation_io(
                            inputs=round2_batch, 
                            outputs=round2_batch, 
                            scores=reward_tensor_round2, 
                            step=self.global_step, 
                            tag="round2_answer_check"
                        )
                        self._add_per_type_reward_metrics(
                            batch=round2_batch,
                            reward_tensor=reward_tensor_round2,
                            metrics=metrics,
                            metric_prefix="reward_round2",
                        )

                        if "is_valid" in round2_batch.non_tensor_batch:
                            is_valid_np = np.asarray(round2_batch.non_tensor_batch["is_valid"], dtype=bool)
                            if "is_padding" in round2_batch.non_tensor_batch:
                                is_padding_np = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool)
                                is_valid_np = is_valid_np[~is_padding_np]
                            invalid_np = ~is_valid_np
                            invalid_count = int(invalid_np.sum())
                            total_count = int(len(is_valid_np))
                            metrics["round2/invalid_count"] = float(invalid_count)
                            metrics["round2/invalid_ratio"] = (float(invalid_count) / float(total_count)) if total_count > 0 else 0.0

                        round2_batch = self._pad_round2_batch_for_updates(round2_batch, metrics=metrics)
                        round2_batch.meta_info["global_token_num"] = torch.sum(
                            round2_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        round2_batch = compute_advantage(
                            round2_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                    
                    if self.use_critic:
                        with timer("update_critic_round2", timing_raw):
                            critic_output_round2 = self.critic_wg.update_critic(round2_batch)
                        critic_metrics_round2 = {f"{k}_round2": v for k, v in reduce_metrics(critic_output_round2.non_tensor_batch).items()}
                        metrics.update(critic_metrics_round2)
                    
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor_round2", timing_raw):
                            actor_output_round2 = self.actor_rollout_ref_wg.update_actor(round2_batch)
                        actor_metrics_round2 = {f"{k}_round2": v for k, v in reduce_metrics(actor_output_round2.non_tensor_batch).items()}
                        metrics.update(actor_metrics_round2)
                    
                    batch = round2_batch
                    skip_generic_ppo_update = True
                    
                else:
                    with timer("gen", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        
                        batch = self._make_batch_data(metrics=metrics)
                        
                        self.actor_rollout_ref_wg.release_rollout_engine()

                if not skip_generic_ppo_update:
                    self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if "token_level_scores" not in batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)

                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        
                        if "old_log_probs" in batch.batch:
                            batch.batch["old_log_probs"] = old_log_probs.batch["old_log_probs"]
                        else:
                            batch = batch.union(old_log_probs)

                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)

                            if "ref_log_probs" in batch.batch:
                                batch.batch["ref_log_probs"] = ref_log_probs.batch["ref_log_probs"]
                            else:
                                batch = batch.union(ref_log_probs)

                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in batch.batch:
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            
                            batch.batch["token_level_scores"] = reward_tensor
                            
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            
                            metrics.update(reward_metrics)

                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        
                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        
                        metrics.update(critic_metrics)

                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                        
                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        
                        metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0 # > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            
            metrics_batch = self._strip_padding_for_metrics(batch)
            metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
            
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
