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
import re
from datetime import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict, List
import requests, re
from filelock import FileLock
from openai import OpenAI

class RewardInput(TypedDict, total=False):
    """Reward Input."""
    response: str
    response_length: int
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str]  # Group id for GRPO.
    round: Optional[int]  # 1 for question generation, 2 for answer generation.
    # Extra fields provided by `RayPPOTrainer`.
    format_score: float
    is_valid: bool
    question_type: str
    question_count: int
    has_invalid_inner_tags: bool  # Invalid nested tags in question/observation blocks.

class RewardScore(TypedDict):
    """Reward Score."""
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

REWARD_NAME = "question_quality"
REWARD_TYPE = "batch"  # Batch-level reward.
VALID_OBSERVATION_FLOOR = 0.1

TASK_ANSWER_HINTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "format_prompt", "task_answer_format_hints.json")
)
DEFAULT_SUPPORTED_TASK_TYPES = {
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

try:
    with open(TASK_ANSWER_HINTS_PATH, "r", encoding="utf-8") as f:
        TASK_ANSWER_HINTS = json.load(f).get("tasks") or {}
except Exception:
    TASK_ANSWER_HINTS = {}

SUPPORTED_TASK_TYPES = set(TASK_ANSWER_HINTS.keys()) or DEFAULT_SUPPORTED_TASK_TYPES

def _resolve_path_from_base(path_value: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    if not path_value:
        return None

    resolved = str(path_value).strip()
    if not resolved:
        return None

    if os.path.isabs(resolved):
        return resolved

    if base_dir:
        return os.path.abspath(os.path.join(base_dir, resolved))
    return os.path.abspath(resolved)

def _resolve_log_dir() -> str:
    env_log = os.environ.get("REWARD_LOG_DIR")
    if env_log:
        resolved_log = _resolve_path_from_base(env_log)
        if resolved_log:
            return resolved_log

    for key in ("TRAINER_SAVE_CHECKPOINT_PATH", "SAVE_CHECKPOINT_PATH", "CHECKPOINT_DIR"):
        cp = os.environ.get(key)
        if cp:
            resolved_cp = _resolve_path_from_base(cp)
            if resolved_cp:
                return os.path.join(resolved_cp, "reward_stats")

    script_path = os.environ.get("TRAIN_SCRIPT_PATH", os.path.join(os.path.dirname(__file__), "../../train_spatialevo.sh"))
    resolved_script_path = _resolve_path_from_base(script_path)
    try:
        if resolved_script_path and os.path.exists(resolved_script_path):
            with open(resolved_script_path, "r") as f:
                text = f.read()
            m = re.search(r"trainer\.save_checkpoint_path=([^\s\\]+)", text)
            if m:
                cp = m.group(1).strip().strip('"').strip("'")
                resolved_cp = _resolve_path_from_base(cp, base_dir=os.path.dirname(resolved_script_path))
                if resolved_cp:
                    return os.path.join(resolved_cp, "reward_stats")
    except Exception:
        pass

    today = datetime.now().strftime("%Y%m%d")
    return os.path.abspath(f"stats/log_{today}")

class GlobalTypeTracker:
    def __init__(self, log_dir=None, filename="global_type_counts.json"):
        if log_dir is None:
            log_dir = _resolve_log_dir()
        self.filepath = os.path.join(log_dir, filename)
        self.lockpath = self.filepath + ".lock"
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize the tracker file on first use.
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({"total": 0, "counts": {}, "details": {}}, f)

    def update_and_get_stats(self, new_batch_info: list[tuple[str, bool]]):
        """Update And Get Stats."""
        with FileLock(self.lockpath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"total": 0, "counts": {}, "details": {}}
            
            counts = data.get("counts", {})
            details = data.get("details", {})
            total = data.get("total", 0)
            
            for t, is_valid in new_batch_info:
                t = t.lower().strip()
                
                # Update per-type counts.
                counts[t] = counts.get(t, 0) + 1
                total += 1
                
                # Update per-type validity stats.
                if t not in details:
                    details[t] = {"valid": 0, "invalid": 0}
                
                if is_valid:
                    details[t]["valid"] += 1
                else:
                    details[t]["invalid"] += 1
            
            data["counts"] = counts
            data["details"] = details
            data["total"] = total
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=2)
                
            return counts, total

# Shared tracker for task-type statistics.
tracker = GlobalTypeTracker()

def _parse_int_like(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            f = float(value)
            return int(f) if f.is_integer() else None
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if not match:
            return None
        try:
            f = float(match.group(0))
            return int(f) if f.is_integer() else None
        except ValueError:
            return None
    return None

def _normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", text)

def _uniform_validity_factor(gt: Any) -> float:
    return 1.0

def _object_counting_validity_factor(gt: Any) -> float:
    gt_count = _parse_int_like(gt)
    if gt_count == 0:
        return 0.0
    if gt_count == 1:
        return 0.5
    return 1.0

def _depth_order_validity_factor(gt: Any) -> float:
    return 0.5 if _normalize_answer_text(gt) == "same" else 1.0

def _elevation_validity_factor(gt: Any) -> float:
    return 0.0 if _normalize_answer_text(gt) == "same_level" else 1.0

def _visibility_compare_validity_factor(gt: Any) -> float:
    return 0.5 if _normalize_answer_text(gt) in {"same", "neither"} else 1.0

TASK_VALIDITY_FACTOR_RULES = {
    "object_counting": _object_counting_validity_factor,
    "object_size": _uniform_validity_factor,
    "absolute_distance": _uniform_validity_factor,
    "relative_distance": _uniform_validity_factor,
    "relative_direction_hard": _uniform_validity_factor,
    "room_size": _uniform_validity_factor,
    "single_image_relative_direction": _uniform_validity_factor,
    "distance_cam_obj": _uniform_validity_factor,
    "depth_order_obj_obj": _depth_order_validity_factor,
    "position_cam_cam": _uniform_validity_factor,
    "elevation_cam_cam": _elevation_validity_factor,
    "visibility_compare": _visibility_compare_validity_factor,
    "position_cam_obj": _uniform_validity_factor,
    "position_cam_reg": _uniform_validity_factor,
    "motion_camera": _uniform_validity_factor,
    "attribute_measurement": _uniform_validity_factor,
}

def compute_validity_factor(question_type: str, is_valid: bool, gt: Any) -> float:
    q_type = str(question_type or "").strip()

    if not is_valid:
        return 0.0

    if q_type == "unknown":
        return 0.0

    scorer = TASK_VALIDITY_FACTOR_RULES.get(q_type)
    if scorer is None:
        return 1.0
    return float(scorer(gt))

def compute_format_factor(
    response_text: str,
    parsed_format_score: Any,
    question_count: int,
) -> float:
    try:
        base_score = float(parsed_format_score)
    except (TypeError, ValueError):
        base_score = 0.0
    base_score = max(0.0, min(base_score, 1.0))

    response = str(response_text or "")
    if not response.strip():
        return 0.0

    if question_count > 1:
        return 0.0

    return base_score

def compute_observation_reward(
    observation: str,
    q_type_str: str,
    raw_question: str,
    *,
    is_valid: bool,
    format_factor: float,
    ak="",
    model="gpt-oss-120b-ldm",
    url="https://example.com/v1",
    timeout=30,
    ) -> float:
    """Compute Observation Reward."""

    if not observation or observation.strip() == "":
        return 0.0
    if format_factor <= 0.0:
        return 0.0
    if not is_valid:
        return 0.0

    scoring_prompt = f"""You are evaluating the quality of a task-conditioned observation written by a 3D spatial reasoning model.

### OBSERVATION:
{observation}

### QUESTION TYPE CHOSEN:
{q_type_str}

### QUESTION GENERATED:
{raw_question}

### CONTEXT:
The Observation should preferably read like a grounded caption of what is visible, rather than just a short phrase or a single isolated fact.
The ideal Observation usually has a global-to-local flow:
1. first summarize the overall scene / image / image-pair layout,
2. then narrow to the local objects, region, or relation most relevant to the assigned task,
3. then naturally lead into the exact target of the generated question,
4. and, when useful, explain why a label is Unique, Non-Unique, or shared.
The Observation does not need to exhaustively describe every visible object, as long as the main visible layout and the task-relevant grounding are clear.

You are NOT grading question correctness directly. You are grading whether the Observation is a good visual-spatial lead-in for that question.
Judge the Observation holistically and semantically. Do NOT use any fixed word-count rule, clause-count rule, or keyword-count rule.
Do NOT require specific cue words such as left/right/front/back if the Observation is still clearly grounded and spatially informative in other wording.
Minor wording mismatch between the Observation and the Question is acceptable.
It is acceptable if the Observation already contains the final directional / visibility / numeric judgment, as long as it arises naturally from grounded visual observation and is not just a bare answer.

### YOUR TASK:
Evaluate the Observation quality based on these criteria together:
1. Groundedness: Is it a genuine observation of the visible content rather than a template or a restatement of the question?
2. Global caption quality: Does it capture the broader visible scene/image/pair before zooming in?
3. Local focus quality: Does it narrow to the task-relevant objects, region, relation, or camera change?
4. Transition quality: Does the Observation move naturally from the broader caption to the question target?
5. Support quality: After reading the Observation, does the generated question feel motivated and well-grounded?
6. Sample-specificity: Does the Observation sound tied to this sample, rather than a reusable template that would still fit many unrelated scenes after swapping object names?

### SCORING RULES:
Score 1.0 if most of the following are clearly true:
- the Observation is clearly grounded,
- gives a meaningful overall caption,
- then narrows to the task-relevant local target or relation,
- makes the final question feel natural and well-supported,
- and is clearly sample-specific rather than generic boilerplate.

Score 0.6 if:
- the Observation is grounded and useful,
- but one major component is weaker: the broader caption is thin, the local focus is vague, the transition to the question is incomplete, or the wording is somewhat generic.

Score 0.3 if:
- the Observation shows some real visual effort,
- but it is fragmentary, overly target-only, generic, list-like, weakly grounded, mostly a paraphrase of the question, or only loosely connected to the question.

Score 0.0 if:
- the Observation is empty, template-like, contradictory, hallucinated, or lacks meaningful visual-spatial content.

IMPORTANT:
- Do not penalize a reusable observation structure by itself. Lower the score only when the content is generic, weakly grounded, or could fit many samples with little meaningful change beyond swapping object names.
- A short observation or a focused observation should not be penalized by length alone. Lower the score only when it is too thin, weakly grounded, or not sufficiently supportive of the question.
- Do not over-penalize the Observation just because it already states the final judgment, if that judgment is naturally grounded in the visible content.
- Focus on whether the observation is visually grounded, structurally coherent, sample-specific, and genuinely useful for motivating the question. Do not demand exhaustive coverage of all visible details.

Think step by step about which score best fits before giving your final answer.

Output ONLY in this exact format, no extra text before or after:
<score>[1.0 / 0.6 / 0.3 / 0.0]</score>"""

    messages = [{"role": "user", "content": scoring_prompt}]
    client = OpenAI(api_key=ak, base_url=url)

    VALID_SCORES = {1.0, 0.6, 0.3, 0.0}
    max_retries = 5

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.0,
                timeout=timeout,
            )
            response_text = chat_completion.choices[0].message.content.strip()

            # Prefer the explicit XML score.
            score_match = re.search(r'<score>\s*([\d.]+)\s*</score>', response_text)
            if score_match:
                score = float(score_match.group(1).strip())
                if score in VALID_SCORES:
                    return score

            # Fallback: the last line is a valid score.
            trailing_match = re.search(r'(?:^|\n)\s*(1\.0|0\.6|0\.3|0\.0)\s*$', response_text)
            if trailing_match:
                return float(trailing_match.group(1))

            # Fallback: use the last valid score mention.
            all_matches = re.findall(r'\b(1\.0|0\.6|0\.3|0\.0)\b', response_text)
            if all_matches:
                # Keep the final mentioned score.
                return float(all_matches[-1])

        except ValueError:
            pass  # Invalid numeric parse.
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                print(f"Observation reward failed: {e}")

    return 0.0

def _resolve_observation_reward_max_workers(kwargs: dict[str, Any]) -> int:
    configured = kwargs.get(
        "observation_reward_max_workers",
        os.environ.get("OBSERVATION_REWARD_MAX_WORKERS", 16),
    )
    try:
        workers = int(configured)
    except (TypeError, ValueError):
        workers = 16
    return max(1, workers)

def compute_score(
    reward_inputs: list[RewardInput],
    **kwargs
) -> list[RewardScore]:
    """Compute Score."""
    
    # Stage A: collect batch-level type statistics.

    def _normalize(s):
        if not isinstance(s, str): return ""
        return re.sub(r'[\s_\[\]【】]+', '', s).lower()

    def get_canonical_type(input_type):
        """Get Canonical Type."""
        norm_input = _normalize(input_type)
        if not norm_input:
            return "unknown"

        for registry_type in SUPPORTED_TASK_TYPES:
            if norm_input == _normalize(registry_type):
                return registry_type

        for registry_type in SUPPORTED_TASK_TYPES:
            norm_registry = _normalize(registry_type)
            if norm_input in norm_registry or norm_registry in norm_input:
                return registry_type

        return "unknown"

    scores = []
    observation_reward_url = kwargs.get("service_url", kwargs.get("url", "https://example.com/v1"))
    observation_reward_timeout = kwargs.get("timeout", 30)
    observation_reward_model = kwargs.get("model", "gpt-oss-120b-ldm")
    observation_reward_api_key = kwargs.get("api_key", "")
    observation_reward_max_workers = _resolve_observation_reward_max_workers(kwargs)
    
    # Keep ordered task types for later pairing.
    current_batch_info = []
    current_batch_types = []
    
    for item in reward_inputs:
        c_type = get_canonical_type(str(item.get("question_type", "")))
        is_valid = bool(item.get("is_valid", False))
        
        current_batch_info.append((c_type, is_valid))
        current_batch_types.append(c_type)
 
    # Update global type-frequency statistics.
    global_counts, global_total = tracker.update_and_get_stats(current_batch_info)
    
    if global_total == 0:
        global_total = 1

    prepared_samples = []
    for item, canonical_type in zip(reward_inputs, current_batch_types):
        response_text = str(item.get("response", ""))
        observation = str(item.get("observation", ""))
        raw_question = str(item.get("question", ""))
        q_type_str = canonical_type
        is_valid = bool(item.get("is_valid", False))
        question_count = int(item.get("question_count", 0))
        has_invalid_inner_tags = bool(item.get("has_invalid_inner_tags", False))
        gt = item.get("gt", "")

        f_format = compute_format_factor(
            response_text=response_text,
            parsed_format_score=item.get("format_score", 0.0),
            question_count=question_count,
        )
        if has_invalid_inner_tags:
            f_format = 0.0

        f_valid = compute_validity_factor(
            question_type=q_type_str,
            is_valid=is_valid,
            gt=gt,
        )

        prepared_samples.append(
            {
                "response_text": response_text,
                "observation": observation,
                "raw_question": raw_question,
                "q_type_str": q_type_str,
                "is_valid": is_valid,
                "question_count": question_count,
                "has_invalid_inner_tags": has_invalid_inner_tags,
                "gt": gt,
                "f_format": f_format,
                "f_valid": f_valid,
            }
        )

    observation_scores = [0.0] * len(prepared_samples)
    observation_jobs = [
        idx for idx, sample in enumerate(prepared_samples)
        if sample["question_count"] <= 1
        and sample["observation"].strip()
        and sample["is_valid"]
        and not sample["has_invalid_inner_tags"]
        and sample["f_format"] > 0.0
    ]

    if observation_jobs:
        if observation_reward_max_workers > 1 and len(observation_jobs) > 1:
            max_workers = min(observation_reward_max_workers, len(observation_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        compute_observation_reward,
                        prepared_samples[idx]["observation"],
                        prepared_samples[idx]["q_type_str"],
                        prepared_samples[idx]["raw_question"],
                        is_valid=prepared_samples[idx]["is_valid"],
                        format_factor=prepared_samples[idx]["f_format"],
                        ak=observation_reward_api_key,
                        model=observation_reward_model,
                        url=observation_reward_url,
                        timeout=observation_reward_timeout,
                    ): idx
                    for idx in observation_jobs
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        observation_scores[idx] = float(future.result())
                    except Exception as exc:
                        print(f"Observation reward parallel call failed (idx={idx}): {exc}")
                        observation_scores[idx] = 0.0
        else:
            for idx in observation_jobs:
                sample = prepared_samples[idx]
                observation_scores[idx] = compute_observation_reward(
                    sample["observation"],
                    sample["q_type_str"],
                    sample["raw_question"],
                    is_valid=sample["is_valid"],
                    format_factor=sample["f_format"],
                    ak=observation_reward_api_key,
                    model=observation_reward_model,
                    url=observation_reward_url,
                    timeout=observation_reward_timeout,
                )

    # Stage B: combine format, validity, and observation rewards.
    for idx, sample in enumerate(prepared_samples):
        q_type_str = sample["q_type_str"]
        question_count = sample["question_count"]
        has_invalid_inner_tags = bool(sample["has_invalid_inner_tags"])
        f_format = sample["f_format"]
        f_valid = sample["f_valid"]
        f_observation = float(observation_scores[idx])
        if (
            sample["is_valid"]
            and sample["question_count"] <= 1
            and sample["observation"].strip()
            and not sample["has_invalid_inner_tags"]
            and sample["f_format"] > 0.0
        ):
            f_observation = max(VALID_OBSERVATION_FLOOR, f_observation)

        # Current weighting: 10% format, 90% observation gated by validity.
        task_quality = (f_format * 0.1) + ((f_valid * f_observation) * 0.9)
        final_reward = task_quality

        if question_count > 1 or has_invalid_inner_tags:
            final_reward = -1.0

        type_count = global_counts.get(q_type_str, 0)
        global_frequency = float(type_count) / float(global_total)

        scores.append({
            "overall": float(final_reward),
            "format": float(f_format),
            "valid_factor": float(f_valid),
            "f_observation": float(f_observation),
            "question_quality": float(task_quality),
            "global_type_count": float(type_count),
            "global_type_frequency": float(global_frequency),
            "invalid_inner_tags": 1.0 if has_invalid_inner_tags else 0.0,
            "accuracy": None
        })
    
    return scores
