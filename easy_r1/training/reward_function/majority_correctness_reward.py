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
import numpy as np
import os
import re
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict
import time
from openai import OpenAI

class RewardInput(TypedDict, total=False):
    """Reward Input."""
    response: str
    response_length: int
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str]  # Group id for GRPO.
    round: Optional[int]  # 1 for question generation, 2 for answer generation.
    gt: str
    question_type: str
    is_valid: bool
    question: str
    round2_mode: str
    round2_task_reference: str
    sim_error: str
    sim_error_code: str
    sim_failure_stage: str
    sim_parsed_params_json: str
    sim_validation_result_json: str
    sim_judge_reference_json: str
    sim_result_json: str

class RewardScore(TypedDict):
    """Reward Score."""
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

REWARD_NAME = "answer_correctness"
REWARD_TYPE = "batch"  # Batch-level reward.

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

# Task groups defined by the answer-format hints.
DIRECTION_MATCH_TASK_TYPES = {
    "relative_direction_hard",
    "single_image_relative_direction",
    "position_cam_cam",
    "position_cam_obj",
    "position_cam_reg",
    "motion_camera",
}
COUNT_TASK_TYPES = {
    "object_counting",
}
METRIC_DISTANCE_TASK_TYPES = {
    "absolute_distance",
    "distance_cam_obj",
}
OBJECT_SIZE_TASK_TYPES = {
    "object_size",
}
ROOM_SIZE_TASK_TYPES = {
    "room_size",
}
EXACT_MATCH_TASK_TYPES = {
    "relative_distance",
    "depth_order_obj_obj",
    "elevation_cam_cam",
    "visibility_compare",
    "attribute_measurement",
}
LEGACY_EXACT_MATCH_TASK_TYPES = {"appearance_order"}
INVALID_REASON_SCORE_LEVELS = (0.0, 0.3, 0.6, 1.0)
ROUND2_FORMAT_WEIGHT = 0.1

def clean_text(text, exclude_chars=['\n', '\r']):
    text = "" if text is None else str(text)

    for char in exclude_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

DIRECTION_ALIAS_MAP = {
    "behind": "back",
    "backward": "back",
    "backwards": "back",
    "rear": "back",
    "forward": "front",
    "forwards": "front",
    "ahead": "front",
    "above": "up",
    "below": "down",
}
DIRECTION_BASE_TOKENS = {"left", "right", "front", "back", "up", "down"}
EXACT_ARTICLES = ("the ", "a ", "an ")
VISIBILITY_EQUIVALENTS = {
    "image1": "image1",
    "image 1": "image1",
    "image one": "image1",
    "the first image": "image1",
    "first image": "image1",
    "image2": "image2",
    "image 2": "image2",
    "image two": "image2",
    "the second image": "image2",
    "second image": "image2",
}
EXACT_EQUIVALENTS = {
    "same level": "same_level",
    "same-level": "same_level",
    "samelevel": "same_level",
}
NUMBER_WORD_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
NUMBER_WORD_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_WORD_SCALES = {
    "hundred": 100,
    "thousand": 1000,
}
NUMBER_PATTERN = re.compile(
    r"(?<![a-zA-Z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![a-zA-Z])"
)

def _normalize_exact_answer(text: str, q_type: str = "") -> str:
    normalized = clean_text(text)
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if q_type == "visibility_compare":
        return VISIBILITY_EQUIVALENTS.get(normalized, normalized)

    normalized = EXACT_EQUIVALENTS.get(normalized, normalized)
    for article in EXACT_ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article):].strip()
            break
    return normalized

@lru_cache(maxsize=256)
def _split_compound_direction_token(token: str) -> tuple[str, ...]:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return ()

    normalized = DIRECTION_ALIAS_MAP.get(normalized, normalized)
    if normalized in DIRECTION_BASE_TOKENS:
        return (normalized,)

    compact = normalized.replace("_", "").replace("-", "")
    if not compact:
        return ()

    candidates: list[tuple[str, ...]] = []

    def _dfs(remaining: str, parts: list[str]) -> None:
        if not remaining:
            candidates.append(tuple(parts))
            return

        for base in sorted(DIRECTION_BASE_TOKENS, key=len, reverse=True):
            if remaining.startswith(base):
                parts.append(base)
                _dfs(remaining[len(base):], parts)
                parts.pop()

    _dfs(compact, [])

    valid_candidates = [parts for parts in candidates if 1 <= len(parts) <= 3]
    if not valid_candidates:
        return ()

    valid_candidates.sort(key=lambda parts: (len(parts), parts))
    return valid_candidates[0]

def _normalize_direction_parts(text: str, q_type: str = "") -> tuple[str, ...]:
    raw = clean_text(text).replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-zA-Z]+", raw)
    normalized_parts: list[str] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "counter" and index + 1 < len(tokens) and tokens[index + 1] == "clockwise":
            token = "counterclockwise"
            index += 1
        elif token == "same" and index + 1 < len(tokens) and tokens[index + 1] == "level":
            token = "same_level"
            index += 1

        if q_type == "motion_camera":
            if token == "clockwise":
                token = "right"
            elif token == "counterclockwise":
                token = "left"

        parts = _split_compound_direction_token(token)
        if not parts:
            token = DIRECTION_ALIAS_MAP.get(token, token)
            if token in DIRECTION_BASE_TOKENS:
                normalized_parts.append(token)
        else:
            normalized_parts.extend(parts)

        index += 1

    return tuple(sorted(set(normalized_parts)))

def direction_reward(content, gt, q_type: str = ""):
    content_parts = _normalize_direction_parts(content, q_type=q_type)
    gt_parts = _normalize_direction_parts(gt, q_type=q_type)

    if not content_parts or not gt_parts:
        return 0.0

    return 1.0 if content_parts == gt_parts else 0.0

ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
GENERIC_XML_TAG_PATTERN = re.compile(r"<\s*/?\s*([A-Za-z][A-Za-z0-9_-]*)[^>]*>")

def _extract_answer_tag_matches(text: str) -> list[str]:
    matches = ANSWER_TAG_PATTERN.findall(str(text or ""))
    return [str(match or "").strip() for match in matches]

def _extract_last_answer_tag(text: str) -> Optional[str]:
    matches = _extract_answer_tag_matches(text)
    if not matches:
        return None
    answer = str(matches[-1] or "").strip()
    return answer or None

def _strip_answer_tags(text: str) -> str:
    return ANSWER_TAG_PATTERN.sub(" ", str(text or ""))

def _parse_round2_response(response: Any) -> dict[str, Any]:
    raw_text = str(response or "").strip()
    answer_tag_matches = _extract_answer_tag_matches(raw_text)
    answer_tag_count = len(answer_tag_matches)
    has_exactly_one_answer_tag = answer_tag_count == 1
    has_multiple_answer_tags = answer_tag_count > 1
    has_missing_answer_tag = answer_tag_count == 0
    tagged_answer = answer_tag_matches[-1] if has_exactly_one_answer_tag else None
    reasoning_text = _strip_answer_tags(raw_text).strip()
    xml_tags = [str(tag or "").strip().lower() for tag in GENERIC_XML_TAG_PATTERN.findall(raw_text)]
    non_answer_xml_tags = [tag for tag in xml_tags if tag != "answer"]
    has_other_xml_tags = bool(non_answer_xml_tags)

    if has_multiple_answer_tags or has_other_xml_tags:
        format_score = -1.0
    elif has_missing_answer_tag:
        format_score = 0.0
    else:
        format_score = 1.0

    extracted_answer = tagged_answer if has_exactly_one_answer_tag else ""
    explanation_text = reasoning_text if reasoning_text else raw_text

    return {
        "raw_text": raw_text,
        "format_score": float(format_score),
        "answer_text": str(extracted_answer or "").strip(),
        "explanation_text": str(explanation_text or "").strip(),
        "answer_tag_count": int(answer_tag_count),
        "has_exactly_one_answer_tag": bool(has_exactly_one_answer_tag),
        "has_missing_answer_tag": bool(has_missing_answer_tag),
        "has_multiple_answer_tags": bool(has_multiple_answer_tags),
        "has_other_xml_tags": bool(has_other_xml_tags),
        "non_answer_xml_tags": list(non_answer_xml_tags),
    }

def single_image_relative_direction_reward(content, gt):
    content_parts = _normalize_direction_parts(content, q_type="single_image_relative_direction")
    gt_parts = _normalize_direction_parts(gt, q_type="single_image_relative_direction")

    if not content_parts or not gt_parts:
        return 0.0

    if content_parts == gt_parts:
        return 1.0

    content_set = set(content_parts)
    gt_set = set(gt_parts)

    if len(gt_set) == 3 and len(content_set) == 2 and content_set.issubset(gt_set):
        return 1.0

    if len(gt_set) == 2 and len(content_set) == 1 and content_set.issubset(gt_set):
        return 0.1

    return 0.0

def to_float(text):
    if text is None:
        return None

    normalized_text = str(text).strip()
    match = NUMBER_PATTERN.search(normalized_text.replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    lowered = clean_text(text).replace("-", " ")
    tokens = [token for token in re.findall(r"[a-zA-Z]+", lowered) if token]
    if not tokens:
        return None

    total = 0
    current = 0
    consumed = 0
    for token in tokens:
        if token == "and":
            consumed += 1
            continue
        if token in NUMBER_WORD_UNITS:
            current += NUMBER_WORD_UNITS[token]
            consumed += 1
            continue
        if token in NUMBER_WORD_TENS:
            current += NUMBER_WORD_TENS[token]
            consumed += 1
            continue
        if token == "hundred":
            current = max(current, 1) * NUMBER_WORD_SCALES[token]
            consumed += 1
            continue
        if token == "thousand":
            total += max(current, 1) * NUMBER_WORD_SCALES[token]
            current = 0
            consumed += 1
            continue
        break

    if consumed > 0:
        return float(total + current)
    return None

def abs_dist_norm(pred, target):
    # Normalize absolute error by the target scale.
    return abs(pred - target) / max(target, 1e-9)

def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    if pred is None or target is None:
        return 0.0

    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, round(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())

def numeric_error_stats(content, solution_answer):
    pred = to_float(clean_text(content))
    target = to_float(clean_text(solution_answer))
    if pred is None or target is None:
        return None, None, None, None
    abs_err = abs(pred - target)
    rel_err = abs_dist_norm(pred, target)
    return pred, target, abs_err, rel_err

def mca_reward(content, solution_answer):
    content = _normalize_exact_answer(content)
    solution_answer = _normalize_exact_answer(solution_answer)
    return 1.0 if content == solution_answer else 0.0

def count_reward(content, solution_answer):
    pred, target, abs_err, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None or abs_err is None:
        return 0.0

    if abs_err <= 1e-6:
        return 1.0
    if abs_err <= 1.0:
        return 0.3
    if abs_err <= 2.0:
        return 0.1
    return 0.0

def metric_distance_reward(content, solution_answer):
    pred, target, _, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None:
        return 0.0
    return mean_relative_accuracy(pred, target)

def object_size_reward(content, solution_answer):
    pred, target, _, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None:
        return 0.0
    return mean_relative_accuracy(pred, target)

def room_size_reward(content, solution_answer):
    pred, target, _, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None:
        return 0.0
    return mean_relative_accuracy(pred, target)

def answer_correctness_score(content, gt, q_type):
    norm_q_type = str(q_type or "").strip()
    gt_text = "" if gt is None else str(gt)

    if "invalid" in gt_text.lower():
        gt_text = ''

    if norm_q_type == "single_image_relative_direction":
        return single_image_relative_direction_reward(content, gt_text)

    if norm_q_type in DIRECTION_MATCH_TASK_TYPES:
        return direction_reward(content, gt_text, q_type=norm_q_type)

    if norm_q_type in COUNT_TASK_TYPES:
        return count_reward(content, gt_text)

    if norm_q_type in METRIC_DISTANCE_TASK_TYPES:
        return metric_distance_reward(content, gt_text)

    if norm_q_type in OBJECT_SIZE_TASK_TYPES:
        return object_size_reward(content, gt_text)

    if norm_q_type in ROOM_SIZE_TASK_TYPES:
        return room_size_reward(content, gt_text)

    if norm_q_type in EXACT_MATCH_TASK_TYPES or norm_q_type in LEGACY_EXACT_MATCH_TASK_TYPES:
        return mca_reward(
            _normalize_exact_answer(content, q_type=norm_q_type),
            _normalize_exact_answer(gt_text, q_type=norm_q_type),
        )

    return mca_reward(
        _normalize_exact_answer(content, q_type=norm_q_type),
        _normalize_exact_answer(gt_text, q_type=norm_q_type),
    )

def _load_optional_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

def _resolve_invalid_reason_judge_max_workers(kwargs: dict[str, Any]) -> int:
    configured = kwargs.get(
        "invalid_reason_judge_max_workers",
        os.environ.get("INVALID_REASON_JUDGE_MAX_WORKERS", 16),
    )
    try:
        workers = int(configured)
    except (TypeError, ValueError):
        workers = 16
    return max(1, workers)

def _format_task_reference_for_judge(task_reference: Any) -> str:
    text = str(task_reference or "").strip()
    return text if text else "(No task reference provided.)"

def _render_simulator_reference_for_judge(
    *,
    sim_error_code: Any,
    sim_failure_stage: Any,
    sim_error: Any,
    sim_judge_reference_json: Any,
    sim_parsed_params_json: Any,
    sim_validation_result_json: Any,
    sim_result_json: Any,
) -> str:
    judge_reference = _load_optional_json(sim_judge_reference_json)
    parsed_params = _load_optional_json(sim_parsed_params_json)
    validation_result = _load_optional_json(sim_validation_result_json)

    if isinstance(judge_reference, dict):
        payload = judge_reference
    else:
        payload = {
            "error_code": str(sim_error_code or ""),
            "failure_stage": str(sim_failure_stage or ""),
            "error": str(sim_error or ""),
            "parsed_params": parsed_params or {},
            "validation_result": validation_result or {},
        }

    final_invalid_reason = payload.get("final_invalid_reason") or payload.get("error") or str(sim_error or "")
    error_code = payload.get("error_code") or str(sim_error_code or "")
    failure_stage = payload.get("failure_stage") or str(sim_failure_stage or "")
    expected_fields = payload.get("expected_extraction_fields", [])
    resolved_fields = payload.get("resolved_extraction_fields", [])
    unresolved_fields = payload.get("unresolved_extraction_fields", [])
    payload_parsed_params = payload.get("parsed_params", parsed_params or {})
    validation_issues = payload.get("validation_issues", [])
    validation_suggestions = payload.get("validation_suggestions", [])

    lines = [
        "These simulator signals are authoritative diagnostics from the deterministic validation pipeline.",
        "Judge the explanation by whether it matches the simulator's actual failure reason.",
        "",
        "[Final Verdict]",
        f"- Final invalid reason: {final_invalid_reason}",
        f"- Error code: {error_code}",
        f"- Failure stage: {failure_stage}",
        "",
        "[Grounding Summary]",
        f"- Expected extraction fields: {json.dumps(expected_fields, ensure_ascii=False)}",
        f"- Unresolved extraction fields: {json.dumps(unresolved_fields, ensure_ascii=False)}",
        f"- Resolved extraction fields: {json.dumps(resolved_fields, ensure_ascii=False)}",
        f"- Parsed params: {json.dumps(payload_parsed_params, ensure_ascii=False, sort_keys=True)}",
    ]
    if validation_issues:
        lines.extend([
            "",
            "[Validation Issues]",
            f"- Issues: {json.dumps(validation_issues, ensure_ascii=False)}",
        ])
        if validation_suggestions:
            lines.append(f"- Suggestions: {json.dumps(validation_suggestions, ensure_ascii=False)}")
    elif validation_suggestions:
        lines.extend([
            "",
            "[Validation Issues]",
            f"- Suggestions: {json.dumps(validation_suggestions, ensure_ascii=False)}",
        ])
    return "\n".join(lines)

def _snap_invalid_reason_score(value: float) -> float:
    return min(INVALID_REASON_SCORE_LEVELS, key=lambda candidate: abs(candidate - value))

def judge_invalid_reason_explanation(
    *,
    task_type: str,
    question: str,
    explanation: str,
    task_reference_text: str,
    sim_error_code: Any,
    sim_failure_stage: Any,
    sim_error: Any,
    sim_judge_reference_json: Any,
    sim_parsed_params_json: Any,
    sim_validation_result_json: Any,
    sim_result_json: Any,
    ak="",
    model="gpt-oss-120b-ldm",
    url="https://example.com/v1",
    timeout=30,
) -> float:
    explanation = str(explanation or "").strip()
    if not explanation:
        return 0.0

    simulator_reference = _render_simulator_reference_for_judge(
        sim_error_code=sim_error_code,
        sim_failure_stage=sim_failure_stage,
        sim_error=sim_error,
        sim_judge_reference_json=sim_judge_reference_json,
        sim_parsed_params_json=sim_parsed_params_json,
        sim_validation_result_json=sim_validation_result_json,
        sim_result_json=sim_result_json,
    )
    task_reference = _format_task_reference_for_judge(task_reference_text)

    judge_prompt = f"""You are judging whether a model correctly explained why a spatial reasoning question is invalid.

The simulator diagnostics below are authoritative deterministic ground truth. They come from the real validation pipeline, not from another language model.
Your job is to check whether the model explanation captures the simulator's actual failure reason.
Prefer the simulator's final conclusion and grounded extraction status over fluent but unsupported explanations.

### ASSIGNED TASK TYPE:
{task_type}

### ORIGINAL QUESTION:
{question}

### TASK REFERENCE:
{task_reference}

### SIMULATOR INVALID SIGNALS:
{simulator_reference}

### MODEL EXPLANATION:
{explanation}

### SCORING RUBRIC:
- 1.0: The explanation clearly identifies the simulator's main invalid reason and correctly grounds it in the task rules, grounded object pool, extraction failure, non-unique reference, or validation issue.
- 0.6: The explanation is mostly correct and captures the main failure, but misses an important supporting detail or states it too vaguely.
- 0.3: The explanation catches only part of the problem, or gives a generic reason that weakly overlaps with the simulator diagnostics.
- 0.0: The explanation is wrong, contradicts the simulator diagnostics, or fails to explain why the question is invalid.

Return ONLY one score from this set:
0
0.3
0.6
1.0
"""

    client = OpenAI(api_key=ak, base_url=url)
    messages = [{"role": "user", "content": judge_prompt}]
    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.0,
                timeout=timeout,
            )
            content = str(chat_completion.choices[0].message.content or "").strip()
            match = re.search(r"0\.3|0\.6|0(?:\.0+)?|1(?:\.0+)?", content)
            if match:
                return _snap_invalid_reason_score(float(match.group(0)))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Invalid-reason judge failed: {e}")

    return 0.0

def compute_score(
    reward_inputs: list[RewardInput],
    **kwargs
) -> list[RewardScore]:
    """Compute Score."""
    judge_url = kwargs.get("invalid_reason_judge_url", kwargs.get("service_url", kwargs.get("url", "https://example.com/v1")))
    judge_timeout = kwargs.get("invalid_reason_judge_timeout", kwargs.get("timeout", 30))
    judge_model = kwargs.get("invalid_reason_judge_model", kwargs.get("model", "gpt-oss-120b-ldm"))
    judge_api_key = kwargs.get("invalid_reason_judge_api_key", kwargs.get("api_key", ""))
    judge_max_workers = _resolve_invalid_reason_judge_max_workers(kwargs)

    sample_details = []
    invalid_jobs = []

    for idx, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        uid = reward_input.get("uid", f"unknown_{idx}")
        gt = reward_input.get("gt", "")
        q_type = reward_input.get("question_type", "")
        is_valid = bool(reward_input.get("is_valid", True))
        round2_mode = str(reward_input.get("round2_mode", "") or "")
        parsed_response = _parse_round2_response(response)
        response_format_score = float(parsed_response["format_score"])
        has_multiple_answer_tags = bool(parsed_response.get("has_multiple_answer_tags", False))
        has_missing_answer_tag = bool(parsed_response.get("has_missing_answer_tag", False))
        has_other_xml_tags = bool(parsed_response.get("has_other_xml_tags", False))
        valid_answer_hard_format_fail = bool(
            is_valid and (has_multiple_answer_tags or has_other_xml_tags)
        )
        if not is_valid:
            response_format_score = -1.0 if has_other_xml_tags else 0.0

        if is_valid:
            content = parsed_response["answer_text"]
            r_acc = answer_correctness_score(content, gt, q_type)
            sample_details.append({
                "uid": uid,
                "is_valid": True,
                "content": content,
                "r_acc": float(r_acc),
                "format_score": response_format_score,
                "invalid_reason_score": 0.0,
                "round2_mode": round2_mode or "answer",
                "has_multiple_answer_tags": has_multiple_answer_tags,
                "has_missing_answer_tag": has_missing_answer_tag,
                "has_other_xml_tags": has_other_xml_tags,
                "valid_answer_hard_format_fail": valid_answer_hard_format_fail,
            })
            continue

        explanation = parsed_response["raw_text"]
        sample_details.append({
            "uid": uid,
            "is_valid": False,
            "content": explanation,
            "r_acc": 0.0,
            "format_score": response_format_score,
            "invalid_reason_score": 0.0,
            "round2_mode": round2_mode or "invalid_explanation",
            "has_multiple_answer_tags": has_multiple_answer_tags,
            "has_missing_answer_tag": has_missing_answer_tag,
            "has_other_xml_tags": has_other_xml_tags,
            "valid_answer_hard_format_fail": False,
            "judge_payload": {
                "task_type": str(q_type or ""),
                "question": str(reward_input.get("question", "") or ""),
                "explanation": explanation,
                "task_reference_text": reward_input.get("round2_task_reference", ""),
                "sim_error_code": reward_input.get("sim_error_code", ""),
                "sim_failure_stage": reward_input.get("sim_failure_stage", ""),
                "sim_error": reward_input.get("sim_error", ""),
                "sim_judge_reference_json": reward_input.get("sim_judge_reference_json", ""),
                "sim_parsed_params_json": reward_input.get("sim_parsed_params_json", ""),
                "sim_validation_result_json": reward_input.get("sim_validation_result_json", ""),
                "sim_result_json": reward_input.get("sim_result_json", ""),
            }
        })
        if explanation and not has_multiple_answer_tags:
            invalid_jobs.append(len(sample_details) - 1)

    if invalid_jobs:
        if judge_max_workers > 1 and len(invalid_jobs) > 1:
            max_workers = min(judge_max_workers, len(invalid_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        judge_invalid_reason_explanation,
                        **sample_details[idx]["judge_payload"],
                        ak=judge_api_key,
                        model=judge_model,
                        url=judge_url,
                        timeout=judge_timeout,
                    ): idx
                    for idx in invalid_jobs
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        sample_details[idx]["invalid_reason_score"] = float(future.result())
                    except Exception as exc:
                        print(f"Invalid-reason reward parallel call failed (idx={idx}): {exc}")
                        sample_details[idx]["invalid_reason_score"] = 0.0
        else:
            for idx in invalid_jobs:
                try:
                    sample_details[idx]["invalid_reason_score"] = float(
                        judge_invalid_reason_explanation(
                            **sample_details[idx]["judge_payload"],
                            ak=judge_api_key,
                            model=judge_model,
                            url=judge_url,
                            timeout=judge_timeout,
                        )
                    )
                except Exception as exc:
                    print(f"Invalid-reason reward failed (idx={idx}): {exc}")
                    sample_details[idx]["invalid_reason_score"] = 0.0

    scores = []
    for detail in sample_details:
        if bool(detail.get("valid_answer_hard_format_fail", False)):
            scores.append({
                "overall": -1.0,
                "format": -1.0,
                "accuracy": 0.0,
                "answer_correctness": 0.0,
                "invalid_reason_score": 0.0,
                "valid_case": 1.0 if detail["is_valid"] else 0.0,
                "invalid_case": 0.0 if detail["is_valid"] else 1.0,
                "format_hard_penalty": 1.0,
            })
            continue

        r_acc = float(detail["r_acc"])
        format_score = float(detail.get("format_score", 0.0))
        invalid_reason_score = float(detail["invalid_reason_score"])
        content_reward = r_acc if detail["is_valid"] else invalid_reason_score
        overall_reward = (ROUND2_FORMAT_WEIGHT * format_score) + ((1.0 - ROUND2_FORMAT_WEIGHT) * content_reward)
        scores.append({
            "overall": overall_reward,
            "format": format_score,
            "accuracy": r_acc,
            "answer_correctness": r_acc,
            "invalid_reason_score": invalid_reason_score,
            "valid_case": 1.0 if detail["is_valid"] else 0.0,
            "invalid_case": 0.0 if detail["is_valid"] else 1.0,
            "format_hard_penalty": 0.0,
        })

    try:
        uid2contents = defaultdict(list)
        for d in sample_details:
            uid2contents[d["uid"]].append(d["content"])
        total_groups = len(uid2contents)
        uniq_ge2 = sum(1 for v in uid2contents.values() if len(set(v)) >= 2)
        uniq_ge3 = sum(1 for v in uid2contents.values() if len(set(v)) >= 3)
        diversity_metrics = {
            "diversity/num_groups": float(total_groups),
            "diversity/uniq_ge2_ratio": (float(uniq_ge2) / float(total_groups)) if total_groups > 0 else 0.0,
            "diversity/uniq_ge3_ratio": (float(uniq_ge3) / float(total_groups)) if total_groups > 0 else 0.0,
        }
        # The trainer reduces these extra metrics across workers.
        if scores:
            scores[0].update(diversity_metrics)
    except Exception:
        pass
    
    return scores
