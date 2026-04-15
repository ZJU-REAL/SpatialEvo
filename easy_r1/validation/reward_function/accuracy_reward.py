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

"""Accuracy Reward."""

import re
from typing import Optional, TypedDict

# Optional exact math verification.
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("[Warning] math_verify not available, will use string comparison for math problems")

class RewardInput(TypedDict, total=False):
    """Reward Input."""
    response: str
    response_length: int
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str]  # Group id for GRPO.
    round: Optional[int]  # 1 for question generation, 2 for answer generation.

class RewardScore(TypedDict):
    """Reward Score."""
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

REWARD_NAME = "validation_accuracy"
REWARD_TYPE = "batch"  # Batch-level reward.

def extract_answer(response: str, pattern: str = r"\\boxed\{(.+?)\}|<answer>(.+?)</answer>") -> Optional[str]:
    """Extract Answer."""
    if not response:
        return None
    
    # Prefer `\\boxed{}` answers.
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, response)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Then fall back to `<answer>` tags.
    xml_pattern = r'<answer>(.+?)</answer>'
    xml_matches = re.findall(xml_pattern, response, re.DOTALL)
    if xml_matches:
        return xml_matches[-1].strip()
    
    # No answer found.
    return None

def normalize_answer(answer: Optional[str]) -> str:
    """Normalize Answer."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = answer.lower()
    answer = re.sub(r'[.,;:!?]$', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer

def compare_answers(predicted: str, ground_truth: str, use_math_verify: bool = True) -> bool:
    """Compare Answers."""
    if not predicted or not ground_truth:
        return False
    
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    if pred_normalized == gt_normalized:
        return True
    
    if use_math_verify and MATH_VERIFY_AVAILABLE:
        try:
            pred_parsed = parse(predicted)
            gt_parsed = parse(ground_truth)
            if verify(gt_parsed, pred_parsed):
                return True
        except Exception:
            pass
    
    return pred_normalized == gt_normalized

def compute_score(
    reward_inputs: list[RewardInput],
    answer_extraction_pattern: str = r"\\boxed\{(.+?)\}|<answer>(.+?)</answer>",
    use_math_verify: bool = True,
) -> list[RewardScore]:
    """Compute Score."""
    scores = []
    
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        
        # Missing labels default to zero reward.
        if not ground_truth:
            print(f"[Warning] Missing ground_truth for sample, returning 0.0")
            scores.append({
                "overall": 0.0,
                "format": None,
                "accuracy": 0.0,
            })
            continue
        
        predicted_answer = extract_answer(response, answer_extraction_pattern)
        
        # Missing extracted answers also get zero reward.
        if predicted_answer is None:
            scores.append({
                "overall": 0.0,
                "format": None,
                "accuracy": 0.0,
            })
            continue
        
        is_correct = compare_answers(predicted_answer, ground_truth, use_math_verify)
        
        accuracy_score = 1.0 if is_correct else 0.0
        scores.append({
            "overall": accuracy_score,
            "format": None,
            "accuracy": accuracy_score,
        })
    
    return scores
