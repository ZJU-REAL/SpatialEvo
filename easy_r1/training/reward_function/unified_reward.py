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

"""Unified Reward."""

from typing import Optional, TypedDict
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from majority_correctness_reward import compute_score as majority_correctness_compute_score
from question_quality_reward import compute_score as question_quality_compute_score

REWARD_NAME = "unified_reward"
REWARD_TYPE = "batch"  # Batch-level reward.

class RewardInput(TypedDict, total=False):
    """Reward Input."""
    response: str
    response_length: int  # Token count.
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str]
    round: Optional[int]  # 1 for question generation, 2 for answer generation.

class RewardScore(TypedDict):
    """Reward Score."""
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

def compute_score(
    reward_inputs: list[RewardInput],
    question_quality_kwargs: Optional[dict] = None,
    majority_correctness_kwargs: Optional[dict] = None,
    **kwargs
) -> list[RewardScore]:
    """Compute Score."""
    if not reward_inputs:
        raise ValueError("Input list is empty")

    # Require a shared round value across the batch.
    round_values = [inp.get("round", 1) for inp in reward_inputs]
    unique_rounds = set(round_values)

    if len(unique_rounds) == 0:
        raise ValueError("Cannot determine the round value because the input list is empty")
    
    if len(unique_rounds) > 1:
        raise ValueError(
            f"Input contains mixed round values: {unique_rounds}. "
            f"All items in one batch must share the same round. "
            f"Call this function separately for Round1 and Round2."
        )
    
    round_value = unique_rounds.pop()
    if round_value not in (1, 2):
        raise ValueError(
            f"round must be 1 or 2, but got {round_value}. "
            f"1=question generation, 2=answer generation."
        )

    # Dispatch to the round-specific reward.
    if round_value == 1:
        scores = question_quality_compute_score(
            reward_inputs,
            **(question_quality_kwargs or {})
        )
    elif round_value == 2:
        scores = majority_correctness_compute_score(
            reward_inputs,
            **(majority_correctness_kwargs or {})
        )
    else:
        raise RuntimeError(f"Unexpected round value: {round_value}")

    # Verify output size.
    if len(scores) != len(reward_inputs):
        raise RuntimeError(
            f"Returned score count ({len(scores)}) does not match input count ({len(reward_inputs)})"
        )

    # Verify required score fields.
    for i, score in enumerate(scores):
        assert isinstance(score, dict), f"Score must be a dict, but got {type(score)}"
        assert "overall" in score, f"Score {i} is missing the overall field"
        assert isinstance(score["overall"], (int, float)), f"Score {i} overall must be numeric"
        # `format` and `accuracy` may be `None`.
        assert "format" in score, f"Score {i} is missing the format field"
        assert "accuracy" in score, f"Score {i} is missing the accuracy field"
    
    return scores
