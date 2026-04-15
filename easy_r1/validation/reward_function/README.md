# Validation Reward Function

## Overview

The validation reward function is used for validation-set evaluation before or during PPO training. It compares the model response with the ground-truth answer and returns a reward when they match.

## Usage

### Example config

By default, validation and training use the same config path: `config.worker.reward`. If you want a separate validation reward function, configure it explicitly:

```yaml
worker:
  reward:
    # Training reward function
    reward_function: ./training/reward_function/unified_reward.py:compute_score
    reward_function_kwargs:
      question_quality_kwargs:
        service_url: "https://example.com/v1"
      majority_correctness_kwargs:
        answer_extraction_pattern: "\\\\boxed\\{(.+?)\\}|<answer>(.+?)</answer>"

    # Validation reward function
    # The current main.py uses the same config.worker.reward for val_reward_fn.
    # Use a separate setup in code if validation must differ from training.
```

### Supported parameters

`accuracy_reward.py` supports:

- `answer_extraction_pattern`: regex used to extract the final answer  
  Default: `r"\\boxed\{(.+?)\}|<answer>(.+?)</answer>"`
- `use_math_verify`: whether to use the math verification backend  
  Default: `True`

### Supported answer formats

1. **LaTeX boxed**: `\boxed{42}`
2. **XML**: `<answer>42</answer>`
3. **Raw answer**: if extraction fails, the function returns `None` and the reward becomes `0.0`

### Comparison modes

1. **String match**: compare normalized strings directly
2. **Math verification**: if `use_math_verify=True` and `math_verify` is available, compare semantic equivalence instead

## File layout

```text
validation/
├── reward_function/
│   ├── __init__.py
│   ├── accuracy_reward.py
│   └── README.md
└── format_prompt/
    └── validation_prompt.jinja
```

## Notes

1. Validation data must include `ground_truth`, or the reward is `0.0`
2. If answer extraction fails, the reward is `0.0`
3. If `math_verify` is unavailable, the function falls back to string matching
4. Keep validation reward functions lightweight to avoid slowing training
