#!/usr/bin/env python3
"""Test Accuracy Reward."""

from accuracy_reward import compute_score, RewardInput, RewardScore

def test_extract_boxed_answer():
    """Test Extract Boxed Answer."""
    inputs = [
        {
            "response": "The answer is \\boxed{42}",
            "ground_truth": "42",
            "response_length": 10
        },
        {
            "response": "\\boxed{x+1} and \\boxed{y-2}",
            "ground_truth": "y-2",
            "response_length": 10
        },
    ]
    
    scores = compute_score(inputs)
    assert scores[0]["overall"] == 1.0, f"Expected 1.0, got {scores[0]['overall']}"
    assert scores[1]["overall"] == 1.0, f"Expected 1.0, got {scores[1]['overall']}"
    print("✓ Passed boxed-answer extraction test")

def test_extract_xml_answer():
    """Test Extract Xml Answer."""
    inputs = [
        {
            "response": "The answer is <answer>A</answer>",
            "ground_truth": "A",
            "response_length": 10
        },
        {
            "response": "<answer>B</answer>",
            "ground_truth": "B",
            "response_length": 10
        },
    ]
    
    scores = compute_score(inputs)
    assert scores[0]["overall"] == 1.0, f"Expected 1.0, got {scores[0]['overall']}"
    assert scores[1]["overall"] == 1.0, f"Expected 1.0, got {scores[1]['overall']}"
    print("✓ Passed XML answer extraction test")

def test_wrong_answer():
    """Test Wrong Answer."""
    inputs = [
        {
            "response": "The answer is \\boxed{42}",
            "ground_truth": "43",
            "response_length": 10
        },
    ]
    
    scores = compute_score(inputs)
    assert scores[0]["overall"] == 0.0, f"Expected 0.0, got {scores[0]['overall']}"
    print("✓ Passed incorrect-answer test")

def test_missing_answer():
    """Test Missing Answer."""
    inputs = [
        {
            "response": "I don't know the answer",
            "ground_truth": "42",
            "response_length": 10
        },
    ]
    
    scores = compute_score(inputs)
    assert scores[0]["overall"] == 0.0, f"Expected 0.0, got {scores[0]['overall']}"
    print("✓ Passed missing-answer extraction test")

def test_missing_ground_truth():
    """Test Missing Ground Truth."""
    inputs = [
        {
            "response": "The answer is \\boxed{42}",
            "ground_truth": "",
            "response_length": 10
        },
    ]
    
    scores = compute_score(inputs)
    assert scores[0]["overall"] == 0.0, f"Expected 0.0, got {scores[0]['overall']}"
    print("✓ Passed missing ground_truth test")

def test_normalize_answer():
    """Test Normalize Answer."""
    inputs = [
        {
            "response": "The answer is \\boxed{42}",
            "ground_truth": "42",
            "response_length": 10
        },
        {
            "response": "The answer is \\boxed{ 42 }",
            "ground_truth": "42",
            "response_length": 10
        },
        {
            "response": "The answer is \\boxed{42.}",
            "ground_truth": "42",
            "response_length": 10
        },
    ]
    
    scores = compute_score(inputs)
    assert scores[0]["overall"] == 1.0, f"Expected 1.0, got {scores[0]['overall']}"
    assert scores[1]["overall"] == 1.0, f"Expected 1.0, got {scores[1]['overall']}"
    # "42." "42" ground_truth "42" "42"
    # normalize_answer
    print("✓ Passed answer normalization test")

if __name__ == "__main__":
    print("Starting validation reward tests...")
    print()
    
    test_extract_boxed_answer()
    test_extract_xml_answer()
    test_wrong_answer()
    test_missing_answer()
    test_missing_ground_truth()
    test_normalize_answer()
    
    print()
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
