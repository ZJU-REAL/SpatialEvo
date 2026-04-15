"""Validator."""

from typing import Dict, Any, Optional
import re

class AnswerValidator:
    """Answer validator."""
    
    def __init__(self):
        self.validation_history = []
    
    def validate_question(
        self,
        question: str,
        task_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate question."""
        result = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        

        if len(question.strip()) < 10:
            result["is_valid"] = False
            result["issues"].append("Question is too short")
        

        if "?" not in question:
            result["issues"].append("Question is missing a question mark")
            result["suggestions"].append("Add a question mark")
        

        if "position" in task_type.lower():

            direction_words = ["left", "right", "front", "back", "behind", "where", "direction"]
            if not any(word in question.lower() for word in direction_words):
                result["issues"].append("Position task is missing direction-related wording")
        

        if "images" in context and not context["images"]:
            result["is_valid"] = False
            result["issues"].append("Required image input is missing")
        
        return result
    
    def validate_answer(
        self,
        answer: Any,
        expected_format: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Validate answer."""
        result = {
            "is_valid": True,
            "format_correct": True,
            "is_correct": None,
            "issues": []
        }
        

        if isinstance(answer, dict):

            if "answer" not in answer:
                result["format_correct"] = False
                result["issues"].append("Answer is missing the `answer` field")
        

        if ground_truth is not None:
            result["is_correct"] = self._compare_answers(answer, ground_truth)
        
        result["is_valid"] = result["format_correct"]
        
        return result
    
    def _compare_answers(self, answer: Any, ground_truth: Any) -> bool:
        """Compare answers."""

        if isinstance(answer, dict):
            answer_value = answer.get("answer", answer)
        else:
            answer_value = answer
        

        if isinstance(answer_value, str) and isinstance(ground_truth, str):
            answer_norm = answer_value.lower().strip()
            gt_norm = ground_truth.lower().strip()
            return answer_norm == gt_norm
        

        if isinstance(answer_value, (int, float)) and isinstance(ground_truth, (int, float)):
            return abs(answer_value - ground_truth) < 1e-3
        

        if isinstance(answer_value, list) and isinstance(ground_truth, list):
            return answer_value == ground_truth
        
        return answer_value == ground_truth
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = len(self.validation_history)
        if total == 0:
            return {"total": 0}
        
        valid_count = sum(1 for v in self.validation_history if v.get("is_valid", False))
        correct_count = sum(1 for v in self.validation_history if v.get("is_correct", False))
        
        return {
            "total": total,
            "valid_count": valid_count,
            "correct_count": correct_count,
            "validity_rate": valid_count / total if total > 0 else 0,
            "accuracy_rate": correct_count / total if total > 0 else 0
        }
