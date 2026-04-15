"""Base task definitions."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable
from enum import Enum


class TaskDifficulty(Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BaseTask(ABC):
    """Base class for all task types."""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        description: str,
        difficulty: TaskDifficulty = TaskDifficulty.MEDIUM,
        task_alias_ids: tuple[str, ...] = (),
    ):
        """Initialize the task."""
        self.task_id = task_id
        self.task_name = task_name
        self.description = description
        self.difficulty = difficulty
        self.task_alias_ids = tuple(task_alias_ids)

    @abstractmethod
    def get_required_tools(self) -> List[str]:
        """Return required tool names."""
        pass

    @abstractmethod
    def get_expected_output_format(self) -> Dict[str, Any]:
        """Return the expected output schema."""
        pass

    def validate_input(self, input_data: Dict) -> bool:
        """Check whether all required inputs are present."""
        required_keys = self.get_required_input_keys()
        return all(key in input_data for key in required_keys)

    @abstractmethod
    def get_required_input_keys(self) -> List[str]:
        """Return required input keys."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return task metadata."""
        return {
            "task_id": self.task_id,
            "task_alias_ids": list(self.task_alias_ids),
            "task_name": self.task_name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "required_tools": self.get_required_tools(),
            "required_inputs": self.get_required_input_keys(),
            "output_format": self.get_expected_output_format(),
            "requires_llm_extraction": self.requires_llm_extraction()
        }

    def requires_llm_extraction(self) -> bool:
        """Return whether LLM-based parameter extraction is required."""
        return False

    def build_extraction_prompt(self, question: str) -> str:
        """Build the extraction prompt."""
        return ""

    def parse_extracted_params(
        self,
        raw_extraction: str,
        map_label: Callable[[str], str]
    ) -> Dict[str, Any]:
        """Parse extracted parameters into structured tool inputs."""
        return {}

    def get_generation_difficulty_score(self) -> float:
        """Return the generation difficulty score."""
        return 1.0

    def get_question_difficulty_score(
        self,
        answer: Any,
        parsed_params: Dict[str, Any]
    ) -> float:
        """Return the question difficulty score."""
        return 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.task_id}', name='{self.task_name}')"
