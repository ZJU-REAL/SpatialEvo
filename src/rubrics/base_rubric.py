"""Base rubric definitions."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RubricStep:
    """One step in a rubric pipeline."""

    step_id: int
    tool_name: str
    description: str
    required_params: List[str]
    optional_params: Optional[List[str]] = None

    def __post_init__(self):
        if self.optional_params is None:
            self.optional_params = []


class BaseRubric(ABC):
    """Base class for rubric definitions."""

    def __init__(
        self,
        rubric_id: str,
        task_id: str,
        description: str,
        rubric_alias_ids: tuple[str, ...] = (),
        task_alias_ids: tuple[str, ...] = (),
    ):
        """Initialize the rubric."""
        self.rubric_id = rubric_id
        self.task_id = task_id
        self.description = description
        self.rubric_alias_ids = tuple(rubric_alias_ids)
        self.task_alias_ids = tuple(task_alias_ids)
        self.steps: List[RubricStep] = []
        self._build_steps()

    @abstractmethod
    def _build_steps(self):
        """Build the rubric step sequence."""
        pass

    def add_step(
        self,
        tool_name: str,
        description: str,
        required_params: List[str],
        optional_params: Optional[List[str]] = None
    ):
        """Append a step to the rubric."""
        step = RubricStep(
            step_id=len(self.steps) + 1,
            tool_name=tool_name,
            description=description,
            required_params=required_params,
            optional_params=optional_params
        )
        self.steps.append(step)

    def get_steps(self) -> List[RubricStep]:
        """Return all rubric steps."""
        return self.steps

    def get_step(self, step_id: int) -> Optional[RubricStep]:
        """Return a step by id."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_required_tools(self) -> List[str]:
        """Return tool names required by the rubric."""
        return [step.tool_name for step in self.steps]

    def validate_step_params(
        self,
        step_id: int,
        params: Dict[str, Any]
    ) -> bool:
        """Check whether all required parameters are present."""
        step = self.get_step(step_id)
        if step is None:
            return False

        return all(key in params for key in step.required_params)

    def get_info(self) -> Dict[str, Any]:
        """Return rubric metadata."""
        return {
            "rubric_id": self.rubric_id,
            "rubric_alias_ids": list(self.rubric_alias_ids),
            "task_id": self.task_id,
            "task_alias_ids": list(self.task_alias_ids),
            "description": self.description,
            "num_steps": len(self.steps),
            "required_tools": self.get_required_tools(),
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "required_params": step.required_params
                }
                for step in self.steps
            ]
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.rubric_id}', steps={len(self.steps)})"
