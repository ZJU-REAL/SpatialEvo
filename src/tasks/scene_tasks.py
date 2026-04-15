"""Scene tasks."""
from typing import Dict, List, Any, Callable
from .base_task import BaseTask, TaskDifficulty

STRICT_EXTRACTION_NOTICE = (
    "Role: Expert Entity Extractor for 3D Scenes.\n"
    "Instruction: Extract objects from the question exactly as they appear.\n"
    "Rules:\n"
    "- Use lowercase and singular forms.\n"
    "- Keep original spelling; do not paraphrase or replace with synonyms.\n"
    "- If missing or uncertain, output null.\n"
    "- Output only the extraction result in required format.\n"
)

class ObjectCountingTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="object_counting",
            task_name="Object Counting",
            description="Count objects of a target category in the room",
            difficulty=TaskDifficulty.EASY,
            task_alias_ids=("vsi_object_counting",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "ambiguity_checker", "object_count_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "images", "scene_data", "target_category"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {"answer": "int", "confidence": "float"}

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the object category being counted.\n"
            "Format: category_name\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        return {"target_category": map_label(raw_extraction.strip().lower())}

    def get_generation_difficulty_score(self) -> float:
        return 0.7

    def get_question_difficulty_score(self, answer: Any, parsed_params: Dict[str, Any]) -> float:
        try:
            value = int(str(answer).strip().split()[0])
        except (TypeError, ValueError, IndexError):
            return 1.0
        return 0.5 if value in (0, 1) else 1.0

class RelativeDistanceTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="relative_distance",
            task_name="Relative Distance",
            description="Judge which object is closest to the target",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("vsi_relative_distance",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "ambiguity_checker", "relative_distance_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "images", "scene_data", "candidates", "target"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {"answer": "str", "distances": "dict", "confidence": "float"}

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Identify the reference target and candidate objects.\n"
            "Format: reference_target; candidate_1, candidate_2, candidate_3\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        parts = raw_extraction.strip().lower().split(";")
        if len(parts) < 2:
            return {}
        target = map_label(parts[0].strip())
        candidates = [map_label(c.strip()) for c in parts[1].split(",") if c.strip()]
        return {"target_label": target, "candidate_labels": candidates}

class RelativeDirectionTask(BaseTask):
    def __init__(self, difficulty_level: str = "medium"):
        difficulty_map = {
            "easy": TaskDifficulty.EASY,
            "medium": TaskDifficulty.MEDIUM,
            "hard": TaskDifficulty.HARD,
        }
        super().__init__(
            task_id=f"relative_direction_{difficulty_level}",
            task_name=f"Relative Direction ({difficulty_level.capitalize()})",
            description=f"Judge relative direction between objects ({difficulty_level})",
            difficulty=difficulty_map.get(difficulty_level, TaskDifficulty.MEDIUM),
            task_alias_ids=(f"vsi_relative_direction_{difficulty_level}",),
        )
        self.difficulty_level = difficulty_level

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "ambiguity_checker", "relative_direction_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "images", "scene_data", "positioning_object", "orienting_object", "querying_object"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        if self.difficulty_level == "easy":
            return {"answer": "str (left/right)"}
        if self.difficulty_level == "medium":
            return {"answer": "str (left/right/back)"}
        return {"answer": "str (front-left/front-right/back-left/back-right)"}

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Identify standing object, facing object and querying object.\n"
            "Format: standing_object, facing_object, querying_object\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 3:
            return {}
        return {
            "positioning_label": map_label(parts[0]),
            "orienting_label": map_label(parts[1]),
            "querying_label": map_label(parts[2]),
            "difficulty": self.difficulty_level,
        }

class ObjectSizeTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="object_size",
            task_name="Object Size",
            description="Measure an object's longest dimension",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("vsi_object_size",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "ambiguity_checker", "object_size_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "images", "scene_data", "target_object"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {"answer": "float (cm)", "dimensions": "dict", "confidence": "float"}

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the object whose size is queried.\n"
            "Format: object_name\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        return {"object_label": map_label(raw_extraction.strip().lower())}

class AbsoluteDistanceTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="absolute_distance",
            task_name="Absolute Distance",
            description="Measure the straight-line distance between two objects",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("vsi_absolute_distance",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "ambiguity_checker", "absolute_distance_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "images", "scene_data", "object1", "object2"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {"answer": "float (meters)", "confidence": "float"}

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the two objects used for absolute distance measurement.\n"
            "Format: object_1, object_2\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"object1_label": map_label(parts[0]), "object2_label": map_label(parts[1])}

class RoomSizeTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="room_size",
            task_name="Room Size",
            description="Compute the room area",
            difficulty=TaskDifficulty.EASY,
            task_alias_ids=("vsi_room_size",),
        )

    def get_required_tools(self) -> List[str]:
        return ["room_size_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "images", "scene_data"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {"answer": "float (square meters)", "dimensions": "dict", "confidence": "float"}

    def get_generation_difficulty_score(self) -> float:
        return 0.6

SCENE_TASK_REGISTRY = {
    "object_counting": ObjectCountingTask,
    "relative_distance": RelativeDistanceTask,
    "relative_direction_hard": lambda: RelativeDirectionTask("hard"),
    "object_size": ObjectSizeTask,
    "absolute_distance": AbsoluteDistanceTask,
    "room_size": RoomSizeTask,
}
