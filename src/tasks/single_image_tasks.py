"""Single image tasks."""
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

class SingleImageRelativeDirectionTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="single_image_relative_direction",
            task_name="Single Image Relative Direction",
            description="Judge the target direction relative to the reference in one frame",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("vsi_single_image_relative_direction",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "ambiguity_checker", "single_image_relative_direction_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "scene_id", "image_path"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {"answer": "str (right/front-left/up-back-right/... )", "confidence": "float"}

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract two objects for relative direction in one image.\n"
            "Rule: answer means target relative to reference.\n"
            "Format: reference_object, target_object\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"reference_label": map_label(parts[0]), "target_label": map_label(parts[1])}

    def get_generation_difficulty_score(self) -> float:
        return 0.8

class DistanceCamObjTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="distance_cam_obj",
            task_name="Distance (Cam.-Obj.)",
            description="Measure camera distance to the target's nearest point",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("vsi_distance_cam_obj",),
        )

    def get_required_tools(self) -> List[str]:
        return ["ambiguity_checker", "distance_cam_obj_tool", "vlm_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "scene_id", "image_path"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {
            "answer": "float (meters)",
            "distance_m": "float",
            "distance_bucket": "str",
            "confidence": "float",
        }

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the object whose distance to the current camera is queried.\n"
            "Format: object_name\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        return {"target_label": map_label(raw_extraction.strip().lower())}

class DepthOrderObjObjTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="depth_order_obj_obj",
            task_name="Depth Order (Obj.-Obj.)",
            description="Judge which object is closer to the camera in one image",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("vsi_depth_order_obj_obj",),
        )

    def get_required_tools(self) -> List[str]:
        return ["ambiguity_checker", "depth_order_tool", "vlm_tool"]

    def get_required_input_keys(self) -> List[str]:
        return ["question", "scene_id", "image_path"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {
            "answer": "str (object1/object2/same)",
            "depth_1_m": "float",
            "depth_2_m": "float",
            "confidence": "float",
        }

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the two objects whose camera-depth order is compared.\n"
            "Format: object_1, object_2\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable[[str], str]) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"object1_label": map_label(parts[0]), "object2_label": map_label(parts[1])}

SINGLE_IMAGE_TASK_REGISTRY = {
    "single_image_relative_direction": SingleImageRelativeDirectionTask,
    "distance_cam_obj": DistanceCamObjTask,
    "depth_order_obj_obj": DepthOrderObjObjTask,
}
