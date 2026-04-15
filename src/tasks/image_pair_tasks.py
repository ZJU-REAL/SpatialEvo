"""Image pair tasks."""

from typing import Dict, List, Any, Callable
from .base_task import BaseTask, TaskDifficulty

STRICT_EXTRACTION_NOTICE = (
    "Role: Expert Entity Extractor for spatial reasoning.\n"
    "Instruction: Extract entities exactly as asked in question.\n"
    "Rules:\n"
    "- output lowercase\n"
    "- keep nouns only\n"
    "- if uncertain output null\n"
    "- output only the required format\n"
)

class _ImagePairBaseTask(BaseTask):
    """Image pair base task."""

    def get_required_input_keys(self) -> List[str]:
        return ["question", "scene_id", "image_paths"]

    def get_expected_output_format(self) -> Dict[str, Any]:
        return {
            "answer": "str",
            "confidence": "float",
            "reasoning": "str",
        }

class PositionCamCamTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="position_cam_cam",
            task_name="Position (Cam.-Cam.)",
            description="Judge Image2 camera position relative to Image1",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("mmsi_position_cam_cam",),
        )

    def get_required_tools(self) -> List[str]:
        return ["camera_pair_tool", "spatial_relation_tool"]

class PositionCamObjTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="position_cam_obj",
            task_name="Position (Cam.-Obj.)",
            description="Judge object direction relative to the current camera",
            difficulty=TaskDifficulty.HARD,
            task_alias_ids=("mmsi_position_cam_obj",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "camera_pair_tool", "object_detection_tool", "spatial_relation_tool"]

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: extract target object relative to camera.\n"
            "Format: target_object\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(
        self,
        raw_extraction: str,
        map_label: Callable[[str], str]
    ) -> Dict[str, Any]:
        target = map_label(raw_extraction.strip().lower())
        return {"target_label": target}

class VisibilityCompareTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="visibility_compare",
            task_name="Visibility Compare",
            description="Compare one object's visibility across two images",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("mmsi_visibility_compare",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "visibility_compare_tool"]

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: extract the object whose visibility is compared across the two images.\n"
            "Format: target_object\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(
        self,
        raw_extraction: str,
        map_label: Callable[[str], str]
    ) -> Dict[str, Any]:
        target = map_label(raw_extraction.strip().lower())
        return {"target_label": target}

class PositionCamRegTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="position_cam_reg",
            task_name="Position (Cam.-Reg.)",
            description="Judge region direction relative to the current camera",
            difficulty=TaskDifficulty.HARD,
            task_alias_ids=("mmsi_position_cam_reg",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "camera_pair_tool", "region_anchor_tool", "spatial_relation_tool"]

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: extract the target region phrase relative to the camera.\n"
            "Important:\n"
            "- If the question asks about an abstract region such as sleeping area, bathroom area, kitchen area, or living area, keep that region phrase verbatim.\n"
            "- Do NOT replace the region phrase with an object label.\n"
            "- For questions like 'Where is X relative to the camera in Image 2?', extract X.\n"
            "Format: region_name\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(
        self,
        raw_extraction: str,
        map_label: Callable[[str], str]
    ) -> Dict[str, Any]:
        return {"region_name": raw_extraction.strip().lower()}

class MotionCameraTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="motion_camera",
            task_name="Motion (Camera)",
            description="Judge camera rotation and translation direction",
            difficulty=TaskDifficulty.EASY,
            task_alias_ids=("mmsi_motion_camera",),
        )

    def get_required_tools(self) -> List[str]:
        return ["camera_pair_tool"]

class ElevationCamCamTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="elevation_cam_cam",
            task_name="Elevation (Cam.-Cam.)",
            description="Judge Image2 camera elevation change relative to Image1",
            difficulty=TaskDifficulty.EASY,
            task_alias_ids=("mmsi_elevation_cam_cam",),
        )

    def get_required_tools(self) -> List[str]:
        return ["camera_pair_tool", "camera_elevation_tool"]

class AttributeMeasurementTask(_ImagePairBaseTask):
    def __init__(self):
        super().__init__(
            task_id="attribute_measurement",
            task_name="Attribute (Measurement)",
            description="Compare the size attributes of two target entities",
            difficulty=TaskDifficulty.HARD,
            task_alias_ids=("mmsi_attribute_measurement",),
        )

    def get_required_tools(self) -> List[str]:
        return ["vlm_tool", "object_detection_tool", "measurement_tool"]

    def requires_llm_extraction(self) -> bool:
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: extract two objects for measurement comparison.\n"
            "Format: object1, object2\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(
        self,
        raw_extraction: str,
        map_label: Callable[[str], str]
    ) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {
            "object1_label": map_label(parts[0]),
            "object2_label": map_label(parts[1]),
        }

IMAGE_PAIR_TASK_REGISTRY = {
    "position_cam_cam": PositionCamCamTask,
    "elevation_cam_cam": ElevationCamCamTask,
    "position_cam_obj": PositionCamObjTask,
    "visibility_compare": VisibilityCompareTask,
    "position_cam_reg": PositionCamRegTask,
    "motion_camera": MotionCameraTask,
    "attribute_measurement": AttributeMeasurementTask,
}
