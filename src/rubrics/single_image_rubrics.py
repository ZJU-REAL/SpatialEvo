"""Single image rubrics."""

from .base_rubric import BaseRubric

class SingleImageRelativeDirectionRubric(BaseRubric):
    """Single image relative direction rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_single_image_relative_direction",
            task_id="single_image_relative_direction",
            description="Single-image object-to-object direction reasoning",
            rubric_alias_ids=("vsi_rubric_single_image_relative_direction",),
            task_alias_ids=("vsi_single_image_relative_direction",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract the reference and target objects",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether the reference and target are uniquely visible",
            required_params=["scene_metadata", "labels"],
            optional_params=["frame_metadata", "min_visibility"]
        )
        self.add_step(
            tool_name="single_image_relative_direction_tool",
            description="Compute the target direction relative to the reference",
            required_params=["frame_metadata", "reference_label", "target_label"],
            optional_params=["min_visibility"]
        )

class DistanceCamObjRubric(BaseRubric):
    """Distance cam obj rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_distance_cam_obj",
            task_id="distance_cam_obj",
            description="Single-image camera-to-object distance measurement",
            rubric_alias_ids=("vsi_rubric_distance_cam_obj",),
            task_alias_ids=("vsi_distance_cam_obj",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract the target object",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether the target is uniquely visible in the frame",
            required_params=["scene_metadata", "labels"],
            optional_params=["frame_metadata", "min_visibility"]
        )
        self.add_step(
            tool_name="distance_cam_obj_tool",
            description="Measure the camera distance to the target's nearest point",
            required_params=["scene_id", "frame_id", "frame_metadata", "target_label"],
            optional_params=["min_visibility"]
        )

class DepthOrderObjObjRubric(BaseRubric):
    """Depth order obj obj rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_depth_order_obj_obj",
            task_id="depth_order_obj_obj",
            description="Single-image depth order reasoning",
            rubric_alias_ids=("vsi_rubric_depth_order_obj_obj",),
            task_alias_ids=("vsi_depth_order_obj_obj",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract two target objects",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether both objects are uniquely visible",
            required_params=["scene_metadata", "labels"],
            optional_params=["frame_metadata", "min_visibility"]
        )
        self.add_step(
            tool_name="depth_order_tool",
            description="Compare both objects' depth in camera coordinates",
            required_params=["frame_metadata", "object1_label", "object2_label"],
            optional_params=["min_visibility"]
        )

SINGLE_IMAGE_RUBRIC_REGISTRY = {
    "single_image_relative_direction": SingleImageRelativeDirectionRubric,
    "distance_cam_obj": DistanceCamObjRubric,
    "depth_order_obj_obj": DepthOrderObjObjRubric,
}
