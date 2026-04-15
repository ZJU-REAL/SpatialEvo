"""Scene rubrics."""

from .base_rubric import BaseRubric

class ObjectCountingRubric(BaseRubric):
    """Object counting rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_object_counting",
            task_id="object_counting",
            description="Standard steps for object counting",
            rubric_alias_ids=("vsi_rubric_object_counting",),
            task_alias_ids=("vsi_object_counting",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract the target category",
            required_params=["question"],
            optional_params=["context"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether the target category is ambiguous",
            required_params=["scene_metadata", "labels"]
        )
        self.add_step(
            tool_name="object_count_tool",
            description="Count target-category objects in the scene",
            required_params=["scene_metadata", "target_category"],
            optional_params=["frame_metadata", "min_visibility"]
        )

class RelativeDistanceRubric(BaseRubric):
    """Relative distance rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_relative_distance",
            task_id="relative_distance",
            description="Standard steps for relative distance reasoning",
            rubric_alias_ids=("vsi_rubric_relative_distance",),
            task_alias_ids=("vsi_relative_distance",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract the target and candidates",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether any involved object is ambiguous",
            required_params=["scene_metadata", "labels"]
        )
        self.add_step(
            tool_name="relative_distance_tool",
            description="Find the candidate closest to the target",
            required_params=["scene_metadata", "target_label", "candidate_labels"]
        )

class RelativeDirectionRubric(BaseRubric):
    """Relative direction rubric."""

    def __init__(self, difficulty: str = "medium"):
        super().__init__(
            rubric_id=f"rubric_relative_direction_{difficulty}",
            task_id=f"relative_direction_{difficulty}",
            description=f"Standard steps for relative direction reasoning ({difficulty})",
            rubric_alias_ids=(f"vsi_rubric_relative_direction_{difficulty}",),
            task_alias_ids=(f"vsi_relative_direction_{difficulty}",),
        )
        self.difficulty = difficulty

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract positioning, orienting, and querying objects",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether the three objects are ambiguous",
            required_params=["scene_metadata", "labels"]
        )
        self.add_step(
            tool_name="relative_direction_tool",
            description="Compute the querying object's direction in the reference frame",
            required_params=["scene_metadata", "positioning_label", "orienting_label", "querying_label", "difficulty"]
        )

class ObjectSizeRubric(BaseRubric):
    """Object size rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_object_size",
            task_id="object_size",
            description="Standard steps for size measurement",
            rubric_alias_ids=("vsi_rubric_object_size",),
            task_alias_ids=("vsi_object_size",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract the target object",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether the target object is ambiguous",
            required_params=["scene_metadata", "labels"]
        )
        self.add_step(
            tool_name="object_size_tool",
            description="Measure the object's longest dimension",
            required_params=["scene_metadata", "object_label"]
        )

class AbsoluteDistanceRubric(BaseRubric):
    """Absolute distance rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_absolute_distance",
            task_id="absolute_distance",
            description="Standard steps for absolute distance measurement",
            rubric_alias_ids=("vsi_rubric_absolute_distance",),
            task_alias_ids=("vsi_absolute_distance",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Use the VLM to extract the two target objects",
            required_params=["question"]
        )
        self.add_step(
            tool_name="ambiguity_checker",
            description="Check whether either object is ambiguous",
            required_params=["scene_metadata", "labels"]
        )
        self.add_step(
            tool_name="absolute_distance_tool",
            description="Measure the straight-line distance between the objects",
            required_params=["scene_metadata", "object1_label", "object2_label"]
        )

class RoomSizeRubric(BaseRubric):
    """Room size rubric."""

    def __init__(self):
        super().__init__(
            rubric_id="rubric_room_size",
            task_id="room_size",
            description="Standard steps for room size measurement",
            rubric_alias_ids=("vsi_rubric_room_size",),
            task_alias_ids=("vsi_room_size",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="room_size_tool",
            description="Get room area from scene metadata",
            required_params=["scene_metadata"]
        )

SCENE_RUBRIC_REGISTRY = {
    "object_counting": ObjectCountingRubric,
    "relative_distance": RelativeDistanceRubric,
    "relative_direction_hard": lambda: RelativeDirectionRubric("hard"),
    "object_size": ObjectSizeRubric,
    "absolute_distance": AbsoluteDistanceRubric,
    "room_size": RoomSizeRubric,
}
