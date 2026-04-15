"""Image pair rubrics."""
from .base_rubric import BaseRubric

class PositionCamCamRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_position_cam_cam",
            task_id="position_cam_cam",
            description="Camera-to-camera direction reasoning",
            rubric_alias_ids=("mmsi_rubric_position_cam_cam",),
            task_alias_ids=("mmsi_position_cam_cam",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="camera_pair_tool",
            description="Extract both cameras and compute relative pose",
            required_params=["scene_data_path", "frame_id_1", "frame_id_2"],
            optional_params=["image_paths"],
        )
        self.add_step(
            tool_name="spatial_relation_tool",
            description="Compute camera2 direction relative to camera1",
            required_params=["entity1", "entity2"],
            optional_params=["reference_frame"],
        )

class PositionCamObjRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_position_cam_obj",
            task_id="position_cam_obj",
            description="Camera-to-object direction reasoning",
            rubric_alias_ids=("mmsi_rubric_position_cam_obj",),
            task_alias_ids=("mmsi_position_cam_obj",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Extract the target object from the question",
            required_params=["question"],
        )
        self.add_step(
            tool_name="camera_pair_tool",
            description="Extract camera parameters for both images",
            required_params=["scene_data_path", "frame_id_1", "frame_id_2"],
            optional_params=["image_paths"],
        )
        self.add_step(
            tool_name="object_detection_tool",
            description="Locate the target object in frame 2",
            required_params=["scene_metadata", "frame_metadata", "target_objects"],
        )
        self.add_step(
            tool_name="spatial_relation_tool",
            description="Compute the target direction relative to camera2",
            required_params=["entity1", "entity2"],
            optional_params=["reference_frame"],
        )

class VisibilityCompareRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_visibility_compare",
            task_id="visibility_compare",
            description="Compare target visibility across two images",
            rubric_alias_ids=("mmsi_rubric_visibility_compare",),
            task_alias_ids=("mmsi_visibility_compare",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Extract the target object for visibility comparison",
            required_params=["question"],
        )
        self.add_step(
            tool_name="visibility_compare_tool",
            description="Compare target visibility in both images",
            required_params=["frame_metadata_1", "frame_metadata_2", "target_label"],
            optional_params=["min_visibility"],
        )

class PositionCamRegRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_position_cam_reg",
            task_id="position_cam_reg",
            description="Camera-to-region direction reasoning",
            rubric_alias_ids=("mmsi_rubric_position_cam_reg",),
            task_alias_ids=("mmsi_position_cam_reg",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Extract the target region from the question",
            required_params=["question"],
        )
        self.add_step(
            tool_name="camera_pair_tool",
            description="Extract camera parameters for both images",
            required_params=["scene_data_path", "frame_id_1", "frame_id_2"],
            optional_params=["image_paths"],
        )
        self.add_step(
            tool_name="region_anchor_tool",
            description="Map the region name to an anchor point",
            required_params=["region_positions", "region_name"],
        )
        self.add_step(
            tool_name="spatial_relation_tool",
            description="Compute the region direction relative to camera2",
            required_params=["entity1", "entity2"],
            optional_params=["reference_frame"],
        )

class MotionCameraRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_motion_camera",
            task_id="motion_camera",
            description="Camera motion reasoning",
            rubric_alias_ids=("mmsi_rubric_motion_camera",),
            task_alias_ids=("mmsi_motion_camera",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="camera_pair_tool",
            description="Compute translation and rotation trends between frames",
            required_params=["scene_data_path", "frame_id_1", "frame_id_2"],
            optional_params=["image_paths"],
        )

class ElevationCamCamRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_elevation_cam_cam",
            task_id="elevation_cam_cam",
            description="Camera elevation comparison reasoning",
            rubric_alias_ids=("mmsi_rubric_elevation_cam_cam",),
            task_alias_ids=("mmsi_elevation_cam_cam",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="camera_pair_tool",
            description="Extract camera parameters for both images",
            required_params=["scene_data_path", "frame_id_1", "frame_id_2"],
            optional_params=["image_paths"],
        )
        self.add_step(
            tool_name="camera_elevation_tool",
            description="Compare target camera elevation against the reference",
            required_params=["camera_entity_1", "camera_entity_2"],
            optional_params=["camera_reference_image_idx", "camera_target_image_idx"],
        )

class AttributeMeasurementRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_attribute_measurement",
            task_id="attribute_measurement",
            description="Attribute measurement and comparison reasoning",
            rubric_alias_ids=("mmsi_rubric_attribute_measurement",),
            task_alias_ids=("mmsi_attribute_measurement",),
        )

    def _build_steps(self):
        self.add_step(
            tool_name="vlm_tool",
            description="Extract the two objects to compare",
            required_params=["question"],
        )
        self.add_step(
            tool_name="object_detection_tool",
            description="Locate the two comparison objects",
            required_params=["scene_metadata", "frame_metadata", "target_objects"],
        )
        self.add_step(
            tool_name="measurement_tool",
            description="Compare the objects' longest dimensions",
            required_params=["measurement_type", "entity1", "entity2"],
        )

IMAGE_PAIR_RUBRIC_REGISTRY = {
    "position_cam_cam": PositionCamCamRubric,
    "elevation_cam_cam": ElevationCamCamRubric,
    "position_cam_obj": PositionCamObjRubric,
    "visibility_compare": VisibilityCompareRubric,
    "position_cam_reg": PositionCamRegRubric,
    "motion_camera": MotionCameraRubric,
    "attribute_measurement": AttributeMeasurementRubric,
}
