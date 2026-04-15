"""Task-level runtime support for the deterministic geometric environment."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

VALID_LABELS = {
    "wall", "chair", "books", "floor", "door", "window", "table",
    "trash can", "pillow", "picture", "ceiling", "box", "doorframe", "monitor",
    "cabinet", "shelves", "towel", "sofa", "sink", "backpack", "lamp", "bed",
    "bookshelf", "mirror", "curtain", "plant", "whiteboard", "radiator", "shoes",
    "toilet", "paper bag", "clothes", "keyboard", "night stand",
    "tv", "dresser", "computer tower", "telephone", "refridgerator", "shower curtain",
    "bathtub", "microwave", "counter", "suitcase", "laptop", "printer",
    "fan", "blanket", "ceiling light", "clock", "floor mat", "trash bin", "basket",
    "paper", "person", "closet", "bucket", "dishwasher", "blinds", "guitar", "piano",
    "bowl", "oven", "washer", "kettle", "coat rack", "fireplace", "power strip",
    "headphones", "crate", "cutting board", "cushion", "shoe rack", "heater", "bag",
    "pan", "cup", "stool",
}

NON_ANCHOR_LABELS = {
    "wall",
    "otherprop",
    "ceiling",
    "otherfurniture",
}

TASK_ALIASES = {
    "relativedirection": "relative_direction_hard",
    "relative_direction": "relative_direction_hard",
    "vsi_relative_direction": "relative_direction_hard",
    "relativedirectioneasy": "relative_direction_hard",
    "relativedirectionmedium": "relative_direction_hard",
    "vsirelativedirectioneasy": "relative_direction_hard",
    "vsirelativedirectionmedium": "relative_direction_hard",
    "singleimagerelativedirection": "single_image_relative_direction",
    "single_image_direction": "single_image_relative_direction",
    "singleimagedirection": "single_image_relative_direction",
    "depthorder": "depth_order_obj_obj",
    "depthcomparison": "depth_order_obj_obj",
    "cameraposition": "position_cam_cam",
    "camerarelativeposition": "position_cam_cam",
    "positioncamcam": "position_cam_cam",
    "cameraobjectposition": "position_cam_obj",
    "positioncamobj": "position_cam_obj",
    "cameraregionposition": "position_cam_reg",
    "positioncamreg": "position_cam_reg",
    "cameramotion": "motion_camera",
    "motioncamera": "motion_camera",
    "objectsizecomparison": "attribute_measurement",
    "sizecomparison": "attribute_measurement",
    "attributemeasurement": "attribute_measurement",
    "visibilitycompare": "visibility_compare",
    "depthorderobjobj": "depth_order_obj_obj",
    "cameradistance": "distance_cam_obj",
    "distancetocamera": "distance_cam_obj",
    "distancefromcamera": "distance_cam_obj",
    "cameraobjectdistance": "distance_cam_obj",
    "objectcameradistance": "distance_cam_obj",
    "distancecamobj": "distance_cam_obj",
    "cameraelevation": "elevation_cam_cam",
    "cameraheight": "elevation_cam_cam",
    "elevationcamcam": "elevation_cam_cam",
}

LEGACY_ALIAS_MARKERS = ("vsi", "mmsi")

TASK_REJECTION_REASONS = {
    "positionobjobj": (
        "Unsupported `position_obj_obj`: cross-view object-object directions change with viewpoint, "
        "so the current DGE cannot provide stable deterministic geometric ground truth."
    ),
    "positionobjreg": (
        "Unsupported `position_obj_reg`: cross-view object-region directions change with viewpoint, "
        "so the current DGE cannot provide stable deterministic geometric ground truth."
    ),
    "positionregreg": (
        "Unsupported `position_reg_reg`: cross-view region-region directions change with viewpoint, "
        "so the current DGE cannot provide stable deterministic geometric ground truth."
    ),
}

EXTRACTION_DEFAULTS = {
    "object_counting": {"target_category": "null"},
    "object_size": {"object_label": "null"},
    "absolute_distance": {"object1_label": "null", "object2_label": "null"},
    "relative_distance": {"target_label": "null", "candidate_labels": []},
    "relative_direction_hard": {
        "positioning_label": "null",
        "orienting_label": "null",
        "querying_label": "null",
    },
    "single_image_relative_direction": {"reference_label": "null", "target_label": "null"},
    "position_cam_obj": {"target_label": "null"},
    "position_cam_reg": {"region_name": "null"},
    "attribute_measurement": {"object1_label": "null", "object2_label": "null"},
    "visibility_compare": {"target_label": "null"},
    "depth_order_obj_obj": {"object1_label": "null", "object2_label": "null"},
    "distance_cam_obj": {"target_label": "null"},
}

EXPECTED_EXTRACTION_FIELDS = {
    "object_counting": ["target_category"],
    "object_size": ["object_label"],
    "absolute_distance": ["object1_label", "object2_label"],
    "single_image_relative_direction": ["reference_label", "target_label"],
    "relative_direction_hard": ["positioning_label", "orienting_label", "querying_label"],
    "position_cam_obj": ["target_label"],
    "attribute_measurement": ["object1_label", "object2_label"],
    "position_cam_reg": ["region_name"],
    "visibility_compare": ["target_label"],
    "depth_order_obj_obj": ["object1_label", "object2_label"],
    "distance_cam_obj": ["target_label"],
}

REGION_OBJECT_ONTOLOGY = {
    "sleep": [
        "bed", "bed frame", "bed cover", "bed sheet", "bedsheet",
        "pillow", "blanket", "night stand", "nightstand",
        "bedside table", "bedside cabinet", "bedside shelf", "bedside lamp",
    ],
    "bedroom": [
        "bed", "bed frame", "bed cover", "bed sheet", "bedsheet",
        "pillow", "blanket", "night stand", "nightstand",
        "dresser", "bedside table", "bedside cabinet", "bedside shelf", "bedside lamp",
    ],
    "bath": [
        "toilet", "sink", "bathroom sink", "mirror", "bathtub", "bath tub",
        "shower curtain", "towel", "bath mat", "toilet paper", "toilet paper dispenser",
        "bathroom cabinet", "bathroom counter", "bathroom shelf",
    ],
    "toilet": [
        "toilet", "sink", "bathroom sink", "mirror", "shower curtain",
        "toilet paper", "toilet paper dispenser", "bathroom cabinet",
    ],
    "kitchen": [
        "refridgerator", "refrigerator", "microwave", "sink", "counter",
        "dishwasher", "kettle", "oven", "stove", "air fryer", "blender",
    ],
    "cooking": [
        "microwave", "oven", "stove", "counter", "kettle", "dishwasher",
        "air fryer", "blender",
    ],
    "living": [
        "sofa", "couch", "armchair", "tv", "tv monitor", "tv_monitor",
        "table", "coffee table", "floor mat", "fireplace", "speakers",
    ],
    "dining": ["table", "chair", "counter", "stool", "barstool", "bench", "bench stool"],
    "study": [
        "table", "chair", "whiteboard", "blackboard", "books",
        "bookshelf", "monitor", "keyboard", "laptop", "shelf",
    ],
    "office": [
        "table", "chair", "monitor", "keyboard", "computer tower",
        "printer", "laptop", "clock", "shelf",
    ],
    "work": [
        "table", "chair", "monitor", "keyboard", "computer tower",
        "printer", "laptop", "shelf",
    ],
    "print": ["printer", "table", "monitor", "computer tower", "binding machine"],
    "entry": ["door", "shoe rack", "shoes", "floor mat", "coat rack"],
    "storage": ["cabinet", "shelf", "shelves", "box", "closet", "basket", "crate", "bucket", "suitcase"],
    "laundry": ["washer", "washing machine", "towel", "basket", "bucket"],
    "media": ["tv", "tv monitor", "tv_monitor", "monitor", "speaker", "speakers", "headphones", "guitar", "piano"],
}

REGION_PHRASE_ALIASES = {
    "sleep": [
        "sleeping area",
        "sleep area",
        "bed area",
        "rest area",
        "resting area",
        "sleeping zone",
        "sleeping region",
    ],
    "bedroom": [
        "bedroom area",
        "bed room",
        "bed room area",
        "sleeping room",
    ],
    "bath": [
        "bathroom area",
        "bath area",
        "bathroom zone",
        "bathroom region",
        "washroom",
        "wash room",
        "restroom",
        "shower area",
        "shower room",
    ],
    "toilet": [
        "toilet area",
        "toilet zone",
        "wc",
        "lavatory",
    ],
    "kitchen": [
        "kitchen area",
        "kitchen zone",
        "cooking area",
        "cooking room",
        "food prep area",
        "food preparation area",
    ],
    "cooking": [
        "cook area",
        "cooking zone",
        "stove area",
        "prep area",
        "preparation area",
    ],
    "living": [
        "living area",
        "living room",
        "lounge area",
        "sitting area",
        "seating area",
        "sofa area",
    ],
    "dining": [
        "dining area",
        "dining room",
        "eating area",
        "meal area",
    ],
    "study": [
        "study area",
        "reading area",
        "reading corner",
        "desk area",
    ],
    "office": [
        "office area",
        "office corner",
        "workspace",
        "workstation",
        "work station",
    ],
    "work": [
        "work area",
        "working area",
        "work zone",
    ],
    "print": [
        "printing area",
        "printer area",
        "copy area",
        "copy corner",
    ],
    "entry": [
        "entryway",
        "entry area",
        "entrance area",
        "entrance zone",
        "foyer",
        "doorway area",
    ],
    "storage": [
        "storage area",
        "storage room",
        "storage zone",
        "closet area",
        "utility area",
    ],
    "laundry": [
        "laundry area",
        "laundry room",
        "washer area",
        "washing machine area",
    ],
    "media": [
        "tv area",
        "television area",
        "media area",
        "entertainment area",
        "audio area",
        "music area",
    ],
}

SUMMARY_TYPE_ALIASES = {
    "scene": "scene",
    "scene_summary": "scene",
    "single_image": "single_image",
    "single": "single_image",
    "image": "single_image",
    "frame": "single_image",
    "single_image_summary": "single_image",
    "image_pair": "multi_image",
    "imagepair": "multi_image",
    "pair": "multi_image",
    "pair_summary": "multi_image",
    "two_image": "multi_image",
    "two_images": "multi_image",
    "pair_image": "multi_image",
    "pair_images": "multi_image",
    "double_image": "multi_image",
    "double_images": "multi_image",
    "multi_image": "multi_image",
    "multi": "multi_image",
    "images": "multi_image",
    "frames": "multi_image",
    "multi_image_summary": "multi_image",
}

TASK_GROUP_ALIASES = {
    "scene": "scene",
    "scenetask": "scene",
    "vsi": "scene",
    "single": "single_image",
    "singleimage": "single_image",
    "single_image": "single_image",
    "image": "single_image",
    "pair": "image_pair",
    "mmsi": "image_pair",
    "imagepair": "image_pair",
    "image_pair": "image_pair",
    "twoimage": "image_pair",
    "twoimages": "image_pair",
    "two_image": "image_pair",
    "two_images": "image_pair",
}

INPUT_MODE_TO_TASK_GROUP = {
    "basic": "scene",
    "single_image": "single_image",
    "image_pair": "image_pair",
}

def normalize_task_group_name(raw_group: Any, input_mode: str = "basic") -> str:
    """Normalize public-facing task group names."""
    if isinstance(raw_group, str):
        normalized = "".join(ch for ch in raw_group.strip().lower() if ch.isalnum() or ch == "_")
        if normalized in TASK_GROUP_ALIASES:
            return TASK_GROUP_ALIASES[normalized]
    return INPUT_MODE_TO_TASK_GROUP.get(str(input_mode).strip().lower(), "scene")

def is_legacy_benchmark_name(raw_name: Any) -> bool:
    """Whether a public-facing alias/name still carries legacy benchmark wording."""
    if not isinstance(raw_name, str):
        return False
    normalized = "".join(ch for ch in raw_name.strip().lower() if ch.isalnum() or ch == "_")
    return any(marker in normalized for marker in LEGACY_ALIAS_MARKERS)

def get_extraction_defaults(task_type: str) -> Dict[str, Any]:
    """Return a fresh extraction-default dict for a task."""
    return deepcopy(EXTRACTION_DEFAULTS.get(task_type, {}))

def get_expected_extraction_fields(task_type: str) -> list[str]:
    """Return the required structured fields after extraction."""
    return list(EXPECTED_EXTRACTION_FIELDS.get(task_type, []))

@dataclass(frozen=True)
class LabelFieldRole:
    """Bind a context field to a human-readable role name."""

    field_name: str
    role_name: str

@dataclass(frozen=True)
class SameLabelConstraint:
    """Two fields must not resolve to the same object label."""

    left_field: str
    right_field: str
    error_message: str

@dataclass(frozen=True)
class DistinctFieldGroupConstraint:
    """All non-null labels in a field group must be distinct."""

    field_names: tuple[str, ...]
    error_message: str

@dataclass(frozen=True)
class ListFieldConstraint:
    """List-field constraints used by deterministic invalid-question checks."""

    list_field: str
    require_non_empty: bool = False
    dedupe_items: bool = False
    element_field: Optional[str] = None
    empty_error: str = ""
    duplicate_error: str = ""
    overlap_error: str = ""

@dataclass(frozen=True)
class DetectionRequirement:
    """A detected entity must exist after object detection."""

    detections_key: str = "detections"
    entity_key: str = "target_entity"
    label_field: str = "target_label"
    error_template: str = "{task_type} invalid: target object `{label}` was not detected in the input image, so relative direction cannot be computed"

@dataclass(frozen=True)
class ObjectDetectionPolicy:
    """Task-level policy for binding object detection tool parameters."""

    frame_source: str = "auto"
    include_scene_metadata: bool = True
    use_camera_location: bool = False
    target_label_fields: tuple[str, ...] = ()

@dataclass(frozen=True)
class SpatialRelationPolicy:
    """Task-level policy for binding spatial-relation tool parameters."""

    binding: str = "default"
    default_reference_frame: Optional[str] = None

@dataclass(frozen=True)
class HeuristicExtractionPolicy:
    """Profile-level heuristic extraction strategy."""

    strategy: str
    target_field: Optional[str] = None
    enable_relative_to_me_fallback: bool = False
    allow_abstract_region_subject: bool = False

@dataclass
class TaskProfile:
    """Single-source task profile for DGE extension and runtime behavior."""

    task_type: str
    task_group: str = ""
    family: str = "custom"
    input_mode: str = "basic"
    aliases: tuple[str, ...] = ()
    extraction_defaults: Dict[str, Any] = field(default_factory=dict)
    expected_extraction_fields: tuple[str, ...] = ()
    context_label_fields: tuple[str, ...] = ()
    candidate_label_scope: str = "prefer_frame_then_scene"
    heuristic_extractor: Optional[Callable[[str, list[str]], Dict[str, Any]]] = None
    heuristic_policy: Optional[HeuristicExtractionPolicy] = None
    allow_heuristic_without_candidates: bool = False
    same_label_constraints: tuple[SameLabelConstraint, ...] = ()
    distinct_field_group_constraints: tuple[DistinctFieldGroupConstraint, ...] = ()
    list_constraints: tuple[ListFieldConstraint, ...] = ()
    detection_requirement: Optional[DetectionRequirement] = None
    ambiguity_object_fields: tuple[LabelFieldRole, ...] = ()
    ambiguity_region_fields: tuple[LabelFieldRole, ...] = ()
    object_detection_policy: Optional[ObjectDetectionPolicy] = None
    detection_result_binding: str = "default"
    spatial_relation_policy: Optional[SpatialRelationPolicy] = None
    measurement_default: Optional[str] = None
    camera_pair_answer_mode: Optional[str] = None
    prefill_image_index_mode: Optional[str] = None
    image_index_parse_mode: Optional[str] = None
    region_anchor_infer_max_labels: int = 1

    def __post_init__(self):
        normalized_group = normalize_task_group_name(
            self.task_group or self.family,
            input_mode=self.input_mode,
        )
        self.task_group = normalized_group
        self.family = normalized_group

def _invert_task_aliases(task_aliases: Dict[str, str]) -> Dict[str, list[str]]:
    grouped: Dict[str, list[str]] = defaultdict(list)
    for alias, task_type in task_aliases.items():
        grouped[task_type].append(alias)
    return grouped

def build_default_task_profiles() -> Dict[str, TaskProfile]:
    """Build the built-in TaskProfile set for current DGE tasks."""
    aliases_by_task = _invert_task_aliases(TASK_ALIASES)

    return {
        "object_counting": TaskProfile(
            task_type="object_counting",
            task_group="scene",
            extraction_defaults={"target_category": "null"},
            expected_extraction_fields=("target_category",),
            context_label_fields=("target_category",),
            candidate_label_scope="scene_observed_counting",
            heuristic_policy=HeuristicExtractionPolicy(
                strategy="single_entity",
                target_field="target_category",
            ),
        ),
        "object_size": TaskProfile(
            task_type="object_size",
            task_group="scene",
            extraction_defaults={"object_label": "null"},
            expected_extraction_fields=("object_label",),
            context_label_fields=("object_label",),
            candidate_label_scope="scene_observed_unique",
            heuristic_policy=HeuristicExtractionPolicy(
                strategy="single_entity",
                target_field="object_label",
            ),
        ),
        "distance_cam_obj": TaskProfile(
            task_type="distance_cam_obj",
            task_group="single_image",
            input_mode="single_image",
            aliases=tuple(aliases_by_task.get("distance_cam_obj", [])),
            extraction_defaults={"target_label": "null"},
            expected_extraction_fields=("target_label",),
            context_label_fields=("target_label",),
            candidate_label_scope="frame_only",
            heuristic_policy=HeuristicExtractionPolicy(
                strategy="single_entity",
                target_field="target_label",
            ),
        ),
        "absolute_distance": TaskProfile(
            task_type="absolute_distance",
            task_group="scene",
            extraction_defaults={"object1_label": "null", "object2_label": "null"},
            expected_extraction_fields=("object1_label", "object2_label"),
            context_label_fields=("object1_label", "object2_label"),
            candidate_label_scope="scene_observed_unique",
            heuristic_policy=HeuristicExtractionPolicy(strategy="pair_measurement"),
            same_label_constraints=(
                SameLabelConstraint(
                    "object1_label",
                    "object2_label",
                    "absolute_distance invalid: `object1_label` and `object2_label` must refer to different objects",
                ),
            ),
        ),
        "depth_order_obj_obj": TaskProfile(
            task_type="depth_order_obj_obj",
            task_group="single_image",
            input_mode="single_image",
            aliases=tuple(aliases_by_task.get("depth_order_obj_obj", [])),
            extraction_defaults={"object1_label": "null", "object2_label": "null"},
            expected_extraction_fields=("object1_label", "object2_label"),
            context_label_fields=("object1_label", "object2_label"),
            candidate_label_scope="frame_only",
            heuristic_policy=HeuristicExtractionPolicy(strategy="pair_measurement"),
            same_label_constraints=(
                SameLabelConstraint(
                    "object1_label",
                    "object2_label",
                    "depth_order_obj_obj invalid: `object1_label` and `object2_label` must refer to different objects",
                ),
            ),
        ),
        "relative_distance": TaskProfile(
            task_type="relative_distance",
            task_group="scene",
            extraction_defaults={"target_label": "null", "candidate_labels": []},
            expected_extraction_fields=("target_label",),
            context_label_fields=("target_label", "candidate_labels"),
            candidate_label_scope="scene_observed_unique",
            heuristic_policy=HeuristicExtractionPolicy(strategy="relative_distance"),
            list_constraints=(
                ListFieldConstraint(
                    list_field="candidate_labels",
                    require_non_empty=True,
                    dedupe_items=True,
                    element_field="target_label",
                    empty_error="relative_distance invalid: `candidate_labels` cannot be empty",
                    duplicate_error="relative_distance invalid: `candidate_labels` contains duplicate candidates",
                    overlap_error="relative_distance invalid: `candidate_labels` contains `target_label`, so the target cannot appear in the candidate list",
                ),
            ),
        ),
        "relative_direction_hard": TaskProfile(
            task_type="relative_direction_hard",
            task_group="scene",
            aliases=tuple(aliases_by_task.get("relative_direction_hard", [])),
            extraction_defaults={
                "positioning_label": "null",
                "orienting_label": "null",
                "querying_label": "null",
            },
            expected_extraction_fields=("positioning_label", "orienting_label", "querying_label"),
            context_label_fields=("positioning_label", "orienting_label", "querying_label"),
            candidate_label_scope="scene_observed_unique",
            heuristic_policy=HeuristicExtractionPolicy(strategy="relative_direction_hard"),
            distinct_field_group_constraints=(
                DistinctFieldGroupConstraint(
                    ("positioning_label", "orienting_label", "querying_label"),
                    "relative_direction_hard invalid: positioning/orienting/querying contain duplicate objects, so the relation is not well-defined",
                ),
            ),
        ),
        "single_image_relative_direction": TaskProfile(
            task_type="single_image_relative_direction",
            task_group="single_image",
            input_mode="single_image",
            aliases=tuple(aliases_by_task.get("single_image_relative_direction", [])),
            extraction_defaults={"reference_label": "null", "target_label": "null"},
            expected_extraction_fields=("reference_label", "target_label"),
            context_label_fields=("reference_label", "target_label"),
            candidate_label_scope="frame_only",
            heuristic_policy=HeuristicExtractionPolicy(strategy="single_image_relative_direction"),
            same_label_constraints=(
                SameLabelConstraint(
                    "reference_label",
                    "target_label",
                    "single_image_relative_direction invalid: `reference_label` and `target_label` must refer to different objects",
                ),
            ),
        ),
        "room_size": TaskProfile(
            task_type="room_size",
            task_group="scene",
        ),
        "position_cam_cam": TaskProfile(
            task_type="position_cam_cam",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("position_cam_cam", [])),
            candidate_label_scope="frame_only",
            spatial_relation_policy=SpatialRelationPolicy(
                binding="camera_pair",
                default_reference_frame="local",
            ),
            camera_pair_answer_mode="position",
            prefill_image_index_mode="pair",
            image_index_parse_mode="pair",
        ),
        "elevation_cam_cam": TaskProfile(
            task_type="elevation_cam_cam",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("elevation_cam_cam", [])),
            prefill_image_index_mode="pair",
            image_index_parse_mode="pair",
        ),
        "position_cam_obj": TaskProfile(
            task_type="position_cam_obj",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("position_cam_obj", [])),
            extraction_defaults={"target_label": "null"},
            expected_extraction_fields=("target_label",),
            context_label_fields=("target_label",),
            candidate_label_scope="reference_frame_unique",
            heuristic_policy=HeuristicExtractionPolicy(
                strategy="single_entity",
                target_field="target_label",
                enable_relative_to_me_fallback=True,
            ),
            allow_heuristic_without_candidates=True,
            detection_requirement=DetectionRequirement(),
            ambiguity_object_fields=(
                LabelFieldRole("target_label", "target object"),
            ),
            object_detection_policy=ObjectDetectionPolicy(
                frame_source="reference_image",
                include_scene_metadata=True,
                use_camera_location=False,
                target_label_fields=("target_label",),
            ),
            detection_result_binding="pick_single_target",
            spatial_relation_policy=SpatialRelationPolicy(
                binding="camera_to_target",
                default_reference_frame="camera",
            ),
            camera_pair_answer_mode="position",
            prefill_image_index_mode="reference",
            image_index_parse_mode="reference",
        ),
        "visibility_compare": TaskProfile(
            task_type="visibility_compare",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("visibility_compare", [])),
            extraction_defaults={"target_label": "null"},
            expected_extraction_fields=("target_label",),
            context_label_fields=("target_label",),
            candidate_label_scope="pair_visibility_compare",
            heuristic_policy=HeuristicExtractionPolicy(
                strategy="single_entity",
                target_field="target_label",
            ),
            ambiguity_object_fields=(
                LabelFieldRole("target_label", "target object"),
            ),
        ),
        "position_cam_reg": TaskProfile(
            task_type="position_cam_reg",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("position_cam_reg", [])),
            extraction_defaults={"region_name": "null"},
            expected_extraction_fields=("region_name",),
            context_label_fields=("region_name",),
            candidate_label_scope="reference_frame_unique",
            heuristic_policy=HeuristicExtractionPolicy(
                strategy="single_entity",
                target_field="region_name",
                enable_relative_to_me_fallback=True,
                allow_abstract_region_subject=True,
            ),
            allow_heuristic_without_candidates=True,
            ambiguity_region_fields=(
                LabelFieldRole("region_name", "region anchor"),
            ),
            spatial_relation_policy=SpatialRelationPolicy(
                binding="camera_to_region",
                default_reference_frame="camera",
            ),
            camera_pair_answer_mode="position",
            prefill_image_index_mode="reference",
            image_index_parse_mode="reference",
            region_anchor_infer_max_labels=1,
        ),
        "motion_camera": TaskProfile(
            task_type="motion_camera",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("motion_camera", [])),
            candidate_label_scope="frame_only",
            camera_pair_answer_mode="motion",
            prefill_image_index_mode="reference",
            image_index_parse_mode="reference",
        ),
        "attribute_measurement": TaskProfile(
            task_type="attribute_measurement",
            task_group="image_pair",
            input_mode="image_pair",
            aliases=tuple(aliases_by_task.get("attribute_measurement", [])),
            extraction_defaults={"object1_label": "null", "object2_label": "null"},
            expected_extraction_fields=("object1_label", "object2_label"),
            context_label_fields=("object1_label", "object2_label"),
            candidate_label_scope="pair_non_ambiguous",
            heuristic_policy=HeuristicExtractionPolicy(strategy="pair_measurement"),
            same_label_constraints=(
                SameLabelConstraint(
                    "object1_label",
                    "object2_label",
                    "attribute_measurement invalid: `object1_label` and `object2_label` must refer to different objects",
                ),
            ),
            ambiguity_object_fields=(
                LabelFieldRole("object1_label", "object 1"),
                LabelFieldRole("object2_label", "object 2"),
            ),
            object_detection_policy=ObjectDetectionPolicy(
                frame_source="both_images",
                include_scene_metadata=False,
                use_camera_location=False,
                target_label_fields=("object1_label", "object2_label"),
            ),
            detection_result_binding="bind_measurement_pair",
            measurement_default="compare_longer",
            prefill_image_index_mode="reference",
            image_index_parse_mode="reference",
        ),
    }

@dataclass(frozen=True)
class TaskResolution:
    """Task name resolution result."""

    task_type: Optional[str]
    rejection_reason: str = ""
    match_type: str = "unresolved"
    normalized_input: str = ""

@dataclass(frozen=True)
class TaskGroupResolution:
    """Task-group resolution result."""

    task_group: Optional[str]
    match_type: str = "unresolved"
    normalized_input: str = ""
    error: str = ""

class TaskResolver:
    """Resolve flexible task aliases while rejecting ambiguous task families."""

    def __init__(
        self,
        task_aliases: Optional[Dict[str, str]] = None,
        task_rejection_reasons: Optional[Dict[str, str]] = None,
    ):
        self.task_aliases = dict(task_aliases or TASK_ALIASES)
        self.task_rejection_reasons = dict(task_rejection_reasons or TASK_REJECTION_REASONS)

    def resolve(
        self,
        input_type: Any,
        supported_task_names: Iterable[str],
        normalize: Callable[[str], str],
    ) -> TaskResolution:
        norm_input = normalize(input_type)
        if not norm_input:
            return TaskResolution(task_type=None, match_type="empty", normalized_input=norm_input)

        rejection_reason = self._match_rejection_reason(norm_input)
        if rejection_reason:
            return TaskResolution(
                task_type=None,
                rejection_reason=rejection_reason,
                match_type="rejected",
                normalized_input=norm_input,
            )

        for registry_type in supported_task_names:
            if norm_input == normalize(registry_type):
                return TaskResolution(
                    task_type=registry_type,
                    match_type="exact",
                    normalized_input=norm_input,
                )

        if norm_input in self.task_aliases:
            return TaskResolution(
                task_type=self.task_aliases[norm_input],
                match_type="alias",
                normalized_input=norm_input,
            )

        for registry_type in supported_task_names:
            norm_registry = normalize(registry_type)
            if norm_input in norm_registry or norm_registry in norm_input:
                return TaskResolution(
                    task_type=registry_type,
                    match_type="fuzzy",
                    normalized_input=norm_input,
                )

        return TaskResolution(
            task_type=None,
            match_type="unsupported",
            normalized_input=norm_input,
        )

    def _match_rejection_reason(self, norm_input: str) -> str:
        for rejected_norm, reason in self.task_rejection_reasons.items():
            if norm_input == rejected_norm or norm_input in rejected_norm or rejected_norm in norm_input:
                return reason
        return ""

    def get_alias_mapping(self) -> Dict[str, str]:
        return dict(self.task_aliases)

    def get_rejection_rules(self) -> Dict[str, str]:
        return dict(self.task_rejection_reasons)

    def remove_aliases_for_task(self, task_type: str) -> None:
        self.task_aliases = {
            alias: mapped_task
            for alias, mapped_task in self.task_aliases.items()
            if mapped_task != task_type
        }

class TaskGroupResolver:
    """Resolve noisy task-group names to canonical public groups."""

    def __init__(self, aliases: Optional[Dict[str, str]] = None):
        self.aliases = dict(aliases or TASK_GROUP_ALIASES)

    @staticmethod
    def _normalize(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        return "".join(ch for ch in text.strip().lower() if ch.isalnum() or ch == "_")

    def resolve(self, raw_group: Any) -> TaskGroupResolution:
        normalized_input = self._normalize(raw_group)
        if not normalized_input:
            return TaskGroupResolution(
                task_group=None,
                match_type="empty",
                normalized_input=normalized_input,
                error="`task_group` cannot be empty",
            )

        if normalized_input in self.aliases:
            canonical = self.aliases[normalized_input]
            match_type = "exact" if normalized_input == canonical else "alias"
            return TaskGroupResolution(
                task_group=canonical,
                match_type=match_type,
                normalized_input=normalized_input,
            )

        return TaskGroupResolution(
            task_group=None,
            match_type="unsupported",
            normalized_input=normalized_input,
            error=f"Unsupported task_group: {raw_group}",
        )

    def get_alias_mapping(self) -> Dict[str, str]:
        return dict(self.aliases)

    def register_alias(self, alias: str, canonical: str) -> None:
        norm_alias = self._normalize(alias)
        norm_canonical = self._normalize(canonical)
        if not norm_alias or not norm_canonical:
            return
        if norm_canonical not in set(self.aliases.values()):
            return
        self.aliases[norm_alias] = norm_canonical

class TaskSupportRegistry:
    """Mutable TaskProfile registry for DGE runtime extension."""

    def __init__(
        self,
        extraction_defaults: Optional[Dict[str, Dict[str, Any]]] = None,
        expected_extraction_fields: Optional[Dict[str, Sequence[str]]] = None,
        region_object_ontology: Optional[Dict[str, Sequence[str]]] = None,
        region_phrase_aliases: Optional[Dict[str, Sequence[str]]] = None,
        task_profiles: Optional[Dict[str, TaskProfile]] = None,
    ):
        default_profiles = build_default_task_profiles()
        self._task_profiles: Dict[str, TaskProfile] = {
            task_type: deepcopy(profile)
            for task_type, profile in (task_profiles or default_profiles).items()
        }

        source_defaults = extraction_defaults or {
            task_type: profile.extraction_defaults
            for task_type, profile in self._task_profiles.items()
        }
        self._extraction_defaults = deepcopy(source_defaults)

        source_fields = expected_extraction_fields or {
            task_type: profile.expected_extraction_fields
            for task_type, profile in self._task_profiles.items()
        }
        self._expected_extraction_fields = {
            key: list(values)
            for key, values in source_fields.items()
        }

        source_ontology = region_object_ontology or REGION_OBJECT_ONTOLOGY
        self._region_object_ontology = {
            key: list(values)
            for key, values in source_ontology.items()
        }

        source_region_aliases = region_phrase_aliases or REGION_PHRASE_ALIASES
        self._region_phrase_aliases = {
            str(key).strip().lower(): [
                str(value).strip()
                for value in values
                if str(value).strip()
            ]
            for key, values in source_region_aliases.items()
            if str(key).strip()
        }
        self._region_phrase_lookup: Dict[str, str] = {}
        self._region_phrase_compact_lookup: Dict[str, str] = {}
        self._rebuild_region_phrase_alias_index()

    @staticmethod
    def _normalize_region_phrase(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        normalized = re.sub(r"[^a-zA-Z0-9]+", " ", text.strip().lower())
        return " ".join(normalized.split())

    def _rebuild_region_phrase_alias_index(self) -> None:
        self._region_phrase_lookup = {}
        self._region_phrase_compact_lookup = {}
        for canonical, aliases in self._region_phrase_aliases.items():
            candidates = [canonical, *aliases]
            for candidate in candidates:
                normalized = self._normalize_region_phrase(candidate)
                if not normalized:
                    continue
                self._region_phrase_lookup[normalized] = canonical
                self._region_phrase_compact_lookup[normalized.replace(" ", "")] = canonical

    def get_extraction_defaults(self, task_type: str) -> Dict[str, Any]:
        profile = self.get_task_profile(task_type)
        if profile is not None:
            return deepcopy(profile.extraction_defaults)
        return deepcopy(self._extraction_defaults.get(task_type, {}))

    def get_expected_extraction_fields(self, task_type: str) -> list[str]:
        profile = self.get_task_profile(task_type)
        if profile is not None:
            return list(profile.expected_extraction_fields)
        return list(self._expected_extraction_fields.get(task_type, []))

    def get_region_object_ontology(self) -> Dict[str, list[str]]:
        return {
            key: list(values)
            for key, values in self._region_object_ontology.items()
        }

    def get_region_phrase_aliases(self) -> Dict[str, list[str]]:
        return {
            key: list(values)
            for key, values in self._region_phrase_aliases.items()
        }

    def resolve_region_phrase_alias(self, text: Any) -> Optional[str]:
        normalized = self._normalize_region_phrase(text)
        if not normalized:
            return None

        direct = self._region_phrase_lookup.get(normalized)
        if direct:
            return direct

        compact = self._region_phrase_compact_lookup.get(normalized.replace(" ", ""))
        if compact:
            return compact

        best_match: Optional[str] = None
        best_len = 0
        padded = f" {normalized} "
        for alias_text, canonical in self._region_phrase_lookup.items():
            if not alias_text:
                continue
            alias_len = len(alias_text)
            alias_padded = f" {alias_text} "
            if alias_padded in padded or padded.strip() in alias_text:
                if alias_len > best_len:
                    best_len = alias_len
                    best_match = canonical
        return best_match

    def get_task_profile(self, task_type: str) -> Optional[TaskProfile]:
        return self._task_profiles.get(task_type)

    def get_all_task_profiles(self) -> Dict[str, TaskProfile]:
        return {
            task_type: deepcopy(profile)
            for task_type, profile in self._task_profiles.items()
        }

    def get_task_group(self, task_type: str) -> Optional[str]:
        profile = self.get_task_profile(task_type)
        return profile.task_group if profile is not None else None

    def get_task_family(self, task_type: str) -> Optional[str]:
        """Backward-compatible alias of `get_task_group`."""
        return self.get_task_group(task_type)

    def get_task_types(
        self,
        task_group: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> list[str]:
        selected_group = task_group if task_group is not None else task_family
        if not isinstance(selected_group, str) or not selected_group.strip():
            return sorted(self._task_profiles.keys())
        resolution = TaskGroupResolver().resolve(selected_group)
        if resolution.task_group is None:
            return []
        normalized_group = resolution.task_group
        return sorted([
            task_type
            for task_type, profile in self._task_profiles.items()
            if profile.task_group == normalized_group
        ])

    def get_supported_task_groups(self) -> list[str]:
        return sorted({profile.task_group for profile in self._task_profiles.values() if profile.task_group})

    def is_task_in_group(self, task_type: str, task_group: str) -> bool:
        resolution = TaskGroupResolver().resolve(task_group)
        if resolution.task_group is None:
            return False
        return self.get_task_group(task_type) == resolution.task_group

    def is_task_in_family(self, task_type: str, family: str) -> bool:
        """Backward-compatible alias of `is_task_in_group`."""
        return self.is_task_in_group(task_type, family)

    def register_task_profile(self, profile: TaskProfile) -> None:
        self._task_profiles[profile.task_type] = deepcopy(profile)
        self._extraction_defaults[profile.task_type] = deepcopy(profile.extraction_defaults)
        self._expected_extraction_fields[profile.task_type] = list(profile.expected_extraction_fields)

    def build_task_profile(
        self,
        task_type: str,
        *,
        task_profile: Optional[TaskProfile] = None,
        aliases: Optional[Sequence[str]] = None,
        extraction_defaults: Optional[Dict[str, Any]] = None,
        expected_extraction_fields: Optional[Sequence[str]] = None,
        heuristic_extractor: Optional[Callable[[str, list[str]], Dict[str, Any]]] = None,
        task_group: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> TaskProfile:
        base_profile = task_profile or self.get_task_profile(task_type) or TaskProfile(task_type=task_type)
        profile = deepcopy(base_profile)
        profile.task_type = task_type

        selected_group = task_group if task_group is not None else task_family
        if isinstance(selected_group, str) and selected_group.strip():
            normalized_group = normalize_task_group_name(selected_group, input_mode=profile.input_mode)
            profile.task_group = normalized_group
            profile.family = normalized_group

        if aliases is not None:
            merged_aliases = list(profile.aliases)
            merged_aliases.extend([
                alias.strip()
                for alias in aliases
                if isinstance(alias, str) and alias.strip()
            ])
            profile.aliases = tuple(dict.fromkeys(merged_aliases))

        if extraction_defaults is not None:
            profile.extraction_defaults = deepcopy(extraction_defaults)

        if expected_extraction_fields is not None:
            profile.expected_extraction_fields = tuple(expected_extraction_fields)

        if heuristic_extractor is not None:
            profile.heuristic_extractor = heuristic_extractor

        return profile

    def register_task_support(
        self,
        task_type: str,
        *,
        extraction_defaults: Optional[Dict[str, Any]] = None,
        expected_extraction_fields: Optional[Sequence[str]] = None,
        task_profile: Optional[TaskProfile] = None,
    ) -> None:
        if task_profile is not None:
            self.register_task_profile(task_profile)
            return

        if extraction_defaults is not None:
            self._extraction_defaults[task_type] = deepcopy(extraction_defaults)
        if expected_extraction_fields is not None:
            self._expected_extraction_fields[task_type] = list(expected_extraction_fields)
        if task_type in self._task_profiles:
            profile = deepcopy(self._task_profiles[task_type])
            if extraction_defaults is not None:
                profile.extraction_defaults = deepcopy(extraction_defaults)
            if expected_extraction_fields is not None:
                profile.expected_extraction_fields = tuple(expected_extraction_fields)
            self._task_profiles[task_type] = profile
        elif extraction_defaults is not None or expected_extraction_fields is not None:
            self._task_profiles[task_type] = TaskProfile(
                task_type=task_type,
                extraction_defaults=deepcopy(extraction_defaults or {}),
                expected_extraction_fields=tuple(expected_extraction_fields or ()),
            )

    def register_region_ontology(self, key: str, candidate_labels: Sequence[str]) -> None:
        if not isinstance(key, str) or not key.strip():
            return
        self._region_object_ontology[key.strip().lower()] = [
            str(value).strip().lower()
            for value in candidate_labels
            if str(value).strip()
        ]

    def register_region_phrase_alias(self, canonical: str, aliases: Sequence[str]) -> None:
        if not isinstance(canonical, str) or not canonical.strip():
            return
        canonical_key = canonical.strip().lower()
        current = list(self._region_phrase_aliases.get(canonical_key, []))
        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text and alias_text not in current:
                current.append(alias_text)
        self._region_phrase_aliases[canonical_key] = current
        self._rebuild_region_phrase_alias_index()

    def remove_task_support(self, task_type: str) -> None:
        self._extraction_defaults.pop(task_type, None)
        self._expected_extraction_fields.pop(task_type, None)
        self._task_profiles.pop(task_type, None)

@dataclass(frozen=True)
class SummaryTypeResolution:
    """Summary-type resolution result."""

    summary_type: Optional[str]
    match_type: str = "unresolved"
    normalized_input: str = ""
    error: str = ""

class SummaryTypeResolver:
    """Resolve noisy summary-type names to canonical environment-summary modes."""

    def __init__(self, aliases: Optional[Dict[str, str]] = None):
        self.aliases = dict(aliases or SUMMARY_TYPE_ALIASES)

    @staticmethod
    def _normalize(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        return "".join(ch for ch in text.strip().lower() if ch.isalnum() or ch == "_")

    def resolve(self, raw_type: Any) -> SummaryTypeResolution:
        normalized_input = self._normalize(raw_type)
        if not normalized_input:
            return SummaryTypeResolution(
                summary_type=None,
                match_type="empty",
                normalized_input=normalized_input,
                error="`summary_type` cannot be empty",
            )

        if normalized_input in self.aliases:
            canonical = self.aliases[normalized_input]
            match_type = "exact" if normalized_input == canonical else "alias"
            return SummaryTypeResolution(
                summary_type=canonical,
                match_type=match_type,
                normalized_input=normalized_input,
            )

        return SummaryTypeResolution(
            summary_type=None,
            match_type="unsupported",
            normalized_input=normalized_input,
            error=f"Unsupported summary_type: {raw_type}",
        )

    def get_alias_mapping(self) -> Dict[str, str]:
        return dict(self.aliases)

    def register_alias(self, alias: str, canonical: str) -> None:
        norm_alias = self._normalize(alias)
        norm_canonical = self._normalize(canonical)
        if not norm_alias or not norm_canonical:
            return
        self.aliases[norm_alias] = norm_canonical
