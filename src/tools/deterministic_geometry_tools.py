"""Deterministic geometry tools."""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base_tool import BaseTool
from .camera_tools import CameraParameterTool
from ..data.scannet_loader import ScanNetLoader
from ..data.metadata_extractor import MetadataExtractor

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _find_visible_candidates(
    frame_metadata: Dict[str, Any],
    label: str,
    min_visibility: float,
) -> List[Dict[str, Any]]:
    if not isinstance(frame_metadata, dict):
        return []
    objects = frame_metadata.get("objects", []) or []
    normalized_label = str(label or "").strip().lower()
    return [
        obj for obj in objects
        if str(obj.get("label", "")).strip().lower() == normalized_label
        and _safe_float(obj.get("visibility", 0.0), 0.0) >= float(min_visibility)
    ]

def _find_unique_visible_object(
    frame_metadata: Dict[str, Any],
    label: str,
    min_visibility: float,
) -> Optional[Dict[str, Any]]:
    candidates = _find_visible_candidates(frame_metadata, label, min_visibility)
    if len(candidates) != 1:
        return None
    return candidates[0]

def _bbox_area_ratio(obj: Dict[str, Any], image_width: float = 640.0, image_height: float = 480.0) -> float:
    bbox = obj.get("2d_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return 0.0
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    denom = max(1.0, float(image_width) * float(image_height))
    return float((width * height) / denom)

class VSIAmbiguityChecker(BaseTool):
    """V s i ambiguity checker."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="ambiguity_checker",
            description="Check whether an object label is ambiguous",
            aliases=("vsi_ambiguity_checker",),
        )
        self.loader = scannet_loader or ScanNetLoader()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        labels: List[str],
        frame_metadata: Optional[Dict[str, Any]] = None,
        min_visibility: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""
        label_counts = {}
        ambiguous_labels = []
        
        for label in labels:
            if frame_metadata:

                visible_objects = self.loader.get_visible_objects(
                    frame_metadata, min_visibility
                )
                count = sum(1 for obj in visible_objects if obj["label"] == label)
            else:

                count = self.loader.count_objects_by_label(scene_metadata, label)
            
            label_counts[label] = count
            
            if count != 1:
                ambiguous_labels.append(label)
        
        return {
            "has_ambiguity": len(ambiguous_labels) > 0,
            "label_counts": label_counts,
            "ambiguous_labels": ambiguous_labels,
            "is_valid": len(ambiguous_labels) == 0
        }

class VSIObjectSizeTool(BaseTool):
    """V s i object size tool."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="object_size_tool",
            description="Measure an object's longest dimension",
            aliases=("vsi_object_size_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.extractor = MetadataExtractor()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        object_label: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""

        obj = self.loader.get_object_by_label(
            scene_metadata, object_label, check_uniqueness=True
        )
        
        if obj is None:
            return {
                "success": False,
                "error": f"Object '{object_label}' not found or ambiguous"
            }
        

        size_cm = self.extractor.get_object_size(obj)
        

        result = {
            "success": True,
            "object_label": object_label,
            "longest_dimension_cm": round(size_cm, 2),
            "answer": f"{round(size_cm, 2)} cm"
        }
        

        if isinstance(obj["size"], dict):
            result["width_m"] = obj["size"]["width"]
            result["length_m"] = obj["size"]["length"]
            result["height_m"] = obj["size"]["height"]
        
        return result

class VSICameraObjectDistanceTool(BaseTool):
    """V s i camera object distance tool."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="distance_cam_obj_tool",
            description="Measure camera distance to the target's nearest point",
            aliases=("vsi_distance_cam_obj_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    @staticmethod
    def _point_to_aabb_distance(point: np.ndarray, bbox: List[float]) -> float:
        mins = np.array(bbox[:3], dtype=float)
        maxs = np.array(bbox[3:6], dtype=float)
        clamped = np.minimum(np.maximum(point, mins), maxs)
        return float(np.linalg.norm(point - clamped))

    @staticmethod
    def _distance_bucket(distance_m: float) -> str:
        if distance_m < 1.0:
            return "near"
        if distance_m < 2.5:
            return "medium"
        return "far"

    def execute(
        self,
        scene_id: str,
        frame_id: Any,
        frame_metadata: Dict[str, Any],
        target_label: str,
        min_visibility: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        visibility_floor = float(getattr(self.loader, "visibility_floor", 0.1))
        min_visibility = max(visibility_floor, float(min_visibility))
        target_obj = _find_unique_visible_object(frame_metadata, target_label, min_visibility)
        if target_obj is None:
            return {
                "success": False,
                "error": f"Target object '{target_label}' not found or not unique in the selected frame"
            }

        scene_data_path = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        camera_params = self.camera_tool.execute(scene_data_path=scene_data_path, frame_id=frame_id)
        camera_pos = np.array(camera_params.get("position", [0.0, 0.0, 0.0]), dtype=float)

        bbox = target_obj.get("3d_bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 6:
            distance_m = self._point_to_aabb_distance(camera_pos, list(bbox[:6]))
        else:
            obj_pos = np.array(target_obj.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
            distance_m = float(np.linalg.norm(camera_pos - obj_pos))

        distance_m = round(distance_m, 2)
        return {
            "success": True,
            "target_label": target_label,
            "frame_id": str(frame_id),
            "distance_m": distance_m,
            "distance_bucket": self._distance_bucket(distance_m),
            "answer": f"{distance_m} meters",
        }

class VSIAbsoluteDistanceTool(BaseTool):
    """V s i absolute distance tool."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="absolute_distance_tool",
            description="Measure the straight-line distance between two objects",
            aliases=("vsi_absolute_distance_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.extractor = MetadataExtractor()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        object1_label: str,
        object2_label: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""

        obj1 = self.loader.get_object_by_label(
            scene_metadata, object1_label, check_uniqueness=True
        )
        obj2 = self.loader.get_object_by_label(
            scene_metadata, object2_label, check_uniqueness=True
        )
        if obj1 is None or obj2 is None:
            print(f"Object {object1_label} or {object2_label} not found or ambiguous")
            return {
                "success": False,
                "error": "One or both objects not found or ambiguous"
            }
        

        distance = self.extractor.compute_object_distance(obj1, obj2)

        return {
            "success": True,
            "object1": object1_label,
            "object2": object2_label,
            "distance_m": round(distance, 2),
            "answer": f"{round(distance, 2)} meters"
        }

class VSIDepthOrderTool(BaseTool):
    """V s i depth order tool."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="depth_order_tool",
            description="Compare two objects by camera depth",
            aliases=("vsi_depth_order_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()

    def execute(
        self,
        frame_metadata: Dict[str, Any],
        object1_label: str,
        object2_label: str,
        min_visibility: float = 0.5,
        same_threshold_m: float = 0.15,
        **kwargs
    ) -> Dict[str, Any]:
        visibility_floor = float(getattr(self.loader, "visibility_floor", 0.1))
        min_visibility = max(visibility_floor, float(min_visibility))
        obj1 = _find_unique_visible_object(frame_metadata, object1_label, min_visibility)
        obj2 = _find_unique_visible_object(frame_metadata, object2_label, min_visibility)

        if obj1 is None or obj2 is None:
            return {
                "success": False,
                "error": "One or both objects are not uniquely visible in the selected frame"
            }

        cam1 = obj1.get("camera_location")
        cam2 = obj2.get("camera_location")
        if not (
            isinstance(cam1, (list, tuple)) and len(cam1) >= 3
            and isinstance(cam2, (list, tuple)) and len(cam2) >= 3
        ):
            return {
                "success": False,
                "error": "Missing camera_location for one or both objects"
            }

        depth1 = _safe_float(cam1[2], 0.0)
        depth2 = _safe_float(cam2[2], 0.0)
        if abs(depth1 - depth2) <= float(same_threshold_m):
            answer = "same"
        else:
            answer = object1_label if depth1 < depth2 else object2_label

        return {
            "success": True,
            "object1_label": object1_label,
            "object2_label": object2_label,
            "depth_1_m": round(depth1, 2),
            "depth_2_m": round(depth2, 2),
            "depth_delta_m": round(depth1 - depth2, 2),
            "answer": answer,
        }

class VSIRelativeDistanceTool(BaseTool):
    """V s i relative distance tool."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="relative_distance_tool",
            description="Find the closest candidate to the target",
            aliases=("vsi_relative_distance_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.extractor = MetadataExtractor()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        target_label: str,
        candidate_labels: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""

        target_obj = self.loader.get_object_by_label(
            scene_metadata, target_label, check_uniqueness=True
        )
        
        if target_obj is None:
            return {
                "success": False,
                "error": f"Target object '{target_label}' not found or ambiguous"
            }
        

        candidate_objs = []
        for label in candidate_labels:
            obj = self.loader.get_object_by_label(
                scene_metadata, label, check_uniqueness=True
            )
            if obj:
                candidate_objs.append(obj)
        
        if not candidate_objs:
            return {
                "success": False,
                "error": "No valid candidate objects found"
            }
        

        distances = {}
        for obj in candidate_objs:
            dist = self.extractor.compute_object_distance(target_obj, obj)
            distances[obj["label"]] = round(dist, 2)
        

        closest_label = min(distances, key=distances.get)
        
        return {
            "success": True,
            "target": target_label,
            "candidates": candidate_labels,
            "distances": distances,
            "closest": closest_label,
            "answer": closest_label
        }

class VSIRelativeDirectionTool(BaseTool):
    """V s i relative direction tool."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="relative_direction_tool",
            description="Compute object relative direction",
            aliases=("vsi_relative_direction_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.extractor = MetadataExtractor()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        positioning_label: str,
        orienting_label: str,
        querying_label: str,
        difficulty: str = "hard",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""

        pos_obj = self.loader.get_object_by_label(
            scene_metadata, positioning_label, check_uniqueness=True
        )
        orient_obj = self.loader.get_object_by_label(
            scene_metadata, orienting_label, check_uniqueness=True
        )
        query_obj = self.loader.get_object_by_label(
            scene_metadata, querying_label, check_uniqueness=True
        )
        
        if not all([pos_obj, orient_obj, query_obj]):
            return {
                "success": False,
                "error": "One or more objects not found or ambiguous"
            }
        

        direction = self.extractor.compute_relative_direction(
            pos_obj, orient_obj, query_obj
        )
        

        if difficulty == "easy":

            if "left" in direction:
                direction = "left"
            else:
                direction = "right"
        elif difficulty == "medium":

            if "front" in direction:
                direction = "front"
            elif "left" in direction:
                direction = "left"
            elif "right" in direction:
                direction = "right"
            else:
                direction = "back"

        
        return {
            "success": True,
            "positioning_object": positioning_label,
            "orienting_object": orienting_label,
            "querying_object": querying_label,
            "direction": direction,
            "answer": direction
        }

class VSIVisibilityCompareTool(BaseTool):
    """V s i visibility compare tool."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="visibility_compare_tool",
            description="Compare target visibility across two images",
            aliases=("vsi_visibility_compare_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()

    @staticmethod
    def _visibility_score(obj: Dict[str, Any]) -> float:
        visibility = _safe_float(obj.get("visibility", 0.0), 0.0)
        area_ratio = _bbox_area_ratio(obj)
        occluded_penalty = 0.03 if bool(obj.get("occluded", False)) else 0.0
        truncated_penalty = 0.02 if bool(obj.get("truncated", False)) else 0.0
        return 0.75 * visibility + 0.25 * area_ratio - occluded_penalty - truncated_penalty

    def _resolve_visible_object(
        self,
        frame_metadata: Dict[str, Any],
        target_label: str,
        min_visibility: float,
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        candidates = _find_visible_candidates(frame_metadata, target_label, min_visibility)
        return (candidates[0] if len(candidates) == 1 else None, len(candidates))

    def execute(
        self,
        frame_metadata_1: Dict[str, Any],
        frame_metadata_2: Dict[str, Any],
        target_label: str,
        min_visibility: float = 0.5,
        same_threshold: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        visibility_floor = float(getattr(self.loader, "visibility_floor", 0.1))
        min_visibility = max(visibility_floor, float(min_visibility))

        obj1, count1 = self._resolve_visible_object(frame_metadata_1, target_label, min_visibility)
        obj2, count2 = self._resolve_visible_object(frame_metadata_2, target_label, min_visibility)

        if count1 > 1 or count2 > 1:
            return {
                "success": False,
                "error": (
                    f"Target object '{target_label}' is ambiguous across provided frames "
                    f"(image1={count1}, image2={count2})"
                )
            }

        if obj1 is None and obj2 is None:
            return {
                "success": True,
                "target_label": target_label,
                "image1_visible": False,
                "image2_visible": False,
                "answer": "neither",
            }
        if obj1 is not None and obj2 is None:
            return {
                "success": True,
                "target_label": target_label,
                "image1_visible": True,
                "image2_visible": False,
                "answer": "image1",
            }
        if obj1 is None and obj2 is not None:
            return {
                "success": True,
                "target_label": target_label,
                "image1_visible": False,
                "image2_visible": True,
                "answer": "image2",
            }

        score1 = self._visibility_score(obj1)
        score2 = self._visibility_score(obj2)
        if abs(score1 - score2) <= float(same_threshold):
            answer = "same"
        else:
            answer = "image1" if score1 > score2 else "image2"

        return {
            "success": True,
            "target_label": target_label,
            "image1_visible": True,
            "image2_visible": True,
            "image1_visibility": round(_safe_float(obj1.get("visibility", 0.0), 0.0), 4),
            "image2_visibility": round(_safe_float(obj2.get("visibility", 0.0), 0.0), 4),
            "image1_bbox_area_ratio": round(_bbox_area_ratio(obj1), 4),
            "image2_bbox_area_ratio": round(_bbox_area_ratio(obj2), 4),
            "image1_score": round(score1, 4),
            "image2_score": round(score2, 4),
            "answer": answer,
        }

class VSISingleImageRelativeDirectionTool(BaseTool):
    """V s i single image relative direction tool."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="single_image_relative_direction_tool",
            description="Compute relative direction from single-frame metadata",
            aliases=("vsi_single_image_relative_direction_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()

    @staticmethod
    def _find_unique_visible_object(
        frame_metadata: Dict[str, Any],
        label: str,
        min_visibility: float
    ) -> Optional[Dict[str, Any]]:
        return _find_unique_visible_object(frame_metadata, label, min_visibility)

    @staticmethod
    def _dominant_signed_axis(
        primary_value: float,
        primary_pos_label: str,
        primary_neg_label: str,
        secondary_value: float,
        secondary_pos_label: str,
        secondary_neg_label: str,
    ) -> str:
        if abs(primary_value) >= abs(secondary_value):
            return primary_pos_label if primary_value >= 0 else primary_neg_label
        return secondary_pos_label if secondary_value >= 0 else secondary_neg_label

    @classmethod
    def _smooth_vertical_diagonal_world(cls, x: float, y: float, z: float) -> str:
        vertical = "up" if z > 0 else "down"
        horizontal = cls._dominant_signed_axis(
            primary_value=x,
            primary_pos_label="right",
            primary_neg_label="left",
            secondary_value=y,
            secondary_pos_label="front",
            secondary_neg_label="back",
        )
        return f"{vertical}-{horizontal}"

    @classmethod
    def _smooth_vertical_diagonal_camera(cls, x: float, y: float, z: float) -> str:
        vertical = "up" if y < 0 else "down"
        horizontal = cls._dominant_signed_axis(
            primary_value=x,
            primary_pos_label="right",
            primary_neg_label="left",
            secondary_value=z,
            secondary_pos_label="front",
            secondary_neg_label="back",
        )
        return f"{vertical}-{horizontal}"

    @staticmethod
    def _vector_to_direction_world(vec: np.ndarray, vertical_ratio_threshold: float = 0.5) -> str:
        """Vector to direction world."""
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        horizontal_norm = float(np.linalg.norm([x, y]))

        if horizontal_norm < 1e-8:
            if z > 0:
                return "up"
            if z < 0:
                return "down"
            return "overlap"

        angle = np.degrees(np.arctan2(y, x)) % 360.0
        horizontal_bins = [
            "right", "front-right", "front", "front-left",
            "left", "back-left", "back", "back-right"
        ]
        idx = int((angle + 22.5) // 45) % 8
        horizontal = horizontal_bins[idx]

        if abs(z) > horizontal_norm * vertical_ratio_threshold:
            return VSISingleImageRelativeDirectionTool._smooth_vertical_diagonal_world(x, y, z)
        return horizontal

    @staticmethod
    def _vector_to_direction_camera(vec: np.ndarray, vertical_ratio_threshold: float = 0.5) -> str:
        """Vector to direction camera."""
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        horizontal_norm = float(np.linalg.norm([x, z]))

        if horizontal_norm < 1e-8:
            if y < 0:
                return "up"
            if y > 0:
                return "down"
            return "overlap"

        angle = np.degrees(np.arctan2(z, x)) % 360.0
        horizontal_bins = [
            "right", "front-right", "front", "front-left",
            "left", "back-left", "back", "back-right"
        ]
        idx = int((angle + 22.5) // 45) % 8
        horizontal = horizontal_bins[idx]

        if abs(y) > horizontal_norm * vertical_ratio_threshold:
            return VSISingleImageRelativeDirectionTool._smooth_vertical_diagonal_camera(x, y, z)
        return horizontal

    @staticmethod
    def _relative_vector_with_frame(
        ref_obj: Dict[str, Any],
        tgt_obj: Dict[str, Any],
    ) -> tuple[np.ndarray, str]:
        """Relative vector with frame."""
        ref_cam = ref_obj.get("camera_location")
        tgt_cam = tgt_obj.get("camera_location")
        if (
            isinstance(ref_cam, (list, tuple))
            and isinstance(tgt_cam, (list, tuple))
            and len(ref_cam) >= 3
            and len(tgt_cam) >= 3
        ):
            ref_cam_vec = np.array(ref_cam[:3], dtype=float)
            tgt_cam_vec = np.array(tgt_cam[:3], dtype=float)
            if np.all(np.isfinite(ref_cam_vec)) and np.all(np.isfinite(tgt_cam_vec)):
                return tgt_cam_vec - ref_cam_vec, "camera"

        ref_loc = np.array(ref_obj.get("3d_location", [0, 0, 0]), dtype=float)
        tgt_loc = np.array(tgt_obj.get("3d_location", [0, 0, 0]), dtype=float)
        return tgt_loc - ref_loc, "world"

    def execute(
        self,
        frame_metadata: Dict[str, Any],
        reference_label: str,
        target_label: str,
        min_visibility: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""
        visibility_floor = float(getattr(self.loader, "visibility_floor", 0.1))
        min_visibility = max(visibility_floor, float(min_visibility))
        ref_obj = self._find_unique_visible_object(
            frame_metadata=frame_metadata,
            label=reference_label,
            min_visibility=min_visibility,
        )
        tgt_obj = self._find_unique_visible_object(
            frame_metadata=frame_metadata,
            label=target_label,
            min_visibility=min_visibility,
        )

        if ref_obj is None or tgt_obj is None:
            return {
                "success": False,
                "error": (
                    "Reference or target object not found/unique under visibility threshold. "
                    "Please ensure both objects are visible and unique in the selected frame."
                )
            }

        relative_vec, coord_frame = self._relative_vector_with_frame(ref_obj, tgt_obj)
        if coord_frame == "camera":
            direction = self._vector_to_direction_camera(relative_vec)
        else:
            direction = self._vector_to_direction_world(relative_vec)

        return {
            "success": True,
            "reference_label": reference_label,
            "target_label": target_label,
            "coordinate_frame": coord_frame,
            "relative_vector": relative_vec.tolist(),
            "direction": direction,
            "answer": direction
        }

class VSIObjectCountTool(BaseTool):
    """V s i object count tool."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="object_count_tool",
            description="Count objects of the target category",
            aliases=("vsi_object_count_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        target_category: str,
        frame_metadata: Optional[Dict[str, Any]] = None,
        min_visibility: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""
        if frame_metadata:

            visible_objects = self.loader.get_visible_objects(
                frame_metadata, min_visibility
            )
            count = sum(1 for obj in visible_objects if obj["label"] == target_category)
        else:

            count = self.loader.count_objects_by_label(scene_metadata, target_category)
        
        return {
            "success": True,
            "category": target_category,
            "count": count,
            "answer": str(count)
        }

class VSIRoomSizeTool(BaseTool):
    """V s i room size tool."""
    
    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="room_size_tool",
            description="Compute the room area",
            aliases=("vsi_room_size_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.extractor = MetadataExtractor()
    
    def execute(
        self,
        scene_metadata: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute."""
        room_size = self.extractor.get_room_size(scene_metadata)
        
        return {
            "success": True,
            "room_area_sqm": round(room_size, 2),
            "answer": f"{round(room_size, 2)} square meters"
        }
