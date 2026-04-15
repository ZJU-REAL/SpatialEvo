"""Metadata extractor."""

import os
import json
import numpy as np
from typing import Dict, List, Any

class MetadataExtractor:
    """Helper methods for scene metadata."""

    def __init__(self):
        pass

    def extract_scene_metadata(
        self,
        scene_dir: str,
        scene_id: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Load scene metadata from the extracted output."""
        # Placeholder wrapper around the original extraction pipeline.
        metadata_file = os.path.join(output_dir, scene_id, f"{scene_id}.json")

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)

        print(f"Metadata not found for {scene_id}. Please run metadata extraction first.")
        return None

    @staticmethod
    def compute_object_distance(obj1: Dict, obj2: Dict) -> float:
        """Compute the Euclidean distance between two objects."""
        loc1 = np.array(obj1["3d_location"])
        loc2 = np.array(obj2["3d_location"])
        return np.linalg.norm(loc1 - loc2)

    @staticmethod
    def compute_relative_direction(
        positioning_obj: Dict,
        orienting_obj: Dict,
        querying_obj: Dict
    ) -> str:
        """Compute the querying object's direction in the local reference frame."""
        pos_loc = np.array(positioning_obj["3d_location"])
        orient_loc = np.array(orienting_obj["3d_location"])
        query_loc = np.array(querying_obj["3d_location"])

        # Build the forward vector from positioning to orienting.
        forward = orient_loc - pos_loc
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Build the query vector from positioning to querying.
        to_query = query_loc - pos_loc
        to_query = to_query / (np.linalg.norm(to_query) + 1e-8)

        # Use the XY plane only and ignore height.
        forward_2d = forward[:2]
        to_query_2d = to_query[:2]

        # Compute the right-hand vector and the two projections.
        right = np.array([forward_2d[1], -forward_2d[0]])
        forward_proj = np.dot(to_query_2d, forward_2d)
        right_proj = np.dot(to_query_2d, right)

        threshold = 0.3

        if abs(forward_proj) < threshold:
            return "left" if right_proj < 0 else "right"
        if abs(right_proj) < threshold:
            return "back" if forward_proj < 0 else "front"
        if forward_proj > 0:
            return "front-left" if right_proj < 0 else "front-right"
        return "back-left" if right_proj < 0 else "back-right"

    @staticmethod
    def get_object_size(obj: Dict) -> float:
        """Return the longest object dimension in centimeters."""
        # New format: `size` is already the longest dimension in centimeters.
        if isinstance(obj["size"], (int, float)):
            return float(obj["size"])

        # Legacy format: `size` stores width/length/height.
        if isinstance(obj["size"], dict):
            dimensions = [obj["size"]["width"], obj["size"]["length"], obj["size"]["height"]]
            return max(dimensions) * 100

        # Fallback: compute the size from the 3D box.
        bbox = obj["3d_bbox"]
        width = abs(bbox[3] - bbox[0]) * 100
        length = abs(bbox[4] - bbox[1]) * 100
        height = abs(bbox[5] - bbox[2]) * 100
        return max(width, length, height)

    @staticmethod
    def get_room_size(scene_metadata: Dict) -> float:
        """Return room area in square meters."""
        room_size = scene_metadata.get("room_size", 0.0)

        # New format: direct numeric area.
        if isinstance(room_size, (int, float)):
            return float(room_size)

        # Legacy format: dictionary with an `area` field.
        if isinstance(room_size, dict):
            return room_size.get("area", 0.0)

        return 0.0

    @staticmethod
    def compute_closest_object(
        target_obj: Dict,
        candidate_objs: List[Dict]
    ) -> Dict:
        """Return the closest candidate object."""
        if not candidate_objs:
            return None

        target_loc = np.array(target_obj["3d_location"])
        min_dist = float('inf')
        closest_obj = None

        for obj in candidate_objs:
            obj_loc = np.array(obj["3d_location"])
            dist = np.linalg.norm(obj_loc - target_loc)

            if dist < min_dist:
                min_dist = dist
                closest_obj = obj

        return closest_obj
