"""Spatial tools."""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from .base_tool import BaseTool

class SpatialRelationTool(BaseTool):
    """Spatial relation tool."""
    
    def __init__(self):
        super().__init__(
            name="spatial_relation_tool",
            description="Compute spatial relations between objects and cameras"
        )
    
    def execute(
        self,
        entity1: Dict,
        entity2: Dict,
        reference_frame: str = "world",
        **kwargs
    ) -> Dict:
        """Execute."""
        pos1 = np.array(entity1.get("position", np.zeros(3)), dtype=float)
        pos2 = np.array(entity2.get("position", np.zeros(3)), dtype=float)
        

        relative_vec = pos2 - pos1
        distance = np.linalg.norm(relative_vec)
        

        if reference_frame == "local" and "rotation" in entity1:
            rotation = entity1["rotation"]
            relative_vec = rotation.T @ relative_vec

        if reference_frame == "camera":
            direction = self._compute_direction_camera(relative_vec)
        else:
            direction = self._compute_direction(relative_vec)
        
        return {
            "distance": distance,
            "direction": direction,
            "relative_vector": relative_vec.tolist(),
            "reference_frame": reference_frame,
            "answer": str(direction).lower()
        }

    @staticmethod
    def _compute_direction_camera(vec: np.ndarray) -> str:
        """Compute direction camera."""
        arr = np.array(vec, dtype=float).reshape(-1)
        x = float(arr[0]) if arr.shape[0] > 0 else 0.0
        z = float(arr[2]) if arr.shape[0] > 2 else 0.0
        angle = np.degrees(np.arctan2(z, x)) % 360.0
        directions = [
            "Right", "Front-Right", "Front", "Front-Left",
            "Left", "Back-Left", "Back", "Back-Right"
        ]
        index = int((angle + 22.5) // 45) % 8
        return directions[index]
    
    def _compute_direction(
        self,
        vec: np.ndarray,
        mode: str = "8-way"
    ) -> str:
        """Compute direction."""
        arr = np.array(vec, dtype=float).reshape(-1)
        x = float(arr[0]) if arr.shape[0] > 0 else 0.0
        y = float(arr[1]) if arr.shape[0] > 1 else 0.0
        z = float(arr[2]) if arr.shape[0] > 2 else 0.0
        
        if mode == "4-way":

            if abs(x) > abs(y):
                return "Right" if x > 0 else "Left"
            else:
                return "Front" if y > 0 else "Back"
        
        elif mode == "8-way":

            angle = np.degrees(np.arctan2(y, x)) % 360.0
            directions = [
                "Right", "Front-Right", "Front", "Front-Left",
                "Left", "Back-Left", "Back", "Back-Right"
            ]
            index = int((angle + 22.5) // 45) % 8
            return directions[index]
        
        else:  # detailed

            horizontal = self._compute_direction(vec[:2], mode="8-way")
            if abs(z) > np.linalg.norm(vec[:2]) * 0.5:
                vertical = "Up" if z > 0 else "Down"
                return f"{vertical}-{horizontal}"
            return horizontal
    
    def compute_relative_distance(
        self,
        entities: List[Dict],
        target_entity: Dict
    ) -> Dict[str, float]:
        """Compute relative distance."""
        target_pos = target_entity.get("position", np.zeros(3))
        distances = {}
        
        for entity in entities:
            name = entity.get("name", "unknown")
            pos = entity.get("position", np.zeros(3))
            distance = np.linalg.norm(pos - target_pos)
            distances[name] = distance
        
        return distances

class ObjectDetectionTool(BaseTool):
    """Object detection tool."""
    
    def __init__(self):
        super().__init__(
            name="object_detection_tool",
            description="Detect objects and their locations in images or scenes"
        )
    
    def execute(
        self,
        image_path: Optional[str] = None,
        scene_data_path: Optional[str] = None,
        scene_metadata: Optional[Dict] = None,
        frame_metadata: Optional[Dict] = None,
        frame_metadata_list: Optional[List[Dict]] = None,
        target_objects: Optional[List[str]] = None,
        use_camera_location: bool = False,
        **kwargs
    ) -> Dict:
        """Execute."""
        targets = set()
        if isinstance(target_objects, list):
            targets = {str(x).strip().lower() for x in target_objects if str(x).strip()}

        detections_by_key: Dict[Any, Dict] = {}
        frame_metadatas: List[Dict] = []
        if isinstance(frame_metadata_list, list):
            for meta in frame_metadata_list:
                if isinstance(meta, dict):
                    frame_metadatas.append(meta)
        if isinstance(frame_metadata, dict):
            frame_metadatas.append(frame_metadata)

        def _frame_detection_key(label: str, obj: Dict) -> Any:
            object_id = obj.get("object_id")
            if object_id is not None:
                return ("object_id", object_id)
            pos = obj.get("3d_location", [0, 0, 0])
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                pos_key = tuple(round(float(value), 4) for value in pos[:3])
            else:
                pos_key = (0.0, 0.0, 0.0)
            return ("label_pos", label, pos_key)

        def _store_detection(obj: Dict) -> None:
            label = str(obj.get("label", "")).strip().lower()
            if targets and label not in targets:
                return
            pos_source = obj.get("3d_location", [0, 0, 0])
            if use_camera_location:
                cam_loc = obj.get("camera_location")
                if isinstance(cam_loc, (list, tuple)) and len(cam_loc) >= 3:
                    pos_source = cam_loc[:3]
            detection = {
                "name": label,
                "category": label,
                "position": np.array(pos_source, dtype=float),
                "bbox": obj.get("3d_bbox", None),
                "visibility": float(obj.get("visibility", 0.0)),
                "object_id": obj.get("object_id"),
                "camera_location": obj.get("camera_location"),
            }
            key = _frame_detection_key(label, obj)
            existing = detections_by_key.get(key)
            if existing is None or float(detection.get("visibility", 0.0)) > float(existing.get("visibility", 0.0)):
                detections_by_key[key] = detection

        for frame_meta in frame_metadatas:
            for obj in frame_meta.get("objects", []) or []:
                _store_detection(obj)

        detections: List[Dict] = list(detections_by_key.values())
        if scene_metadata:
            by_id = {d.get("object_id"): d for d in detections if d.get("object_id") is not None}
            for obj in scene_metadata.get("objects", []) or []:
                label = str(obj.get("label", "")).strip().lower()
                if targets and label not in targets:
                    continue
                object_id = obj.get("object_id")
                if object_id in by_id:
                    continue
                detections.append(
                    {
                        "name": label,
                        "category": label,
                        "position": np.array(obj.get("3d_location", [0, 0, 0]), dtype=float),
                        "bbox": obj.get("3d_bbox", None),
                        "size": obj.get("size"),
                        "object_id": object_id,
                    }
                )

        result: Dict = {
            "success": True,
            "detections": detections,
        }
        if len(detections) == 1:
            result["target_entity"] = detections[0]
            result["answer"] = detections[0]["name"]
        elif len(detections) >= 2:
            result["entity1"] = detections[0]
            result["entity2"] = detections[1]
        return result
    
    def count_objects(
        self,
        detections: List[Dict],
        category: Optional[str] = None
    ) -> int:
        """Count objects."""
        if category is None:
            return len(detections)
        
        count = sum(1 for det in detections if det.get("category") == category)
        return count

class MeasurementTool(BaseTool):
    """Measurement tool."""
    
    def __init__(self):
        super().__init__(
            name="measurement_tool",
            description="Measure object size, distance, and related properties"
        )
    
    def execute(
        self,
        measurement_type: str,
        entities: Optional[List[Dict]] = None,
        entity1: Optional[Dict] = None,
        entity2: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Execute."""
        if entities is None:
            entities = []
        if entity1 is not None:
            entities = [entity1] + entities
        if entity2 is not None:
            entities = entities + [entity2]

        if measurement_type == "size" and len(entities) >= 1:
            return self._measure_size(entities[0])
        elif measurement_type == "distance" and len(entities) >= 2:
            return self._measure_distance(entities[0], entities[1])
        elif measurement_type == "volume" and len(entities) >= 1:
            return self._measure_volume(entities[0])
        elif measurement_type == "compare_longer" and len(entities) >= 2:
            m1 = self._measure_size(entities[0])
            m2 = self._measure_size(entities[1])
            l1 = float(m1.get("longest_dimension", 0.0))
            l2 = float(m2.get("longest_dimension", 0.0))
            winner = entities[0].get("name", "entity1") if l1 >= l2 else entities[1].get("name", "entity2")
            return {
                "entity1": entities[0].get("name", "entity1"),
                "entity2": entities[1].get("name", "entity2"),
                "length1": l1,
                "length2": l2,
                "winner": winner,
                "answer": winner,
            }
        else:
            return {}
    
    def _measure_size(self, entity: Dict) -> Dict:
        """Measure size."""
        bbox = entity.get("bbox")
        if isinstance(bbox, dict) and "min" in bbox and "max" in bbox:
            dimensions = np.array(bbox["max"], dtype=float) - np.array(bbox["min"], dtype=float)
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 6:
            dimensions = np.array(
                [
                    abs(float(bbox[3]) - float(bbox[0])),
                    abs(float(bbox[4]) - float(bbox[1])),
                    abs(float(bbox[5]) - float(bbox[2])),
                ],
                dtype=float,
            )
        else:
            size = entity.get("size")
            if isinstance(size, (int, float)):
                dimensions = np.array([float(size), float(size), float(size)], dtype=float)
            else:
                dimensions = np.array([1.0, 1.0, 1.0], dtype=float)
        
        return {
            "width": dimensions[0],
            "depth": dimensions[1],
            "height": dimensions[2],
            "longest_dimension": max(dimensions)
        }
    
    def _measure_distance(self, entity1: Dict, entity2: Dict) -> Dict:
        """Measure distance."""
        pos1 = np.array(entity1.get("position", [0, 0, 0]))
        pos2 = np.array(entity2.get("position", [0, 0, 0]))
        distance = np.linalg.norm(pos2 - pos1)
        
        return {
            "distance": distance,
            "unit": "meters"
        }
    
    def _measure_volume(self, entity: Dict) -> Dict:
        """Measure volume."""
        size = self._measure_size(entity)
        volume = size["width"] * size["depth"] * size["height"]
        
        return {
            "volume": volume,
            "unit": "cubic_meters"
        }

class RegionAnchorTool(BaseTool):
    """Region anchor tool."""

    def __init__(self):
        super().__init__(
            name="region_anchor_tool",
            description="Map region names to anchor coordinates"
        )

    @staticmethod
    def _build_region_entity(region_name: str, region_positions: Dict) -> Optional[Dict]:
        if not isinstance(region_name, str) or not region_name.strip():
            return None
        if not isinstance(region_positions, dict):
            return None
        pos = region_positions.get(region_name)
        if pos is None:
            pos = region_positions.get(region_name.lower())
        if pos is None:
            return None
        arr = np.array(pos, dtype=float)
        if arr.shape[0] < 3:
            return None
        return {
            "name": region_name,
            "type": "region",
            "position": arr[:3],
        }

    def execute(
        self,
        region_positions: Dict,
        region_name: Optional[str] = None,
        region1: Optional[str] = None,
        region2: Optional[str] = None,
        **kwargs
    ) -> Dict:
        if region_name is not None:
            entity = self._build_region_entity(region_name, region_positions)
            if entity is None:
                return {"success": False, "error": f"Region anchor not found: {region_name}"}
            return {"success": True, "region_entity": entity}

        if region1 is not None and region2 is not None:
            e1 = self._build_region_entity(region1, region_positions)
            e2 = self._build_region_entity(region2, region_positions)
            if e1 is None or e2 is None:
                return {"success": False, "error": f"Region anchor not found: {region1}/{region2}"}
            return {"success": True, "entity1": e1, "entity2": e2}

        return {"success": False, "error": "region_anchor_tool requires `region_name` or `region1` + `region2`"}
