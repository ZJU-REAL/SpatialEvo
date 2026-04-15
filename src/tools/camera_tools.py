"""Camera tools."""

import os
import re
import numpy as np
from typing import Dict, Tuple, Optional
from .base_tool import BaseTool

class CameraParameterTool(BaseTool):
    """Camera parameter tool."""
    
    def __init__(self):
        super().__init__(
            name="camera_parameter_tool",
            description="Extract camera parameters for an image"
        )
    
    def execute(
        self,
        scene_data_path: str,
        frame_id: int,
        **kwargs
    ) -> Dict:
        """Execute."""
        frame_id_str = str(frame_id).replace(".jpg", "").strip()
        pose_file = os.path.join(scene_data_path, "pose", f"{frame_id_str}.txt")

        extrinsics = np.eye(4)
        if os.path.exists(pose_file):
            extrinsics = np.loadtxt(pose_file)

        axis_align = self._load_axis_align_matrix(scene_data_path)
        extrinsics = axis_align @ extrinsics

        rotation = extrinsics[:3, :3]
        position = extrinsics[:3, 3]

        camera_params = {
            "frame_id": frame_id_str,
            "scene_data_path": scene_data_path,
            "position": position,  # x, y, z
            "rotation": rotation,
            "intrinsics": {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 320.0,
                "cy": 240.0
            },
            "extrinsics": extrinsics,
            "entity": {
                "name": f"camera_{frame_id_str}",
                "type": "camera",
                "position": position,
                "rotation": rotation,
                "frame_id": frame_id_str,
            }
        }
        
        return camera_params

    @staticmethod
    def _load_axis_align_matrix(scene_data_path: str) -> np.ndarray:
        """Load axis align matrix."""
        try:
            scene_id = os.path.basename(os.path.normpath(scene_data_path))
            meta_file = os.path.join(scene_data_path, f"{scene_id}.txt")
            if not os.path.exists(meta_file):
                return np.eye(4)
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "axisAlignment" not in line:
                        continue
                    rhs = line.split("=", 1)[1].strip()
                    vals = [float(x) for x in rhs.split()]
                    if len(vals) == 16:
                        return np.array(vals, dtype=float).reshape(4, 4)
            return np.eye(4)
        except Exception:
            return np.eye(4)
    
    def compute_relative_position(
        self,
        camera1_params: Dict,
        camera2_params: Dict
    ) -> Dict:
        """Compute relative position."""
        pos1 = camera1_params["position"]
        pos2 = camera2_params["position"]
        

        relative_vec = pos2 - pos1
        

        rotation1 = camera1_params["rotation"]
        local_vec = rotation1.T @ relative_vec
        

        direction = self._vector_to_direction(local_vec)
        
        return {
            "relative_vector": relative_vec,
            "local_vector": local_vec,
            "direction": direction,
            "distance": np.linalg.norm(relative_vec)
        }
    
    def _vector_to_direction(self, vec: np.ndarray) -> str:
        """Vector to direction."""
        x, y, z = vec
        

        if abs(x) > abs(y) and abs(x) > abs(z):
            return "Left" if x < 0 else "Right"
        elif abs(y) > abs(x) and abs(y) > abs(z):
            return "Back" if y < 0 else "Front"
        else:
            return "Down" if z < 0 else "Up"

class CameraPairTool(BaseTool):
    """Camera pair tool."""

    def __init__(self):
        super().__init__(
            name="camera_pair_tool",
            description="Extract both cameras and compute their relation"
        )
        self.single_tool = CameraParameterTool()

    @staticmethod
    def _vector_to_direction_xy(vec: np.ndarray) -> str:
        x, y = float(vec[0]), float(vec[1])
        angle = np.degrees(np.arctan2(y, x)) % 360.0
        bins = [
            "Right", "Front-Right", "Front", "Front-Left",
            "Left", "Back-Left", "Back", "Back-Right"
        ]
        idx = int((angle + 22.5) // 45) % 8
        return bins[idx]

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _vector_to_direction_6(vec: np.ndarray) -> str:
        """Vector to direction 6."""
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        axis = int(np.argmax(np.abs([x, z, y])))
        if axis == 0:
            return "right" if x > 0 else "left"
        if axis == 1:
            return "front" if z > 0 else "back"
        return "up" if y < 0 else "down"

    @classmethod
    def _vector_to_motion_direction(
        cls,
        vec: np.ndarray,
        *,
        secondary_ratio_threshold: float = 0.35,
        secondary_norm_threshold: float = 0.25,
    ) -> str:
        """Vector to motion direction."""
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        axis_values = {
            "lateral": ("right" if x > 0 else "left", abs(x)),
            "depth": ("front" if z > 0 else "back", abs(z)),
            "vertical": ("up" if y < 0 else "down", abs(y)),
        }
        ranked = sorted(
            [(axis_name, label, magnitude) for axis_name, (label, magnitude) in axis_values.items()],
            key=lambda item: item[2],
            reverse=True,
        )
        primary_axis, primary_label, primary_mag = ranked[0]
        if primary_mag < 1e-8:
            return "same"

        norm = float(np.linalg.norm([x, y, z]))
        if norm < 1e-8:
            return primary_label

        secondary_axis, secondary_label, secondary_mag = ranked[1]
        secondary_ratio = secondary_mag / max(primary_mag, 1e-8)
        secondary_norm_ratio = secondary_mag / norm
        if (
            secondary_mag < 1e-8
            or secondary_ratio < float(secondary_ratio_threshold)
            or secondary_norm_ratio < float(secondary_norm_threshold)
        ):
            return primary_label

        order = {"lateral": 0, "depth": 1, "vertical": 2}
        composite_labels = [
            label
            for _, label, _ in sorted(
                [(primary_axis, primary_label, primary_mag), (secondary_axis, secondary_label, secondary_mag)],
                key=lambda item: order[item[0]],
            )
        ]
        return "-".join(composite_labels)

    @staticmethod
    def _forward_xz_from_rotation(rotation: np.ndarray) -> Optional[np.ndarray]:
        """Forward xz from rotation."""
        if not isinstance(rotation, np.ndarray) or rotation.shape[0] < 3 or rotation.shape[1] < 3:
            return None
        f = np.array([float(rotation[0, 2]), float(rotation[2, 2])], dtype=float)  # world x,z
        norm = float(np.linalg.norm(f))
        if norm < 1e-8:
            return None
        return f / norm

    @classmethod
    def _signed_yaw_delta_y(cls, rot_from: np.ndarray, rot_to: np.ndarray) -> float:
        """Signed yaw delta y."""
        v1 = cls._forward_xz_from_rotation(rot_from)
        v2 = cls._forward_xz_from_rotation(rot_to)
        if v1 is None or v2 is None:
            return 0.0
        cross_y = float(v1[0] * v2[1] - v1[1] * v2[0])
        dot = float(np.clip(v1[0] * v2[0] + v1[1] * v2[1], -1.0, 1.0))
        return float(np.degrees(np.arctan2(cross_y, dot)))

    @staticmethod
    def _parse_int_frame_id(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _collect_intermediate_frame_ids(
        self,
        scene_data_path: str,
        frame_id_1: int,
        frame_id_2: int,
    ) -> list:
        pose_dir = os.path.join(scene_data_path, "pose")
        if not os.path.isdir(pose_dir):
            return [frame_id_1, frame_id_2]
        ids = []
        lo, hi = min(frame_id_1, frame_id_2), max(frame_id_1, frame_id_2)
        for fn in os.listdir(pose_dir):
            if not fn.endswith(".txt"):
                continue
            fid = self._parse_int_frame_id(fn.replace(".txt", ""))
            if fid is None:
                continue
            if lo <= fid <= hi:
                ids.append(fid)
        if frame_id_1 not in ids:
            ids.append(frame_id_1)
        if frame_id_2 not in ids:
            ids.append(frame_id_2)
        ids = sorted(set(ids))
        if frame_id_1 > frame_id_2:
            ids = list(reversed(ids))
        return ids

    def _compute_cumulative_yaw(
        self,
        scene_data_path: str,
        frame_id_1: int,
        frame_id_2: int,
    ) -> float:
        frame_ids = self._collect_intermediate_frame_ids(scene_data_path, frame_id_1, frame_id_2)
        if len(frame_ids) < 2:
            return 0.0
        cams = [self.single_tool.execute(scene_data_path=scene_data_path, frame_id=fid) for fid in frame_ids]
        yaw_sum = 0.0
        for i in range(len(cams) - 1):
            r1 = np.array(cams[i]["rotation"], dtype=float)
            r2 = np.array(cams[i + 1]["rotation"], dtype=float)
            yaw_sum += self._signed_yaw_delta_y(r1, r2)
        return float(yaw_sum)

    @staticmethod
    def _parse_frame_id(image_path: str) -> Optional[str]:
        if not isinstance(image_path, str):
            return None
        match = re.search(r"(\d+)(?:\.[A-Za-z0-9]+)?$", os.path.basename(image_path))
        if match:
            return match.group(1)
        return None

    def execute(
        self,
        scene_data_path: str,
        frame_id_1: Optional[int] = None,
        frame_id_2: Optional[int] = None,
        image_paths: Optional[list] = None,
        answer_mode: str = "position",
        motion_rot_threshold_deg: float = 8.0,
        **kwargs
    ) -> Dict:
        if (frame_id_1 is None or frame_id_2 is None) and isinstance(image_paths, list) and len(image_paths) >= 2:
            if frame_id_1 is None:
                frame_id_1 = self._parse_frame_id(image_paths[0])
            if frame_id_2 is None:
                frame_id_2 = self._parse_frame_id(image_paths[1])

        frame_id_1 = self._parse_int_frame_id(frame_id_1)
        frame_id_2 = self._parse_int_frame_id(frame_id_2)
        if frame_id_1 is None or frame_id_2 is None:
            return {"success": False, "error": "camera_pair_tool requires `frame_id_1` and `frame_id_2`"}

        cam1 = self.single_tool.execute(scene_data_path=scene_data_path, frame_id=frame_id_1)
        cam2 = self.single_tool.execute(scene_data_path=scene_data_path, frame_id=frame_id_2)

        rel_vec_world = np.array(cam2["position"]) - np.array(cam1["position"])
        rel_vec_local = np.array(cam1["rotation"]).T @ rel_vec_world
        rel_direction = self._vector_to_direction_xy(rel_vec_local)
        rel_distance = float(np.linalg.norm(rel_vec_world))

        rel_rotation = np.array(cam2["rotation"]) @ np.array(cam1["rotation"]).T
        yaw_deg_pair = self._signed_yaw_delta_y(np.array(cam1["rotation"]), np.array(cam2["rotation"]))
        yaw_deg_sum = self._compute_cumulative_yaw(scene_data_path, frame_id_1, frame_id_2)
        yaw_deg = yaw_deg_sum if answer_mode == "motion" else yaw_deg_pair
        if abs(yaw_deg) >= float(motion_rot_threshold_deg):

            motion_direction = "left" if yaw_deg > 0 else "right"
            motion_rule = (
                f"abs(cumulative_yaw_deg)>={float(motion_rot_threshold_deg):.1f} "
                "=> rotation-dominant (yaw sign around world-y)"
            )
        else:

            motion_direction = self._vector_to_motion_direction(rel_vec_local)
            motion_rule = (
                f"abs(cumulative_yaw_deg)<{float(motion_rot_threshold_deg):.1f} "
                "=> translation-dominant (camera-axis dominant, composite direction enabled)"
            )

        return {
            "success": True,
            "camera1_params": cam1,
            "camera2_params": cam2,
            "camera_entity_1": cam1["entity"],
            "camera_entity_2": cam2["entity"],
            "relative_vector_world": rel_vec_world.tolist(),
            "relative_vector_local": rel_vec_local.tolist(),
            "rotation_delta_matrix": rel_rotation.tolist(),
            "yaw_delta_deg": yaw_deg_pair,
            "yaw_cumulative_deg": yaw_deg_sum,
            "relative_distance": rel_distance,
            "relative_direction": rel_direction,
            "camera_motion_direction": motion_direction,
            "camera_motion_rule": motion_rule,
            "answer": (motion_direction if answer_mode == "motion" else rel_direction).lower(),
        }

class CameraElevationTool(BaseTool):
    """Camera elevation tool."""

    def __init__(self):
        super().__init__(
            name="camera_elevation_tool",
            description="Compare target camera elevation against the reference"
        )

    @staticmethod
    def _select_camera_entity(
        camera_entity_1: Dict,
        camera_entity_2: Dict,
        image_idx: int,
    ) -> Optional[Dict]:
        if int(image_idx) == 1:
            return camera_entity_1
        if int(image_idx) == 2:
            return camera_entity_2
        return None

    def execute(
        self,
        camera_entity_1: Dict,
        camera_entity_2: Dict,
        camera_reference_image_idx: int = 1,
        camera_target_image_idx: int = 2,
        same_level_threshold_m: float = 0.15,
        **kwargs
    ) -> Dict:
        ref_camera = self._select_camera_entity(
            camera_entity_1,
            camera_entity_2,
            int(camera_reference_image_idx),
        )
        tgt_camera = self._select_camera_entity(
            camera_entity_1,
            camera_entity_2,
            int(camera_target_image_idx),
        )
        if not isinstance(ref_camera, dict) or not isinstance(tgt_camera, dict):
            return {
                "success": False,
                "error": "camera_elevation_tool requires valid reference and target cameras"
            }

        ref_pos = np.array(ref_camera.get("position", [0.0, 0.0, 0.0]), dtype=float)
        tgt_pos = np.array(tgt_camera.get("position", [0.0, 0.0, 0.0]), dtype=float)
        delta_z = float(tgt_pos[2] - ref_pos[2])

        if abs(delta_z) <= float(same_level_threshold_m):
            answer = "same_level"
        else:
            answer = "higher" if delta_z > 0 else "lower"

        return {
            "success": True,
            "reference_image_idx": int(camera_reference_image_idx),
            "target_image_idx": int(camera_target_image_idx),
            "reference_height_m": round(float(ref_pos[2]), 3),
            "target_height_m": round(float(tgt_pos[2]), 3),
            "height_delta_m": round(delta_z, 3),
            "answer": answer,
        }
