"""View tools."""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from .base_tool import BaseTool

class BirdEyeViewTool(BaseTool):
    """Bird eye view tool."""
    
    def __init__(self):
        super().__init__(
            name="bird_eye_view_tool",
            description="Generate a top-down view with camera and object markers"
        )
    
    def execute(
        self,
        scene_data_path: str,
        scene_id: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        camera_positions: Optional[List[Dict]] = None,
        object_positions: Optional[List[Dict]] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Execute."""
        if metadata_dir and scene_id:
            vis_path = os.path.join(metadata_dir, scene_id, f"{scene_id}_3d_vis.jpg")
            if os.path.exists(vis_path):
                return vis_path

        if scene_id:
            vis_path = os.path.join(scene_data_path, "..", "metadata", scene_id, f"{scene_id}_3d_vis.jpg")
            if os.path.exists(vis_path):
                return vis_path

        if output_path is None:
            output_path = "bird_eye_view.png"
        

        
        return output_path
    
    def mark_positions_on_view(
        self,
        base_view_path: str,
        positions: List[Tuple[float, float]],
        labels: List[str],
        output_path: str,
        **kwargs
    ) -> str:
        """Mark positions on view."""

        print(f"Plotting {len(positions)} positions on the bird's-eye view")
        return output_path

class PointCloudTool(BaseTool):
    """Point cloud tool."""
    
    def __init__(self):
        super().__init__(
            name="point_cloud_tool",
            description="Load and process 3D point clouds"
        )
    
    def execute(
        self,
        point_cloud_path: str,
        **kwargs
    ) -> np.ndarray:
        """Execute."""

        

        return np.zeros((1000, 3))
    
    def get_room_bounds(
        self,
        point_cloud: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Get room bounds."""
        min_coords = point_cloud.min(axis=0)
        max_coords = point_cloud.max(axis=0)
        
        return {
            "x": (min_coords[0], max_coords[0]),
            "y": (min_coords[1], max_coords[1]),
            "z": (min_coords[2], max_coords[2])
        }
