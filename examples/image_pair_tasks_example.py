"""Image-pair task example for the public DGE interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.world_simulator import WorldSimulator


SCENE_ID = "scene0000_01"
SCANNET_ROOT = "/mnt/jfs/lidingm/data/dataset/ScanNet/train"
METADATA_ROOT = "/mnt/jfs/lidingm/data/dataset/ScanNet/metadata"


def color_image(frame_id: int | str, scene_id: str = SCENE_ID) -> str:
    return f"{SCANNET_ROOT}/{scene_id}/color/{int(frame_id)}.jpg"


def build_simulator() -> WorldSimulator:
    simulator = WorldSimulator(
        scannet_root=SCANNET_ROOT,
        enable_vlm=True,
        enable_invalid_recovery=False,
    )
    retries = 2
    timeout = 20
    for tool_name in ("vlm_tool", "invalid_recovery_tool"):
        tool = simulator.tools.get(tool_name)
        if tool is None:
            continue
        if hasattr(tool, "max_retries"):
            tool.max_retries = retries
        if hasattr(tool, "timeout"):
            tool.timeout = timeout
        backend = getattr(tool, "backend", None)
        if backend is not None:
            if hasattr(backend, "max_retries"):
                backend.max_retries = retries
            if hasattr(backend, "timeout"):
                backend.timeout = timeout
    return simulator


def print_json(title: str, value: Any) -> None:
    print(f"{title}:")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


PAIR_IMAGES = [color_image(960), color_image(1060)]

REQUESTS = [
    {
        "name": "image_pair_summary",
        "mode": "summary",
        "input": {
            "summary_type": "image_pair",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "min_visibility": 0.1,
        },
    },
    {
        "name": "position_cam_cam_valid",
        "mode": "validate",
        "input": {
            "task_type": "position_cam_cam",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": [color_image(120), color_image(240)],
            "question": "When you took Image one, where was the camera for Image 2 relative to you?",
        },
    },
    {
        "name": "visibility_compare_valid",
        "mode": "validate",
        "input": {
            "task_type": "visibility_compare",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "In which image is the guitar more visible?",
        },
    },
    {
        "name": "motion_camera_valid",
        "mode": "validate",
        "input": {
            "task_type": "motion_camera",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Based on the continuous images, in which direction is the camera rotating?",
        },
    },
    {
        "name": "elevation_cam_cam_valid",
        "mode": "validate",
        "input": {
            "task_type": "elevation_cam_cam",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Is the camera for Image 2 higher or lower than Image 1?",
        },
    },
    {
        "name": "position_cam_obj_valid",
        "mode": "validate",
        "input": {
            "task_type": "position_cam_obj",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Where is the sink relative to the camera in Image 1?",
        },
    },
    {
        "name": "position_cam_reg_valid",
        "mode": "validate",
        "input": {
            "task_type": "position_cam_reg",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Where is the sleeping area relative to the camera in Image 2?",
        },
    },
    {
        "name": "attribute_measurement_valid",
        "mode": "validate",
        "input": {
            "task_type": "attribute_measurement",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Which object is longer in Image 2, the guitar or the sink?",
        },
    },
    {
        "name": "visibility_compare_invalid_missing_label",
        "mode": "validate",
        "input": {
            "task_type": "visibility_compare",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "In which image is the guitar more visible?",
        },
    },
    {
        "name": "position_cam_reg_invalid_unknown_region",
        "mode": "validate",
        "input": {
            "task_type": "position_cam_reg",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Where is the garage area relative to the camera in Image 2?",
        },
    },
    {
        "name": "attribute_measurement_invalid_same_object",
        "mode": "validate",
        "input": {
            "task_type": "attribute_measurement",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": PAIR_IMAGES,
            "question": "Which object is longer in Image 2, the sink or the night stand?",
        },
    },
]


def main() -> int:
    simulator = build_simulator()

    for idx, item in enumerate(REQUESTS, 1):
        print("-" * 80)
        print(f"[{idx:02d}] {item['name']}")
        print_json("input", item["input"])
        if item["mode"] == "summary":
            output = simulator.get_environment_summary(item["input"])
        else:
            output = simulator.validate_and_answer(item["input"], return_intermediate=False)
        print_json("output", output)
    print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
