"""Invalid-recovery example for the public DGE interface."""

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
        enable_invalid_recovery=True,
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


REQUESTS = [
    {
        "name": "recovery_scene_relative_direction",
        "mode": "validate",
        "input": {
            "task_type": "relative_direction_hard",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "If I stand by the stool and face the counter, where is the microwave?",
        },
    },
    {
        "name": "recovery_pair_missing_object",
        "mode": "validate",
        "input": {
            "task_type": "position_cam_obj",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "image_paths": [color_image(1200), color_image(1220)],
            "question": "Where is the clock relative to the camera in Image 2?",
        },
    },
]


def main() -> int:
    simulator = build_simulator()

    for idx, item in enumerate(REQUESTS, 1):
        print("-" * 80)
        print(f"[{idx:02d}] {item['name']}")
        print_json("input", item["input"])
        output = simulator.validate_and_answer(item["input"], return_intermediate=True)
        print_json("output", output)
    print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
