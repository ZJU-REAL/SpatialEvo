"""Scene-task example for the public DGE interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.world_simulator import WorldSimulator


# SCENE_ID = "scene0000_01"
# SCANNET_ROOT = "/mnt/jfs/lidingm/data/dataset/ScanNet/train"
# METADATA_ROOT = "/mnt/jfs/lidingm/data/dataset/ScanNet/metadata"

# SCENE_ID = "scene0a5c013435_00"
# SCANNET_ROOT = "/mnt/jfs/lidingm/data/dataset/scannetpp/train"
# METADATA_ROOT = "/mnt/jfs/lidingm/data/dataset/scannetpp/metadata"

SCENE_ID = "scene41048113_00"
SCANNET_ROOT = "/mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like/train"
METADATA_ROOT = "/mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like/metadata"

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


REQUESTS = [
    {
        "name": "scene_summary",
        "mode": "summary",
        "input": {
            "summary_type": "scene",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
        },
    },
    {
        "name": "object_counting_valid",
        "mode": "validate",
        "input": {
            "task_type": "object_counting",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "How many tables are in this room?",
        },
    },
    {
        "name": "object_size_valid",
        "mode": "validate",
        "input": {
            "task_type": "object_size",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "What is the size of the sink in centimeters?",
        },
    },
    {
        "name": "absolute_distance_valid",
        "mode": "validate",
        "input": {
            "task_type": "absolute_distance",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "What is the direct distance between the ladder and the sink in meters?",
        },
    },
    {
        "name": "relative_distance_valid",
        "mode": "validate",
        "input": {
            "task_type": "relative_distance",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "Which of these objects, ladder, floor cleaner, or cleaning trolley, is closest to the sink?",
        },
    },
    {
        "name": "relative_direction_hard_valid",
        "mode": "validate",
        "input": {
            "task_type": "relative_direction",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "If I stand by the floor cleaner and face the sink, where is the ladder?",
        },
    },
    {
        "name": "room_size_valid",
        "mode": "validate",
        "input": {
            "task_type": "room_size",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "What is the size of this room in square meters?",
        },
    },
    {
        "name": "absolute_distance_invalid_same_object",
        "mode": "validate",
        "input": {
            "task_type": "absolute_distance",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "What is the direct distance between the towel and the bike in meters?",
        },
    },
    {
        "name": "relative_distance_invalid_overlap_target",
        "mode": "validate",
        "input": {
            "task_type": "relative_distance",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "Which of these objects, stool, table, or door, is closest to the stool?",
        },
    },
    {
        "name": "relative_direction_hard_invalid_duplicate_roles",
        "mode": "validate",
        "input": {
            "task_type": "relative_direction_hard",
            "scene_id": SCENE_ID,
            "metadata_dir": METADATA_ROOT,
            "question": "If I stand by the stool and face the stool, where is the microwave?",
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
