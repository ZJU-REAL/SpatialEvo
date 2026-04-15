#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent / "data_scripts"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from scannetpp_pipeline import DEFAULT_OUTPUT_ROOT  # type: ignore
from scannetpp_pipeline import DEFAULT_RAW_ROOT  # type: ignore
from scannetpp_pipeline import convert_one_scene  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one local ScanNet++ scene into the ScanNet-like scene schema.",
    )
    parser.add_argument("--scene-id", required=True, help="ScanNet++ scene id, e.g. 036bce3393")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--depth-root",
        default="/mnt/jfs/lidingm/data/dataset/scannetpp/processed",
        help="Directory containing pre-rendered depth for ScanNet++ scenes.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-name", default=None, help="Default: scene<scene-id>_00")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all valid frames")
    parser.add_argument("--frame-processed-max", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene_name_pattern = args.scene_name or "scene{scene_id}_00"
    if "{scene_id}" not in scene_name_pattern and args.scene_name is not None:
        scene_name_pattern = args.scene_name

    result = convert_one_scene(
        raw_root=Path(args.raw_root),
        depth_root=Path(args.depth_root) if args.depth_root else None,
        output_root=Path(args.output_root),
        split=args.split,
        scene_id=args.scene_id,
        scene_name_pattern=scene_name_pattern,
        max_frames=args.max_frames,
        frame_processed_max=args.frame_processed_max,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
