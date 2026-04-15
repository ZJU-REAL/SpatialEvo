#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "/mnt/jfs/lidingm/data/dataset/ScanNet"


def load_manifest(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        result: dict[str, list[str]] = {}
        for scene_id, frame_ids in payload.items():
            if not isinstance(scene_id, str) or not isinstance(frame_ids, list):
                continue
            normalized = []
            for frame_id in frame_ids:
                frame_name = str(frame_id).strip()
                if not frame_name:
                    continue
                if "." not in frame_name:
                    frame_name = f"{frame_name}.jpg"
                normalized.append(frame_name)
            result[scene_id] = normalized
        return result

    if isinstance(payload, list):
        result = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            scene_id = str(item.get("scene_id", "")).strip()
            frame_ids = item.get("frame_ids", [])
            if not scene_id or not isinstance(frame_ids, list):
                continue
            normalized = []
            for frame_id in frame_ids:
                frame_name = str(frame_id).strip()
                if not frame_name:
                    continue
                if "." not in frame_name:
                    frame_name = f"{frame_name}.jpg"
                normalized.append(frame_name)
            result[scene_id] = normalized
        return result

    raise TypeError(f"Unsupported manifest format: {path}")


def parse_scene_ids(scene_ids_text: str | None, scene_ids_file: str | None) -> list[str]:
    scene_ids: list[str] = []
    if scene_ids_text:
        scene_ids.extend([item.strip() for item in scene_ids_text.split(",") if item.strip()])
    if scene_ids_file:
        with open(scene_ids_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    scene_ids.append(line)
    return sorted(set(scene_ids))


def materialize_scene_frames(
    scene_dir: Path,
    frame_names: list[str],
    frame_type: str,
    link_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    color_dir = scene_dir / "color"
    frame_dir = scene_dir / frame_type
    if not color_dir.is_dir():
        raise FileNotFoundError(f"Missing color directory: {color_dir}")

    if overwrite and frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []
    for frame_name in frame_names:
        source = color_dir / frame_name
        target = frame_dir / frame_name
        if not source.is_file():
            missing.append(frame_name)
            continue
        if target.exists():
            if not overwrite:
                continue
            target.unlink()

        if link_mode == "symlink":
            target.symlink_to(source)
        else:
            shutil.copy2(source, target)
        copied += 1

    return {"copied": copied, "missing": missing}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize ScanNet selected frames into each scene's frame_processed directory.",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest", required=True, help="JSON mapping scene_id -> frame jpg names")
    parser.add_argument("--scene-ids", default=None, help="Comma separated scene ids")
    parser.add_argument("--scene-ids-file", default=None, help="One scene id per line")
    parser.add_argument("--frame-type", default="frame_processed")
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    scene_root = dataset_root / args.split
    manifest = load_manifest(Path(args.manifest))

    requested_scene_ids = parse_scene_ids(args.scene_ids, args.scene_ids_file)
    if requested_scene_ids:
        scene_ids = [scene_id for scene_id in requested_scene_ids if scene_id in manifest]
    else:
        scene_ids = sorted(manifest.keys())

    results: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_dir = scene_root / scene_id
        if not scene_dir.is_dir():
            results.append({"scene_id": scene_id, "status": "missing_scene_dir"})
            continue
        result = materialize_scene_frames(
            scene_dir=scene_dir,
            frame_names=manifest.get(scene_id, []),
            frame_type=args.frame_type,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        results.append(
            {
                "scene_id": scene_id,
                "status": "done",
                "copied": result["copied"],
                "missing": result["missing"],
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    summary = {
        "scene_count": len(results),
        "done": sum(1 for item in results if item["status"] == "done"),
        "missing_scene_dir": sum(1 for item in results if item["status"] == "missing_scene_dir"),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
