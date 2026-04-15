#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCANNET_PROCESS_ROOT = Path(__file__).resolve().parents[1]
if str(SCANNET_PROCESS_ROOT) not in sys.path:
    sys.path.append(str(SCANNET_PROCESS_ROOT))

from utils.bbox_extractor import align_mesh_vertices  # type: ignore
from utils.bbox_extractor import export_2d_bbox  # type: ignore
from utils.bbox_extractor import export_3d_bbox  # type: ignore
from utils.bbox_extractor import get_object_id_to_label_id  # type: ignore
from utils.bbox_extractor import read_label_mapping  # type: ignore
from utils.bbox_extractor import read_mesh_vertices_rgb  # type: ignore
from utils.bbox_extractor import read_scene_axis_alignment  # type: ignore
from utils.metadata_builder import export_frame_metadata  # type: ignore
from utils.metadata_builder import export_scene_metadata  # type: ignore
from utils.room_size_extractor import calculate_room_area  # type: ignore


DEFAULT_DATASET_ROOT = "/mnt/jfs/lidingm/data/dataset/ScanNet"
DEFAULT_LABEL_MAP_FILE = str(SCANNET_PROCESS_ROOT / "scannet_metadata" / "scannetv2-labels-v2.tsv")


def load_existing_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def sorted_frame_names(frame_dir: Path) -> list[str]:
    frames = [path.name for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    frames.sort(key=lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else Path(name).stem)
    return frames


def process_scene(
    scene_id: str,
    scene_frames: list[str],
    scene_root: str,
    metadata_root: str,
    raw_id_map: dict[str, int],
    id_label_map: dict[int, str],
    frame_type: str,
    overwrite: bool,
    visualize: bool = False,
) -> dict[str, Any]:
    scene_root_path = Path(scene_root)
    metadata_root_path = Path(metadata_root)
    scene_dir = scene_root_path / scene_id
    ply_file = scene_dir / f"{scene_id}_vh_clean.ply"
    agg_file = scene_dir / f"{scene_id}_vh_clean.aggregation.json"
    seg_file = scene_dir / f"{scene_id}_vh_clean.segs.json"
    meta_file = scene_dir / f"{scene_id}.txt"
    scene_metadata_out_dir = metadata_root_path / scene_id
    frame_metadata_out_dir = scene_metadata_out_dir / frame_type

    scene_metadata_out_dir.mkdir(parents=True, exist_ok=True)
    frame_metadata_out_dir.mkdir(parents=True, exist_ok=True)

    mesh_vertices = read_mesh_vertices_rgb(str(ply_file))
    axis_align_matrix = read_scene_axis_alignment(str(meta_file))
    aligned_mesh_vertices = align_mesh_vertices(mesh_vertices, axis_align_matrix)
    object_id_to_label_id, object_id_to_segs, instance_ids = get_object_id_to_label_id(
        str(agg_file),
        str(seg_file),
        raw_id_map,
    )

    scene_metadata_file = scene_metadata_out_dir / f"{scene_id}.json"
    scene_metadata = None if overwrite else load_existing_json(scene_metadata_file)
    if scene_metadata is None:
        bboxes_3d = export_3d_bbox(
            axis_align_matrix,
            aligned_mesh_vertices,
            object_id_to_label_id,
            object_id_to_segs,
            instance_ids,
            id_label_map,
            str(scene_dir),
            visualize=visualize,
        )
        room_size = calculate_room_area(aligned_mesh_vertices, visualize=visualize)
        scene_metadata = export_scene_metadata(scene_id, bboxes_3d, room_size)
        scene_metadata_file.write_text(json.dumps(scene_metadata, ensure_ascii=False, indent=4), encoding="utf-8")

    obj_id_to_3d_bbox = {int(obj["object_id"]): obj["3d_bbox"] for obj in scene_metadata["objects"]}
    obj_id_to_3d_loc = {int(obj["object_id"]): obj["3d_location"] for obj in scene_metadata["objects"]}
    obj_id_to_size = {int(obj["object_id"]): obj["size"] for obj in scene_metadata["objects"]}

    camera_intrinsic_file = scene_dir / "intrinsic_color.txt"
    frame_written = 0
    for frame_name in scene_frames:
        color_file = scene_dir / frame_type / frame_name
        depth_file = scene_dir / "depth" / frame_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        pose_file = scene_dir / "pose" / frame_name.replace(".jpg", ".txt").replace(".jpeg", ".txt")

        frame_stem = Path(frame_name).stem
        frame_metadata_file = frame_metadata_out_dir / f"{frame_stem}.json"
        if frame_metadata_file.exists() and not overwrite:
            continue
        if not color_file.exists() or not depth_file.exists() or not pose_file.exists():
            continue

        bboxes_2d = export_2d_bbox(
            axis_align_matrix,
            aligned_mesh_vertices,
            object_id_to_label_id,
            object_id_to_segs,
            instance_ids,
            str(camera_intrinsic_file),
            str(color_file),
            str(depth_file),
            str(pose_file),
            id_label_map,
            visualize=visualize,
        )
        frame_metadata = export_frame_metadata(
            scene_id,
            frame_name,
            bboxes_2d,
            obj_id_to_3d_bbox,
            obj_id_to_3d_loc,
            obj_id_to_size,
        )
        frame_metadata_file.write_text(json.dumps(frame_metadata, ensure_ascii=False, indent=4), encoding="utf-8")
        frame_written += 1

    return {
        "scene_id": scene_id,
        "status": "processed",
        "frames": len(scene_frames),
        "frame_metadata_written": frame_written,
        "objects": len(scene_metadata["objects"]),
    }


def process_scene_wrapper(task: tuple[Any, ...]) -> dict[str, Any]:
    return process_scene(*task)


def chunk_list(items: list[str], chunk_count: int) -> list[list[str]]:
    if chunk_count <= 0:
        return [items]
    k, m = divmod(len(items), chunk_count)
    return [items[index * k + min(index, m) : (index + 1) * k + min(index + 1, m)] for index in range(chunk_count)]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract object and frame metadata from ScanNet-like ScanNet scenes.",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--metadata-root", default=None, help="Default: <dataset-root>/metadata")
    parser.add_argument("--label-map-file", default=DEFAULT_LABEL_MAP_FILE)
    parser.add_argument("--scene-ids", default=None, help="Comma separated scene ids")
    parser.add_argument("--scene-ids-file", default=None, help="One scene id per line")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-type", default="frame_processed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    scene_root = dataset_root / args.split
    metadata_root = Path(args.metadata_root) if args.metadata_root else dataset_root / "metadata"

    raw_id_map = read_label_mapping(args.label_map_file, label_from="raw_category", label_to="nyu40id")
    id_label_map = read_label_mapping(args.label_map_file, label_from="nyu40id", label_to="nyu40class")

    requested_scene_ids = parse_scene_ids(args.scene_ids, args.scene_ids_file)
    if requested_scene_ids:
        scene_ids = requested_scene_ids
    else:
        scene_ids = sorted(path.name for path in scene_root.iterdir() if path.is_dir())

    tasks: list[tuple[Any, ...]] = []
    for scene_id in scene_ids:
        frame_dir = scene_root / scene_id / args.frame_type
        if not frame_dir.is_dir():
            continue
        scene_frames = sorted_frame_names(frame_dir)
        if not scene_frames:
            continue
        tasks.append(
            (
                scene_id,
                scene_frames,
                str(scene_root),
                str(metadata_root),
                raw_id_map,
                id_label_map,
                args.frame_type,
                args.overwrite,
                args.visualize,
            )
        )

    if not tasks:
        print(json.dumps({"summary": {"scene_count": 0, "processed": 0, "failed": 0}}, ensure_ascii=False, indent=2))
        return

    batch_count = max(1, (len(tasks) // max(1, args.batch_size)) + 1)
    task_batches = chunk_list(tasks, batch_count)
    results = {"processed": 0, "failed": 0}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch in tqdm(task_batches, desc="Processing batches"):
            futures = [executor.submit(process_scene_wrapper, task) for task in batch]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Scenes {len(batch)}"):
                try:
                    result = future.result()
                    results["processed"] += 1
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                except Exception as exc:
                    results["failed"] += 1
                    print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)

    summary = {
        "scene_count": len(tasks),
        "processed": results["processed"],
        "failed": results["failed"],
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
