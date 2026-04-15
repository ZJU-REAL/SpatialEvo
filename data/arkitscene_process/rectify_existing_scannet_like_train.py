#!/usr/bin/env python3
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import export_scannet_like as util


RECTIFIED_MARKER_NAME = "upright_rectified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectify existing ScanNet-like ARKitScenes train scenes in-place.",
    )
    parser.add_argument(
        "--train-root",
        required=True,
        help="Path to ARKitScenes_scannet_like/train",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Parallel workers.",
    )
    parser.add_argument(
        "--selected-frame-count",
        type=int,
        default=32,
        help="Uniformly sampled frame_processed image count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run scenes even if upright_rectified marker exists.",
    )
    parser.add_argument(
        "--scene-id",
        nargs="+",
        default=None,
        help="Optional scene ids to process.",
    )
    return parser.parse_args()


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=float)


def matrix_to_intrinsic_record(
    path: Path,
    matrix: np.ndarray,
    image_path: Path,
) -> util.IntrinsicRecord:
    with Image.open(image_path) as image:
        width, height = image.size
    return util.IntrinsicRecord(
        timestamp=0.0,
        path=path,
        width=width,
        height=height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
    )


def load_manifest(scene_dir: Path) -> Dict:
    return json.loads((scene_dir / "manifest.json").read_text())


def load_frame_map(scene_dir: Path) -> Dict[int, Dict[str, str]]:
    frame_map_path = scene_dir / "frame_map.csv"
    if not frame_map_path.exists():
        return {}

    rows: Dict[int, Dict[str, str]] = {}
    with frame_map_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[int(row["export_index"])] = row
    return rows


def load_pose_map(traj_file: Optional[Path]) -> Dict[str, np.ndarray]:
    pose_map: Dict[str, np.ndarray] = {}
    if traj_file is None or not traj_file.exists():
        return pose_map

    with traj_file.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            timestamp, pose = util.traj_string_to_matrix(line)
            pose_map[util.normalize_timestamp(timestamp)] = pose
    return pose_map


def infer_depth_intrinsic_path(source_kind: str, row: Dict[str, str]) -> Optional[Path]:
    intrinsic_path_str = row.get("intrinsic_path", "")
    intrinsic_path = Path(intrinsic_path_str) if intrinsic_path_str else None
    depth_path_str = row.get("depth_path", "")
    if not depth_path_str:
        return intrinsic_path

    depth_path = Path(depth_path_str)
    if source_kind == "raw_vga":
        candidate = depth_path.parent.parent / "lowres_wide_intrinsics" / f"{depth_path.stem}.pincam"
        if candidate.exists():
            return candidate
    return intrinsic_path


def current_color_path(scene_dir: Path, frame_index: int) -> Optional[Path]:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = scene_dir / "color" / f"{frame_index}{suffix}"
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return None


def current_depth_path(scene_dir: Path, frame_index: int) -> Optional[Path]:
    candidate = scene_dir / "depth" / f"{frame_index}.png"
    if candidate.exists() or candidate.is_symlink():
        return candidate
    return None


def write_manifest(path: Path, manifest: Dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def process_scene(
    scene_dir: Path,
    overwrite: bool,
    selected_frame_count: int,
) -> str:
    marker = scene_dir / RECTIFIED_MARKER_NAME
    if marker.exists() and not overwrite:
        return f"[skip] {scene_dir.name} already rectified"

    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        return f"[skip] {scene_dir.name} missing manifest.json"

    manifest = load_manifest(scene_dir)
    metadata = manifest.get("metadata", {})
    sky_direction = util.normalize_sky_direction(metadata.get("sky_direction")) or util.normalize_sky_direction(
        manifest.get("sky_direction")
    )
    if sky_direction is None:
        sky_direction = "Up"

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    frame_processed_dir = scene_dir / "frame_processed"

    frame_map = load_frame_map(scene_dir)
    pose_map = load_pose_map(Path(manifest["traj_file"])) if manifest.get("traj_file") else {}
    source_kind = manifest.get("source_kind", "")

    if frame_map:
        frame_indices = sorted(frame_map)
    else:
        frame_indices = sorted(
            int(path.stem)
            for path in pose_dir.glob("*.txt")
            if path.stem.isdigit()
        )

    if not frame_indices:
        return f"[skip] {scene_dir.name} has no frame indices"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    available_frame_indices: List[int] = []
    first_color_intrinsic: Optional[util.IntrinsicRecord] = None
    first_depth_intrinsic: Optional[util.IntrinsicRecord] = None
    skipped_frames = 0

    for frame_index in frame_indices:
        row = frame_map.get(frame_index)
        color_dst = color_dir / f"{frame_index}.jpg"
        depth_dst = depth_dir / f"{frame_index}.png"
        pose_dst = pose_dir / f"{frame_index}.txt"
        intrinsic_dst = intrinsic_dir / f"{frame_index}.txt"

        if (
            sky_direction == "Up"
            and color_dst.exists()
            and depth_dst.exists()
            and pose_dst.exists()
            and intrinsic_dst.exists()
        ):
            available_frame_indices.append(frame_index)
            continue

        color_src = current_color_path(scene_dir, frame_index)
        depth_src = current_depth_path(scene_dir, frame_index)
        pose_matrix: Optional[np.ndarray] = read_matrix(pose_dst) if pose_dst.exists() else None
        color_intrinsic: Optional[util.IntrinsicRecord] = None
        depth_intrinsic: Optional[util.IntrinsicRecord] = None

        if row is not None:
            if row.get("color_path"):
                color_src = Path(row["color_path"])
            if row.get("depth_path"):
                depth_src = Path(row["depth_path"])

            if row.get("timestamp"):
                pose_matrix = pose_map.get(util.normalize_timestamp(float(row["timestamp"])), pose_matrix)

            intrinsic_path_str = row.get("intrinsic_path", "")
            if intrinsic_path_str:
                color_intrinsic_path = Path(intrinsic_path_str)
                if color_intrinsic_path.exists():
                    color_intrinsic = util.parse_pincam(color_intrinsic_path)

            depth_intrinsic_path = infer_depth_intrinsic_path(source_kind, row)
            if depth_intrinsic_path is not None and depth_intrinsic_path.exists():
                depth_intrinsic = util.parse_pincam(depth_intrinsic_path)

        if color_src is None or depth_src is None or not color_src.exists() or not depth_src.exists() or pose_matrix is None:
            skipped_frames += 1
            continue

        if color_intrinsic is None:
            if intrinsic_dst.exists():
                color_intrinsic = matrix_to_intrinsic_record(intrinsic_dst, read_matrix(intrinsic_dst), color_src)
            elif (scene_dir / "intrinsic_color.txt").exists():
                color_intrinsic = matrix_to_intrinsic_record(
                    scene_dir / "intrinsic_color.txt",
                    read_matrix(scene_dir / "intrinsic_color.txt"),
                    color_src,
                )
        if color_intrinsic is None:
            skipped_frames += 1
            continue

        if depth_intrinsic is None:
            depth_intrinsic_txt = scene_dir / "intrinsic_depth.txt"
            if depth_intrinsic_txt.exists():
                depth_intrinsic = matrix_to_intrinsic_record(
                    depth_intrinsic_txt,
                    read_matrix(depth_intrinsic_txt),
                    depth_src,
                )
            else:
                depth_intrinsic = color_intrinsic

        rectified_pose = util.transform_pose_for_sky_direction(pose_matrix, sky_direction)
        rectified_color_intrinsic = util.transform_intrinsic(color_intrinsic, sky_direction)
        rectified_depth_intrinsic = util.transform_intrinsic(depth_intrinsic, sky_direction)

        util.materialize_color_frame(color_src, color_dst, "symlink", sky_direction)
        util.materialize_depth_frame(depth_src, depth_dst, "symlink", sky_direction)
        util.write_matrix(pose_dst, rectified_pose)
        util.write_matrix(intrinsic_dst, rectified_color_intrinsic.matrix_4x4)

        if first_color_intrinsic is None:
            first_color_intrinsic = rectified_color_intrinsic
        if first_depth_intrinsic is None:
            first_depth_intrinsic = rectified_depth_intrinsic

        available_frame_indices.append(frame_index)

    if not available_frame_indices:
        return f"[skip] {scene_dir.name} no usable frames (skipped={skipped_frames})"

    available_frame_indices = sorted(set(available_frame_indices))
    selected_frame_ids = util.select_exported_frame_ids(available_frame_indices, selected_frame_count)

    if frame_processed_dir.exists():
        util.remove_tree(frame_processed_dir)
    frame_processed_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in selected_frame_ids:
        color_path = color_dir / f"{frame_index}.jpg"
        if color_path.exists() or color_path.is_symlink():
            util.materialize_path(color_path, frame_processed_dir / f"{frame_index}.jpg", "symlink")

    if first_color_intrinsic is not None:
        util.write_matrix(scene_dir / "intrinsic_color.txt", first_color_intrinsic.matrix_4x4)
    if first_depth_intrinsic is not None:
        util.write_matrix(scene_dir / "intrinsic_depth.txt", first_depth_intrinsic.matrix_4x4)

    manifest["sky_direction"] = sky_direction
    manifest["selected_frame_ids"] = selected_frame_ids
    manifest["selected_frame_count"] = len(selected_frame_ids)
    manifest["upright_rectified"] = True
    manifest["upright_rectified_version"] = "scene_train_v1"
    write_manifest(manifest_path, manifest)
    marker.write_text("ok\n")

    return (
        f"[done] {scene_dir.name} sky={sky_direction} "
        f"frames={len(available_frame_indices)} frame_processed={len(selected_frame_ids)} "
        f"skipped_frames={skipped_frames}"
    )


def collect_scenes(train_root: Path, requested_scene_ids: Optional[List[str]]) -> List[Path]:
    requested = set(requested_scene_ids or [])
    scenes = []
    for scene_dir in sorted(train_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if requested and scene_dir.name not in requested:
            continue
        scenes.append(scene_dir)
    return scenes


def main() -> int:
    args = parse_args()
    train_root = Path(args.train_root).resolve()
    scene_dirs = collect_scenes(train_root, args.scene_id)
    total = len(scene_dirs)
    if total == 0:
        print("No scenes selected.")
        return 0

    print(f"Rectifying {total} existing train scenes with jobs={args.jobs}")
    if args.jobs <= 1 or total == 1:
        for idx, scene_dir in enumerate(scene_dirs, start=1):
            print(f"[{idx}/{total}] {process_scene(scene_dir, args.overwrite, args.selected_frame_count)}", flush=True)
        return 0

    max_workers = min(args.jobs, total)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_scene = {
            executor.submit(process_scene, scene_dir, args.overwrite, args.selected_frame_count): scene_dir
            for scene_dir in scene_dirs
        }
        completed = 0
        for future in as_completed(future_to_scene):
            completed += 1
            scene_dir = future_to_scene[future]
            try:
                message = future.result()
            except Exception as error:
                message = f"[error] {scene_dir.name}: {error}"
            print(f"[{completed}/{total}] {message}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
