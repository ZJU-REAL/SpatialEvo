#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image


ARKIT_BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
TRAINING = "Training"
VALIDATION = "Validation"
SPLIT_MAP = {
    TRAINING: "train",
    VALIDATION: "val",
}
SKY_DIRECTION_ORDER = ["Up", "Left", "Down", "Right"]


@dataclass
class IntrinsicRecord:
    timestamp: float
    path: Path
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix_4x4(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[0, 0] = self.fx
        matrix[1, 1] = self.fy
        matrix[0, 2] = self.cx
        matrix[1, 2] = self.cy
        return matrix


@dataclass
class FrameRecord:
    index: int
    timestamp: float
    color_path: Path
    depth_path: Path
    pose: np.ndarray
    color_intrinsic: IntrinsicRecord
    depth_intrinsic: Optional[IntrinsicRecord]


@dataclass
class SceneExportPlan:
    video_id: str
    visit_id: Optional[int]
    split: str
    split_short: str
    scene_id: str
    source_kind: str
    color_dir: Path
    depth_dir: Path
    traj_file: Path
    intrinsic_dir: Path
    frames: List[FrameRecord]
    mesh_path: Optional[Path]
    annotation_path: Optional[Path]
    raw_scene_dir: Optional[Path]
    laser_scanner_dir: Optional[Path]
    metadata_row: Dict[str, str]
    sky_direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ARKitScenes into a ScanNet-like layout.",
    )
    parser.add_argument(
        "--download-root",
        required=True,
        help="Root directory of downloaded ARKitScenes assets.",
    )
    parser.add_argument(
        "--export-root",
        required=True,
        help="Destination root for ScanNet-like export.",
    )
    parser.add_argument(
        "--link-mode",
        default="symlink",
        choices=["symlink", "copy"],
        help="Whether to create symlinks or copies for large assets.",
    )
    parser.add_argument(
        "--ensure-raw-wide",
        action="store_true",
        help="Download raw wide/wide_intrinsics assets for upsampling subset before export.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files when downloading missing raw wide assets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild scenes even if a processed marker exists.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for auxiliary download/export steps.",
    )
    parser.add_argument(
        "--selected-frame-count",
        type=int,
        default=32,
        help="Number of ScanNet-like `frame_processed` images to export per scene.",
    )
    parser.add_argument(
        "--scene-id",
        nargs="+",
        default=None,
        help="Optional ScanNet-like scene ids to export, e.g. scene40777079_00.",
    )
    parser.add_argument(
        "--video-id",
        nargs="+",
        default=None,
        help="Optional ARKitScenes video ids to export, e.g. 40777079.",
    )
    return parser.parse_args()


def angle_axis_to_matrix3(angle_axis: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(angle_axis)
    if theta < 1e-12:
        return np.eye(3, dtype=float)

    axis = angle_axis / theta
    x, y, z = axis
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    one_minus_cos = 1.0 - cos_t

    return np.array(
        [
            [
                cos_t + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_t,
                x * z * one_minus_cos + y * sin_t,
            ],
            [
                y * x * one_minus_cos + z * sin_t,
                cos_t + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_t,
            ],
            [
                z * x * one_minus_cos - y * sin_t,
                z * y * one_minus_cos + x * sin_t,
                cos_t + z * z * one_minus_cos,
            ],
        ],
        dtype=float,
    )


def traj_string_to_matrix(traj_str: str) -> Tuple[float, np.ndarray]:
    tokens = traj_str.split()
    if len(tokens) != 7:
        raise ValueError(f"Unexpected traj line: {traj_str}")

    timestamp = float(tokens[0])
    angle_axis = np.asarray([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=float)
    translation = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=float)

    extrinsics = np.eye(4, dtype=float)
    extrinsics[:3, :3] = angle_axis_to_matrix3(angle_axis)
    extrinsics[:3, 3] = translation
    pose = np.linalg.inv(extrinsics)
    return timestamp, pose


def normalize_timestamp(value: float) -> str:
    return f"{value:.3f}"


def parse_frame_timestamp(path: Path) -> float:
    stem = path.stem
    if "_" in stem:
        maybe_video_id, maybe_timestamp = stem.split("_", 1)
        if maybe_video_id.isdigit():
            return float(maybe_timestamp)
    return float(stem)


def parse_pincam(path: Path) -> IntrinsicRecord:
    values = np.loadtxt(str(path), dtype=float)
    width, height, fx, fy, cx, cy = values.tolist()
    return IntrinsicRecord(
        timestamp=parse_frame_timestamp(path),
        path=path,
        width=int(round(width)),
        height=int(round(height)),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), matrix, fmt="%.6f")


def materialize_path(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        return

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def materialize_color_frame(src: Path, dst: Path, mode: str, sky_direction: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    needs_render = sky_direction != "Up" or src.suffix.lower() not in {".jpg", ".jpeg"}
    if not needs_render:
        materialize_path(src, dst, mode)
        return

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    with Image.open(src) as image:
        image = image.convert("RGB")
        image = rotate_image(image, sky_direction)
        image.save(dst, format="JPEG", quality=95)


def materialize_depth_frame(src: Path, dst: Path, mode: str, sky_direction: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sky_direction == "Up":
        materialize_path(src, dst, mode)
        return

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    with Image.open(src) as image:
        image = rotate_image(image, sky_direction)
        image.save(dst, format="PNG")


def remove_tree(path: Path, retries: int = 5, sleep_seconds: float = 0.2) -> None:
    last_error: Optional[OSError] = None
    for attempt in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except OSError as error:
            last_error = error
            time.sleep(sleep_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error


def list_files(directory: Path, suffixes: Iterable[str]) -> List[Path]:
    if not directory.exists():
        return []
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffix_set
    )


def build_numeric_index(values: Iterable[float]) -> List[float]:
    unique_values = sorted(set(float(v) for v in values))
    return unique_values


def nearest_value(value: float, sorted_values: List[float], tolerance: float) -> Optional[float]:
    if not sorted_values:
        return None

    idx = bisect.bisect_left(sorted_values, value)
    candidates: List[float] = []
    if idx < len(sorted_values):
        candidates.append(sorted_values[idx])
    if idx > 0:
        candidates.append(sorted_values[idx - 1])
    if not candidates:
        return None

    best = min(candidates, key=lambda item: abs(item - value))
    if abs(best - value) > tolerance:
        return None
    return best


def build_traj_lookup(traj_file: Path) -> Tuple[Dict[float, np.ndarray], List[float]]:
    poses: Dict[float, np.ndarray] = {}
    timestamps: List[float] = []
    with traj_file.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            timestamp, pose = traj_string_to_matrix(line)
            rounded = float(normalize_timestamp(timestamp))
            poses[rounded] = pose
            timestamps.append(rounded)
    return poses, build_numeric_index(timestamps)


def build_intrinsic_lookup(intrinsic_dir: Path) -> Tuple[Dict[float, IntrinsicRecord], List[float]]:
    records: Dict[float, IntrinsicRecord] = {}
    timestamps: List[float] = []
    for path in list_files(intrinsic_dir, {".pincam"}):
        record = parse_pincam(path)
        rounded = float(normalize_timestamp(record.timestamp))
        records[rounded] = record
        timestamps.append(rounded)
    return records, build_numeric_index(timestamps)


def build_frame_map(directory: Path, suffixes: Iterable[str]) -> Dict[float, Path]:
    frame_map: Dict[float, Path] = {}
    for path in list_files(directory, suffixes):
        rounded = float(normalize_timestamp(parse_frame_timestamp(path)))
        frame_map[rounded] = path
    return frame_map


def make_scene_id(video_id: str) -> str:
    return f"scene{video_id}_00"


def first_existing_path(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_3dod_frame_assets(frames_dir: Path) -> Optional[Dict[str, Path]]:
    asset_candidates = {
        "traj_file": [frames_dir / "lowres_wide.traj", frames_dir / "color.traj"],
        "intrinsic_dir": [frames_dir / "lowres_wide_intrinsics", frames_dir / "color_intrinsics"],
        "color_dir": [frames_dir / "lowres_wide", frames_dir / "wide"],
        "depth_dir": [frames_dir / "lowres_depth", frames_dir / "depth_densified"],
    }

    resolved_assets: Dict[str, Path] = {}
    for asset_name, candidates in asset_candidates.items():
        selected_path = first_existing_path(candidates)
        if selected_path is None:
            return None
        resolved_assets[asset_name] = selected_path
    return resolved_assets


def load_export_metadata(download_root: Path) -> List[Dict[str, str]]:
    candidate_paths = [
        download_root / "raw" / "metadata.csv",
        download_root / "3dod" / "metadata.csv",
    ]
    for metadata_path in candidate_paths:
        if metadata_path.exists():
            with metadata_path.open("r", newline="") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(
        f"Missing export metadata. Expected one of: {candidate_paths}"
    )


def select_exported_frame_ids(frame_indices: List[int], target_count: int) -> List[int]:
    frame_indices = [int(frame_index) for frame_index in frame_indices]
    frame_count = len(frame_indices)
    if frame_count <= 0:
        return []
    if target_count <= 0 or frame_count <= target_count:
        return frame_indices
    if target_count == 1:
        return [frame_indices[0]]

    selected: List[int] = []
    for sample_idx in range(target_count):
        position = round(sample_idx * (frame_count - 1) / (target_count - 1))
        if not selected or position != selected[-1]:
            selected.append(position)

    if len(selected) < target_count:
        existing = set(selected)
        for position in range(frame_count):
            if position not in existing:
                selected.append(position)
                existing.add(position)
            if len(selected) >= target_count:
                break

    return [frame_indices[position] for position in sorted(selected[:target_count])]


def normalize_sky_direction(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().capitalize()
    if normalized in SKY_DIRECTION_ORDER:
        return normalized
    return None


def decide_pose_orientation_index(pose: np.ndarray) -> int:
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    corr = np.matmul(z_orien, z_vec)
    return int(np.argmax(corr))


def sky_direction_from_pose(pose: np.ndarray) -> str:
    return SKY_DIRECTION_ORDER[decide_pose_orientation_index(pose)]


def resolve_sky_direction(row: Dict[str, str], fallback_pose: Optional[np.ndarray]) -> str:
    sky_direction = normalize_sky_direction(row.get("sky_direction"))
    if sky_direction is not None:
        return sky_direction
    if fallback_pose is not None:
        return sky_direction_from_pose(fallback_pose)
    return "Up"


def rotate_image(image: Image.Image, sky_direction: str) -> Image.Image:
    if sky_direction == "Up":
        return image
    if sky_direction == "Left":
        return image.transpose(Image.Transpose.ROTATE_270)
    if sky_direction == "Down":
        return image.transpose(Image.Transpose.ROTATE_180)
    if sky_direction == "Right":
        return image.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Unsupported sky direction: {sky_direction}")


def transform_intrinsic(record: IntrinsicRecord, sky_direction: str) -> IntrinsicRecord:
    width = int(record.width)
    height = int(record.height)
    fx = float(record.fx)
    fy = float(record.fy)
    cx = float(record.cx)
    cy = float(record.cy)

    if sky_direction == "Up":
        new_width, new_height = width, height
        new_fx, new_fy = fx, fy
        new_cx, new_cy = cx, cy
    elif sky_direction == "Left":
        new_width, new_height = height, width
        new_fx, new_fy = fy, fx
        new_cx, new_cy = height - 1 - cy, cx
    elif sky_direction == "Down":
        new_width, new_height = width, height
        new_fx, new_fy = fx, fy
        new_cx, new_cy = width - 1 - cx, height - 1 - cy
    elif sky_direction == "Right":
        new_width, new_height = height, width
        new_fx, new_fy = fy, fx
        new_cx, new_cy = cy, width - 1 - cx
    else:
        raise ValueError(f"Unsupported sky direction: {sky_direction}")

    return IntrinsicRecord(
        timestamp=record.timestamp,
        path=record.path,
        width=int(new_width),
        height=int(new_height),
        fx=float(new_fx),
        fy=float(new_fy),
        cx=float(new_cx),
        cy=float(new_cy),
    )


def camera_rotation_matrix_for_sky_direction(sky_direction: str) -> np.ndarray:
    if sky_direction == "Up":
        return np.eye(3, dtype=float)
    if sky_direction == "Left":
        return np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if sky_direction == "Down":
        return np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if sky_direction == "Right":
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported sky direction: {sky_direction}")


def transform_pose_for_sky_direction(pose: np.ndarray, sky_direction: str) -> np.ndarray:
    camera_rotation = np.eye(4, dtype=float)
    camera_rotation[:3, :3] = camera_rotation_matrix_for_sky_direction(sky_direction)
    return pose @ camera_rotation.T


def load_point_cloud_mapping(download_root: Path) -> Dict[int, List[str]]:
    mapping_path = download_root / "laser_scanner_point_clouds_mapping.csv"
    if not mapping_path.exists():
        return {}
    mapping: Dict[int, List[str]] = {}
    with mapping_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            visit_id = int(row["visit_id"])
            mapping.setdefault(visit_id, []).append(row["laser_scanner_point_clouds_id"])
    return mapping


def build_3dod_index(
    download_root: Path,
    allowed_video_ids: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, Path]]:
    index: Dict[Tuple[str, str], Dict[str, Path]] = {}
    threedod_root = download_root / "3dod"
    if not threedod_root.exists():
        return index

    for split in [TRAINING, VALIDATION]:
        split_dir = threedod_root / split
        if not split_dir.exists():
            continue
        if allowed_video_ids is not None:
            candidate_frame_dirs = [
                split_dir / video_id / f"{video_id}_frames" for video_id in sorted(allowed_video_ids)
            ]
        else:
            candidate_frame_dirs = split_dir.rglob("*_frames")

        for frames_dir in candidate_frame_dirs:
            if not frames_dir.is_dir() or not frames_dir.name.endswith("_frames"):
                continue
            video_id = frames_dir.name[: -len("_frames")]
            scene_dir = frames_dir.parent
            resolved_assets = resolve_3dod_frame_assets(frames_dir)
            if resolved_assets is None:
                continue

            index[(split, video_id)] = {
                "scene_dir": scene_dir,
                "frames_dir": frames_dir,
                "annotation_path": scene_dir / f"{video_id}_3dod_annotation.json",
                "mesh_path": scene_dir / f"{video_id}_3dod_mesh.ply",
                **resolved_assets,
            }
    return index


def bool_from_string(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_visit_id(value: str) -> Optional[int]:
    if value in {"", "NA", "nan", "NaN", None}:
        return None
    return int(float(value))


def download_to_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(dst)


def download_and_extract_raw_wide_asset(task: Tuple[str, str, str, str], keep_zip: bool) -> int:
    download_root_str, split, video_id, asset_name = task
    raw_scene_dir = Path(download_root_str) / "raw" / split / video_id
    extracted_dir = raw_scene_dir / asset_name.replace(".zip", "")
    zip_path = raw_scene_dir / asset_name
    if extracted_dir.exists():
        return 0

    url = f"{ARKIT_BASE_URL}/raw/{split}/{video_id}/{asset_name}"
    print(f"Downloading {asset_name} for video {video_id}")
    download_to_file(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_scene_dir)
    if not keep_zip:
        zip_path.unlink()
    return 1

def ensure_raw_wide_assets(
    download_root: Path,
    keep_zip: bool,
    jobs: int,
    allowed_video_ids: Optional[Set[str]] = None,
) -> None:
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "depth_upsampling" / "upsampling_train_val_splits.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing upsampling split CSV: {csv_path}")

    print("Ensuring raw wide/wide_intrinsics assets for upsampling subset...")
    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    tasks: List[Tuple[str, str, str, str]] = []
    for row in rows:
        video_id = row["video_id"]
        if allowed_video_ids is not None and video_id not in allowed_video_ids:
            continue
        split = row["fold"]
        for asset_name in ("wide.zip", "wide_intrinsics.zip"):
            raw_scene_dir = download_root / "raw" / split / video_id
            extracted_dir = raw_scene_dir / asset_name.replace(".zip", "")
            if not extracted_dir.exists():
                tasks.append((str(download_root), split, video_id, asset_name))

    total_downloaded = 0
    if jobs <= 1:
        for task in tasks:
            total_downloaded += download_and_extract_raw_wide_asset(task, keep_zip)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(download_and_extract_raw_wide_asset, task, keep_zip) for task in tasks]
            for future in as_completed(futures):
                total_downloaded += future.result()

    print(f"Finished ensuring extra raw wide assets. New downloads: {total_downloaded}")


def choose_source_plan(
    row: Dict[str, str],
    download_root: Path,
    threedod_index: Dict[Tuple[str, str], Dict[str, Path]],
    point_cloud_mapping: Dict[int, List[str]],
) -> Optional[SceneExportPlan]:
    video_id = str(row["video_id"]).strip()
    split = row["fold"]
    split_short = SPLIT_MAP[split]
    scene_id = make_scene_id(video_id)
    visit_id = safe_visit_id(row.get("visit_id", ""))

    mesh_path: Optional[Path] = None
    annotation_path: Optional[Path] = None
    raw_scene_dir = download_root / "raw" / split / video_id
    laser_scanner_dir: Optional[Path] = None
    if visit_id is not None and visit_id in point_cloud_mapping:
        candidate_dir = download_root / "laser_scanner_point_clouds" / str(visit_id)
        if candidate_dir.exists():
            laser_scanner_dir = candidate_dir

    threedod_scene = threedod_index.get((split, video_id))
    if threedod_scene is not None:
        mesh_path = threedod_scene["mesh_path"] if threedod_scene["mesh_path"].exists() else None
        annotation_path = (
            threedod_scene["annotation_path"] if threedod_scene["annotation_path"].exists() else None
        )

    if raw_scene_dir.exists():
        raw_mesh = raw_scene_dir / f"{video_id}_3dod_mesh.ply"
        raw_annotation = raw_scene_dir / f"{video_id}_3dod_annotation.json"
        if raw_mesh.exists():
            mesh_path = raw_mesh
        if raw_annotation.exists():
            annotation_path = raw_annotation

        highres_frames = build_frames(
            color_dir=raw_scene_dir / "wide",
            depth_dir=raw_scene_dir / "highres_depth",
            traj_file=raw_scene_dir / "lowres_wide.traj",
            color_intrinsic_dir=raw_scene_dir / "wide_intrinsics",
            depth_intrinsic_dir=raw_scene_dir / "wide_intrinsics",
            pose_tolerance=0.05,
            intrinsic_tolerance=0.05,
        )
        if highres_frames:
            sky_direction = resolve_sky_direction(row, highres_frames[0].pose)
            return SceneExportPlan(
                video_id=video_id,
                visit_id=visit_id,
                split=split,
                split_short=split_short,
                scene_id=scene_id,
                source_kind="raw_wide",
                color_dir=raw_scene_dir / "wide",
                depth_dir=raw_scene_dir / "highres_depth",
                traj_file=raw_scene_dir / "lowres_wide.traj",
                intrinsic_dir=raw_scene_dir / "wide_intrinsics",
                frames=highres_frames,
                mesh_path=mesh_path,
                annotation_path=annotation_path,
                raw_scene_dir=raw_scene_dir,
                laser_scanner_dir=laser_scanner_dir,
                metadata_row=row,
                sky_direction=sky_direction,
            )

        raw_vga_frames = build_frames(
            color_dir=raw_scene_dir / "vga_wide",
            depth_dir=raw_scene_dir / "lowres_depth",
            traj_file=raw_scene_dir / "lowres_wide.traj",
            color_intrinsic_dir=raw_scene_dir / "vga_wide_intrinsics",
            depth_intrinsic_dir=raw_scene_dir / "lowres_wide_intrinsics",
            pose_tolerance=0.05,
            intrinsic_tolerance=0.05,
        )
        if raw_vga_frames:
            sky_direction = resolve_sky_direction(row, raw_vga_frames[0].pose)
            return SceneExportPlan(
                video_id=video_id,
                visit_id=visit_id,
                split=split,
                split_short=split_short,
                scene_id=scene_id,
                source_kind="raw_vga",
                color_dir=raw_scene_dir / "vga_wide",
                depth_dir=raw_scene_dir / "lowres_depth",
                traj_file=raw_scene_dir / "lowres_wide.traj",
                intrinsic_dir=raw_scene_dir / "vga_wide_intrinsics",
                frames=raw_vga_frames,
                mesh_path=mesh_path,
                annotation_path=annotation_path,
                raw_scene_dir=raw_scene_dir,
                laser_scanner_dir=laser_scanner_dir,
                metadata_row=row,
                sky_direction=sky_direction,
            )

        lowres_frames = build_frames(
            color_dir=raw_scene_dir / "lowres_wide",
            depth_dir=raw_scene_dir / "lowres_depth",
            traj_file=raw_scene_dir / "lowres_wide.traj",
            color_intrinsic_dir=raw_scene_dir / "lowres_wide_intrinsics",
            depth_intrinsic_dir=raw_scene_dir / "lowres_wide_intrinsics",
            pose_tolerance=0.05,
            intrinsic_tolerance=0.05,
        )
        if lowres_frames:
            sky_direction = resolve_sky_direction(row, lowres_frames[0].pose)
            return SceneExportPlan(
                video_id=video_id,
                visit_id=visit_id,
                split=split,
                split_short=split_short,
                scene_id=scene_id,
                source_kind="raw_lowres",
                color_dir=raw_scene_dir / "lowres_wide",
                depth_dir=raw_scene_dir / "lowres_depth",
                traj_file=raw_scene_dir / "lowres_wide.traj",
                intrinsic_dir=raw_scene_dir / "lowres_wide_intrinsics",
                frames=lowres_frames,
                mesh_path=mesh_path,
                annotation_path=annotation_path,
                raw_scene_dir=raw_scene_dir,
                laser_scanner_dir=laser_scanner_dir,
                metadata_row=row,
                sky_direction=sky_direction,
            )

    if threedod_scene is not None:
        frames = build_frames(
            color_dir=threedod_scene["color_dir"],
            depth_dir=threedod_scene["depth_dir"],
            traj_file=threedod_scene["traj_file"],
            color_intrinsic_dir=threedod_scene["intrinsic_dir"],
            depth_intrinsic_dir=threedod_scene["intrinsic_dir"],
            pose_tolerance=0.05,
            intrinsic_tolerance=0.05,
        )
        if frames:
            sky_direction = resolve_sky_direction(row, frames[0].pose)
            return SceneExportPlan(
                video_id=video_id,
                visit_id=visit_id,
                split=split,
                split_short=split_short,
                scene_id=scene_id,
                source_kind="3dod",
                color_dir=threedod_scene["color_dir"],
                depth_dir=threedod_scene["depth_dir"],
                traj_file=threedod_scene["traj_file"],
                intrinsic_dir=threedod_scene["intrinsic_dir"],
                frames=frames,
                mesh_path=mesh_path,
                annotation_path=annotation_path,
                raw_scene_dir=raw_scene_dir if raw_scene_dir.exists() else None,
                laser_scanner_dir=laser_scanner_dir,
                metadata_row=row,
                sky_direction=sky_direction,
            )

    return None


def build_frames(
    color_dir: Path,
    depth_dir: Path,
    traj_file: Path,
    color_intrinsic_dir: Path,
    depth_intrinsic_dir: Optional[Path],
    pose_tolerance: float,
    intrinsic_tolerance: float,
) -> List[FrameRecord]:
    if not (
        color_dir.exists()
        and depth_dir.exists()
        and traj_file.exists()
        and color_intrinsic_dir.exists()
    ):
        return []

    color_map = build_frame_map(color_dir, {".png", ".jpg", ".jpeg"})
    depth_map = build_frame_map(depth_dir, {".png"})
    pose_map, pose_times = build_traj_lookup(traj_file)
    color_intrinsic_map, color_intrinsic_times = build_intrinsic_lookup(color_intrinsic_dir)
    if depth_intrinsic_dir is not None and depth_intrinsic_dir.exists():
        depth_intrinsic_map, depth_intrinsic_times = build_intrinsic_lookup(depth_intrinsic_dir)
    else:
        depth_intrinsic_map, depth_intrinsic_times = color_intrinsic_map, color_intrinsic_times

    common_timestamps = sorted(set(color_map.keys()) & set(depth_map.keys()))
    frames: List[FrameRecord] = []

    for export_index, timestamp in enumerate(common_timestamps):
        pose_key = nearest_value(timestamp, pose_times, pose_tolerance)
        color_intrinsic_key = nearest_value(timestamp, color_intrinsic_times, intrinsic_tolerance)
        depth_intrinsic_key = nearest_value(timestamp, depth_intrinsic_times, intrinsic_tolerance)
        if pose_key is None or color_intrinsic_key is None or depth_intrinsic_key is None:
            continue

        frames.append(
            FrameRecord(
                index=export_index,
                timestamp=timestamp,
                color_path=color_map[timestamp],
                depth_path=depth_map[timestamp],
                pose=pose_map[pose_key],
                color_intrinsic=color_intrinsic_map[color_intrinsic_key],
                depth_intrinsic=depth_intrinsic_map[depth_intrinsic_key],
            )
        )

    return frames


def write_scene_summary(
    plan: SceneExportPlan,
    scene_dir: Path,
    first_color_intrinsic: IntrinsicRecord,
    first_depth_intrinsic: IntrinsicRecord,
) -> None:
    axis_alignment = np.eye(4, dtype=float).reshape(-1)
    lines = [
        "dataset = ARKitScenes",
        f"sourceKind = {plan.source_kind}",
        f"video_id = {plan.video_id}",
        f"visit_id = {plan.visit_id if plan.visit_id is not None else 'NA'}",
        f"fold = {plan.split}",
        f"skyDirection = {plan.sky_direction}",
        f"axisAlignment = {' '.join(f'{value:.6f}' for value in axis_alignment)}",
        f"colorHeight = {first_color_intrinsic.height}",
        f"colorWidth = {first_color_intrinsic.width}",
        f"depthHeight = {first_depth_intrinsic.height}",
        f"depthWidth = {first_depth_intrinsic.width}",
        f"fx_color = {first_color_intrinsic.fx:.6f}",
        f"fy_color = {first_color_intrinsic.fy:.6f}",
        f"mx_color = {first_color_intrinsic.cx:.6f}",
        f"my_color = {first_color_intrinsic.cy:.6f}",
        f"fx_depth = {first_depth_intrinsic.fx:.6f}",
        f"fy_depth = {first_depth_intrinsic.fy:.6f}",
        f"mx_depth = {first_depth_intrinsic.cx:.6f}",
        f"my_depth = {first_depth_intrinsic.cy:.6f}",
        f"numColorFrames = {len(plan.frames)}",
        f"numDepthFrames = {len(plan.frames)}",
        "sceneType = ARKitScenes",
    ]
    (scene_dir / f"{plan.scene_id}.txt").write_text("\n".join(lines) + "\n")


def write_manifest(plan: SceneExportPlan, scene_dir: Path, selected_frame_ids: List[int]) -> None:
    manifest = {
        "dataset": "ARKitScenes",
        "scene_id": plan.scene_id,
        "video_id": plan.video_id,
        "visit_id": plan.visit_id,
        "fold": plan.split,
        "source_kind": plan.source_kind,
        "sky_direction": plan.sky_direction,
        "color_dir": str(plan.color_dir),
        "depth_dir": str(plan.depth_dir),
        "traj_file": str(plan.traj_file),
        "intrinsic_dir": str(plan.intrinsic_dir),
        "mesh_path": str(plan.mesh_path) if plan.mesh_path else None,
        "annotation_path": str(plan.annotation_path) if plan.annotation_path else None,
        "laser_scanner_dir": str(plan.laser_scanner_dir) if plan.laser_scanner_dir else None,
        "num_exported_frames": len(plan.frames),
        "selected_frame_count": len(selected_frame_ids),
        "selected_frame_ids": selected_frame_ids,
        "metadata": plan.metadata_row,
        "missing_scannet_equivalents": [
            "ScanNet-style semantic segment files (*.segs.json)",
            "ScanNet-style labeled mesh (*.labels.ply)",
            "ScanNet-style aggregation.json",
        ],
        "available_arkitscenes_assets": {
            "arkit_bbox_annotation": scene_dir.joinpath(f"{plan.scene_id}_3dod_annotation.json").name
            if plan.annotation_path
            else None,
            "laser_scanner_point_clouds": scene_dir.joinpath("laser_scanner_point_clouds").name
            if plan.laser_scanner_dir
            else None,
        },
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_frame_map(plan: SceneExportPlan, scene_dir: Path) -> None:
    with (scene_dir / "frame_map.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "export_index",
                "timestamp",
                "color_path",
                "depth_path",
                "color_intrinsic_path",
                "depth_intrinsic_path",
            ]
        )
        for frame in plan.frames:
            writer.writerow(
                [
                    frame.index,
                    f"{frame.timestamp:.3f}",
                    str(frame.color_path),
                    str(frame.depth_path),
                    str(frame.color_intrinsic.path),
                    str(frame.depth_intrinsic.path if frame.depth_intrinsic is not None else frame.color_intrinsic.path),
                ]
            )


def export_scene(
    plan: SceneExportPlan,
    export_root: Path,
    link_mode: str,
    overwrite: bool,
    selected_frame_count: int,
) -> None:
    scene_dir = export_root / plan.split_short / plan.scene_id
    processed_marker = scene_dir / "processed"
    if processed_marker.exists() and not overwrite:
        print(f"[skip] {plan.scene_id} already exported")
        return

    if scene_dir.exists() and overwrite:
        remove_tree(scene_dir)

    color_out = scene_dir / "color"
    depth_out = scene_dir / "depth"
    pose_out = scene_dir / "pose"
    intrinsic_out = scene_dir / "intrinsic"
    frame_processed_out = scene_dir / "frame_processed"
    color_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)
    pose_out.mkdir(parents=True, exist_ok=True)
    intrinsic_out.mkdir(parents=True, exist_ok=True)
    frame_processed_out.mkdir(parents=True, exist_ok=True)

    first_color_intrinsic = plan.frames[0].color_intrinsic
    first_depth_intrinsic = plan.frames[0].depth_intrinsic or plan.frames[0].color_intrinsic
    first_color_intrinsic = transform_intrinsic(first_color_intrinsic, plan.sky_direction)
    first_depth_intrinsic = transform_intrinsic(first_depth_intrinsic, plan.sky_direction)
    selected_frame_ids = select_exported_frame_ids([frame.index for frame in plan.frames], selected_frame_count)
    selected_frame_id_set = set(selected_frame_ids)
    frame_by_index = {frame.index: frame for frame in plan.frames}

    for frame in plan.frames:
        rectified_color_intrinsic = transform_intrinsic(frame.color_intrinsic, plan.sky_direction)
        rectified_pose = transform_pose_for_sky_direction(frame.pose, plan.sky_direction)

        materialize_color_frame(frame.color_path, color_out / f"{frame.index}.jpg", link_mode, plan.sky_direction)
        materialize_depth_frame(frame.depth_path, depth_out / f"{frame.index}.png", link_mode, plan.sky_direction)
        write_matrix(pose_out / f"{frame.index}.txt", rectified_pose)
        write_matrix(intrinsic_out / f"{frame.index}.txt", rectified_color_intrinsic.matrix_4x4)
        if frame.index in selected_frame_id_set:
            materialize_path(color_out / f"{frame.index}.jpg", frame_processed_out / f"{frame.index}.jpg", link_mode)

    for frame_index in selected_frame_ids:
        selected_frame_path = frame_processed_out / f"{frame_index}.jpg"
        if selected_frame_path.exists() or selected_frame_path.is_symlink():
            continue
        materialize_path(color_out / f"{frame_index}.jpg", selected_frame_path, link_mode)

    identity = np.eye(4, dtype=float)
    write_matrix(scene_dir / "intrinsic_color.txt", first_color_intrinsic.matrix_4x4)
    write_matrix(scene_dir / "intrinsic_depth.txt", first_depth_intrinsic.matrix_4x4)
    write_matrix(scene_dir / "extrinsic_color.txt", identity)
    write_matrix(scene_dir / "extrinsic_depth.txt", identity)
    write_scene_summary(plan, scene_dir, first_color_intrinsic, first_depth_intrinsic)
    write_frame_map(plan, scene_dir)
    write_manifest(plan, scene_dir, selected_frame_ids)

    if plan.mesh_path is not None and plan.mesh_path.exists():
        materialize_path(plan.mesh_path, scene_dir / f"{plan.scene_id}_vh_clean.ply", link_mode)
        materialize_path(plan.mesh_path, scene_dir / f"{plan.scene_id}_vh_clean_2.ply", link_mode)
        materialize_path(plan.mesh_path, scene_dir / f"{plan.scene_id}_3dod_mesh.ply", link_mode)

    if plan.annotation_path is not None and plan.annotation_path.exists():
        materialize_path(
            plan.annotation_path,
            scene_dir / f"{plan.scene_id}_3dod_annotation.json",
            link_mode,
        )

    if plan.laser_scanner_dir is not None and plan.laser_scanner_dir.exists():
        materialize_path(plan.laser_scanner_dir, scene_dir / "laser_scanner_point_clouds", link_mode)

    processed_marker.write_text("prepared_scannet_like\n")


def export_scene_plans(
    plans: List[SceneExportPlan],
    export_root: Path,
    link_mode: str,
    overwrite: bool,
    jobs: int,
    selected_frame_count: int,
) -> None:
    total_plans = len(plans)
    if total_plans == 0:
        return

    if jobs <= 1 or total_plans == 1:
        for idx, plan in enumerate(plans, start=1):
            print(
                f"[{idx}/{total_plans}] exporting {plan.scene_id} "
                f"from {plan.source_kind} with {len(plan.frames)} frames"
            )
            export_scene(plan, export_root, link_mode, overwrite, selected_frame_count)
        return

    max_workers = min(jobs, total_plans)
    print(f"Exporting {total_plans} scenes with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plan = {}
        for idx, plan in enumerate(plans, start=1):
            print(
                f"[queue {idx}/{total_plans}] exporting {plan.scene_id} "
                f"from {plan.source_kind} with {len(plan.frames)} frames"
            )
            future = executor.submit(
                export_scene,
                plan,
                export_root,
                link_mode,
                overwrite,
                selected_frame_count,
            )
            future_to_plan[future] = plan

        for future in as_completed(future_to_plan):
            plan = future_to_plan[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed exporting {plan.scene_id}") from exc
            print(f"[done] exported {plan.scene_id}")


def write_scene_index(
    export_root: Path,
    exported_plans: List[SceneExportPlan],
) -> None:
    export_root.mkdir(parents=True, exist_ok=True)
    with (export_root / "scene_id_map.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scene_id",
                "split",
                "video_id",
                "visit_id",
                "source_kind",
                "num_exported_frames",
            ]
        )
        for plan in exported_plans:
            writer.writerow(
                [
                    plan.scene_id,
                    plan.split_short,
                    plan.video_id,
                    plan.visit_id if plan.visit_id is not None else "",
                    plan.source_kind,
                    len(plan.frames),
                ]
            )

    for split_short in {"train", "val"}:
        scene_ids = [plan.scene_id for plan in exported_plans if plan.split_short == split_short]
        scene_list_path = export_root / f"{split_short}_scenes.txt"
        if scene_ids:
            scene_list_path.write_text("\n".join(scene_ids) + "\n")
        elif scene_list_path.exists():
            scene_list_path.unlink()


def main() -> int:
    args = parse_args()
    download_root = Path(args.download_root).resolve()
    export_root = Path(args.export_root).resolve()
    requested_scene_ids = set(args.scene_id or [])
    explicit_video_ids = set(args.video_id or [])
    requested_video_ids = set(explicit_video_ids)

    if requested_scene_ids:
        requested_video_ids.update(
            scene_id.removeprefix("scene").rsplit("_", 1)[0]
            for scene_id in requested_scene_ids
            if scene_id.startswith("scene") and "_" in scene_id
        )

    export_metadata = load_export_metadata(download_root)
    selected_metadata: List[Dict[str, str]] = []
    matched_scene_ids: Set[str] = set()
    matched_video_ids: Set[str] = set()
    for row in export_metadata:
        row_video_id = str(row["video_id"]).strip()
        row_scene_id = make_scene_id(row_video_id)
        if requested_video_ids and row_video_id not in requested_video_ids:
            continue
        if requested_scene_ids and row_scene_id not in requested_scene_ids:
            continue
        selected_metadata.append(row)
        matched_scene_ids.add(row_scene_id)
        matched_video_ids.add(row_video_id)

    missing_scene_ids = requested_scene_ids - matched_scene_ids
    if missing_scene_ids:
        raise SystemExit(f"Scene ids not found in metadata: {', '.join(sorted(missing_scene_ids))}")

    missing_video_ids = explicit_video_ids - matched_video_ids
    if missing_video_ids:
        raise SystemExit(f"Video ids not found in metadata: {', '.join(sorted(missing_video_ids))}")

    if args.ensure_raw_wide:
        ensure_raw_wide_assets(
            download_root,
            keep_zip=args.keep_zip,
            jobs=max(1, args.jobs),
            allowed_video_ids=requested_video_ids or None,
        )

    point_cloud_mapping = load_point_cloud_mapping(download_root)
    threedod_index = build_3dod_index(download_root, requested_video_ids or None)

    export_candidates: List[SceneExportPlan] = []
    skipped_rows = 0

    for idx, row in enumerate(selected_metadata, start=1):
        plan = choose_source_plan(row, download_root, threedod_index, point_cloud_mapping)
        if plan is None or not plan.frames:
            skipped_rows += 1
            print(f"[warn] no usable aligned assets for video {row['video_id']} ({row['fold']})")
            continue
        print(
            f"[plan {idx}/{len(selected_metadata)}] exporting {plan.scene_id} "
            f"from {plan.source_kind} with {len(plan.frames)} frames"
        )
        export_candidates.append(plan)

    export_scene_plans(
        export_candidates,
        export_root,
        args.link_mode,
        args.overwrite,
        jobs=max(1, args.jobs),
        selected_frame_count=max(1, args.selected_frame_count),
    )
    exported_plans = export_candidates

    write_scene_index(export_root, exported_plans)

    print(f"Export finished. Scenes exported: {len(exported_plans)}")
    if skipped_rows:
        print(f"Scenes skipped due to missing aligned assets: {skipped_rows}")
    print(f"ScanNet-like export root: {export_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
