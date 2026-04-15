#!/usr/bin/env python3
import argparse
import functools
import json
import math
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial import ConvexHull

DEFAULT_LABEL_MAP_PATH = Path(__file__).resolve().with_name("arkitscenes_label_harmonization.json")


@dataclass
class SceneObject:
    object_id: int
    uid: str
    source_label: str
    label: str
    label_family: str
    scannet_ref_label: str
    scannet_ref_id: int
    nyu40class: str
    nyu40id: int
    mpcat40: str
    center: np.ndarray
    size_xyz: np.ndarray
    rotation: np.ndarray
    corners: np.ndarray
    sample_points: np.ndarray
    bbox_3d: List[float]
    location_3d: List[float]
    size: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ScanNet-like metadata for exported ARKitScenes scenes.",
    )
    parser.add_argument(
        "--export-root",
        required=True,
        help="Root of ScanNet-like ARKitScenes export.",
    )
    parser.add_argument(
        "--scene-id",
        nargs="+",
        default=None,
        help="Optional scene ids, e.g. scene40777079_00.",
    )
    parser.add_argument(
        "--video-id",
        nargs="+",
        default=None,
        help="Optional video ids, e.g. 40777079.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Parallel frame workers for a scene.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=20,
        help="Fallback: use every Nth exported frame when `frame_processed` is unavailable.",
    )
    parser.add_argument(
        "--target-frame-count",
        type=int,
        default=32,
        help="Fallback target count when `frame_processed` is unavailable.",
    )
    parser.add_argument(
        "--samples-per-axis",
        type=int,
        default=7,
        help="Per-axis grid density for box surface sampling.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="Minimum projected visibility ratio to keep an object in a frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata scene folders.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save frame and 3D visualization images.",
    )
    parser.add_argument(
        "--label-map",
        default=str(DEFAULT_LABEL_MAP_PATH),
        help="JSON file that harmonizes ARKitScenes labels to ScanNet/NYU40-friendly labels.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n")


def normalize_label_text(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


@functools.lru_cache(maxsize=8)
def load_label_map(label_map_path: str) -> Tuple[Dict[str, Dict], Dict]:
    path = Path(label_map_path)
    if not path.is_file():
        default_meta = {
            "label_family": "object",
            "scannet_ref_label": "object",
            "scannet_ref_id": -1,
            "nyu40class": "object",
            "nyu40id": -1,
            "mpcat40": "object",
        }
        return {}, default_meta

    payload = load_json(path)
    default_meta = {
        "label_family": normalize_label_text(payload.get("default", {}).get("label_family", "object")) or "object",
        "scannet_ref_label": normalize_label_text(payload.get("default", {}).get("scannet_ref_label", "object")) or "object",
        "scannet_ref_id": int(payload.get("default", {}).get("scannet_ref_id", -1)),
        "nyu40class": normalize_label_text(payload.get("default", {}).get("nyu40class", "object")) or "object",
        "nyu40id": int(payload.get("default", {}).get("nyu40id", -1)),
        "mpcat40": normalize_label_text(payload.get("default", {}).get("mpcat40", "object")) or "object",
    }

    lookup: Dict[str, Dict] = {}
    for source_label, raw_meta in payload.get("labels", {}).items():
        meta = dict(default_meta)
        meta.update(
            {
                "label": normalize_label_text(raw_meta.get("label", source_label)) or normalize_label_text(source_label) or "object",
                "label_family": normalize_label_text(raw_meta.get("label_family", default_meta["label_family"])) or default_meta["label_family"],
                "scannet_ref_label": normalize_label_text(raw_meta.get("scannet_ref_label", raw_meta.get("label", source_label))) or default_meta["scannet_ref_label"],
                "scannet_ref_id": int(raw_meta.get("scannet_ref_id", default_meta["scannet_ref_id"])),
                "nyu40class": normalize_label_text(raw_meta.get("nyu40class", default_meta["nyu40class"])) or default_meta["nyu40class"],
                "nyu40id": int(raw_meta.get("nyu40id", default_meta["nyu40id"])),
                "mpcat40": normalize_label_text(raw_meta.get("mpcat40", default_meta["mpcat40"])) or default_meta["mpcat40"],
            }
        )

        aliases = [
            source_label,
            raw_meta.get("label", source_label),
            raw_meta.get("scannet_ref_label", source_label),
            *raw_meta.get("aliases", []),
        ]
        for alias in aliases:
            normalized_alias = normalize_label_text(alias)
            if normalized_alias:
                lookup[normalized_alias] = dict(meta)
    return lookup, default_meta


def resolve_label_meta(source_label: str, label_map_path: Path) -> Dict:
    lookup, default_meta = load_label_map(str(label_map_path.resolve()))
    normalized_source = normalize_label_text(source_label)
    meta = dict(default_meta)
    meta.update(lookup.get(normalized_source, {}))
    meta.setdefault("label", normalized_source or "object")
    meta["source_label"] = str(source_label)
    return meta


def numeric_stem(path: Path) -> int:
    return int(path.stem)


def select_evenly_spaced_frame_numbers(frame_numbers: Sequence[int], target_count: int) -> List[int]:
    frame_numbers = sorted(int(frame_num) for frame_num in frame_numbers)
    if not frame_numbers:
        return []
    if target_count <= 0 or len(frame_numbers) <= target_count:
        return frame_numbers
    if target_count == 1:
        return [frame_numbers[0]]

    selected_positions: List[int] = []
    for sample_idx in range(target_count):
        position = round(sample_idx * (len(frame_numbers) - 1) / (target_count - 1))
        if not selected_positions or position != selected_positions[-1]:
            selected_positions.append(position)

    if len(selected_positions) < target_count:
        existing = set(selected_positions)
        for position in range(len(frame_numbers)):
            if position not in existing:
                selected_positions.append(position)
                existing.add(position)
            if len(selected_positions) >= target_count:
                break

    return [frame_numbers[position] for position in sorted(selected_positions[:target_count])]


def list_scene_dirs(export_root: Path) -> List[Path]:
    scene_dirs: List[Path] = []
    for split in ("train", "val"):
        split_dir = export_root / split
        if not split_dir.exists():
            continue
        scene_dirs.extend(sorted((path for path in split_dir.iterdir() if path.is_dir()), key=lambda p: p.name))
    return scene_dirs


def select_scene_dirs(export_root: Path, scene_ids: Optional[Sequence[str]], video_ids: Optional[Sequence[str]]) -> List[Path]:
    requested_scene_ids = set(scene_ids or [])
    requested_video_ids = set(video_ids or [])
    if requested_scene_ids:
        requested_video_ids.update(
            scene_id.removeprefix("scene").rsplit("_", 1)[0]
            for scene_id in requested_scene_ids
            if scene_id.startswith("scene") and "_" in scene_id
        )

    all_scenes = list_scene_dirs(export_root)
    if not requested_scene_ids and not requested_video_ids:
        return all_scenes

    selected: List[Path] = []
    for scene_dir in all_scenes:
        scene_id = scene_dir.name
        video_id = scene_id.removeprefix("scene").rsplit("_", 1)[0]
        if requested_scene_ids and scene_id not in requested_scene_ids:
            continue
        if requested_video_ids and video_id not in requested_video_ids:
            continue
        selected.append(scene_dir)
    return selected


def load_manifest(scene_dir: Path) -> Dict:
    manifest_path = scene_dir / "manifest.json"
    return load_json(manifest_path) if manifest_path.exists() else {}


def resolve_selected_frame_nums(
    scene_dir: Path,
    manifest: Dict,
    frame_step: int,
    target_frame_count: int,
) -> Tuple[List[int], str]:
    frame_processed_dir = scene_dir / "frame_processed"
    if frame_processed_dir.exists():
        frame_processed = sorted(frame_processed_dir.glob("*.jpg"), key=numeric_stem)
        if frame_processed:
            return [numeric_stem(path) for path in frame_processed], "frame_processed"

    manifest_frame_ids = manifest.get("selected_frame_ids") or []
    if manifest_frame_ids:
        return sorted(int(frame_id) for frame_id in manifest_frame_ids), "manifest"

    color_frame_nums = [numeric_stem(path) for path in sorted((scene_dir / "color").glob("*.jpg"), key=numeric_stem)]
    if target_frame_count > 0 and color_frame_nums:
        return select_evenly_spaced_frame_numbers(color_frame_nums, target_frame_count), "fallback_even"

    return color_frame_nums[:: max(1, frame_step)], "fallback_step"


def resolve_annotation_path(scene_dir: Path, manifest: Dict) -> Path:
    local_path = scene_dir / f"{scene_dir.name}_3dod_annotation.json"
    if local_path.exists():
        return local_path
    manifest_path = manifest.get("annotation_path")
    if manifest_path:
        path = Path(manifest_path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing annotation for {scene_dir}")


def resolve_mesh_path(scene_dir: Path, manifest: Dict) -> Path:
    local_path = scene_dir / f"{scene_dir.name}_3dod_mesh.ply"
    if local_path.exists():
        return local_path
    local_path = scene_dir / f"{scene_dir.name}_vh_clean.ply"
    if local_path.exists():
        return local_path
    manifest_path = manifest.get("mesh_path")
    if manifest_path:
        path = Path(manifest_path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing mesh for {scene_dir}")


def load_matrix(path: Path, shape: Tuple[int, int] = (4, 4)) -> np.ndarray:
    values = [float(value) for value in path.read_text().split()]
    return np.asarray(values, dtype=float).reshape(shape)


def read_mesh_vertices(mesh_path: Path) -> np.ndarray:
    ply_data = PlyData.read(str(mesh_path))
    vertices = ply_data["vertex"]
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(float)


def compute_box_corners(size_xyz: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    x_half, y_half, z_half = [float(value) / 2.0 for value in size_xyz]
    x_corners = [x_half, x_half, -x_half, -x_half, x_half, x_half, -x_half, -x_half]
    y_corners = [y_half, -y_half, -y_half, y_half, y_half, -y_half, -y_half, y_half]
    z_corners = [z_half, z_half, z_half, z_half, -z_half, -z_half, -z_half, -z_half]
    local_points = np.column_stack([x_corners, y_corners, z_corners])
    return local_points @ rotation + center.reshape(1, 3)


def sample_box_surface_points(
    size_xyz: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    samples_per_axis: int,
) -> np.ndarray:
    samples_per_axis = max(2, samples_per_axis)
    x_half, y_half, z_half = [float(value) / 2.0 for value in size_xyz]
    x_values = np.linspace(-x_half, x_half, samples_per_axis)
    y_values = np.linspace(-y_half, y_half, samples_per_axis)
    z_values = np.linspace(-z_half, z_half, samples_per_axis)

    local_points: List[np.ndarray] = []
    yy, zz = np.meshgrid(y_values, z_values, indexing="xy")
    for x_fixed in (-x_half, x_half):
        local_points.append(np.column_stack([np.full(yy.size, x_fixed), yy.reshape(-1), zz.reshape(-1)]))

    xx, zz = np.meshgrid(x_values, z_values, indexing="xy")
    for y_fixed in (-y_half, y_half):
        local_points.append(np.column_stack([xx.reshape(-1), np.full(xx.size, y_fixed), zz.reshape(-1)]))

    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    for z_fixed in (-z_half, z_half):
        local_points.append(np.column_stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, z_fixed)]))

    local_points_arr = np.concatenate(local_points, axis=0)
    local_points_arr = np.unique(np.round(local_points_arr, 6), axis=0)
    return local_points_arr @ rotation + center.reshape(1, 3)


def load_scene_objects(annotation_path: Path, samples_per_axis: int, label_map_path: Path) -> List[SceneObject]:
    annotation = load_json(annotation_path)
    scene_objects: List[SceneObject] = []

    for index, item in enumerate(annotation.get("data", []), start=1):
        segments = item.get("segments", {})
        box = segments.get("obbAligned") or segments.get("obb")
        if not box:
            continue

        center = np.asarray(box["centroid"], dtype=float)
        size_xyz = np.asarray(box["axesLengths"], dtype=float)
        if "obbAligned" not in segments:
            center = center / 100.0
            size_xyz = size_xyz / 100.0
        rotation = np.asarray(box["normalizedAxes"], dtype=float).reshape(3, 3)
        corners = compute_box_corners(size_xyz, center, rotation)
        bbox_min = np.min(corners, axis=0)
        bbox_max = np.max(corners, axis=0)
        bbox_3d = [float(value) for value in np.concatenate([bbox_min, bbox_max])]
        location_3d = [float((bbox_3d[idx] + bbox_3d[idx + 3]) / 2.0) for idx in range(3)]
        size = float(max(bbox_3d[idx + 3] - bbox_3d[idx] for idx in range(3)) * 100.0)
        sample_points = sample_box_surface_points(size_xyz, center, rotation, samples_per_axis)
        label_meta = resolve_label_meta(str(item.get("label", "object")), label_map_path)

        scene_objects.append(
            SceneObject(
                object_id=index,
                uid=str(item.get("uid", f"obj_{index}")),
                source_label=label_meta["source_label"],
                label=label_meta["label"],
                label_family=label_meta["label_family"],
                scannet_ref_label=label_meta["scannet_ref_label"],
                scannet_ref_id=int(label_meta["scannet_ref_id"]),
                nyu40class=label_meta["nyu40class"],
                nyu40id=int(label_meta["nyu40id"]),
                mpcat40=label_meta["mpcat40"],
                center=center,
                size_xyz=size_xyz,
                rotation=rotation,
                corners=corners,
                sample_points=sample_points,
                bbox_3d=bbox_3d,
                location_3d=location_3d,
                size=size,
            )
        )

    return scene_objects


def downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points
    min_bound = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def estimate_room_size(mesh_vertices: np.ndarray) -> float:
    if len(mesh_vertices) < 3:
        return 0.0

    z_coords = mesh_vertices[:, 2]
    base_height = np.percentile(z_coords, 40)
    floor_points = mesh_vertices[z_coords < base_height + 0.1][:, :2]
    if len(floor_points) < 3:
        floor_points = mesh_vertices[:, :2]
    floor_points = downsample_points(floor_points, voxel_size=0.05)
    if len(floor_points) < 3:
        xy_min = np.min(mesh_vertices[:, :2], axis=0)
        xy_max = np.max(mesh_vertices[:, :2], axis=0)
        return float(np.prod(np.maximum(0.0, xy_max - xy_min)))

    try:
        hull = ConvexHull(floor_points)
        return float(hull.volume)
    except Exception:
        xy_min = np.min(floor_points, axis=0)
        xy_max = np.max(floor_points, axis=0)
        return float(np.prod(np.maximum(0.0, xy_max - xy_min)))


def load_depth(depth_path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(depth_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def project_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    pose: np.ndarray,
    image_size: Tuple[int, int],
    depth: Optional[np.ndarray],
    depth_scale: float = 1000.0,
    occlusion_threshold: float = 0.001,
) -> Tuple[np.ndarray, float, bool, bool, np.ndarray]:
    depth_height, depth_width = 0, 0
    depth_to_color_scale_x, depth_to_color_scale_y = 1.0, 1.0
    truncated = False
    occluded = False

    if depth is not None:
        depth_height, depth_width = depth.shape[:2]
        depth_to_color_scale_x = image_size[0] / depth_width
        depth_to_color_scale_y = image_size[1] / depth_height

    world_to_cam = np.linalg.inv(pose)
    world_points = np.column_stack([points, np.ones(len(points), dtype=float)])
    cam_points = world_points @ world_to_cam.T

    visibility = cam_points[:, 2] > 0
    projected = cam_points @ intrinsic.T
    projected_xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)

    in_image = (
        (projected_xy[:, 0] >= 0)
        & (projected_xy[:, 0] < image_size[0])
        & (projected_xy[:, 1] >= 0)
        & (projected_xy[:, 1] < image_size[1])
    )
    if int(np.sum(in_image)) != 0:
        truncated = True

    visibility &= in_image

    if depth is not None:
        for index in range(len(projected_xy)):
            if not visibility[index]:
                continue
            color_x, color_y = int(projected_xy[index, 0]), int(projected_xy[index, 1])
            depth_x = int(color_x / depth_to_color_scale_x)
            depth_y = int(color_y / depth_to_color_scale_y)
            if not (0 <= depth_x < depth_width and 0 <= depth_y < depth_height):
                continue

            actual_depth = float(depth[depth_y, depth_x]) / depth_scale
            calculated_depth = float(cam_points[index, 2])
            relative_error = (calculated_depth - actual_depth) / (actual_depth + 1e-6)
            if actual_depth > 0 and relative_error > occlusion_threshold:
                visibility[index] = False
                occluded = True

    visible_points = projected_xy[visibility]
    visibility_ratio = float(len(visible_points) / len(projected_xy)) if len(projected_xy) else 0.0
    if int(np.sum(visibility)) != 0:
        visible_cam_points = cam_points[visibility][:, :3]
        cam_loc = (np.min(visible_cam_points, axis=0) + np.max(visible_cam_points, axis=0)) / 2.0
    else:
        cam_loc = np.asarray([-1.0, -1.0, -1.0], dtype=float)
    return visible_points, round(visibility_ratio, 4), truncated, occluded, cam_loc


def frame_intrinsic_path(scene_dir: Path, frame_num: int) -> Path:
    intrinsic_path = scene_dir / "intrinsic" / f"{frame_num}.txt"
    if intrinsic_path.exists():
        return intrinsic_path
    return scene_dir / "intrinsic_color.txt"


def build_frame_metadata(scene_id: str, frame_id: str, objects: List[Dict]) -> Dict:
    return {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "objects": objects,
    }


def build_scene_metadata(scene_id: str, room_size: float, objects: Iterable[SceneObject], first_frames: Dict[int, str]) -> Dict:
    payload = {
        "scene_id": scene_id,
        "room_size": room_size,
        "objects": [],
    }
    for obj in objects:
        payload["objects"].append(
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "source_label": obj.source_label,
                "label_family": obj.label_family,
                "scannet_ref_label": obj.scannet_ref_label,
                "scannet_ref_id": obj.scannet_ref_id,
                "nyu40class": obj.nyu40class,
                "nyu40id": obj.nyu40id,
                "mpcat40": obj.mpcat40,
                "size": obj.size,
                "3d_bbox": obj.bbox_3d,
                "3d_location": obj.location_3d,
                "first_frame": first_frames.get(obj.object_id, "-1"),
            }
        )
    return payload


def draw_frame_visualization(color_path: Path, object_records: List[Dict], out_path: Path) -> None:
    image = Image.open(color_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = list(mcolors.TABLEAU_COLORS.values())

    for record in object_records:
        x1, y1, x2, y2 = [int(value) for value in record["2d_bbox"]]
        color = palette[(int(record["object_id"]) - 1) % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{record['object_id']}: {record['label']} {record['visibility']:.3f}"
        text_bbox = draw.textbbox((x1 + 2, y1 + 2), text)
        draw.rectangle(text_bbox, fill="white")
        draw.text((x1 + 2, y1 + 2), text, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def draw_scene_visualization(scene_objects: Iterable[SceneObject], out_path: Path) -> None:
    scene_objects = list(scene_objects)
    palette = list(mcolors.TABLEAU_COLORS.values())
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    bounds: List[np.ndarray] = []
    for obj in scene_objects:
        corners = obj.corners
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[1], corners[2], corners[6], corners[5]],
            [corners[0], corners[3], corners[7], corners[4]],
        ]
        color = palette[(obj.object_id - 1) % len(palette)]
        ax.add_collection3d(
            Poly3DCollection(faces, alpha=0.2, facecolor=color, edgecolor="black", linewidth=0.8)
        )
        ax.text(*obj.center.tolist(), f"{obj.object_id}:{obj.label}", fontsize=7, color="black")
        bounds.append(corners)

    if bounds:
        stacked = np.concatenate(bounds, axis=0)
        mins = np.min(stacked, axis=0)
        maxs = np.max(stacked, axis=0)
    else:
        mins = np.asarray([-1.0, -1.0, -1.0])
        maxs = np.asarray([1.0, 1.0, 1.0])

    margin = 0.3
    ax.set_xlim(mins[0] - margin, maxs[0] + margin)
    ax.set_ylim(mins[1] - margin, maxs[1] + margin)
    ax.set_zlim(mins[2] - margin, maxs[2] + margin)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=90, azim=-90)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def process_single_frame(
    scene_dir: Path,
    scene_id: str,
    frame_num: int,
    scene_objects: List[SceneObject],
    min_visibility: float,
    visualize: bool,
    vis_dir: Path,
) -> Tuple[int, Dict]:
    color_path = scene_dir / "color" / f"{frame_num}.jpg"
    depth_path = scene_dir / "depth" / f"{frame_num}.png"
    pose_path = scene_dir / "pose" / f"{frame_num}.txt"
    intrinsic_path = frame_intrinsic_path(scene_dir, frame_num)

    color_image = Image.open(color_path)
    image_size = color_image.size
    depth = load_depth(depth_path)
    pose = load_matrix(pose_path)
    intrinsic = load_matrix(intrinsic_path)

    object_records: List[Dict] = []
    for obj in scene_objects:
        visible_points, visibility, truncated, occluded, cam_loc = project_points(
            obj.sample_points,
            intrinsic,
            pose,
            image_size,
            depth,
        )
        if len(visible_points) == 0 or visibility < min_visibility:
            continue

        xmin = float(np.min(visible_points[:, 0]))
        ymin = float(np.min(visible_points[:, 1]))
        xmax = float(np.max(visible_points[:, 0]))
        ymax = float(np.max(visible_points[:, 1]))
        margin = 2
        bbox_2d = [
            min(image_size[0], max(0, int(xmin) - margin)),
            min(image_size[1], max(0, int(ymin) - margin)),
            min(image_size[0], int(xmax + 0.5) + margin),
            min(image_size[1], int(ymax + 0.5) + margin),
        ]

        object_records.append(
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "source_label": obj.source_label,
                "label_family": obj.label_family,
                "scannet_ref_label": obj.scannet_ref_label,
                "scannet_ref_id": obj.scannet_ref_id,
                "nyu40class": obj.nyu40class,
                "nyu40id": obj.nyu40id,
                "mpcat40": obj.mpcat40,
                "size": obj.size,
                "visibility": visibility,
                "truncated": truncated,
                "occluded": occluded,
                "2d_bbox": bbox_2d,
                "camera_location": [float(value) for value in cam_loc.tolist()],
                "3d_bbox": obj.bbox_3d,
                "3d_location": obj.location_3d,
            }
        )

    object_records.sort(key=lambda item: item["object_id"])
    frame_metadata = build_frame_metadata(scene_id, f"{frame_num}.jpg", object_records)
    if visualize:
        draw_frame_visualization(color_path, object_records, vis_dir / f"{frame_num}.jpg")
    return frame_num, frame_metadata


def process_scene(scene_dir: Path, export_root: Path, args: argparse.Namespace) -> Dict[str, int]:
    scene_id = scene_dir.name
    metadata_scene_dir = export_root / "metadata" / scene_id
    frame_metadata_dir = metadata_scene_dir / "frame_processed"
    frame_vis_dir = metadata_scene_dir / "frame_processed_vis"

    if metadata_scene_dir.exists() and args.overwrite:
        shutil.rmtree(metadata_scene_dir)
    if metadata_scene_dir.exists() and not args.overwrite:
        print(f"[skip] metadata already exists for {scene_id}")
        return {"processed_frames": 0, "processed_objects": 0}

    metadata_scene_dir.mkdir(parents=True, exist_ok=True)
    frame_metadata_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        frame_vis_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(scene_dir)
    annotation_path = resolve_annotation_path(scene_dir, manifest)
    mesh_path = resolve_mesh_path(scene_dir, manifest)
    scene_objects = load_scene_objects(annotation_path, args.samples_per_axis, Path(args.label_map))
    mesh_vertices = read_mesh_vertices(mesh_path)
    room_size = estimate_room_size(mesh_vertices)

    selected_frame_nums, selected_source = resolve_selected_frame_nums(
        scene_dir,
        manifest,
        args.frame_step,
        args.target_frame_count,
    )

    print(
        f"[scene] {scene_id} objects={len(scene_objects)} selected_frames={len(selected_frame_nums)} "
        f"selection={selected_source}"
    )

    frame_results: List[Tuple[int, Dict]] = []
    if args.jobs <= 1:
        for frame_num in selected_frame_nums:
            frame_results.append(
                process_single_frame(
                    scene_dir,
                    scene_id,
                    frame_num,
                    scene_objects,
                    args.min_visibility,
                    args.visualize,
                    frame_vis_dir,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {
                executor.submit(
                    process_single_frame,
                    scene_dir,
                    scene_id,
                    frame_num,
                    scene_objects,
                    args.min_visibility,
                    args.visualize,
                    frame_vis_dir,
                ): frame_num
                for frame_num in selected_frame_nums
            }
            for future in as_completed(futures):
                frame_results.append(future.result())

    frame_results.sort(key=lambda item: item[0])
    first_frames: Dict[int, str] = {}
    for frame_num, frame_metadata in frame_results:
        write_json(frame_metadata_dir / f"{frame_num}.json", frame_metadata)
        for obj in frame_metadata["objects"]:
            first_frames.setdefault(int(obj["object_id"]), frame_metadata["frame_id"])

    scene_metadata = build_scene_metadata(scene_id, room_size, scene_objects, first_frames)
    write_json(metadata_scene_dir / f"{scene_id}.json", scene_metadata)

    if args.visualize:
        draw_scene_visualization(scene_objects, metadata_scene_dir / f"{scene_id}_3d_vis.jpg")

    return {
        "processed_frames": len(frame_results),
        "processed_objects": len(scene_objects),
    }


def main() -> int:
    args = parse_args()
    export_root = Path(args.export_root).resolve()
    scene_dirs = select_scene_dirs(export_root, args.scene_id, args.video_id)

    if not scene_dirs:
        raise SystemExit("No matching exported scenes found.")

    total_frames = 0
    total_objects = 0
    for scene_dir in scene_dirs:
        result = process_scene(scene_dir, export_root, args)
        total_frames += result["processed_frames"]
        total_objects += result["processed_objects"]

    print(
        f"Metadata generation finished. scenes={len(scene_dirs)} "
        f"frames={total_frames} objects={total_objects}"
    )
    print(f"Metadata root: {export_root / 'metadata'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
