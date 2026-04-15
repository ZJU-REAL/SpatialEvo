#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCANNET_PROCESS_ROOT = PROJECT_ROOT / "data" / "scannet_process"

DEFAULT_RAW_ROOT = "/mnt/jfs/lidingm/data/dataset/scannetpp_raw"
DEFAULT_OUTPUT_ROOT = "/mnt/jfs/lidingm/data/dataset/scannetpp"
DEFAULT_SPLIT = "train"
INVALID_SCANNET_LABEL_IDS = {
    0,
    1,
    2,
    3,
    11,
    13,
    15,
    16,
    17,
    20,
    21,
    22,
    23,
    26,
    28,
    29,
    31,
    34,
    37,
    38,
    39,
    40,
    81,
}

_SCANNET_UTILS: dict[str, Any] | None = None
_SCANNET_LABEL_LOOKUP: dict[str, tuple[int, str]] | None = None


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def resolve_scene_name(scene_id: str, scene_name_pattern: str) -> str:
    return scene_name_pattern.format(scene_id=scene_id)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload is not a dict: {path}")
    return payload


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_matrix(path: Path, matrix: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in matrix:
            file.write(" ".join(f"{value:.8f}" for value in row))
            file.write("\n")


def normalize_label_text(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def build_scannet_label_lookup() -> dict[str, tuple[int, str]]:
    global _SCANNET_LABEL_LOOKUP
    if _SCANNET_LABEL_LOOKUP is not None:
        return _SCANNET_LABEL_LOOKUP

    label_map_file = SCANNET_PROCESS_ROOT / "scannet_metadata" / "scannetv2-labels-v2.tsv"
    lookup: dict[str, tuple[int, str]] = {}
    if label_map_file.is_file():
        with label_map_file.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                try:
                    nyu40_id = int(row["nyu40id"])
                except Exception:
                    continue
                canonical_name = (
                    normalize_label_text(row.get("nyu40class", ""))
                    or normalize_label_text(row.get("category", ""))
                    or normalize_label_text(row.get("raw_category", ""))
                    or "object"
                )
                aliases = [
                    row.get("raw_category", ""),
                    row.get("category", ""),
                    row.get("nyuClass", ""),
                    row.get("nyu40class", ""),
                    row.get("mpcat40", ""),
                ]
                for alias in aliases:
                    normalized = normalize_label_text(alias)
                    if normalized and normalized not in lookup:
                        lookup[normalized] = (nyu40_id, canonical_name)

    _SCANNET_LABEL_LOOKUP = lookup
    return lookup


def build_label_maps_from_aggregation(agg_file: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = load_json(agg_file)
    seg_groups = payload.get("segGroups", [])
    if not isinstance(seg_groups, list):
        seg_groups = []

    scannet_lookup = build_scannet_label_lookup()
    raw_id_map: dict[str, int] = {}
    id_label_map: dict[int, str] = {}
    fallback_label_id = 1000

    for group in seg_groups:
        if not isinstance(group, dict):
            continue
        raw_label = str(group.get("label", "object"))
        normalized_label = normalize_label_text(raw_label) or "object"

        if normalized_label in scannet_lookup:
            label_id, canonical_name = scannet_lookup[normalized_label]
        else:
            label_id = fallback_label_id
            fallback_label_id += 1
            canonical_name = normalized_label

        raw_id_map[raw_label] = label_id
        if label_id not in id_label_map:
            id_label_map[label_id] = canonical_name

    return raw_id_map, id_label_map


def collect_skipped_labels(aggregation_payload: dict[str, Any] | None) -> list[str]:
    if not aggregation_payload:
        return []

    scannet_lookup = build_scannet_label_lookup()
    skipped: set[str] = set()
    seg_groups = aggregation_payload.get("segGroups", [])
    if not isinstance(seg_groups, list):
        return []

    for group in seg_groups:
        if not isinstance(group, dict):
            continue
        normalized_label = normalize_label_text(group.get("label", ""))
        if not normalized_label:
            continue
        label_info = scannet_lookup.get(normalized_label)
        if label_info and label_info[0] in INVALID_SCANNET_LABEL_IDS:
            skipped.add(normalized_label)
    return sorted(skipped)


def get_scannet_utils() -> dict[str, Any]:
    global _SCANNET_UTILS
    if _SCANNET_UTILS is not None:
        return _SCANNET_UTILS

    if str(SCANNET_PROCESS_ROOT) not in sys.path:
        sys.path.append(str(SCANNET_PROCESS_ROOT))

    from utils.bbox_extractor import align_mesh_vertices  # type: ignore
    from utils.bbox_extractor import export_2d_bbox  # type: ignore
    from utils.bbox_extractor import export_3d_bbox  # type: ignore
    from utils.bbox_extractor import get_object_id_to_label_id  # type: ignore
    from utils.bbox_extractor import read_mesh_vertices_rgb  # type: ignore
    from utils.bbox_extractor import read_scene_axis_alignment  # type: ignore
    from utils.metadata_builder import export_frame_metadata  # type: ignore
    from utils.metadata_builder import export_scene_metadata  # type: ignore
    from utils.room_size_extractor import calculate_room_area  # type: ignore

    _SCANNET_UTILS = {
        "align_mesh_vertices": align_mesh_vertices,
        "export_2d_bbox": export_2d_bbox,
        "export_3d_bbox": export_3d_bbox,
        "get_object_id_to_label_id": get_object_id_to_label_id,
        "read_mesh_vertices_rgb": read_mesh_vertices_rgb,
        "read_scene_axis_alignment": read_scene_axis_alignment,
        "export_frame_metadata": export_frame_metadata,
        "export_scene_metadata": export_scene_metadata,
        "calculate_room_area": calculate_room_area,
    }
    return _SCANNET_UTILS


def parse_scene_ids_arg(scene_ids_text: str | None, scene_ids_file: str | None) -> list[str]:
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


def list_local_scene_ids(raw_root: Path) -> list[str]:
    if not raw_root.exists():
        return []

    scene_ids: list[str] = []
    for path in sorted(raw_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        nerfstudio_dir = path / "dslr" / "nerfstudio"
        scans_dir = path / "scans"
        if nerfstudio_dir.is_dir() and scans_dir.is_dir():
            scene_ids.append(path.name)
    return scene_ids


def select_valid_frames(transforms: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    frames = transforms.get("frames", [])
    if not isinstance(frames, list):
        return []
    valid = [frame for frame in frames if isinstance(frame, dict) and not frame.get("is_bad", False)]
    if max_frames > 0:
        valid = valid[:max_frames]
    return valid


def select_evenly_spaced_values(values: list[str], target_count: int) -> list[str]:
    if target_count <= 0 or len(values) <= target_count:
        return list(values)
    if target_count == 1:
        return [values[0]]

    selected_positions: list[int] = []
    last_position = len(values) - 1
    for sample_idx in range(target_count):
        position = round(sample_idx * last_position / (target_count - 1))
        if not selected_positions or position != selected_positions[-1]:
            selected_positions.append(position)

    if len(selected_positions) < target_count:
        existing = set(selected_positions)
        for position in range(len(values)):
            if position not in existing:
                selected_positions.append(position)
                existing.add(position)
            if len(selected_positions) >= target_count:
                break

    return [values[position] for position in sorted(selected_positions[:target_count])]


def copy_frame_processed(color_dir: Path, frame_processed_dir: Path, frame_ids: list[str], max_keep: int) -> int:
    selected_ids = select_evenly_spaced_values(frame_ids, max_keep)
    for frame_id in selected_ids:
        shutil.copy2(color_dir / f"{frame_id}.jpg", frame_processed_dir / f"{frame_id}.jpg")
    return len(selected_ids)


def resolve_transforms_path(raw_scene_dir: Path) -> Path:
    candidates = [
        raw_scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json",
        raw_scene_dir / "dslr" / "nerfstudio" / "transforms.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find transforms json under {raw_scene_dir}")


def resolve_color_source_dir(raw_scene_dir: Path) -> Path:
    candidates = [
        raw_scene_dir / "dslr" / "undistorted_images",
        raw_scene_dir / "dslr" / "resized_images",
        raw_scene_dir / "dslr" / "images",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot find DSLR image directory under {raw_scene_dir}")


def resolve_depth_source_dir(scene_id: str, raw_scene_dir: Path, depth_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    if depth_root is not None:
        candidates.extend(
            [
                depth_root / scene_id / "dslr" / "undistorted_depths",
                depth_root / scene_id / "dslr" / "render_depth",
            ]
        )
    candidates.extend(
        [
            raw_scene_dir / "dslr" / "undistorted_depths",
            raw_scene_dir / "dslr" / "render_depth",
        ]
    )

    for path in candidates:
        if path.is_dir():
            return path
    return None


def build_intrinsic_matrix(transforms: dict[str, Any]) -> list[list[float]]:
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    return [
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def write_scene_info_file(path: Path, transforms: dict[str, Any], num_frames: int) -> None:
    width = int(round(float(transforms["w"])))
    height = int(round(float(transforms["h"])))
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    path.write_text(
        "\n".join(
            [
                "axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                f"colorHeight = {height}",
                "colorToDepthExtrinsics = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                f"colorWidth = {width}",
                f"depthHeight = {height}",
                f"depthWidth = {width}",
                f"fx_color = {fx:.6f}",
                f"fx_depth = {fx:.6f}",
                f"fy_color = {fy:.6f}",
                f"fy_depth = {fy:.6f}",
                f"mx_color = {cx:.6f}",
                f"mx_depth = {cx:.6f}",
                f"my_color = {cy:.6f}",
                f"my_depth = {cy:.6f}",
                f"numColorFrames = {num_frames}",
                f"numDepthFrames = {num_frames}",
                "numIMUmeasurements = 0",
                "sceneType = scannetpp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def convert_aggregation_payload(scene_name: str, raw_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw_payload:
        return None

    raw_groups = raw_payload.get("segGroups", [])
    if not isinstance(raw_groups, list):
        return None

    seg_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            continue
        segments = group.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        normalized_segments: list[int] = []
        for segment_id in segments:
            try:
                normalized_segments.append(int(segment_id))
            except Exception:
                continue

        seg_groups.append(
            {
                "id": group_index,
                "objectId": group_index,
                "label": normalize_label_text(group.get("label", "object")) or "object",
                "segments": normalized_segments,
            }
        )

    return {
        "sceneId": scene_name,
        "appId": "stk.v1",
        "segmentsFile": f"{scene_name}_vh_clean.segs.json",
        "segGroups": seg_groups,
    }


def materialize_scannet_geometry(
    scene_dir: Path,
    scene_name: str,
    mesh_path: Path | None,
    semantic_mesh_path: Path | None,
    segmentation_payload: dict[str, Any] | None,
    aggregation_payload: dict[str, Any] | None,
) -> None:
    if mesh_path and mesh_path.is_file():
        shutil.copy2(mesh_path, scene_dir / f"{scene_name}_vh_clean.ply")
        shutil.copy2(mesh_path, scene_dir / f"{scene_name}_vh_clean_2.ply")
    if semantic_mesh_path and semantic_mesh_path.is_file():
        shutil.copy2(semantic_mesh_path, scene_dir / f"{scene_name}_vh_clean_2.labels.ply")
    elif mesh_path and mesh_path.is_file():
        shutil.copy2(mesh_path, scene_dir / f"{scene_name}_vh_clean_2.labels.ply")

    if segmentation_payload:
        write_json(scene_dir / f"{scene_name}_vh_clean.segs.json", segmentation_payload)
        write_json(scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json", segmentation_payload)

    if aggregation_payload:
        write_json(scene_dir / f"{scene_name}.aggregation.json", aggregation_payload)
        write_json(scene_dir / f"{scene_name}_vh_clean.aggregation.json", aggregation_payload)


def convert_one_scene(
    raw_root: Path,
    depth_root: Path | None,
    output_root: Path,
    split: str,
    scene_id: str,
    scene_name_pattern: str,
    max_frames: int,
    frame_processed_max: int,
    overwrite: bool,
) -> dict[str, Any]:
    raw_scene_dir = raw_root / scene_id
    if not raw_scene_dir.is_dir():
        raise FileNotFoundError(f"raw scene directory does not exist: {raw_scene_dir}")

    scene_name = resolve_scene_name(scene_id, scene_name_pattern)
    scene_dir = output_root / split / scene_name
    if overwrite and scene_dir.exists():
        shutil.rmtree(scene_dir, ignore_errors=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    processed_marker = scene_dir / "processed"
    if processed_marker.exists() and not overwrite:
        return {"scene_id": scene_id, "scene_name": scene_name, "status": "skip_existing"}

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    frame_processed_dir = scene_dir / "frame_processed"
    for folder in [color_dir, depth_dir, pose_dir, frame_processed_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    transforms_path = resolve_transforms_path(raw_scene_dir)
    transforms = load_json(transforms_path)
    frames = select_valid_frames(transforms, max_frames=max_frames)
    if not frames:
        return {"scene_id": scene_id, "scene_name": scene_name, "status": "no_valid_frames"}

    color_source_dir = resolve_color_source_dir(raw_scene_dir)
    depth_source_dir = resolve_depth_source_dir(scene_id, raw_scene_dir, depth_root)
    if depth_source_dir is None:
        raise FileNotFoundError(
            f"Cannot find depth directory for scene {scene_id}. "
            "Expected <depth-root>/<scene-id>/dslr/undistorted_depths or render_depth."
        )

    train_test_lists = load_optional_json(raw_scene_dir / "dslr" / "train_test_lists.json") or {}
    test_frame_names = set(train_test_lists.get("test", [])) if isinstance(train_test_lists, dict) else set()

    intrinsic = build_intrinsic_matrix(transforms)
    identity = identity_matrix()
    write_matrix(scene_dir / "intrinsic_color.txt", intrinsic)
    write_matrix(scene_dir / "intrinsic_depth.txt", intrinsic)
    write_matrix(scene_dir / "extrinsic_color.txt", identity)
    write_matrix(scene_dir / "extrinsic_depth.txt", identity)
    write_scene_info_file(scene_dir / f"{scene_name}.txt", transforms, num_frames=len(frames))

    frame_index: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    frame_ids: list[str] = []

    for frame_id, frame in enumerate(frames):
        file_path = str(frame.get("file_path", ""))
        source_name = Path(file_path).name
        source_stem = Path(source_name).stem
        source_color = color_source_dir / source_name
        source_depth = depth_source_dir / f"{source_stem}.png"

        if not source_color.is_file():
            raise FileNotFoundError(f"Missing color frame: {source_color}")
        if not source_depth.is_file():
            raise FileNotFoundError(f"Missing depth frame: {source_depth}")

        target_stem = str(frame_id)
        shutil.copy2(source_color, color_dir / f"{target_stem}.jpg")
        shutil.copy2(source_depth, depth_dir / f"{target_stem}.png")
        write_matrix(pose_dir / f"{target_stem}.txt", frame["transform_matrix"])

        frame_ids.append(target_stem)
        frame_index.append(
            {
                "frame_id": frame_id,
                "source_name": source_name,
                "source_stem": source_stem,
                "is_test_frame": source_name in test_frame_names,
            }
        )
        source_records.append(
            {
                "frame_id": target_stem,
                "source_name": source_name,
                "source_stem": source_stem,
                "source_color": str(source_color.resolve()),
                "source_depth": str(source_depth.resolve()),
                "is_test_frame": source_name in test_frame_names,
            }
        )

    frame_processed_count = copy_frame_processed(color_dir, frame_processed_dir, frame_ids, frame_processed_max)

    scans_dir = raw_scene_dir / "scans"
    mesh_path = scans_dir / "mesh_aligned_0.05.ply"
    semantic_mesh_path = scans_dir / "mesh_aligned_0.05_semantic.ply"
    segmentation_payload = load_optional_json(scans_dir / "segments.json")
    raw_aggregation_payload = load_optional_json(scans_dir / "segments_anno.json")
    aggregation_payload = convert_aggregation_payload(scene_name, raw_aggregation_payload)
    materialize_scannet_geometry(
        scene_dir=scene_dir,
        scene_name=scene_name,
        mesh_path=mesh_path if mesh_path.is_file() else None,
        semantic_mesh_path=semantic_mesh_path if semantic_mesh_path.is_file() else None,
        segmentation_payload=segmentation_payload,
        aggregation_payload=aggregation_payload,
    )

    asset_manifest = {
        "mesh": str(mesh_path.resolve()) if mesh_path.is_file() else None,
        "semantic_mesh": str(semantic_mesh_path.resolve()) if semantic_mesh_path.is_file() else None,
        "segments": str((scans_dir / "segments.json").resolve()) if (scans_dir / "segments.json").is_file() else None,
        "segments_anno": str((scans_dir / "segments_anno.json").resolve())
        if (scans_dir / "segments_anno.json").is_file()
        else None,
    }
    write_json(scene_dir / "asset_manifest.json", asset_manifest)
    write_json(scene_dir / "frame_index.json", frame_index)

    source_manifest = {
        "scene_id": scene_id,
        "source_scene_id": scene_id,
        "scene_name": scene_name,
        "raw_scene_dir": str(raw_scene_dir.resolve()),
        "transforms_path": str(transforms_path.resolve()),
        "color_source_dir": str(color_source_dir.resolve()),
        "depth_source_dir": str(depth_source_dir.resolve()),
        "num_frames_valid": len(frame_ids),
        "num_frames_frame_processed": frame_processed_count,
        "num_objects": len(aggregation_payload.get("segGroups", [])) if aggregation_payload else 0,
        "skipped_labels": collect_skipped_labels(raw_aggregation_payload),
        "ready_for_metadata": bool(
            mesh_path.is_file() and segmentation_payload is not None and aggregation_payload is not None
        ),
        "frame_records": source_records,
    }
    write_json(scene_dir / "scannetpp_source.json", source_manifest)

    processed_marker.write_text("processed\n", encoding="utf-8")
    return {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "status": "converted",
        "frames": len(frame_ids),
        "frame_processed": frame_processed_count,
        "ready_for_metadata": source_manifest["ready_for_metadata"],
    }


def sorted_frame_files(frame_dir: Path) -> list[Path]:
    files = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    files.sort(key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    return files


def extract_metadata_for_scene(
    scene_dir: Path,
    metadata_root: Path,
    frame_type: str,
    overwrite: bool,
    visualize: bool,
    output_root: Path,
    allow_empty_object_metadata: bool,
) -> dict[str, Any]:
    scene_name = scene_dir.name
    mesh_file = scene_dir / f"{scene_name}_vh_clean.ply"
    agg_file = scene_dir / f"{scene_name}_vh_clean.aggregation.json"
    seg_file = scene_dir / f"{scene_name}_vh_clean.segs.json"
    meta_txt = scene_dir / f"{scene_name}.txt"
    frame_dir = scene_dir / frame_type

    required = [mesh_file, agg_file, seg_file, meta_txt, frame_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing and not allow_empty_object_metadata:
        return {"scene_name": scene_name, "status": "missing_inputs", "missing": missing}
    if missing:
        return extract_camera_only_metadata_for_scene(
            scene_dir=scene_dir,
            metadata_root=metadata_root,
            frame_type=frame_type,
            overwrite=overwrite,
            output_root=output_root,
        )

    utils = get_scannet_utils()
    align_mesh_vertices = utils["align_mesh_vertices"]
    export_2d_bbox = utils["export_2d_bbox"]
    export_3d_bbox = utils["export_3d_bbox"]
    get_object_id_to_label_id = utils["get_object_id_to_label_id"]
    read_mesh_vertices_rgb = utils["read_mesh_vertices_rgb"]
    read_scene_axis_alignment = utils["read_scene_axis_alignment"]
    export_frame_metadata = utils["export_frame_metadata"]
    export_scene_metadata = utils["export_scene_metadata"]
    calculate_room_area = utils["calculate_room_area"]

    scene_out_dir = metadata_root / scene_name
    frame_out_dir = scene_out_dir / frame_type
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    frame_out_dir.mkdir(parents=True, exist_ok=True)
    scene_meta_file = scene_out_dir / f"{scene_name}.json"

    scene_metadata = None
    if scene_meta_file.exists() and not overwrite:
        scene_metadata = load_optional_json(scene_meta_file)

    if scene_metadata is None:
        raw_id_map, id_label_map = build_label_maps_from_aggregation(agg_file)
        mesh_vertices = read_mesh_vertices_rgb(str(mesh_file))
        axis_align_matrix = read_scene_axis_alignment(str(meta_txt))
        aligned_mesh_vertices = align_mesh_vertices(mesh_vertices, axis_align_matrix)
        object_id_to_label_id, object_id_to_segs, instance_ids = get_object_id_to_label_id(
            str(agg_file),
            str(seg_file),
            raw_id_map,
        )

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
        scene_metadata = export_scene_metadata(scene_name, bboxes_3d, room_size)
        scene_meta_file.write_text(json.dumps(scene_metadata, ensure_ascii=False, indent=4), encoding="utf-8")

    obj_id_to_3d_bbox = {int(obj["object_id"]): obj["3d_bbox"] for obj in scene_metadata["objects"]}
    obj_id_to_3d_loc = {int(obj["object_id"]): obj["3d_location"] for obj in scene_metadata["objects"]}
    obj_id_to_size = {int(obj["object_id"]): obj["size"] for obj in scene_metadata["objects"]}

    raw_id_map, id_label_map = build_label_maps_from_aggregation(agg_file)
    mesh_vertices = read_mesh_vertices_rgb(str(mesh_file))
    axis_align_matrix = read_scene_axis_alignment(str(meta_txt))
    aligned_mesh_vertices = align_mesh_vertices(mesh_vertices, axis_align_matrix)
    object_id_to_label_id, object_id_to_segs, instance_ids = get_object_id_to_label_id(
        str(agg_file),
        str(seg_file),
        raw_id_map,
    )

    camera_intrinsic_file = scene_dir / "intrinsic_color.txt"
    frame_files = sorted_frame_files(frame_dir)
    frame_written = 0
    for frame_path in frame_files:
        frame_stem = frame_path.stem
        frame_meta_path = frame_out_dir / f"{frame_stem}.json"
        if frame_meta_path.exists() and not overwrite:
            continue

        depth_file = scene_dir / "depth" / f"{frame_stem}.png"
        pose_file = scene_dir / "pose" / f"{frame_stem}.txt"
        if not depth_file.exists() or not pose_file.exists():
            continue

        bboxes_2d = export_2d_bbox(
            axis_align_matrix,
            aligned_mesh_vertices,
            object_id_to_label_id,
            object_id_to_segs,
            instance_ids,
            str(camera_intrinsic_file),
            str(frame_path),
            str(depth_file),
            str(pose_file),
            id_label_map,
            visualize=visualize,
        )
        frame_metadata = export_frame_metadata(
            scene_name,
            frame_path.name,
            bboxes_2d,
            obj_id_to_3d_bbox,
            obj_id_to_3d_loc,
            obj_id_to_size,
        )
        frame_meta_path.write_text(json.dumps(frame_metadata, ensure_ascii=False, indent=4), encoding="utf-8")
        frame_written += 1

    return {
        "scene_name": scene_name,
        "status": "metadata_done",
        "objects": len(scene_metadata["objects"]),
        "frame_metadata_written": frame_written,
    }


def extract_camera_only_metadata_for_scene(
    scene_dir: Path,
    metadata_root: Path,
    frame_type: str,
    overwrite: bool,
    output_root: Path,
) -> dict[str, Any]:
    scene_name = scene_dir.name
    source_info = load_optional_json(scene_dir / "scannetpp_source.json") or {}
    source_scene_id = str(source_info.get("source_scene_id", source_info.get("scene_id", "")))
    frame_records = source_info.get("frame_records", [])
    frame_id_to_source: dict[str, dict[str, Any]] = {}
    if isinstance(frame_records, list):
        for record in frame_records:
            if isinstance(record, dict) and "frame_id" in record:
                frame_id_to_source[str(record["frame_id"])] = record

    coverage_json = load_optional_json(output_root / "coverage.json") or {}
    scene_coverage = coverage_json.get(source_scene_id, {}) if source_scene_id else {}
    if not isinstance(scene_coverage, dict):
        scene_coverage = {}

    scene_out_dir = metadata_root / scene_name
    frame_out_dir = scene_out_dir / frame_type
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    scene_meta = {
        "scene_id": scene_name,
        "room_size": None,
        "objects": [],
        "metadata_type": "camera_only",
        "source_scene_id": source_scene_id,
        "notes": "No mesh/segmentation/aggregation available, object-level metadata omitted.",
    }
    scene_meta_file = scene_out_dir / f"{scene_name}.json"
    if overwrite or not scene_meta_file.exists():
        scene_meta_file.write_text(json.dumps(scene_meta, ensure_ascii=False, indent=4), encoding="utf-8")

    frame_dir = scene_dir / frame_type
    frame_files = sorted_frame_files(frame_dir)
    frame_written = 0
    for frame_path in frame_files:
        frame_stem = frame_path.stem
        frame_meta_path = frame_out_dir / f"{frame_stem}.json"
        if frame_meta_path.exists() and not overwrite:
            continue

        coverage_for_frame = scene_coverage.get(frame_stem)
        frame_meta = {
            "scene_id": scene_name,
            "frame_id": frame_path.name,
            "objects": [],
            "source": frame_id_to_source.get(frame_stem, {}),
            "coverage": coverage_for_frame if isinstance(coverage_for_frame, (dict, float, int)) else None,
            "metadata_type": "camera_only",
        }
        frame_meta_path.write_text(json.dumps(frame_meta, ensure_ascii=False, indent=4), encoding="utf-8")
        frame_written += 1

    return {
        "scene_name": scene_name,
        "status": "metadata_camera_only",
        "objects": 0,
        "frame_metadata_written": frame_written,
    }


def run_convert(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    depth_root = Path(args.depth_root) if args.depth_root else output_root / "processed"
    output_root.mkdir(parents=True, exist_ok=True)

    scene_ids = parse_scene_ids_arg(args.scene_ids, args.scene_ids_file)
    if not scene_ids:
        scene_ids = list_local_scene_ids(raw_root)
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    log(f"convert scenes: {len(scene_ids)}")
    results: list[dict[str, Any]] = []
    for index, scene_id in enumerate(scene_ids, start=1):
        log(f"[{index}/{len(scene_ids)}] convert scene={scene_id}")
        try:
            result = convert_one_scene(
                raw_root=raw_root,
                depth_root=depth_root,
                output_root=output_root,
                split=args.split,
                scene_id=scene_id,
                scene_name_pattern=args.scene_name_pattern,
                max_frames=args.max_frames,
                frame_processed_max=args.frame_processed_max,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            result = {"scene_id": scene_id, "status": "failed", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    return results


def run_extract_metadata(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    metadata_root = output_root / "metadata"
    scene_root = output_root / args.split
    scene_ids = parse_scene_ids_arg(args.scene_ids, args.scene_ids_file)

    if scene_ids:
        scene_dirs = [scene_root / resolve_scene_name(scene_id, args.scene_name_pattern) for scene_id in scene_ids]
    else:
        if not scene_root.exists():
            log(f"scene root does not exist: {scene_root}")
            return []
        scene_dirs = [path for path in scene_root.iterdir() if path.is_dir()]
        scene_dirs.sort(key=lambda path: path.name)

    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]

    log(f"extract metadata scenes: {len(scene_dirs)}")
    results: list[dict[str, Any]] = []
    for index, scene_dir in enumerate(scene_dirs, start=1):
        log(f"[{index}/{len(scene_dirs)}] metadata scene={scene_dir.name}")
        try:
            result = extract_metadata_for_scene(
                scene_dir=scene_dir,
                metadata_root=metadata_root,
                frame_type=args.frame_type,
                overwrite=args.overwrite_metadata,
                visualize=args.visualize,
                output_root=output_root,
                allow_empty_object_metadata=not args.strict_object_metadata,
            )
        except Exception as exc:
            result = {"scene_name": scene_dir.name, "status": "failed", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local ScanNet++ processing pipeline: raw scene -> ScanNet-like scene -> metadata",
    )
    parser.add_argument(
        "--mode",
        choices=["list-scenes", "convert", "extract-metadata", "convert-and-extract"],
        required=True,
    )
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--depth-root",
        default=None,
        help="Directory containing pre-rendered depth, e.g. /mnt/jfs/lidingm/data/dataset/scannetpp/processed",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--scene-ids", default=None, help="Comma separated scene ids")
    parser.add_argument("--scene-ids-file", default=None, help="One scene id per line")
    parser.add_argument("--scene-name-pattern", default="scene{scene_id}_00")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means all")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all valid frames")
    parser.add_argument("--frame-processed-max", type=int, default=32)
    parser.add_argument("--frame-type", default="frame_processed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite converted scene if it exists")
    parser.add_argument("--overwrite-metadata", action="store_true", help="Overwrite metadata json if it exists")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization in bbox extraction")
    parser.add_argument(
        "--strict-object-metadata",
        action="store_true",
        help="Fail when mesh/segmentation/aggregation is missing.",
    )
    return parser


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in results:
        status = str(item.get("status", "unknown"))
        summary[status] = summary.get(status, 0) + 1
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "list-scenes":
        scene_ids = list_local_scene_ids(Path(args.raw_root))
        print(json.dumps({"scene_count": len(scene_ids), "scene_ids": scene_ids}, ensure_ascii=False, indent=2))
        return

    if args.mode == "convert":
        convert_results = run_convert(args)
        print(json.dumps({"summary": summarize(convert_results)}, ensure_ascii=False, indent=2))
        return

    if args.mode == "extract-metadata":
        metadata_results = run_extract_metadata(args)
        print(json.dumps({"summary": summarize(metadata_results)}, ensure_ascii=False, indent=2))
        return

    convert_results = run_convert(args)
    metadata_results = run_extract_metadata(args)
    print(
        json.dumps(
            {
                "convert_summary": summarize(convert_results),
                "metadata_summary": summarize(metadata_results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
