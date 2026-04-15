#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


INDEX_FILE_PATTERN = re.compile(r"^(\d+)\.(jpg|jpeg|png|txt)$", re.IGNORECASE)


def parse_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(value) for value in line.split()])
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"Matrix in {path} is not 4x4")
    return rows


def indexed_files(directory: Path, expected_extensions: set[str]) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = INDEX_FILE_PATTERN.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        extension = match.group(2).lower()
        if extension in expected_extensions:
            output[index] = path
    return output


def evenly_sample_indices(all_indices: list[int], max_items: int) -> list[int]:
    if max_items <= 0 or len(all_indices) <= max_items:
        return all_indices
    if max_items == 1:
        return [all_indices[0]]

    sampled: list[int] = []
    last_pos = len(all_indices) - 1
    for i in range(max_items):
        pos = round(i * last_pos / (max_items - 1))
        sampled.append(all_indices[pos])

    deduped: list[int] = []
    for index in sampled:
        if not deduped or deduped[-1] != index:
            deduped.append(index)
    return deduped


def create_rgb_grid(
    color_files: dict[int, Path],
    sample_indices: list[int],
    output_path: Path,
) -> None:
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    images = [np.array(Image.open(color_files[index]).convert("RGB")) for index in sample_indices]
    columns = max(1, min(4, math.ceil(math.sqrt(len(sample_indices)))))
    rows = math.ceil(len(sample_indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.2))
    axes_array = np.array(axes, ndmin=1).reshape(-1)

    for axis in axes_array:
        axis.axis("off")
    for axis, image, index in zip(axes_array, images, sample_indices):
        axis.imshow(image)
        axis.set_title(f"idx={index}", fontsize=8)
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def create_depth_grid(
    depth_files: dict[int, Path],
    sample_indices: list[int],
    output_path: Path,
) -> None:
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    depth_images = [np.array(Image.open(depth_files[index]), dtype=np.float32) for index in sample_indices]
    valid_values = [depth[depth > 0] for depth in depth_images if (depth > 0).any()]
    if valid_values:
        merged = np.concatenate(valid_values)
        vmin = float(np.percentile(merged, 2))
        vmax = float(np.percentile(merged, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    columns = max(1, min(4, math.ceil(math.sqrt(len(sample_indices)))))
    rows = math.ceil(len(sample_indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.2))
    axes_array = np.array(axes, ndmin=1).reshape(-1)

    for axis in axes_array:
        axis.axis("off")
    for axis, depth, index in zip(axes_array, depth_images, sample_indices):
        axis.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
        axis.set_title(f"idx={index}", fontsize=8)
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def create_trajectory_plot(
    pose_files: dict[int, Path],
    ordered_indices: list[int],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for index in ordered_indices:
        matrix = parse_matrix(pose_files[index])
        xs.append(matrix[0][3])
        ys.append(matrix[1][3])
        zs.append(matrix[2][3])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    points = len(xs)

    axes[0].plot(xs, ys, "-o", linewidth=1.2, markersize=2)
    axes[0].set_title(f"Trajectory XY (n={points})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, zs, "-o", linewidth=1.2, markersize=2)
    axes[1].set_title(f"Trajectory XZ (n={points})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].grid(alpha=0.3)

    for axis in axes:
        if len(xs) >= 2:
            axis.set_aspect("equal", adjustable="box")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ScanNet-like scene schema.")
    parser.add_argument("--scene-dir", required=True, help="Path to one ScanNet-like scene directory")
    parser.add_argument(
        "--require-all-scannet-files",
        action="store_true",
        help="Fail if ScanNet mesh/aggregation/txt optional files are missing.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate RGB/depth/trajectory visualization outputs.",
    )
    parser.add_argument(
        "--viz-max-frames",
        type=int,
        default=12,
        help="Max number of frames used for RGB/depth grids.",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Output visualization directory. Default: <scene-dir>/visualization",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    print(f"[CMD] verify scene dir: {scene_dir}")
    if not scene_dir.exists():
        raise SystemExit(f"[ERR] scene_dir does not exist: {scene_dir}")

    required_dirs = ["color", "depth", "pose", "frame_processed"]
    required_files = [
        "intrinsic_color.txt",
        "intrinsic_depth.txt",
        "extrinsic_color.txt",
        "extrinsic_depth.txt",
        "processed",
    ]
    missing_dirs = [name for name in required_dirs if not (scene_dir / name).is_dir()]
    missing_files = [name for name in required_files if not (scene_dir / name).is_file()]
    if missing_dirs or missing_files:
        print(f"[ERR] missing dirs: {missing_dirs}")
        print(f"[ERR] missing files: {missing_files}")
        raise SystemExit(1)

    color_files = indexed_files(scene_dir / "color", {"jpg", "jpeg", "png"})
    depth_files = indexed_files(scene_dir / "depth", {"png"})
    pose_files = indexed_files(scene_dir / "pose", {"txt"})
    frame_processed_files = indexed_files(scene_dir / "frame_processed", {"jpg", "jpeg", "png"})

    if not color_files or not depth_files or not pose_files:
        print("[ERR] color/depth/pose contains empty indexed files")
        raise SystemExit(1)

    color_indices = set(color_files.keys())
    depth_indices = set(depth_files.keys())
    pose_indices = set(pose_files.keys())
    if color_indices != depth_indices or color_indices != pose_indices:
        print("[ERR] index mismatch among color/depth/pose")
        print(f"[ERR] color count={len(color_indices)} depth count={len(depth_indices)} pose count={len(pose_indices)}")
        raise SystemExit(1)

    parse_matrix(scene_dir / "intrinsic_color.txt")
    parse_matrix(scene_dir / "intrinsic_depth.txt")
    parse_matrix(scene_dir / "extrinsic_color.txt")
    parse_matrix(scene_dir / "extrinsic_depth.txt")
    ordered_indices = sorted(pose_indices)
    for index in ordered_indices:
        parse_matrix(pose_files[index])

    optional_scene_files = [
        f"{scene_dir.name}.txt",
        f"{scene_dir.name}.aggregation.json",
        f"{scene_dir.name}_vh_clean.aggregation.json",
        f"{scene_dir.name}_vh_clean.ply",
        f"{scene_dir.name}_vh_clean.segs.json",
        f"{scene_dir.name}_vh_clean_2.0.010000.segs.json",
        f"{scene_dir.name}_vh_clean_2.ply",
        f"{scene_dir.name}_vh_clean_2.labels.ply",
    ]
    present_optional = [name for name in optional_scene_files if (scene_dir / name).exists()]
    missing_optional = [name for name in optional_scene_files if not (scene_dir / name).exists()]

    visualization: dict[str, object] = {
        "enabled": args.visualize,
        "viz_dir": None,
        "sample_indices": [],
        "generated_files": [],
        "error": None,
    }

    if args.visualize:
        viz_dir = Path(args.viz_dir) if args.viz_dir else scene_dir / "visualization"
        sample_indices = evenly_sample_indices(ordered_indices, args.viz_max_frames)
        rgb_grid_path = viz_dir / "rgb_grid.jpg"
        depth_grid_path = viz_dir / "depth_grid.png"
        trajectory_path = viz_dir / "trajectory.png"
        try:
            create_rgb_grid(color_files, sample_indices, rgb_grid_path)
            create_depth_grid(depth_files, sample_indices, depth_grid_path)
            create_trajectory_plot(pose_files, ordered_indices, trajectory_path)
            visualization = {
                "enabled": True,
                "viz_dir": str(viz_dir),
                "sample_indices": sample_indices,
                "generated_files": [
                    str(rgb_grid_path),
                    str(depth_grid_path),
                    str(trajectory_path),
                ],
                "error": None,
            }
            print(f"[OUT] visualization dir: {viz_dir}")
            print(f"[OUT] rgb grid: {rgb_grid_path}")
            print(f"[OUT] depth grid: {depth_grid_path}")
            print(f"[OUT] trajectory: {trajectory_path}")
        except Exception as exc:
            visualization = {
                "enabled": True,
                "viz_dir": str(viz_dir),
                "sample_indices": sample_indices,
                "generated_files": [],
                "error": str(exc),
            }
            print(f"[WARN] visualization generation failed: {exc}")

    report = {
        "scene_dir": str(scene_dir),
        "required_ok": True,
        "require_all_scannet_files": args.require_all_scannet_files,
        "color_count": len(color_files),
        "depth_count": len(depth_files),
        "pose_count": len(pose_files),
        "frame_processed_count": len(frame_processed_files),
        "optional_present": present_optional,
        "optional_missing": missing_optional,
        "visualization": visualization,
    }
    report_path = scene_dir / "verify_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OUT] required schema check: PASS")
    print(f"[OUT] color/depth/pose count: {len(color_files)}/{len(depth_files)}/{len(pose_files)}")
    print(f"[OUT] frame_processed count: {len(frame_processed_files)}")
    print(f"[OUT] optional files present: {len(present_optional)}")
    print(f"[OUT] optional files missing: {len(missing_optional)}")
    if args.require_all_scannet_files and missing_optional:
        print("[ERR] require-all mode failed due to missing optional ScanNet files")
        raise SystemExit(1)
    print(f"[OUT] verify report: {report_path}")


if __name__ == "__main__":
    main()
