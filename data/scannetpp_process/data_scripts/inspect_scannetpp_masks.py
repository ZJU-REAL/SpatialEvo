#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_S3_PREFIX = "s3://wjdownload/pretrain_datas/scannetpp/scannetpp"
DEFAULT_OUTPUT_ROOT = "/mnt/jfs/lidingm/data/dataset/scannetpp"


@dataclass(frozen=True)
class AwsConfig:
    aws_bin: str
    profile: str | None
    endpoint_url: str | None
    no_verify_ssl: bool


def build_aws_base(config: AwsConfig) -> list[str]:
    cmd = [config.aws_bin]
    if config.profile and str(config.profile).strip():
        cmd.extend(["--profile", config.profile])
    if config.endpoint_url:
        cmd.extend(["--endpoint-url", config.endpoint_url])
    if config.no_verify_ssl:
        cmd.append("--no-verify-ssl")
    return cmd


def run_aws(config: AwsConfig, args: list[str], capture_output: bool = True) -> str:
    cmd = build_aws_base(config) + ["s3"] + args
    print("[CMD]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=True, text=True, capture_output=capture_output)
    return proc.stdout if capture_output else ""


def load_transforms(config: AwsConfig, s3_prefix: str, scene_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"scannetpp_tf_{scene_id}_") as tmp_dir:
        tmp_json = Path(tmp_dir) / "transforms_undistorted.json"
        run_aws(
            config,
            [
                "cp",
                f"{s3_prefix.rstrip('/')}/processed/{scene_id}/dslr/nerfstudio/transforms_undistorted.json",
                str(tmp_json),
            ],
            capture_output=False,
        )
        with tmp_json.open("r", encoding="utf-8") as f:
            return json.load(f)


def download_mask(
    config: AwsConfig,
    s3_prefix: str,
    scene_id: str,
    mask_name: str,
    local_path: Path,
) -> bool:
    source = f"{s3_prefix.rstrip('/')}/processed/{scene_id}/dslr/undistorted_anon_masks/{mask_name}"
    try:
        run_aws(config, ["cp", source, str(local_path)], capture_output=False)
        return True
    except subprocess.CalledProcessError:
        return False


def load_depth(scene_dir: Path, frame_id: str) -> np.ndarray | None:
    depth_file = scene_dir / "depth" / f"{frame_id}.png"
    if not depth_file.exists():
        return None
    depth = np.array(Image.open(depth_file), dtype=np.float32)
    return depth


def infer_mask_type(unique_values: np.ndarray, nonzero_ratio: float) -> str:
    uniq = unique_values.size
    max_val = int(unique_values.max()) if uniq > 0 else 0
    if uniq <= 3 and max_val <= 255:
        return "likely_binary_or_anon_mask"
    if uniq > 20 and max_val > 10:
        return "possible_instance_or_label_mask"
    if nonzero_ratio < 0.15:
        return "likely_sparse_redaction_mask"
    return "uncertain"


def backproject_mask_points(
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
) -> dict[str, Any]:
    valid = (mask > 0) & (depth > 0)
    ys, xs = np.where(valid)
    if xs.size == 0:
        return {"count": 0, "centroid_world": None}

    z = depth[ys, xs]
    # Most PNG depth in ScanNet-like pipeline is millimeter scale.
    if np.percentile(z, 95) > 100:
        z = z / 1000.0

    x = (xs.astype(np.float32) - cx) / fx * z
    y = (ys.astype(np.float32) - cy) / fy * z

    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world = (c2w @ cam.T).T[:, :3]
    centroid = world.mean(axis=0)
    return {
        "count": int(world.shape[0]),
        "centroid_world": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ScanNet++ mask_path availability and mask semantics.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-name-pattern", default="scene{scene_id}_00")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-masks", type=int, default=8)
    parser.add_argument("--save-masks", action="store_true", help="Copy sampled masks to local scene/mask")
    parser.add_argument("--compute-mask-3d", action="store_true", help="Compute masked-point centroid in world coord")
    parser.add_argument("--aws-bin", default="aws")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--no-verify-ssl", action="store_true")
    args = parser.parse_args()

    config = AwsConfig(
        aws_bin=args.aws_bin,
        profile=args.profile,
        endpoint_url=args.endpoint_url,
        no_verify_ssl=args.no_verify_ssl,
    )

    transforms = load_transforms(config, args.s3_prefix, args.scene_id)
    frames = [f for f in transforms.get("frames", []) if not f.get("is_bad", False)]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise SystemExit("No valid frames in transforms_undistorted.json")

    scene_name = args.scene_name_pattern.format(scene_id=args.scene_id)
    scene_dir = Path(args.output_root) / args.split / scene_name
    local_mask_dir = scene_dir / "mask"
    if args.save_masks:
        local_mask_dir.mkdir(parents=True, exist_ok=True)

    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])

    sampled = frames[: max(1, min(len(frames), args.sample_masks))]
    sampled_reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"scannetpp_mask_{args.scene_id}_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, frame in enumerate(sampled):
            mask_name = str(frame.get("mask_path", "")).strip()
            if not mask_name:
                sampled_reports.append({"idx": idx, "frame_file": frame.get("file_path"), "mask_exists": False})
                continue
            tmp_mask = tmp_root / f"{idx}_{Path(mask_name).name}"
            exists = download_mask(config, args.s3_prefix, args.scene_id, mask_name, tmp_mask)
            if not exists:
                sampled_reports.append(
                    {"idx": idx, "frame_file": frame.get("file_path"), "mask_path": mask_name, "mask_exists": False}
                )
                continue

            mask = np.array(Image.open(tmp_mask))
            unique = np.unique(mask)
            nonzero_ratio = float((mask > 0).mean())
            report: dict[str, Any] = {
                "idx": idx,
                "frame_file": frame.get("file_path"),
                "mask_path": mask_name,
                "mask_exists": True,
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
                "unique_count": int(unique.size),
                "unique_preview": [int(v) for v in unique[:16]],
                "nonzero_ratio": nonzero_ratio,
                "mask_type_guess": infer_mask_type(unique, nonzero_ratio),
            }

            if args.save_masks:
                target = local_mask_dir / Path(mask_name).name
                target.write_bytes(tmp_mask.read_bytes())
                report["saved_local"] = str(target)

            if args.compute_mask_3d:
                frame_id = str(idx)
                depth = load_depth(scene_dir, frame_id)
                c2w = np.array(frame["transform_matrix"], dtype=np.float64)
                if depth is None:
                    report["mask_3d"] = {"count": 0, "centroid_world": None, "note": "local depth missing"}
                else:
                    report["mask_3d"] = backproject_mask_points(mask, depth, fx, fy, cx, cy, c2w)
            sampled_reports.append(report)

    exists_count = sum(1 for r in sampled_reports if r.get("mask_exists"))
    guess_counter: dict[str, int] = {}
    for r in sampled_reports:
        guess = str(r.get("mask_type_guess", "unknown"))
        guess_counter[guess] = guess_counter.get(guess, 0) + 1

    summary = {
        "scene_id": args.scene_id,
        "scene_name": scene_name,
        "valid_frames": len(frames),
        "sampled_frames": len(sampled_reports),
        "sampled_mask_exists": exists_count,
        "mask_type_guess_counts": guess_counter,
        "object_level_possible": any(
            "instance" in key or "label" in key for key in guess_counter.keys()
        ),
        "note": "Object-level 3D requires instance-level mask/segmentation; anon mask is insufficient.",
        "sampled_reports": sampled_reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
