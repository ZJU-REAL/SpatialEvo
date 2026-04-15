"""Recovery-only tools for invalid-question fallback.

These tools are intentionally decoupled from the main deterministic rubric
execution path and are only registered when invalid recovery is enabled.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .base_tool import BaseTool
from .vlm_tools import DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, VLMTool

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _object_id(obj: Dict[str, Any]) -> str:
    return str(
        obj.get("object_id", obj.get("id", obj.get("instance_id", "?")))
    ).strip()

def _normalize_label(label: Any) -> str:
    return str(label or "").strip().lower()

def _color_for_label(label: str):
    normalized = _normalize_label(label)
    if not normalized:
        return plt.cm.tab20(0)
    index = sum(ord(ch) for ch in normalized) % 20
    return plt.cm.tab20(index)

def _ensure_output_dir(output_dir: Optional[str | Path]) -> Path:
    if output_dir is None:
        root = Path(tempfile.gettempdir()) / "spatialevo_invalid_recovery"
    else:
        root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root

class InvalidRecoveryTool(BaseTool):
    """Single-call recovery VLM tool.

    This tool delegates to the existing `VLMTool`, but is registered separately
    so that invalid recovery can be enabled/disabled independently from the
    deterministic VLM extraction path.
    """

    def __init__(
        self,
        *,
        vlm_backend: Optional[Any] = None,
        model: str = "gpt-oss-120b-ldm",
        vision_model: str = "qwen3vl-8b",
        api_key: str = DEFAULT_LLM_API_KEY,
        base_url: str = DEFAULT_LLM_BASE_URL,
        timeout: int = 30,
        max_retries: int = 20,
    ):
        super().__init__(
            name="invalid_recovery_tool",
            description="Try a single LLM/VLM recovery when DGE fails",
        )
        self.backend = vlm_backend or VLMTool(
            model=model,
            vision_model=vision_model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def execute(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_vision: Optional[bool] = None,
        **kwargs,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            return "Error: Empty recovery prompt"
        if not hasattr(self.backend, "execute"):
            return "Error: Invalid recovery backend"
        response = self.backend.execute(
            prompt=prompt,
            image_paths=image_paths or [],
            temperature=temperature,
            max_tokens=max_tokens,
            use_vision=use_vision,
            **kwargs,
        )
        return str(response).strip()

class SceneBEVEvidenceTool(BaseTool):
    """Generate a lightweight bird's-eye-view evidence image from scene metadata."""

    def __init__(self):
        super().__init__(
            name="scene_bev_evidence_tool",
            description="Build a recovery top-down view from scene metadata",
        )

    def execute(
        self,
        scene_id: str,
        scene_metadata: Dict[str, Any],
        output_dir: Optional[str] = None,
        highlight_labels: Optional[Sequence[str]] = None,
        max_objects: int = 80,
        **kwargs,
    ) -> Dict[str, Any]:
        objects = scene_metadata.get("objects", []) if isinstance(scene_metadata, dict) else []
        if not isinstance(objects, list) or not objects:
            return {
                "success": False,
                "error": "No visualizable objects found in `scene_metadata`",
            }

        highlight_set = {_normalize_label(x) for x in (highlight_labels or []) if str(x).strip()}
        selected = sorted(
            [obj for obj in objects if isinstance(obj, dict)],
            key=lambda obj: (
                _normalize_label(obj.get("label")) not in highlight_set,
                _normalize_label(obj.get("label")),
                _object_id(obj),
            ),
        )[: max(1, int(max_objects))]

        fig, ax = plt.subplots(figsize=(10, 8))
        for obj in selected:
            label = _normalize_label(obj.get("label"))
            obj_id = _object_id(obj)
            color = "red" if label in highlight_set else _color_for_label(label)
            bbox = obj.get("3d_bbox")
            location = obj.get("3d_location", [0.0, 0.0, 0.0])

            if isinstance(bbox, (list, tuple)) and len(bbox) >= 6:
                try:
                    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in bbox[:6]]
                    width = max(1e-3, x_max - x_min)
                    height = max(1e-3, y_max - y_min)
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        width,
                        height,
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.20,
                    )
                    ax.add_patch(rect)
                    text_x = x_min + width * 0.5
                    text_y = y_min + height * 0.5
                except (TypeError, ValueError):
                    xyz = [float(v) for v in (location[:3] if isinstance(location, (list, tuple)) else [0.0, 0.0, 0.0])]
                    ax.scatter(xyz[0], xyz[1], s=80, c=[color])
                    text_x, text_y = xyz[0], xyz[1]
            else:
                xyz = [float(v) for v in (location[:3] if isinstance(location, (list, tuple)) else [0.0, 0.0, 0.0])]
                ax.scatter(xyz[0], xyz[1], s=80, c=[color])
                text_x, text_y = xyz[0], xyz[1]

            ax.text(
                text_x,
                text_y,
                f"{obj_id}:{label}",
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 0.2, "edgecolor": "none"},
            )

        ax.set_title(f"Scene BEV Recovery Evidence: {scene_id}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        out_dir = _ensure_output_dir(output_dir) / "bev"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}_bev.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {
            "success": True,
            "image_path": str(out_path),
            "num_objects": len(selected),
            "highlight_labels": sorted(highlight_set),
        }

class BBoxEvidenceTool(BaseTool):
    """Render 2D bbox evidence from a raw image and visible object metadata."""

    def __init__(self):
        super().__init__(
            name="bbox_evidence_tool",
            description="Build bbox evidence images from visible-object data",
        )

    @staticmethod
    def _sort_objects(
        objects: Iterable[Dict[str, Any]],
        highlight_labels: Sequence[str],
        min_visibility: float,
        max_objects: int,
    ) -> List[Dict[str, Any]]:
        highlight_set = {_normalize_label(x) for x in highlight_labels if str(x).strip()}
        selected: List[Dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("2d_bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue
            visibility = _as_float(obj.get("visibility"), 1.0)
            if visibility < float(min_visibility):
                continue
            selected.append(obj)
        selected.sort(
            key=lambda obj: (
                _normalize_label(obj.get("label")) not in highlight_set,
                -_as_float(obj.get("visibility"), 1.0),
                _object_id(obj),
            )
        )
        return selected[: max(1, int(max_objects))]

    def execute(
        self,
        scene_id: str,
        frame_id: Any,
        image_path: str,
        objects: Sequence[Dict[str, Any]],
        output_dir: Optional[str] = None,
        highlight_labels: Optional[Sequence[str]] = None,
        min_visibility: float = 0.1,
        max_objects: int = 25,
        **kwargs,
    ) -> Dict[str, Any]:
        image_file = Path(str(image_path))
        if not image_file.exists():
            return {"success": False, "error": f"Image not found: {image_path}"}
        if not isinstance(objects, Sequence) or len(objects) == 0:
            return {"success": False, "error": "No bbox object info available"}

        selected = self._sort_objects(
            objects=objects,
            highlight_labels=highlight_labels or [],
            min_visibility=min_visibility,
            max_objects=max_objects,
        )
        if not selected:
            return {"success": False, "error": "No bbox object passes the visibility threshold"}

        image = plt.imread(str(image_file))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.axis("off")

        highlight_set = {_normalize_label(x) for x in (highlight_labels or []) if str(x).strip()}
        for obj in selected:
            bbox = obj.get("2d_bbox", [0, 0, 0, 0])
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                continue
            label = _normalize_label(obj.get("label"))
            obj_id = _object_id(obj)
            visibility = _as_float(obj.get("visibility"), 1.0)
            color = "red" if label in highlight_set else "yellow"
            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0.0, y1 - 4.0),
                f"ID:{obj_id} {label} vis={visibility:.2f}",
                fontsize=8,
                color="black",
                bbox={"facecolor": color, "alpha": 0.5, "pad": 0.2, "edgecolor": "none"},
            )

        out_dir = _ensure_output_dir(output_dir) / "bbox"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}_{str(frame_id).strip()}_bbox.jpg"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

        return {
            "success": True,
            "image_path": str(out_path),
            "num_boxes": len(selected),
            "highlight_labels": sorted(highlight_set),
        }
