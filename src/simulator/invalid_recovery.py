"""Invalid-question recovery controller.

This module keeps the invalid-recovery logic isolated from the main simulator
execution path. It assembles concise scene/image evidence, performs a single
LLM/VLM recovery call, and parses the final `<answer>...</answer>` tag.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


RECOVERABLE_ERROR_CODES = {
    "QUESTION_INVALID",
    "INVALID_QUESTION_RULE",
    "EXECUTION_FAILED",
    "EMPTY_ANSWER",
    "MISSING_DEPENDENCY",
    "MISSING_TOOL",
}

SUPPORTED_RECOVERY_TASK_GROUPS = {"scene", "single_image", "image_pair"}
HARD_BLOCKED_TASK_TYPES = {
    "position_obj_obj",
    "position_obj_reg",
    "position_reg_reg",
}
INVALID_ANSWER_SET = {
    "",
    "invalid",
    "null",
    "none",
    "unknown",
    "unanswerable",
    "cannot answer",
    "not answerable",
    "not solvable",
}
ANSWER_HINTS = {
    "visibility_compare": "If answerable, answer with exactly one of: image1, image2, same, neither.",
    "elevation_cam_cam": "If answerable, answer with exactly one of: higher, lower, same_level.",
    "position_cam_cam": "If answerable, answer with a concise stable direction such as left, right, front, back, up, or down.",
    "position_cam_obj": "If answerable, answer with a concise stable direction such as left, right, front, back, up, or down.",
    "position_cam_reg": "If answerable, answer with a concise stable direction such as left, right, front, back, up, or down.",
    "single_image_relative_direction": "If answerable, answer with a concise relative direction such as left, right, front, or back.",
    "relative_direction_hard": "If answerable, answer with a concise relative direction such as left, right, front, back, front-left, front-right, back-left, or back-right.",
    "depth_order_obj_obj": "If answerable, answer with the nearer object label or same.",
    "motion_camera": "If answerable, answer with a concise camera motion direction such as left, right, up, down, front, back, left-front, left-up, front-up, clockwise, or counterclockwise.",
}


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


@dataclass
class RecoveryResult:
    attempted: bool = False
    recovered: bool = False
    answer: str = ""
    raw_response: str = ""
    reason: str = ""
    prompt: str = ""
    image_paths: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class InvalidRecoveryController:
    """Single-call invalid recovery orchestrator."""

    def __init__(
        self,
        *,
        tools: Dict[str, Any],
        scannet_root: str,
        get_task_group: Callable[[str], Optional[str]],
        get_scene_summary: Callable[..., Dict[str, Any]],
        get_single_image_summary: Callable[..., Dict[str, Any]],
        get_multi_image_summary: Callable[..., Dict[str, Any]],
        extract_frame_id_from_image_path: Callable[[str], Optional[str]],
        get_visibility_floor: Callable[[], float],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.tools = tools
        self.scannet_root = scannet_root
        self.get_task_group = get_task_group
        self.get_scene_summary = get_scene_summary
        self.get_single_image_summary = get_single_image_summary
        self.get_multi_image_summary = get_multi_image_summary
        self.extract_frame_id_from_image_path = extract_frame_id_from_image_path
        self.get_visibility_floor = get_visibility_floor
        cfg = config if isinstance(config, dict) else {}
        self.config = {
            "max_scene_images": int(cfg.get("max_scene_images", 0)),
            "max_scene_object_lines": int(cfg.get("max_scene_object_lines", 60)),
            "max_frame_object_lines": int(cfg.get("max_frame_object_lines", 25)),
            "max_tokens": int(cfg.get("max_tokens", 512)),
            "temperature": float(cfg.get("temperature", 0.0)),
            "attach_scene_bev": bool(cfg.get("attach_scene_bev", True)),
            "attach_bbox_overlay": bool(cfg.get("attach_bbox_overlay", True)),
        }

    def recover(
        self,
        *,
        task_type: str,
        question: str,
        input_data: Dict[str, Any],
        error_code: str,
        error: str,
        parsed_params: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryResult:
        can_attempt, gate_reason = self._should_attempt(task_type=task_type, error_code=error_code)
        if not can_attempt:
            return RecoveryResult(attempted=False, reason=gate_reason)

        recovery_tool = self.tools.get("invalid_recovery_tool")
        if recovery_tool is None:
            return RecoveryResult(attempted=False, reason="missing_invalid_recovery_tool")

        evidence = self._build_evidence(
            task_type=task_type,
            question=question,
            input_data=input_data,
            parsed_params=parsed_params or {},
            execution_context=execution_context or {},
        )
        prompt = self._build_prompt(
            task_type=task_type,
            task_group=self.get_task_group(task_type) or "unknown",
            question=question,
            error_code=error_code,
            error=error,
            parsed_params=parsed_params or {},
            validation_result=validation_result,
            evidence=evidence,
        )

        image_paths = [
            str(path) for path in evidence.get("image_paths", [])
            if isinstance(path, str) and path.strip()
        ]
        try:
            raw_response = recovery_tool.execute(
                prompt=prompt,
                image_paths=image_paths,
                temperature=float(self.config.get("temperature", 0.0)),
                max_tokens=int(self.config.get("max_tokens", 512)),
                use_vision=len(image_paths) > 0,
            )
        except Exception as exc:
            return RecoveryResult(
                attempted=True,
                recovered=False,
                answer="",
                reason=f"invalid_recovery_tool_failed: {exc}",
                prompt=prompt,
                image_paths=image_paths,
                evidence=evidence,
            )

        answer = self._normalize_answer(
            task_type=task_type,
            raw_answer=self._extract_answer(raw_response),
        )
        recovered = answer not in INVALID_ANSWER_SET
        return RecoveryResult(
            attempted=True,
            recovered=recovered,
            answer=answer if recovered else "",
            raw_response=str(raw_response or ""),
            reason="recovered" if recovered else "model_returned_invalid",
            prompt=prompt,
            image_paths=image_paths,
            evidence=evidence,
        )

    def _should_attempt(self, *, task_type: str, error_code: str) -> tuple[bool, str]:
        if task_type in HARD_BLOCKED_TASK_TYPES:
            return False, "task_type_hard_blocked"
        task_group = self.get_task_group(task_type) or ""
        if task_group not in SUPPORTED_RECOVERY_TASK_GROUPS:
            return False, f"task_group_not_supported:{task_group}"
        if error_code not in RECOVERABLE_ERROR_CODES:
            return False, f"error_code_not_recoverable:{error_code}"
        return True, ""

    def _default_metadata_dir(self) -> str:
        return str(Path(self.scannet_root).parent / "metadata")

    def _build_evidence(
        self,
        *,
        task_type: str,
        question: str,
        input_data: Dict[str, Any],
        parsed_params: Dict[str, Any],
        execution_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = {}
        context.update(input_data if isinstance(input_data, dict) else {})
        context.update(execution_context if isinstance(execution_context, dict) else {})

        scene_id = str(context.get("scene_id", "")).strip()
        metadata_dir = context.get("metadata_dir") or self._default_metadata_dir()
        min_visibility = self._resolve_min_visibility(context.get("min_visibility"))
        task_group = self.get_task_group(task_type) or "unknown"
        relevant_labels = self._extract_relevant_labels(
            question=question,
            parsed_params=parsed_params,
            context=context,
        )

        scene_summary = self.get_scene_summary(
            scene_id=scene_id,
            metadata_dir=metadata_dir,
            include_objects=True,
        )

        evidence: Dict[str, Any] = {
            "task_group": task_group,
            "scene_summary": scene_summary,
            "relevant_labels": relevant_labels,
            "image_paths": [],
        }

        attached_images: List[str] = []
        frame_entries: List[Dict[str, Any]] = []

        if task_group == "scene":
            sampled_scene_images = self._resolve_scene_image_paths(
                scene_id=scene_id,
                context=context,
                max_images=self.config["max_scene_images"],
            )
            attached_images.extend(sampled_scene_images)
            if self.config.get("attach_scene_bev", True):
                bev_result = self._build_scene_bev(
                    scene_id=scene_id,
                    scene_summary=scene_summary,
                    relevant_labels=relevant_labels,
                )
                if bev_result is not None:
                    evidence["scene_bev"] = bev_result
                    bev_path = bev_result.get("image_path")
                    if isinstance(bev_path, str) and bev_path.strip():
                        attached_images.append(bev_path)
            evidence["scene_image_paths"] = sampled_scene_images

        elif task_group == "single_image":
            image_path = self._resolve_single_image_path(scene_id=scene_id, context=context)
            single_summary = self.get_single_image_summary(
                scene_id=scene_id,
                image_path=image_path,
                frame_id=context.get("frame_id"),
                frame_type=context.get("frame_type", "frame_processed"),
                metadata_dir=metadata_dir,
                min_visibility=min_visibility,
                include_objects=True,
            )
            evidence["single_image_summary"] = single_summary
            if isinstance(image_path, str) and image_path.strip():
                attached_images.append(image_path)
            if self.config.get("attach_bbox_overlay", True):
                overlay = self._build_bbox_overlay(
                    scene_id=scene_id,
                    frame_summary=single_summary,
                    image_path=image_path,
                    relevant_labels=relevant_labels,
                    min_visibility=min_visibility,
                )
                if overlay is not None:
                    evidence["single_image_bbox"] = overlay
                    overlay_path = overlay.get("image_path")
                    if isinstance(overlay_path, str) and overlay_path.strip():
                        attached_images.append(overlay_path)

        elif task_group == "image_pair":
            image_paths = self._resolve_image_pair_paths(scene_id=scene_id, context=context)
            multi_summary = self.get_multi_image_summary(
                scene_id=scene_id,
                image_paths=image_paths or None,
                frame_ids=context.get("frame_ids"),
                frame_type=context.get("frame_type", "frame_processed"),
                metadata_dir=metadata_dir,
                min_visibility=min_visibility,
                include_objects=True,
            )
            evidence["multi_image_summary"] = multi_summary
            attached_images.extend(image_paths)
            frame_entries = list(multi_summary.get("frame_summaries", []))
            if self.config.get("attach_bbox_overlay", True):
                bbox_paths: List[str] = []
                for frame_summary in frame_entries[:2]:
                    overlay = self._build_bbox_overlay(
                        scene_id=scene_id,
                        frame_summary=frame_summary,
                        image_path=frame_summary.get("image_path"),
                        relevant_labels=relevant_labels,
                        min_visibility=min_visibility,
                    )
                    if overlay is None:
                        continue
                    bbox_paths.append(overlay.get("image_path", ""))
                bbox_paths = [path for path in bbox_paths if isinstance(path, str) and path.strip()]
                if bbox_paths:
                    evidence["image_pair_bbox_paths"] = bbox_paths
                    attached_images.extend(bbox_paths)
            if task_type == "position_cam_reg" and self.config.get("attach_scene_bev", True):
                bev_result = self._build_scene_bev(
                    scene_id=scene_id,
                    scene_summary=scene_summary,
                    relevant_labels=relevant_labels,
                )
                if bev_result is not None:
                    evidence["scene_bev"] = bev_result
                    bev_path = bev_result.get("image_path")
                    if isinstance(bev_path, str) and bev_path.strip():
                        attached_images.append(bev_path)

        evidence["image_paths"] = self._dedupe_paths(attached_images)
        evidence["scene_text"] = self._format_scene_summary(
            scene_summary=scene_summary,
            task_group=task_group,
            relevant_labels=relevant_labels,
        )
        if "single_image_summary" in evidence:
            evidence["single_image_text"] = self._format_single_image_summary(
                single_summary=evidence["single_image_summary"],
                relevant_labels=relevant_labels,
            )
        if "multi_image_summary" in evidence:
            evidence["multi_image_text"] = self._format_multi_image_summary(
                multi_summary=evidence["multi_image_summary"],
                relevant_labels=relevant_labels,
            )
        return evidence

    def _build_prompt(
        self,
        *,
        task_type: str,
        task_group: str,
        question: str,
        error_code: str,
        error: str,
        parsed_params: Dict[str, Any],
        validation_result: Optional[Dict[str, Any]],
        evidence: Dict[str, Any],
    ) -> str:
        answer_hint = ANSWER_HINTS.get(
            task_type,
            "If answerable, answer concisely and only with the final answer phrase.",
        )
        validation_text = self._format_validation_result(validation_result)
        parsed_text = json.dumps(parsed_params, ensure_ascii=False, sort_keys=True) if parsed_params else "{}"
        evidence_lines = [
            "You are the invalid-question recovery assistant for a deterministic geometric environment.",
            "The deterministic pipeline failed to produce a valid answer, but the question may still be answerable from the evidence below.",
            "",
            "Rules:",
            "1. Use both the attached images and the provided structured scene/object evidence.",
            "2. If the question can be answered with a stable interpretation, answer it.",
            "3. If the question has no stable answer, has no unique grounding, or evidence is insufficient, output invalid.",
            "4. Do not return JSON.",
            "5. The LAST line must be exactly one tag: <answer>YOUR_FINAL_ANSWER</answer>.",
            "6. If the question is not answerable, the LAST line must be: <answer>invalid</answer>.",
            "",
            f"Task type: {task_type}",
            f"Task group: {task_group}",
            f"Original question: {question}",
            f"Deterministic failure: [{error_code}] {error}",
            f"Partial parsed params from deterministic path: {parsed_text}",
        ]
        if validation_text:
            evidence_lines.append(f"Validator notes: {validation_text}")
        evidence_lines.extend(
            [
                answer_hint,
                "",
                "Scene evidence:",
                evidence.get("scene_text", "(none)"),
            ]
        )
        if evidence.get("single_image_text"):
            evidence_lines.extend(["", "Single-image evidence:", evidence["single_image_text"]])
        if evidence.get("multi_image_text"):
            evidence_lines.extend(["", "Multi-image evidence:", evidence["multi_image_text"]])

        image_note = self._format_attached_image_note(task_group=task_group, evidence=evidence)
        if image_note:
            evidence_lines.extend(["", "Attached image note:", image_note])
        return "\n".join(line for line in evidence_lines if isinstance(line, str))

    def _format_attached_image_note(self, *, task_group: str, evidence: Dict[str, Any]) -> str:
        notes: List[str] = []
        if task_group == "scene":
            scene_images = evidence.get("scene_image_paths", []) or []
            if scene_images:
                notes.append(
                    f"Scene views come from the scene-level frame set ({len(scene_images)} attached, preferring frame_processed)."
                )
            if evidence.get("scene_bev"):
                notes.append("A BEV evidence image from the full scene metadata is attached.")
        elif task_group == "single_image":
            notes.append("The original single image is attached.")
            if evidence.get("single_image_bbox"):
                notes.append("An additional bbox overlay evidence image is attached.")
        elif task_group == "image_pair":
            notes.append("The original image pair is attached.")
            bbox_paths = evidence.get("image_pair_bbox_paths", []) or []
            if bbox_paths:
                notes.append(f"Additional bbox overlay evidence images are attached ({len(bbox_paths)}).")
            if evidence.get("scene_bev"):
                notes.append("A BEV evidence image is also attached for region grounding.")
        return " ".join(notes)

    def _format_validation_result(self, validation_result: Optional[Dict[str, Any]]) -> str:
        if not isinstance(validation_result, dict):
            return ""
        issues = validation_result.get("issues", [])
        if not isinstance(issues, list) or not issues:
            return ""
        concise = [str(item).strip() for item in issues if str(item).strip()]
        return " | ".join(concise[:6])

    def _format_scene_summary(
        self,
        *,
        scene_summary: Dict[str, Any],
        task_group: str,
        relevant_labels: Sequence[str],
    ) -> str:
        if not isinstance(scene_summary, dict) or not scene_summary.get("success", False):
            return "Scene summary unavailable."
        lines = [
            f"- scene_id: {scene_summary.get('scene_id', '')}",
            f"- total_objects: {scene_summary.get('total_objects', 0)}",
            f"- unique_label_count: {scene_summary.get('unique_label_count', 0)}",
            f"- room_size: {scene_summary.get('room_size', 'unknown')}",
        ]
        label_counts = scene_summary.get("label_counts", {})
        if isinstance(label_counts, dict) and label_counts:
            items = [f"{label}={cnt}" for label, cnt in sorted(label_counts.items())]
            lines.append("- label_counts: " + ", ".join(items[:40]))

        objects = scene_summary.get("objects", []) if isinstance(scene_summary.get("objects"), list) else []
        selected = self._select_scene_objects(
            objects=objects,
            relevant_labels=relevant_labels,
            task_group=task_group,
            max_lines=self.config["max_scene_object_lines"],
        )
        if selected:
            lines.append("- scene_objects_compact:")
            for obj in selected:
                lines.append(f"  {self._format_scene_object_line(obj)}")
        return "\n".join(lines)

    def _format_single_image_summary(
        self,
        *,
        single_summary: Dict[str, Any],
        relevant_labels: Sequence[str],
    ) -> str:
        if not isinstance(single_summary, dict) or not single_summary.get("success", False):
            return "Single-image summary unavailable."
        lines = [
            f"- frame_id: {single_summary.get('frame_id', '')}",
            f"- visible_objects_count: {single_summary.get('visible_objects_count', 0)}",
            f"- visible_unique_label_count: {single_summary.get('visible_unique_label_count', 0)}",
        ]
        visible_counts = single_summary.get("visible_label_counts", {})
        if isinstance(visible_counts, dict) and visible_counts:
            items = [f"{label}={cnt}" for label, cnt in sorted(visible_counts.items())]
            lines.append("- visible_label_counts: " + ", ".join(items[:30]))
        objects = single_summary.get("visible_objects", []) if isinstance(single_summary.get("visible_objects"), list) else []
        if objects:
            lines.append("- visible_objects_compact:")
            for obj in self._select_frame_objects(objects, relevant_labels, self.config["max_frame_object_lines"]):
                lines.append(f"  {self._format_frame_object_line(obj)}")
        return "\n".join(lines)

    def _format_multi_image_summary(
        self,
        *,
        multi_summary: Dict[str, Any],
        relevant_labels: Sequence[str],
    ) -> str:
        if not isinstance(multi_summary, dict) or not multi_summary.get("success", False):
            return "Multi-image summary unavailable."
        lines = [
            f"- requested_frames: {multi_summary.get('requested_frames', [])}",
            f"- num_frames_succeeded: {multi_summary.get('num_frames_succeeded', 0)}",
        ]
        aggregate = multi_summary.get("aggregate", {})
        if isinstance(aggregate, dict):
            union_labels = aggregate.get("union_visible_labels", [])
            if union_labels:
                lines.append("- union_visible_labels: " + ", ".join(map(str, union_labels[:30])))
            intersection = aggregate.get("intersection_visible_labels", [])
            if intersection:
                lines.append("- intersection_visible_labels: " + ", ".join(map(str, intersection[:30])))

        frame_summaries = multi_summary.get("frame_summaries", [])
        for idx, frame_summary in enumerate(frame_summaries[:2], 1):
            lines.append(f"- image{idx}_frame_id: {frame_summary.get('frame_id', '')}")
            counts = frame_summary.get("visible_label_counts", {})
            if isinstance(counts, dict) and counts:
                items = [f"{label}={cnt}" for label, cnt in sorted(counts.items())]
                lines.append(f"  visible_label_counts: {', '.join(items[:20])}")
            objects = frame_summary.get("visible_objects", []) if isinstance(frame_summary.get("visible_objects"), list) else []
            for obj in self._select_frame_objects(objects, relevant_labels, self.config["max_frame_object_lines"]):
                lines.append(f"  {self._format_frame_object_line(obj)}")
        return "\n".join(lines)

    def _select_scene_objects(
        self,
        *,
        objects: Sequence[Dict[str, Any]],
        relevant_labels: Sequence[str],
        task_group: str,
        max_lines: int,
    ) -> List[Dict[str, Any]]:
        relevant_set = {str(x).strip().lower() for x in relevant_labels if str(x).strip()}
        filtered = [obj for obj in objects if isinstance(obj, dict)]
        if task_group != "scene" and relevant_set:
            preferred = [obj for obj in filtered if _normalize_label(obj.get("label")) in relevant_set]
            if preferred:
                filtered = preferred
        filtered = sorted(
            filtered,
            key=lambda obj: (
                _normalize_label(obj.get("label")) not in relevant_set,
                _normalize_label(obj.get("label")),
                str(obj.get("object_id", obj.get("id", ""))),
            ),
        )
        return filtered[: max(1, int(max_lines))]

    def _select_frame_objects(
        self,
        objects: Sequence[Dict[str, Any]],
        relevant_labels: Sequence[str],
        max_lines: int,
    ) -> List[Dict[str, Any]]:
        relevant_set = {str(x).strip().lower() for x in relevant_labels if str(x).strip()}
        filtered = [obj for obj in objects if isinstance(obj, dict)]
        filtered = sorted(
            filtered,
            key=lambda obj: (
                _normalize_label(obj.get("label")) not in relevant_set,
                -self._safe_float(obj.get("visibility"), 1.0),
                str(obj.get("object_id", obj.get("id", ""))),
            ),
        )
        return filtered[: max(1, int(max_lines))]

    @staticmethod
    def _format_scene_object_line(obj: Dict[str, Any]) -> str:
        obj_id = obj.get("object_id", obj.get("id", "?"))
        label = obj.get("label", "unknown")
        loc = obj.get("3d_location", [0.0, 0.0, 0.0])
        size = obj.get("size", {})
        if isinstance(size, dict):
            size_text = f"size=({size.get('width', '?')}, {size.get('length', '?')}, {size.get('height', '?')})"
        else:
            size_text = f"size={size}"
        return f"id={obj_id} label={label} loc={loc} {size_text}"

    @staticmethod
    def _format_frame_object_line(obj: Dict[str, Any]) -> str:
        obj_id = obj.get("object_id", obj.get("id", "?"))
        label = obj.get("label", "unknown")
        visibility = obj.get("visibility", "?")
        bbox = obj.get("2d_bbox", [])
        loc = obj.get("3d_location", obj.get("location", []))
        return f"id={obj_id} label={label} visibility={visibility} bbox={bbox} loc={loc}"

    def _build_scene_bev(
        self,
        *,
        scene_id: str,
        scene_summary: Dict[str, Any],
        relevant_labels: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        tool = self.tools.get("scene_bev_evidence_tool")
        if tool is None:
            return None
        try:
            result = tool.execute(
                scene_id=scene_id,
                scene_metadata=scene_summary,
                highlight_labels=list(relevant_labels),
            )
        except Exception:
            return None
        if isinstance(result, dict) and result.get("success", False):
            return result
        return None

    def _build_bbox_overlay(
        self,
        *,
        scene_id: str,
        frame_summary: Dict[str, Any],
        image_path: Optional[str],
        relevant_labels: Sequence[str],
        min_visibility: float,
    ) -> Optional[Dict[str, Any]]:
        tool = self.tools.get("bbox_evidence_tool")
        if tool is None or not isinstance(image_path, str) or not image_path.strip():
            return None
        visible_objects = frame_summary.get("visible_objects", [])
        if not isinstance(visible_objects, list) or not visible_objects:
            return None
        try:
            result = tool.execute(
                scene_id=scene_id,
                frame_id=frame_summary.get("frame_id", ""),
                image_path=image_path,
                objects=visible_objects,
                highlight_labels=list(relevant_labels),
                min_visibility=min_visibility,
            )
        except Exception:
            return None
        if isinstance(result, dict) and result.get("success", False):
            return result
        return None

    def _extract_relevant_labels(
        self,
        *,
        question: str,
        parsed_params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        labels: List[str] = []

        def _push(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    _push(item)
                return
            if not isinstance(value, str):
                return
            text = value.strip().lower()
            if text in {"", "null", "none", "unknown", "easy", "medium", "hard"}:
                return
            if text not in labels:
                labels.append(text)

        for value in parsed_params.values():
            _push(value)

        scene_metadata = context.get("scene_metadata", {})
        objects = scene_metadata.get("objects", []) if isinstance(scene_metadata, dict) else []
        q = str(question or "").lower()
        for obj in objects:
            label = _normalize_label(obj.get("label"))
            if label and label in q and label not in labels:
                labels.append(label)
        return labels

    def _resolve_min_visibility(self, raw_value: Any) -> float:
        floor = float(self.get_visibility_floor())
        try:
            return max(floor, float(raw_value))
        except (TypeError, ValueError):
            return floor

    def _resolve_scene_image_paths(
        self,
        *,
        scene_id: str,
        context: Dict[str, Any],
        max_images: int,
    ) -> List[str]:
        explicit_paths = context.get("image_paths")
        if isinstance(explicit_paths, list):
            selected = [
                str(path).strip()
                for path in explicit_paths
                if isinstance(path, str) and str(path).strip()
            ]
            if selected:
                return self._truncate_or_sample_paths(selected, max_images=max_images)
        return self._sample_scene_images(scene_id, max_images)

    def _sample_scene_images(self, scene_id: str, max_images: int) -> List[str]:
        candidate_dirs = [
            Path(self.scannet_root) / scene_id / "frame_processed",
            Path(self.scannet_root) / scene_id / "color",
        ]
        files: List[Path] = []
        for candidate_dir in candidate_dirs:
            if not candidate_dir.exists():
                continue
            files = sorted(
                [path for path in candidate_dir.glob("*.jpg") if path.is_file()],
                key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
            )
            if files:
                break
        if not files:
            return []
        return self._truncate_or_sample_paths([str(path) for path in files], max_images=max_images)

    @staticmethod
    def _truncate_or_sample_paths(paths: Sequence[str], max_images: int) -> List[str]:
        paths = [str(path) for path in paths if isinstance(path, str) and str(path).strip()]
        if max_images <= 0:
            return list(paths)
        if len(paths) <= max_images:
            return list(paths)
        if max_images <= 1:
            return [str(paths[0])]
        indices = {
            int(round(i * (len(paths) - 1) / float(max_images - 1)))
            for i in range(max_images)
        }
        return [str(paths[idx]) for idx in sorted(indices)]

    def _resolve_single_image_path(self, *, scene_id: str, context: Dict[str, Any]) -> Optional[str]:
        image_path = context.get("image_path")
        if isinstance(image_path, str) and image_path.strip():
            return image_path
        frame_id = context.get("frame_id")
        if frame_id is None:
            image_paths = context.get("image_paths")
            if isinstance(image_paths, list) and image_paths:
                candidate = image_paths[0]
                if isinstance(candidate, str) and candidate.strip():
                    return candidate
            return None
        return str(Path(self.scannet_root) / scene_id / "color" / f"{str(frame_id).strip()}.jpg")

    def _resolve_image_pair_paths(self, *, scene_id: str, context: Dict[str, Any]) -> List[str]:
        image_paths = context.get("image_paths")
        resolved: List[str] = []
        if isinstance(image_paths, list):
            for path in image_paths[:2]:
                if isinstance(path, str) and path.strip():
                    resolved.append(path)
        if len(resolved) >= 2:
            return resolved[:2]
        for key in ("image_path_1", "image_path_2"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                resolved.append(value)
        if len(resolved) >= 2:
            return resolved[:2]
        frame_ids = context.get("frame_ids")
        if isinstance(frame_ids, list):
            for frame_id in frame_ids[:2]:
                if frame_id is None:
                    continue
                resolved.append(str(Path(self.scannet_root) / scene_id / "color" / f"{str(frame_id).strip()}.jpg"))
        if len(resolved) >= 2:
            return resolved[:2]
        for key in ("frame_id_1", "frame_id_2"):
            frame_id = context.get(key)
            if frame_id is None:
                continue
            resolved.append(str(Path(self.scannet_root) / scene_id / "color" / f"{str(frame_id).strip()}.jpg"))
        return self._dedupe_paths(resolved)[:2]

    @staticmethod
    def _dedupe_paths(paths: Sequence[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                continue
            normalized = path.strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _extract_answer(raw_response: Any) -> str:
        text = str(raw_response or "").strip()
        if not text:
            return "invalid"
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).strip().split())
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "invalid"
        last = lines[-1]
        answer_prefix = re.match(r"answer\s*:\s*(.+)$", last, flags=re.IGNORECASE)
        if answer_prefix:
            return " ".join(answer_prefix.group(1).strip().split())
        return "invalid"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_answer(self, *, task_type: str, raw_answer: str) -> str:
        answer = " ".join(str(raw_answer or "").strip().split())
        if not answer:
            return "invalid"
        lowered = answer.lower().strip(" .,:;!?")
        if lowered in INVALID_ANSWER_SET:
            return "invalid"

        if task_type == "visibility_compare":
            compressed = lowered.replace(" ", "").replace("_", "")
            mapping = {
                "image1": "image1",
                "img1": "image1",
                "firstimage": "image1",
                "image2": "image2",
                "img2": "image2",
                "secondimage": "image2",
                "same": "same",
                "equal": "same",
                "neither": "neither",
                "none": "neither",
            }
            return mapping.get(compressed, answer)

        if task_type == "elevation_cam_cam":
            compressed = lowered.replace(" ", "").replace("_", "")
            mapping = {
                "higher": "higher",
                "lower": "lower",
                "same": "same_level",
                "samelevel": "same_level",
                "sameheight": "same_level",
            }
            return mapping.get(compressed, answer)

        if task_type in {
            "position_cam_cam",
            "position_cam_obj",
            "position_cam_reg",
            "single_image_relative_direction",
            "relative_direction_hard",
            "motion_camera",
        }:
            normalized = lowered.replace("_", "-").replace(" ", "-")
            mapping = {
                "frontleft": "front-left",
                "frontright": "front-right",
                "backleft": "back-left",
                "backright": "back-right",
                "leftfront": "left-front",
                "rightfront": "right-front",
                "leftback": "left-back",
                "rightback": "right-back",
                "leftup": "left-up",
                "rightup": "right-up",
                "leftdown": "left-down",
                "rightdown": "right-down",
                "frontup": "front-up",
                "frontdown": "front-down",
                "backup": "back-up",
                "backdown": "back-down",
                "same-level": "same_level",
                "samelevel": "same_level",
            }
            compact = normalized.replace("-", "")
            if compact in mapping:
                return mapping[compact]
            if normalized in {"left", "right", "front", "back", "up", "down", "forward", "backward", "clockwise", "counterclockwise"}:
                return normalized
            if normalized in {
                "front-left", "front-right", "back-left", "back-right",
                "left-front", "right-front", "left-back", "right-back",
                "left-up", "right-up", "left-down", "right-down",
                "front-up", "front-down", "back-up", "back-down",
            }:
                return normalized
        return answer
