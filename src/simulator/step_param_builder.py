"""Build tool-call parameters for rubric execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import numpy as np


class StepParamBuilder:
    """Keep task-specific tool parameter binding outside `WorldSimulator`."""

    def __init__(
        self,
        *,
        get_task_profile: Callable[[str], Any],
        collect_labels_from_context: Callable[[Dict[str, Any]], List[str]],
        build_vlm_extraction_params: Callable[[Any, Dict[str, Any], Any], Dict[str, Any]],
        build_region_positions_from_context: Callable[[Dict[str, Any]], Dict[str, List[float]]],
        infer_anchor_labels_from_question: Callable[[str, List[str], int], List[str]],
        resolve_region_anchor_name_with_llm: Callable[[Dict[str, Any], str, List[str]], str],
        to_camera_frame_entity: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any] | None],
    ):
        self.get_task_profile = get_task_profile
        self.collect_labels_from_context = collect_labels_from_context
        self.build_vlm_extraction_params = build_vlm_extraction_params
        self.build_region_positions_from_context = build_region_positions_from_context
        self.infer_anchor_labels_from_question = infer_anchor_labels_from_question
        self.resolve_region_anchor_name_with_llm = resolve_region_anchor_name_with_llm
        self.to_camera_frame_entity = to_camera_frame_entity

    def build(
        self,
        *,
        step,
        context: Dict[str, Any],
        task,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        if step.tool_name == "vlm_tool" and task.requires_llm_extraction():
            params = self.build_vlm_extraction_params(step, context, task)
        else:
            for param_name in step.required_params:
                if param_name in context:
                    params[param_name] = context[param_name]
                elif param_name == "labels":
                    params["labels"] = self.collect_labels_from_context(context)

            if step.optional_params:
                for param_name in step.optional_params:
                    if param_name in context:
                        params[param_name] = context[param_name]

        task_type = context.get("task_type", "")
        profile = self.get_task_profile(task_type)

        if step.tool_name == "object_detection_tool":
            detection_policy = profile.object_detection_policy if profile is not None else None
            if "frame_metadata" not in params:
                frame_source = detection_policy.frame_source if detection_policy is not None else "auto"
                if frame_source == "reference_image":
                    ref_idx = int(context.get("camera_reference_image_idx", 2))
                    primary_key = "frame_metadata_1" if ref_idx == 1 else "frame_metadata_2"
                    secondary_key = "frame_metadata_2" if ref_idx == 1 else "frame_metadata_1"
                    if primary_key in context:
                        params["frame_metadata"] = context[primary_key]
                    elif secondary_key in context:
                        params["frame_metadata"] = context[secondary_key]
                elif frame_source == "second_image":
                    if "frame_metadata_2" in context:
                        params["frame_metadata"] = context["frame_metadata_2"]
                elif frame_source == "both_images":
                    frame_metadata_list = []
                    if "frame_metadata_1" in context:
                        frame_metadata_list.append(context["frame_metadata_1"])
                    if "frame_metadata_2" in context:
                        frame_metadata_list.append(context["frame_metadata_2"])
                    if frame_metadata_list:
                        params["frame_metadata_list"] = frame_metadata_list
                        params["frame_metadata"] = frame_metadata_list[-1]
                if "frame_metadata" not in params:
                    if "frame_metadata_2" in context:
                        params["frame_metadata"] = context["frame_metadata_2"]
                    elif "frame_metadata_1" in context:
                        params["frame_metadata"] = context["frame_metadata_1"]

            if detection_policy is not None and not detection_policy.include_scene_metadata:
                params.pop("scene_metadata", None)

            params["use_camera_location"] = bool(
                detection_policy.use_camera_location
            ) if detection_policy is not None else False

            if "target_objects" not in params:
                targets = []
                label_fields = (
                    detection_policy.target_label_fields
                    if detection_policy is not None and detection_policy.target_label_fields
                    else ("target_label", "reference_label", "object1_label", "object2_label")
                )
                for field_name in label_fields:
                    if field_name in context:
                        targets.append(context[field_name])
                if targets:
                    params["target_objects"] = targets

        if step.tool_name == "camera_pair_tool" and "answer_mode" not in params:
            answer_mode = profile.camera_pair_answer_mode if profile is not None else None
            params["answer_mode"] = answer_mode or "position"

        if step.tool_name == "region_anchor_tool":
            if "region_positions" not in params:
                params["region_positions"] = self.build_region_positions_from_context(context)

            available_region_names = []
            if isinstance(params.get("region_positions"), dict):
                available_region_names = list(params["region_positions"].keys())

            inferred = (
                self.infer_anchor_labels_from_question(
                    context.get("question", ""),
                    available_region_names,
                    int(getattr(profile, "region_anchor_infer_max_labels", 1)),
                )
                if available_region_names else []
            )

            if "region_name" not in params and "region_name" in context:
                params["region_name"] = context["region_name"]
            if str(params.get("region_name", "")).strip().lower() in {"", "null", "none"} and inferred:
                params["region_name"] = inferred[0]
            if "region_name" in params and available_region_names:
                params["region_name"] = self.resolve_region_anchor_name_with_llm(
                    context,
                    params["region_name"],
                    available_region_names,
                )

        if step.tool_name == "spatial_relation_tool":
            spatial_policy = profile.spatial_relation_policy if profile is not None else None
            if spatial_policy is not None and spatial_policy.binding == "route_entities":
                entities = context.get("entities")
                target_entity = context.get("target_entity")
                if "entity1" not in params and isinstance(target_entity, dict) and target_entity:
                    params["entity1"] = target_entity
                if "entity2" not in params and isinstance(entities, list) and len(entities) >= 1 and isinstance(entities[0], dict):
                    params["entity2"] = entities[0]
                if "entity1" not in params and isinstance(entities, list) and len(entities) >= 1 and isinstance(entities[0], dict):
                    params["entity1"] = entities[0]
                if "entity2" not in params and isinstance(entities, list) and len(entities) >= 2 and isinstance(entities[1], dict):
                    params["entity2"] = entities[1]

            if spatial_policy is not None and spatial_policy.binding == "camera_to_region":
                params["entity1"] = {"name": "camera_origin", "type": "camera", "position": np.zeros(3)}
                if isinstance(context.get("region_entity"), dict):
                    entity2 = context["region_entity"]
                    ref_idx = int(context.get("camera_reference_image_idx", 2))
                    cam_key = "camera_entity_1" if ref_idx == 1 else "camera_entity_2"
                    camera_ref = context.get(cam_key)
                    if isinstance(camera_ref, dict):
                        converted = self.to_camera_frame_entity(entity2, camera_ref)
                        if isinstance(converted, dict):
                            entity2 = converted
                    params["entity2"] = entity2
            elif spatial_policy is not None and spatial_policy.binding == "camera_to_target":
                params["entity1"] = {"name": "camera_origin", "type": "camera", "position": np.zeros(3)}
                if isinstance(context.get("target_entity"), dict):
                    entity2 = context["target_entity"]
                    ref_idx = int(context.get("camera_reference_image_idx", 2))
                    cam_key = "camera_entity_1" if ref_idx == 1 else "camera_entity_2"
                    camera_ref = context.get(cam_key)
                    if isinstance(camera_ref, dict):
                        converted = self.to_camera_frame_entity(entity2, camera_ref)
                        if isinstance(converted, dict):
                            entity2 = converted
                    params["entity2"] = entity2
            elif spatial_policy is not None and spatial_policy.binding == "camera_pair":
                reference_idx = int(context.get("camera_reference_image_idx", 1))
                target_idx = int(context.get("camera_target_image_idx", 2))
                ref_key = "camera_entity_1" if reference_idx == 1 else "camera_entity_2"
                tgt_key = "camera_entity_1" if target_idx == 1 else "camera_entity_2"
                if isinstance(context.get(ref_key), dict):
                    params["entity1"] = context[ref_key]
                if isinstance(context.get(tgt_key), dict):
                    params["entity2"] = context[tgt_key]

            if "entity1" not in params:
                if spatial_policy is not None and spatial_policy.binding == "camera_pair" and "camera_entity_1" in context:
                    params["entity1"] = context["camera_entity_1"]
                elif spatial_policy is not None and spatial_policy.binding in {"camera_to_target", "camera_to_region"}:
                    ref_idx = int(context.get("camera_reference_image_idx", 2))
                    cam_key = "camera_entity_1" if ref_idx == 1 else "camera_entity_2"
                    if cam_key in context:
                        params["entity1"] = context[cam_key]
                elif "reference_entity" in context:
                    params["entity1"] = context["reference_entity"]
                elif "entity1" in context:
                    params["entity1"] = context["entity1"]

            if "entity2" not in params:
                if spatial_policy is not None and spatial_policy.binding == "camera_pair" and "camera_entity_2" in context:
                    params["entity2"] = context["camera_entity_2"]
                elif "target_entity" in context:
                    params["entity2"] = context["target_entity"]
                elif spatial_policy is not None and spatial_policy.binding == "camera_to_region" and "region_entity" in context:
                    params["entity2"] = context["region_entity"]
                elif "entity2" in context:
                    params["entity2"] = context["entity2"]

            if "reference_frame" not in params and spatial_policy is not None and spatial_policy.default_reference_frame:
                params["reference_frame"] = spatial_policy.default_reference_frame

        if step.tool_name == "measurement_tool":
            if "measurement_type" not in params:
                default_measurement = profile.measurement_default if profile is not None else None
                params["measurement_type"] = context.get("measurement_type", default_measurement or "compare_longer")
            if "entity1" not in params and "entity1" in context:
                params["entity1"] = context["entity1"]
            if "entity2" not in params and "entity2" in context:
                params["entity2"] = context["entity2"]

        if step.tool_name == "bird_eye_view_tool":
            if "scene_id" not in params and "scene_id" in context:
                params["scene_id"] = context["scene_id"]
            if "metadata_dir" not in params and "metadata_dir" in context:
                params["metadata_dir"] = context["metadata_dir"]

        return params
