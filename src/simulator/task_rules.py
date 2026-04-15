"""Task-specific validity rules for the deterministic geometric environment."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

class InvalidQuestionRuleEngine:
    """Centralize task-level invalid-question rules outside the simulator shell."""

    def __init__(
        self,
        *,
        label_mapper: Callable[[str], str],
        is_null_like_label: Callable[[Any], bool],
        same_label: Callable[[Any, Any], bool],
        question_indicates_target_in_candidates: Callable[[str, str], bool],
        get_expected_extraction_fields: Callable[[str], list[str]],
        get_task_profile: Callable[[str], Any],
    ):
        self.label_mapper = label_mapper
        self.is_null_like_label = is_null_like_label
        self.same_label = same_label
        self.question_indicates_target_in_candidates = question_indicates_target_in_candidates
        self.get_expected_extraction_fields = get_expected_extraction_fields
        self.get_task_profile = get_task_profile

    def get_invalid_question_reason(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Return the first matched invalid-question reason, or None."""
        profile = self.get_task_profile(task_type)
        base_reason = self.get_invalid_distance_reason(task_type, context)
        if base_reason:
            return base_reason

        if profile is not None:
            for constraint in profile.same_label_constraints:
                if self.same_label(context.get(constraint.left_field), context.get(constraint.right_field)):
                    return constraint.error_message

            for constraint in profile.distinct_field_group_constraints:
                concrete = [
                    context.get(field_name)
                    for field_name in constraint.field_names
                    if not self.is_null_like_label(context.get(field_name))
                ]
                for idx in range(len(concrete)):
                    for jdx in range(idx + 1, len(concrete)):
                        if self.same_label(concrete[idx], concrete[jdx]):
                            return constraint.error_message

            for constraint in profile.list_constraints:
                list_value = context.get(constraint.list_field)
                if list_value is None:
                    if context.get("_vlm_extraction_done") and constraint.require_non_empty:
                        return constraint.empty_error or f"{task_type} invalid: `{constraint.list_field}` cannot be empty"
                    continue
                if not isinstance(list_value, list):
                    if context.get("_vlm_extraction_done"):
                        return f"{task_type} invalid: failed to extract `{constraint.list_field}`"
                    continue

                normalized_items = []
                for candidate in list_value:
                    if self.is_null_like_label(candidate):
                        continue
                    normalized_items.append(self.label_mapper(str(candidate).strip().lower()))

                if constraint.dedupe_items and len(normalized_items) >= 2 and len(set(normalized_items)) != len(normalized_items):
                    return constraint.duplicate_error or f"{task_type} invalid: `{constraint.list_field}` contains duplicate items"

                if constraint.require_non_empty:
                    anchor_field = constraint.element_field
                    anchor_ok = True
                    if anchor_field is not None:
                        anchor_ok = not self.is_null_like_label(context.get(anchor_field))
                    if anchor_ok and not normalized_items:
                        return constraint.empty_error or f"{task_type} invalid: `{constraint.list_field}` cannot be empty"

                if constraint.element_field and normalized_items:
                    anchor_value = context.get(constraint.element_field)
                    if not self.is_null_like_label(anchor_value):
                        anchor_label = self.label_mapper(str(anchor_value).strip().lower())
                        if anchor_label in normalized_items:
                            return constraint.overlap_error or f"{task_type} invalid: `{constraint.element_field}` cannot appear in `{constraint.list_field}`"

            detection_requirement = profile.detection_requirement
            if detection_requirement is not None:
                detections = context.get(detection_requirement.detections_key)
                entity = context.get(detection_requirement.entity_key)
                if isinstance(detections, list) and (not isinstance(entity, dict)):
                    label = context.get(detection_requirement.label_field)
                    if not self.is_null_like_label(label):
                        return detection_requirement.error_template.format(
                            task_type=task_type,
                            label=label,
                        )

        needed_fields = self.get_expected_extraction_fields(task_type)
        if needed_fields and context.get("_vlm_extraction_done"):
            for field_name in needed_fields:
                if self.is_null_like_label(context.get(field_name)):
                    return f"{task_type} invalid: failed to extract `{field_name}`"

        return None

    def get_invalid_distance_reason(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Filter distance questions that cannot yield deterministic ground truth."""
        if "relative_distance" not in task_type:
            return None

        target_label = context.get("target_label")
        candidate_labels = context.get("candidate_labels", [])
        if (
            target_label
            and isinstance(candidate_labels, list)
            and target_label in candidate_labels
        ):
            return (
                "relative_distance invalid: `candidate_labels` contains `target_label`, "
                "so the target cannot appear in the candidate list"
            )

        if (
            target_label
            and isinstance(context.get("question", ""), str)
            and self.question_indicates_target_in_candidates(
                question=context.get("question", ""),
                target_label=str(target_label),
            )
        ):
            return (
                "relative_distance invalid: the candidate list in the question text contains `target_label`, "
                "so the target cannot appear in the candidate list"
            )

        return None
