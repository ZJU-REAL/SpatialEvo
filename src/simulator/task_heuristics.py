"""Task-specific heuristic extraction fallback for DGE."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

class HeuristicTaskParser:
    """Fallback parser used when model extraction is unavailable or unstable."""

    def __init__(
        self,
        *,
        get_task_profile: Callable[[str], Any],
        find_candidate_mentions_in_question: Callable[[str, List[str]], List[str]],
        resolve_object_name_with_candidates: Callable[[str, List[str]], str],
        extract_parenthesis_segment: Callable[[str], str],
        is_null_like_label: Callable[[Any], bool],
        is_abstract_region_phrase: Callable[[str], bool],
    ):
        self.get_task_profile = get_task_profile
        self.find_candidate_mentions_in_question = find_candidate_mentions_in_question
        self.resolve_object_name_with_candidates = resolve_object_name_with_candidates
        self.extract_parenthesis_segment = extract_parenthesis_segment
        self.is_null_like_label = is_null_like_label
        self.is_abstract_region_phrase = is_abstract_region_phrase
        self._task_extractors: Dict[str, Callable[[str, List[str]], Dict[str, Any]]] = {}

    def register_task_extractor(
        self,
        task_type: str,
        extractor: Callable[[str, List[str]], Dict[str, Any]],
    ) -> None:
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("`task_type` must be a non-empty string")
        self._task_extractors[task_type.strip()] = extractor

    def unregister_task_extractor(self, task_type: str) -> None:
        if not isinstance(task_type, str) or not task_type.strip():
            return
        self._task_extractors.pop(task_type.strip(), None)

    def parse(
        self,
        *,
        task_type: str,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        if not isinstance(question, str):
            return {}

        extractor = self._task_extractors.get(task_type)
        if extractor is not None:
            return extractor(question, candidate_labels)

        profile = self.get_task_profile(task_type)
        if profile is not None and callable(getattr(profile, "heuristic_extractor", None)):
            return profile.heuristic_extractor(question, candidate_labels)

        heuristic_policy = getattr(profile, "heuristic_policy", None)
        if heuristic_policy is None:
            return {}

        return self._parse_with_policy(
            heuristic_policy=heuristic_policy,
            question=question,
            candidate_labels=candidate_labels,
        )

    def _parse_with_policy(
        self,
        *,
        heuristic_policy: Any,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        strategy = str(getattr(heuristic_policy, "strategy", "")).strip().lower()
        if strategy == "single_entity":
            return self._extract_single_entity_task(
                target_field=getattr(heuristic_policy, "target_field", None),
                enable_relative_to_me_fallback=bool(
                    getattr(heuristic_policy, "enable_relative_to_me_fallback", False)
                ),
                allow_abstract_region_subject=bool(
                    getattr(heuristic_policy, "allow_abstract_region_subject", False)
                ),
                question=question,
                candidate_labels=candidate_labels,
            )
        if strategy == "pair_measurement":
            return self._extract_pair_measurement_task(question, candidate_labels)
        if strategy == "single_image_relative_direction":
            return self._extract_single_image_relative_direction(question, candidate_labels)
        if strategy == "relative_distance":
            return self._extract_relative_distance(question, candidate_labels)
        if strategy == "relative_direction_hard":
            return self._extract_relative_direction_hard(question, candidate_labels)
        return {}

    def _base_mentions(
        self,
        question: str,
        candidate_labels: List[str],
    ) -> tuple[str, str, List[str], str]:
        q = question.lower()
        mentions = self.find_candidate_mentions_in_question(q, candidate_labels)
        first = mentions[0] if len(mentions) >= 1 else "null"
        second = mentions[1] if len(mentions) >= 2 else "null"
        return first, second, mentions, q

    def _extract_subject_before_relative_to_me(
        self,
        *,
        question_lower: str,
        candidate_labels: List[str],
        allow_abstract_region_subject: bool,
    ) -> str:
        patterns = [
            r"where\s+is\s+(.+?)\s+relative\s+to\s+me",
            r"which\s+direction\s+is\s+(.+?)\s+relative\s+to\s+me",
            r"in\s+which\s+direction\s+is\s+(.+?)\s+relative\s+to\s+me",
            r"where\s+is\s+(.+?)\s+relative\s+to\s+the\s+camera(?:\s+in\s+image\s*[12])?",
            r"which\s+direction\s+is\s+(.+?)\s+relative\s+to\s+the\s+camera(?:\s+in\s+image\s*[12])?",
            r"in\s+which\s+direction\s+is\s+(.+?)\s+relative\s+to\s+the\s+camera(?:\s+in\s+image\s*[12])?",
            r"where\s+is\s+(.+?)\s+relative\s+to\s+camera(?:\s+in\s+image\s*[12])?",
            r"which\s+direction\s+is\s+(.+?)\s+relative\s+to\s+camera(?:\s+in\s+image\s*[12])?",
            r"in\s+which\s+direction\s+is\s+(.+?)\s+relative\s+to\s+camera(?:\s+in\s+image\s*[12])?",
        ]
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if not match:
                continue
            raw = re.sub(r"^(?:the|a|an)\s+", "", str(match.group(1)).strip().lower())
            if allow_abstract_region_subject and self.is_abstract_region_phrase(raw):
                return raw
            resolved = self.resolve_object_name_with_candidates(raw, candidate_labels)
            if not self.is_null_like_label(resolved):
                return resolved
        return "null"

    def _extract_single_entity_task(
        self,
        *,
        target_field: Any,
        enable_relative_to_me_fallback: bool,
        allow_abstract_region_subject: bool,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        target_key = str(target_field).strip()
        if not target_key:
            return {}
        first, _, _, q = self._base_mentions(question, candidate_labels)
        value = first
        if self.is_null_like_label(value) and enable_relative_to_me_fallback:
            value = self._extract_subject_before_relative_to_me(
                question_lower=q,
                candidate_labels=candidate_labels,
                allow_abstract_region_subject=allow_abstract_region_subject,
            )
        return {target_key: value}

    def _extract_pair_measurement_task(
        self,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        first, second, _, q = self._base_mentions(question, candidate_labels)
        between_match = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:[\?\.,]|$)", q)
        if between_match:
            left = self.resolve_object_name_with_candidates(between_match.group(1), candidate_labels)
            right = self.resolve_object_name_with_candidates(between_match.group(2), candidate_labels)
            return {"object1_label": left, "object2_label": right}
        return {"object1_label": first, "object2_label": second}

    def _extract_single_image_relative_direction(
        self,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        first, second, _, q = self._base_mentions(question, candidate_labels)
        rel_match = re.search(r"where\s+is\s+(.+?)\s+relative\s+to\s+(.+?)[\?\.,]?$", q)
        if rel_match:
            left = self.resolve_object_name_with_candidates(rel_match.group(1), candidate_labels)
            right = self.resolve_object_name_with_candidates(rel_match.group(2), candidate_labels)
            return {"reference_label": right, "target_label": left}
        return {"reference_label": first, "target_label": second}

    def _extract_relative_distance(
        self,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        _, _, mentions, q = self._base_mentions(question, candidate_labels)
        target = "null"
        match_target = re.search(r"(?:closest|nearest)\s+to\s+(.+?)(?:[\?\.,]|$)", q)
        if match_target:
            target = self.resolve_object_name_with_candidates(match_target.group(1), candidate_labels)

        segment = self.extract_parenthesis_segment(q)
        if not segment:
            list_patterns = [
                r"which\s+of\s+these\s+objects\s*,\s*(.+?)\s*,?\s*is\s+(?:closest|nearest)\s+to\s+",
                r"which\s+of\s+the(?:se)?\s+objects\s*,\s*(.+?)\s*,?\s*is\s+(?:closest|nearest)\s+to\s+",
                r"which\s+of\s+(.+?)\s*,?\s*is\s+(?:closest|nearest)\s+to\s+",
                r"out\s+of\s+these\s+objects\s*,?\s*(.+?)\s*,?\s*(?:which\s+one\s+)?is\s+(?:closest|nearest)\s+to\s+",
            ]
            for pattern in list_patterns:
                match = re.search(pattern, q)
                if match:
                    segment = str(match.group(1)).strip()
                    break
        candidates = []
        raw_items = re.split(r",|\band\b|\bor\b|;", segment) if segment else []
        for item in raw_items:
            mapped = self.resolve_object_name_with_candidates(item, candidate_labels)
            if mapped != "null" and mapped not in candidates:
                candidates.append(mapped)

        if not candidates and mentions:
            candidates = mentions[:3]
        return {"target_label": target, "candidate_labels": candidates}

    def _extract_relative_direction_hard(
        self,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        first, second, mentions, q = self._base_mentions(question, candidate_labels)
        match = re.search(
            r"(?:stand|standing)\s+(?:by|at|near)\s+(.+?)\s+(?:and|,)\s*(?:face|facing)\s+(.+?)\s*,?\s*where\s+is\s+(.+?)(?:[\?\.,]|$)",
            q,
        )
        if match:
            positioning = self.resolve_object_name_with_candidates(match.group(1), candidate_labels)
            orienting = self.resolve_object_name_with_candidates(match.group(2), candidate_labels)
            querying = self.resolve_object_name_with_candidates(match.group(3), candidate_labels)
            return {
                "positioning_label": positioning,
                "orienting_label": orienting,
                "querying_label": querying,
            }

        return {
            "positioning_label": first,
            "orienting_label": second,
            "querying_label": mentions[2] if len(mentions) >= 3 else "null",
        }
