"""World simulator."""

from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple
import time
import re
import json
import numpy as np
from pathlib import Path
from ..tasks.image_pair_tasks import IMAGE_PAIR_TASK_REGISTRY
from ..tasks.scene_tasks import SCENE_TASK_REGISTRY
from ..tasks.single_image_tasks import SINGLE_IMAGE_TASK_REGISTRY
from ..rubrics.image_pair_rubrics import IMAGE_PAIR_RUBRIC_REGISTRY
from ..rubrics.scene_rubrics import SCENE_RUBRIC_REGISTRY
from ..rubrics.single_image_rubrics import SINGLE_IMAGE_RUBRIC_REGISTRY
from ..tools.camera_tools import CameraElevationTool, CameraParameterTool, CameraPairTool
from ..tools.view_tools import BirdEyeViewTool, PointCloudTool
from ..tools.spatial_tools import SpatialRelationTool, ObjectDetectionTool, MeasurementTool, RegionAnchorTool
from ..tools.deterministic_geometry_tools import (
    VSIAmbiguityChecker, VSIObjectSizeTool, VSIAbsoluteDistanceTool,
    VSICameraObjectDistanceTool, VSIDepthOrderTool, VSIRelativeDistanceTool,
    VSIRelativeDirectionTool, VSIObjectCountTool, VSIRoomSizeTool, VSIVisibilityCompareTool,
    VSISingleImageRelativeDirectionTool
)
from .validator import AnswerValidator
from .step_param_builder import StepParamBuilder
from .task_heuristics import HeuristicTaskParser
from .task_rules import InvalidQuestionRuleEngine
from .task_support import (
    NON_ANCHOR_LABELS,
    REGION_OBJECT_ONTOLOGY,
    REGION_PHRASE_ALIASES,
    SUMMARY_TYPE_ALIASES,
    TASK_REJECTION_REASONS,
    VALID_LABELS,
    SummaryTypeResolver,
    TaskGroupResolver,
    TaskProfile,
    TaskResolver,
    TaskSupportRegistry,
    is_legacy_benchmark_name,
    normalize_task_group_name,
)
from ..data.scannet_loader import ScanNetLoader
from ..utils.io_utils import load_config
import difflib

class WorldSimulator:
    """World simulator."""

    RESPONSE_VERSION = "1.2"
    ERROR_CODES = {
        "OK": "",
        "INVALID_INPUT": "INVALID_INPUT",
        "INVALID_SUMMARY_TYPE": "INVALID_SUMMARY_TYPE",
        "UNSUPPORTED_TASK": "UNSUPPORTED_TASK",
        "TASK_DISABLED": "TASK_DISABLED",
        "QUESTION_INVALID": "QUESTION_INVALID",
        "MISSING_DEPENDENCY": "MISSING_DEPENDENCY",
        "MISSING_TOOL": "MISSING_TOOL",
        "EXECUTION_FAILED": "EXECUTION_FAILED",
        "INVALID_QUESTION_RULE": "INVALID_QUESTION_RULE",
        "EMPTY_ANSWER": "EMPTY_ANSWER",
    }

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
    
    def __init__(
        self,
        vlm_model: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        scannet_root: str = "/mnt/jfs/lidingm/data/dataset/ScanNet/train",
        enable_vlm: bool = False,
        enable_invalid_recovery: bool = False,
        verbose: bool = True,
    ):
        """Init."""
        self.verbose = bool(verbose)

        self.task_registry = {
            **IMAGE_PAIR_TASK_REGISTRY,
            **SCENE_TASK_REGISTRY,
            **SINGLE_IMAGE_TASK_REGISTRY,
        }
        self.rubric_registry = {
            **IMAGE_PAIR_RUBRIC_REGISTRY,
            **SCENE_RUBRIC_REGISTRY,
            **SINGLE_IMAGE_RUBRIC_REGISTRY,
        }

        self.config = self._load_runtime_config()
        

        visibility_floor = self._get_visibility_floor()
        self.scannet_loader = ScanNetLoader(
            scannet_root=scannet_root,
            visibility_floor=visibility_floor,
        )
        

        self.tools = {}
        for tool in (
            VSIAmbiguityChecker(self.scannet_loader),
            VSIObjectSizeTool(self.scannet_loader),
            VSICameraObjectDistanceTool(self.scannet_loader),
            VSIAbsoluteDistanceTool(self.scannet_loader),
            VSIDepthOrderTool(self.scannet_loader),
            VSIRelativeDistanceTool(self.scannet_loader),
            VSIRelativeDirectionTool(self.scannet_loader),
            VSIVisibilityCompareTool(self.scannet_loader),
            VSISingleImageRelativeDirectionTool(self.scannet_loader),
            VSIObjectCountTool(self.scannet_loader),
            VSIRoomSizeTool(self.scannet_loader),
        ):
            self.register_tool(tool.name, tool)
        self.BASE_VALID_LABELS = set(VALID_LABELS)
        self.VALID_LABELS = set(self.BASE_VALID_LABELS)
        self.EXACT_ONLY_LABELS = set()
        self._load_external_label_inventories()
        self.NON_ANCHOR_LABELS = set(NON_ANCHOR_LABELS)
        self.task_support_registry = TaskSupportRegistry(
            region_object_ontology=REGION_OBJECT_ONTOLOGY,
            region_phrase_aliases=REGION_PHRASE_ALIASES,
        )
        self.all_task_names = set(self.task_registry.keys())
        alias_mapping = {}
        for task_type, profile in self.task_support_registry.get_all_task_profiles().items():
            for alias in profile.aliases:
                alias_mapping[self._normalize(alias)] = task_type
        self.task_resolver = TaskResolver(
            task_aliases=alias_mapping,
            task_rejection_reasons=TASK_REJECTION_REASONS,
        )
        self.summary_type_resolver = SummaryTypeResolver(
            aliases=SUMMARY_TYPE_ALIASES,
        )
        self.task_group_resolver = TaskGroupResolver()
        self.enable_vlm = enable_vlm
        self.enable_invalid_recovery = bool(enable_invalid_recovery)

        vlm_cfg = self.config.get("vlm", {})
        effective_vlm_model = str(vlm_model or vlm_cfg.get("model", "gpt-oss-120b-ldm")).strip()
        effective_vlm_vision_model = str(vlm_cfg.get("vision_model", "qwen3vl-8b")).strip()
        effective_vlm_api_key = str(vlm_api_key or vlm_cfg.get("api_key", "EMPTY")).strip()
        effective_vlm_base_url = str(vlm_base_url or vlm_cfg.get("base_url", "http://stepcast-router.shai-core:9200/v1")).strip()
        effective_vlm_timeout = int(vlm_cfg.get("timeout", 30))
        effective_vlm_max_retries = int(vlm_cfg.get("max_retries", 20))

        if enable_vlm:
            try:
                from ..tools.vlm_tools import VLMTool
                self.register_tool("vlm_tool", VLMTool(
                    model=effective_vlm_model,
                    vision_model=effective_vlm_vision_model,
                    api_key=effective_vlm_api_key,
                    base_url=effective_vlm_base_url,
                    timeout=effective_vlm_timeout,
                    max_retries=effective_vlm_max_retries,
                ))
            except ImportError:
                self._log("Warning: VLM tool not available (openai package not installed)")
        

        try:
            for tool_name, tool in {
                "camera_parameter_tool": CameraParameterTool(),
                "camera_pair_tool": CameraPairTool(),
                "camera_elevation_tool": CameraElevationTool(),
                "bird_eye_view_tool": BirdEyeViewTool(),
                "point_cloud_tool": PointCloudTool(),
                "spatial_relation_tool": SpatialRelationTool(),
                "object_detection_tool": ObjectDetectionTool(),
                "measurement_tool": MeasurementTool(),
                "region_anchor_tool": RegionAnchorTool(),
            }.items():
                self.register_tool(tool_name, tool)
        except Exception as e:
            self._log(f"Warning: Some image-pair tools not available: {e}")

        if self.enable_invalid_recovery:
            try:
                from ..tools.recovery_tools import (
                    BBoxEvidenceTool,
                    InvalidRecoveryTool,
                    SceneBEVEvidenceTool,
                )
                self.register_tool("scene_bev_evidence_tool", SceneBEVEvidenceTool())
                self.register_tool("bbox_evidence_tool", BBoxEvidenceTool())
                invalid_recovery_cfg = self.config.get("invalid_recovery", {})
                invalid_model = str(invalid_recovery_cfg.get("model", effective_vlm_model)).strip()
                invalid_vision_model = str(invalid_recovery_cfg.get("vision_model", effective_vlm_vision_model)).strip()
                invalid_api_key = str(invalid_recovery_cfg.get("api_key", effective_vlm_api_key)).strip()
                invalid_base_url = str(invalid_recovery_cfg.get("base_url", effective_vlm_base_url)).strip()
                invalid_timeout = int(invalid_recovery_cfg.get("timeout", effective_vlm_timeout))
                invalid_max_retries = int(invalid_recovery_cfg.get("max_retries", effective_vlm_max_retries))
                reuse_main_vlm_backend = (
                    invalid_model == effective_vlm_model
                    and invalid_vision_model == effective_vlm_vision_model
                    and invalid_api_key == effective_vlm_api_key
                    and invalid_base_url == effective_vlm_base_url
                    and invalid_timeout == effective_vlm_timeout
                    and invalid_max_retries == effective_vlm_max_retries
                )
                self.register_tool(
                    "invalid_recovery_tool",
                    InvalidRecoveryTool(
                        vlm_backend=self.tools.get("vlm_tool") if reuse_main_vlm_backend else None,
                        model=invalid_model,
                        vision_model=invalid_vision_model,
                        api_key=invalid_api_key,
                        base_url=invalid_base_url,
                        timeout=invalid_timeout,
                        max_retries=invalid_max_retries,
                    ),
                )
            except ImportError as e:
                self._log(f"Warning: invalid recovery backend not available yet: {e}")
            except Exception as e:
                self._log(f"Warning: failed to initialize invalid recovery tools: {e}")
        

        self.validator = AnswerValidator()
        self.rule_engine = InvalidQuestionRuleEngine(
            label_mapper=self.map_to_standard_label,
            is_null_like_label=self._is_null_like_label,
            same_label=self._same_label,
            question_indicates_target_in_candidates=self._question_indicates_target_in_candidates,
            get_expected_extraction_fields=self.task_support_registry.get_expected_extraction_fields,
            get_task_profile=self.task_support_registry.get_task_profile,
        )
        self.step_param_builder = StepParamBuilder(
            get_task_profile=self.task_support_registry.get_task_profile,
            collect_labels_from_context=self._collect_labels_from_context,
            build_vlm_extraction_params=self._prepare_vlm_extraction_params,
            build_region_positions_from_context=self._build_region_positions_from_context,
            infer_anchor_labels_from_question=self._infer_anchor_labels_from_question,
            resolve_region_anchor_name_with_llm=self._resolve_region_anchor_name_with_llm,
            to_camera_frame_entity=self._to_camera_frame_entity,
        )
        self.heuristic_task_parser = HeuristicTaskParser(
            get_task_profile=self.task_support_registry.get_task_profile,
            find_candidate_mentions_in_question=self._find_candidate_mentions_in_question,
            resolve_object_name_with_candidates=self._resolve_object_name_with_candidates,
            extract_parenthesis_segment=self._extract_parenthesis_segment,
            is_null_like_label=self._is_null_like_label,
            is_abstract_region_phrase=self._is_abstract_region_phrase,
        )
        self.invalid_recovery_controller = None
        if self.enable_invalid_recovery:
            try:
                from .invalid_recovery import InvalidRecoveryController
                self.invalid_recovery_controller = InvalidRecoveryController(
                    tools=self.tools,
                    scannet_root=self.scannet_loader.scannet_root,
                    get_task_group=self.task_support_registry.get_task_group,
                    get_scene_summary=self.get_scene_summary,
                    get_single_image_summary=self.get_single_image_summary,
                    get_multi_image_summary=self.get_multi_image_summary,
                    extract_frame_id_from_image_path=self._extract_frame_id_from_image_path,
                    get_visibility_floor=self._get_visibility_floor,
                    config=self.config.get("invalid_recovery", {}),
                )
            except Exception as e:
                self._log(f"Warning: invalid recovery controller not available: {e}")
        

        self.execution_history = []

    def _load_runtime_config(self) -> Dict[str, Any]:
        """Load runtime config."""
        root = Path(__file__).resolve().parents[2]
        task_cfg_path = root / "config" / "task_config.yaml"
        rubric_cfg_path = root / "config" / "rubric_config.yaml"

        cfg: Dict[str, Any] = {
            "scoring": {
                "base_confidence": 1.0,
                "llm_extraction_penalty": 0.05,
                "min_confidence": 0.2,
            },
            "constraints": {
                "visibility_floor": 0.1,
            },
            "tasks": {},
            "vlm": {
                "model": "gpt-oss-120b-ldm",
                "vision_model": "qwen3vl-8b",
                "api_key": "EMPTY",
                "base_url": "http://stepcast-router.shai-core:9200/v1",
                "max_retries": 20,
                "max_tokens": 20480,
                "temperature": 0.0,
                "extraction_temperature": 0.0,
                "timeout": 30,
                "extraction_use_vision": False,
            },
            "invalid_recovery": {
                "model": None,
                "vision_model": None,
                "api_key": None,
                "base_url": None,
                "timeout": 30,
                "max_retries": 20,
                "max_scene_images": 0,
                "max_scene_object_lines": 60,
                "max_frame_object_lines": 25,
                "max_tokens": 512,
                "temperature": 0.0,
                "attach_scene_bev": True,
                "attach_bbox_overlay": True,
            },
            "label_inventory": {
                "enabled": True,
                "fuzzy_sources": [],
                "exact_only_sources": [],
            },
        }

        if task_cfg_path.exists():
            loaded = load_config(str(task_cfg_path))
            if isinstance(loaded, dict):
                if isinstance(loaded.get("scoring"), dict):
                    cfg["scoring"].update(loaded["scoring"])
                if isinstance(loaded.get("constraints"), dict):
                    cfg["constraints"].update(loaded["constraints"])
                if isinstance(loaded.get("tasks"), dict):
                    cfg["tasks"] = loaded["tasks"]
                if isinstance(loaded.get("invalid_recovery"), dict):
                    cfg["invalid_recovery"].update(loaded["invalid_recovery"])
                if isinstance(loaded.get("label_inventory"), dict):
                    cfg["label_inventory"].update(loaded["label_inventory"])

        if rubric_cfg_path.exists():
            loaded = load_config(str(rubric_cfg_path))
            if isinstance(loaded, dict):
                vlm_cfg = loaded.get("tools", {}).get("vlm_tool", {})
                if isinstance(vlm_cfg, dict):
                    cfg["vlm"].update(
                        {
                            "model": vlm_cfg.get("model", cfg["vlm"]["model"]),
                            "vision_model": vlm_cfg.get("vision_model", cfg["vlm"]["vision_model"]),
                            "api_key": vlm_cfg.get("api_key", cfg["vlm"]["api_key"]),
                            "base_url": vlm_cfg.get("base_url", cfg["vlm"]["base_url"]),
                            "max_retries": vlm_cfg.get("max_retries", cfg["vlm"]["max_retries"]),
                            "max_tokens": vlm_cfg.get("max_tokens", cfg["vlm"]["max_tokens"]),
                            "temperature": vlm_cfg.get("temperature", cfg["vlm"]["temperature"]),
                            "extraction_temperature": vlm_cfg.get(
                                "extraction_temperature",
                                cfg["vlm"].get("extraction_temperature", 0.0),
                            ),
                            "timeout": vlm_cfg.get("timeout", cfg["vlm"]["timeout"]),
                            "extraction_use_vision": vlm_cfg.get(
                                "extraction_use_vision",
                                cfg["vlm"]["extraction_use_vision"],
                            ),
                        }
                    )
                invalid_recovery_cfg = loaded.get("tools", {}).get("invalid_recovery_tool", {})
                if isinstance(invalid_recovery_cfg, dict):
                    cfg["invalid_recovery"].update(
                        {
                            "model": invalid_recovery_cfg.get("model", cfg["invalid_recovery"]["model"]),
                            "vision_model": invalid_recovery_cfg.get("vision_model", cfg["invalid_recovery"]["vision_model"]),
                            "api_key": invalid_recovery_cfg.get("api_key", cfg["invalid_recovery"]["api_key"]),
                            "base_url": invalid_recovery_cfg.get("base_url", cfg["invalid_recovery"]["base_url"]),
                            "timeout": invalid_recovery_cfg.get("timeout", cfg["invalid_recovery"]["timeout"]),
                            "max_retries": invalid_recovery_cfg.get("max_retries", cfg["invalid_recovery"]["max_retries"]),
                            "max_scene_images": invalid_recovery_cfg.get(
                                "max_scene_images",
                                cfg["invalid_recovery"]["max_scene_images"],
                            ),
                            "attach_scene_bev": invalid_recovery_cfg.get(
                                "attach_scene_bev",
                                cfg["invalid_recovery"]["attach_scene_bev"],
                            ),
                            "attach_bbox_overlay": invalid_recovery_cfg.get(
                                "attach_bbox_overlay",
                                cfg["invalid_recovery"]["attach_bbox_overlay"],
                            ),
                            "max_tokens": invalid_recovery_cfg.get("max_tokens", cfg["invalid_recovery"]["max_tokens"]),
                            "temperature": invalid_recovery_cfg.get("temperature", cfg["invalid_recovery"]["temperature"]),
                        }
                    )

        return cfg

    @staticmethod
    def _flatten_label_inventory_payload(payload: Any) -> List[str]:
        flattened: List[str] = []
        if isinstance(payload, str):
            text = payload.strip()
            if text:
                flattened.append(text)
            return flattened
        if isinstance(payload, dict):
            for value in payload.values():
                flattened.extend(WorldSimulator._flatten_label_inventory_payload(value))
            return flattened
        if isinstance(payload, (list, tuple, set)):
            for item in payload:
                flattened.extend(WorldSimulator._flatten_label_inventory_payload(item))
            return flattened
        return flattened

    @staticmethod
    def _label_inventory_variants(raw_label: Any) -> List[str]:
        text = str(raw_label or "").strip().lower()
        if not text:
            return []
        normalized = re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip()
        variants: List[str] = []
        for value in (text, normalized):
            if value and value not in variants:
                variants.append(value)
        return variants

    def _load_label_inventory_file(self, path_str: str) -> set[str]:
        path = Path(str(path_str).strip())
        if not path.exists() or not path.is_file():
            return set()
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return set()

        labels = set()
        for item in self._flatten_label_inventory_payload(payload):
            for variant in self._label_inventory_variants(item):
                labels.add(variant)
        return labels

    def _load_external_label_inventories(self) -> None:
        label_cfg = self.config.get("label_inventory", {})
        if not isinstance(label_cfg, dict) or not bool(label_cfg.get("enabled", True)):
            return

        fuzzy_sources = label_cfg.get("fuzzy_sources", [])
        exact_only_sources = label_cfg.get("exact_only_sources", [])
        if not isinstance(fuzzy_sources, list):
            fuzzy_sources = []
        if not isinstance(exact_only_sources, list):
            exact_only_sources = []

        fuzzy_loaded = set()
        exact_loaded = set()
        for source in fuzzy_sources:
            fuzzy_loaded.update(self._load_label_inventory_file(str(source)))
        for source in exact_only_sources:
            exact_loaded.update(self._load_label_inventory_file(str(source)))

        self.BASE_VALID_LABELS.update(fuzzy_loaded)
        self.VALID_LABELS.update(fuzzy_loaded)
        self.EXACT_ONLY_LABELS.update(exact_loaded - self.VALID_LABELS)

    def _get_visibility_floor(self) -> float:
        """Get visibility floor."""
        constraints_cfg = self.config.get("constraints", {})
        if isinstance(constraints_cfg, dict):
            try:
                return float(constraints_cfg.get("visibility_floor", 0.1))
            except (TypeError, ValueError):
                return 0.1
        return 0.1

    def _normalize(self, s):

        if not isinstance(s, str): return ""

        return re.sub(r'[\s_\[\]【】]+', '', s).lower()

    def _resolve_task_type(self, input_type):
        """Resolve task type."""
        return self.task_resolver.resolve(
            input_type=input_type,
            supported_task_names=self.all_task_names,
            normalize=self._normalize,
        )
    
    def _find_best_task_type(self, input_type):
        """Find best task type."""
        return self._resolve_task_type(input_type).task_type

    def _is_usable_anchor_label(self, label: str) -> bool:
        text = str(label or "").strip().lower()
        if not text:
            return False
        return text not in self.NON_ANCHOR_LABELS

    def _validate_input_contract(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate input contract."""
        if not isinstance(input_data, dict):
            return False, "`input_data` must be a dict"
        task_type = input_data.get("task_type")
        if not isinstance(task_type, str) or not task_type.strip():
            return False, "Missing `task_type`"
        question = input_data.get("question")
        if not isinstance(question, str) or not question.strip():
            return False, "`question` must be a string"
        scene_id = input_data.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            return False, "Missing `scene_id`"
        image_path = input_data.get("image_path")
        if image_path is not None and (not isinstance(image_path, str) or not image_path.strip()):
            return False, "`image_path` must be a non-empty string when provided"
        return True, ""

    def _validate_task_specific_input(
        self,
        task_type: str,
        input_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate task specific input."""
        profile = self.task_support_registry.get_task_profile(task_type)
        input_mode = profile.input_mode if profile is not None else "basic"

        if input_mode == "single_image":
            image_path = input_data.get("image_path")
            if not isinstance(image_path, str) or not image_path.strip():
                return False, f"{task_type} requires `image_path`"

        if input_mode == "image_pair":
            image_paths = input_data.get("image_paths")
            image_path_1 = input_data.get("image_path_1")
            image_path_2 = input_data.get("image_path_2")
            has_list = isinstance(image_paths, list) and len(image_paths) >= 2
            has_pair = (
                isinstance(image_path_1, str) and image_path_1.strip()
                and isinstance(image_path_2, str) and image_path_2.strip()
            )
            if not has_list and not has_pair:
                return False, f"{task_type} requires two image inputs (`image_paths` or `image_path_1`/`image_path_2`)"

        return True, ""

    def _build_response(
        self,
        *,
        is_valid: bool,
        task_type: str,
        question: str,
        answer: str = "",
        error: str = "",
        error_code: str = "",
        parsed_params: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        generation_difficulty_score: float = 1.0,
        question_difficulty_score: float = 1.0,
        validation_result: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build response."""
        failure_stage = self._infer_failure_stage(is_valid=is_valid, error_code=error_code)
        result = {
            "response_version": self.RESPONSE_VERSION,
            "is_valid": is_valid,
            "task_type": task_type,
            "question": question,
            "answer": answer,
            "error": error,
            "error_code": error_code,
            "parsed_params": parsed_params or {},
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "generation_difficulty_score": round(max(0.0, min(1.0, generation_difficulty_score)), 4),
            "question_difficulty_score": round(max(0.0, min(1.0, question_difficulty_score)), 4),
            "failure_stage": failure_stage,
        }
        if validation_result is not None:
            result["validation_result"] = validation_result
            if isinstance(validation_result, dict):
                result["validation_issues"] = list(validation_result.get("issues") or [])
                result["validation_suggestions"] = list(validation_result.get("suggestions") or [])
        if not is_valid:
            expected_extraction_fields = []
            unresolved_extraction_fields = []
            resolved_extraction_fields = []
            try:
                expected_extraction_fields = list(
                    self.task_support_registry.get_expected_extraction_fields(task_type) or []
                )
            except Exception:
                expected_extraction_fields = []
            if expected_extraction_fields:
                for field_name in expected_extraction_fields:
                    value = result["parsed_params"].get(field_name)
                    if self._is_null_like_label(value):
                        unresolved_extraction_fields.append(field_name)
                    else:
                        resolved_extraction_fields.append(field_name)
            judge_reference = {
                "task_type": task_type,
                "final_invalid_reason": error,
                "error_code": error_code,
                "error": error,
                "failure_stage": failure_stage,
                "parsed_params": deepcopy(result["parsed_params"]),
                "expected_extraction_fields": expected_extraction_fields,
                "resolved_extraction_fields": resolved_extraction_fields,
                "unresolved_extraction_fields": unresolved_extraction_fields,
            }
            if isinstance(validation_result, dict):
                judge_reference["validation_issues"] = list(validation_result.get("issues") or [])
                judge_reference["validation_suggestions"] = list(validation_result.get("suggestions") or [])
            result["judge_reference"] = judge_reference
        if extra:
            result.update(extra)
        return result

    def _infer_failure_stage(self, *, is_valid: bool, error_code: str) -> str:
        if is_valid:
            return "ok"

        normalized = str(error_code or "").strip()
        if normalized in {
            self.ERROR_CODES["INVALID_INPUT"],
            self.ERROR_CODES["INVALID_SUMMARY_TYPE"],
            self.ERROR_CODES["UNSUPPORTED_TASK"],
            self.ERROR_CODES["TASK_DISABLED"],
        }:
            return "input"
        if normalized == self.ERROR_CODES["MISSING_DEPENDENCY"]:
            return "dependency"
        if normalized == self.ERROR_CODES["QUESTION_INVALID"]:
            return "validation"
        if normalized == self.ERROR_CODES["INVALID_QUESTION_RULE"]:
            return "rule"
        if normalized in {self.ERROR_CODES["EXECUTION_FAILED"], self.ERROR_CODES["MISSING_TOOL"]}:
            return "execution"
        if normalized == self.ERROR_CODES["EMPTY_ANSWER"]:
            return "answer"
        return "unknown"

    def _build_invalid_response_with_recovery(
        self,
        *,
        task_type: str,
        question: str,
        input_data: Dict[str, Any],
        error: str,
        error_code: str,
        parsed_params: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        return_intermediate: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._build_response(
            is_valid=False,
            task_type=task_type,
            question=question,
            error=error,
            error_code=error_code,
            parsed_params=parsed_params,
            validation_result=validation_result,
            extra=extra,
        )
        controller = getattr(self, "invalid_recovery_controller", None)
        if not self.enable_invalid_recovery or controller is None:
            return result

        recovery = controller.recover(
            task_type=task_type,
            question=question,
            input_data=input_data,
            error_code=error_code,
            error=error,
            parsed_params=parsed_params or {},
            validation_result=validation_result,
            execution_context=(execution_result or {}).get("context", {}) if isinstance(execution_result, dict) else {},
        )
        if not recovery.attempted:
            return result

        result["recovery_attempted"] = True
        result["recovered"] = bool(recovery.recovered)
        result["recovery_reason"] = recovery.reason
        result["recovery_image_paths"] = list(recovery.image_paths)
        if recovery.recovered:
            result["answer"] = recovery.answer
            result["answer_source"] = "invalid_recovery"
        else:
            result["answer"] = ""
        if return_intermediate:
            result["recovery_prompt"] = recovery.prompt
            result["recovery_raw_response"] = recovery.raw_response
            result["recovery_evidence"] = recovery.evidence
        return result
    
    def validate_and_answer(
        self,
        input_data: Dict[str, Any],
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """Validate and answer."""
        start = time.perf_counter()
        is_contract_valid, contract_error = self._validate_input_contract(input_data)
        raw_task_type = input_data.get("task_type")
        question = input_data.get("question", "")
        if not is_contract_valid:
            return self._build_response(
                is_valid=False,
                task_type=str(raw_task_type or ""),
                question=question,
                error=contract_error,
                error_code=self.ERROR_CODES["INVALID_INPUT"]
            )

        task_resolution = self._resolve_task_type(raw_task_type)
        task_type = task_resolution.task_type
        normalized_input = input_data.copy()
        normalized_input["task_type"] = task_type

        if task_resolution.rejection_reason:
            return self._build_response(
                is_valid=False,
                task_type=str(raw_task_type),
                question=question,
                error=task_resolution.rejection_reason,
                error_code=self.ERROR_CODES["UNSUPPORTED_TASK"],
            )

        if not task_type:
            return self._build_response(
                is_valid=False,
                task_type=str(raw_task_type),
                question=question,
                error=f"Unsupported task type: {raw_task_type}",
                error_code=self.ERROR_CODES["UNSUPPORTED_TASK"]
            )

        task_cfg = self.config.get("tasks", {}).get(task_type, {})
        if isinstance(task_cfg, dict) and task_cfg.get("enabled") is False:
            return self._build_response(
                is_valid=False,
                task_type=task_type,
                question=question,
                error=f"Task disabled by config: {task_type}",
                error_code=self.ERROR_CODES["TASK_DISABLED"]
            )
        task_input_valid, task_input_error = self._validate_task_specific_input(task_type, normalized_input)
        if not task_input_valid:
            return self._build_response(
                is_valid=False,
                task_type=task_type,
                question=question,
                error=task_input_error,
                error_code=self.ERROR_CODES["INVALID_INPUT"]
            )

        task_class = self.task_registry.get(task_type)
        rubric_class = self.rubric_registry.get(task_type)
        
        if task_class is None or rubric_class is None:
            return self._build_response(
                is_valid=False,
                task_type=task_type,
                question=question,
                error=f"Unsupported task type: {task_type}",
                error_code=self.ERROR_CODES["UNSUPPORTED_TASK"]
            )
        

        task = task_class() if callable(task_class) else task_class
        rubric = rubric_class() if callable(rubric_class) else rubric_class
        has_prefilled_extraction = self._has_prefilled_context_labels(task_type, normalized_input)
        if task.requires_llm_extraction() and (not has_prefilled_extraction) and "vlm_tool" not in self.tools:
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error=(
                    f"Task {task_type} requires `vlm_tool` for entity extraction, "
                    "but it is not enabled or available. Install `openai` and set `enable_vlm=True`."
                ),
                error_code=self.ERROR_CODES["MISSING_DEPENDENCY"],
                return_intermediate=return_intermediate,
            )
        

        validation_result = self.validator.validate_question(
            question=question,
            task_type=task_type,
            context=normalized_input
        )
        
        if not validation_result["is_valid"]:
            self._log(f"Question validation failed: {validation_result}")
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error="Question validation failed",
                error_code=self.ERROR_CODES["QUESTION_INVALID"],
                validation_result=validation_result,
                return_intermediate=return_intermediate,
            )
        

        try:
            execution_result = self._execute_rubric(
                rubric=rubric,
                task=task,
                input_data=normalized_input
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error=f"Execution failed: {str(e)}",
                error_code=self.ERROR_CODES["EXECUTION_FAILED"],
                return_intermediate=return_intermediate,
            )
        if execution_result.get("step_error"):
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error=execution_result["step_error"],
                error_code=execution_result.get("step_error_code", self.ERROR_CODES["EXECUTION_FAILED"]),
                parsed_params=execution_result.get("parsed_params", {}),
                execution_result=execution_result,
                return_intermediate=return_intermediate,
            )
        if execution_result.get("invalid_reason"):
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error=execution_result["invalid_reason"],
                error_code=self.ERROR_CODES["INVALID_QUESTION_RULE"],
                parsed_params=execution_result.get("parsed_params", {}),
                execution_result=execution_result,
                return_intermediate=return_intermediate,
            )
        final_answer = execution_result.get("final_answer", "")
        if self._is_null_like_label(final_answer):
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error="Answer is empty or invalid (`null`/`none`)",
                error_code=self.ERROR_CODES["EMPTY_ANSWER"],
                parsed_params=execution_result.get("parsed_params", {}),
                execution_result=execution_result,
                return_intermediate=return_intermediate,
            )

        parsed_params = execution_result.get("parsed_params", {})
        post_answer_invalid_reason = self._get_post_answer_invalid_reason(
            task_type=task_type,
            final_answer=final_answer,
            parsed_params=parsed_params,
            context=execution_result.get("context", {}),
        )
        if post_answer_invalid_reason:
            return self._build_invalid_response_with_recovery(
                task_type=task_type,
                question=question,
                input_data=normalized_input,
                error=post_answer_invalid_reason,
                error_code=self.ERROR_CODES["INVALID_QUESTION_RULE"],
                parsed_params=parsed_params,
                execution_result=execution_result,
                return_intermediate=return_intermediate,
            )

        confidence = self._estimate_confidence(
            llm_extraction_calls=execution_result.get("llm_extraction_calls", 0),
            validation_result=validation_result
        )
        generation_score, question_score = self._estimate_difficulty_scores(
            task=task,
            task_type=task_type,
            answer=final_answer,
            parsed_params=parsed_params
        )

        result = self._build_response(
            is_valid=True,
            task_type=task_type,
            question=question,
            answer=final_answer,
            error_code=self.ERROR_CODES["OK"],
            parsed_params=parsed_params,
            confidence=confidence,
            generation_difficulty_score=generation_score,
            question_difficulty_score=question_score,
            extra={
                "reasoning": execution_result.get("reasoning", ""),
                "score_detail": {
                    "llm_extraction_calls": execution_result.get("llm_extraction_calls", 0),
                    "base_confidence": self.config.get("scoring", {}).get("base_confidence", 1.0),
                    "llm_extraction_penalty": self.config.get("scoring", {}).get("llm_extraction_penalty", 0.05),
                },
                "runtime_ms": round((time.perf_counter() - start) * 1000, 3),
            }
        )
        
        if return_intermediate:
            result["intermediate_results"] = execution_result.get("step_results", [])
            result["tool_result"] = execution_result.get("tool_result", {})
        

        self.execution_history.append(result)
        
        return result

    def _estimate_confidence(
        self,
        llm_extraction_calls: int,
        validation_result: Dict[str, Any]
    ) -> float:
        scoring_cfg = self.config.get("scoring", {})
        base_conf = float(scoring_cfg.get("base_confidence", 1.0))
        penalty = float(scoring_cfg.get("llm_extraction_penalty", 0.05))
        min_conf = float(scoring_cfg.get("min_confidence", 0.2))

        issue_penalty = 0.01 * len(validation_result.get("issues", []))
        confidence = base_conf - llm_extraction_calls * penalty - issue_penalty
        return max(min_conf, min(1.0, confidence))

    def _estimate_difficulty_scores(
        self,
        task,
        task_type: str,
        answer: Any,
        parsed_params: Dict[str, Any]
    ) -> Tuple[float, float]:
        task_cfg = self._get_task_cfg(task_type)
        difficulty_cfg = task_cfg.get("difficulty", {}) if isinstance(task_cfg, dict) else {}
        generation_score = self._resolve_generation_difficulty_score(
            task=task,
            task_cfg=task_cfg,
            difficulty_cfg=difficulty_cfg,
        )
        question_score = self._resolve_question_difficulty_score(
            task=task,
            task_cfg=task_cfg,
            difficulty_cfg=difficulty_cfg,
            answer=answer,
            parsed_params=parsed_params,
        )
        return generation_score, question_score

    @staticmethod
    def _normalize_answer_token(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        return re.sub(r"_+", "_", text)

    @staticmethod
    def _parse_int_like_answer(value: Any) -> Optional[int]:
        match = re.search(r"-?\d+", str(value or ""))
        if not match:
            return None
        try:
            return int(match.group(0))
        except (TypeError, ValueError):
            return None

    def _get_post_answer_invalid_reason(
        self,
        *,
        task_type: str,
        final_answer: Any,
        parsed_params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        normalized_task_type = str(task_type or "").strip()

        if normalized_task_type == "object_counting":
            count_value = self._parse_int_like_answer(final_answer)
            if count_value == 0:
                target_category = ""
                if isinstance(parsed_params, dict):
                    target_category = str(parsed_params.get("target_category", "") or "").strip()
                label_hint = f"（target_category={target_category}）" if target_category else ""
                return (
                    "object_counting invalid: the answer is 0, which means the question targets "
                    f"a category that does not exist in the scene{label_hint}; zero-count questions are not used as valid training samples."
                )

        if normalized_task_type == "elevation_cam_cam":
            if self._normalize_answer_token(final_answer) == "same_level":
                return (
                    "elevation_cam_cam invalid: the camera height relation between the two images is `same_level`, "
                    "so there is no meaningful elevation change; `same_level` questions are not used as valid training samples."
                )

        return None

    def _get_task_cfg(self, task_type: str) -> Dict[str, Any]:
        tasks_cfg = self.config.get("tasks", {})
        if not isinstance(tasks_cfg, dict):
            return {}
        task_cfg = tasks_cfg.get(task_type, {})
        return task_cfg if isinstance(task_cfg, dict) else {}

    @staticmethod
    def _to_unit_float(value: Any, default: float = 1.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _parse_answer_value(answer: Any, parser: str) -> Any:
        if parser == "int_first_token":
            try:
                token = str(answer).strip().split()[0]
                return int(token)
            except (TypeError, ValueError, IndexError):
                return None
        if parser == "float_first_token":
            try:
                token = str(answer).strip().split()[0]
                return float(token)
            except (TypeError, ValueError, IndexError):
                return None
        text = str(answer).strip().lower()
        if text == "":
            return None
        if parser == "auto":
            try:
                return int(text.split()[0])
            except (TypeError, ValueError, IndexError):
                return text
        return text

    def _resolve_generation_difficulty_score(
        self,
        task,
        task_cfg: Dict[str, Any],
        difficulty_cfg: Dict[str, Any],
    ) -> float:

        if "generation_difficulty_score" in task_cfg:
            return self._to_unit_float(
                task_cfg.get("generation_difficulty_score"),
                default=self._to_unit_float(task.get_generation_difficulty_score()),
            )

        generation_cfg = (
            difficulty_cfg.get("generation", {})
            if isinstance(difficulty_cfg, dict)
            else {}
        )
        if isinstance(generation_cfg, dict):
            mode = generation_cfg.get("mode", "task_default")
            if mode == "fixed":
                return self._to_unit_float(
                    generation_cfg.get("value"),
                    default=self._to_unit_float(task.get_generation_difficulty_score()),
                )

        return self._to_unit_float(task.get_generation_difficulty_score())

    def _resolve_question_difficulty_score(
        self,
        task,
        task_cfg: Dict[str, Any],
        difficulty_cfg: Dict[str, Any],
        answer: Any,
        parsed_params: Dict[str, Any],
    ) -> float:

        if "question_difficulty_score" in task_cfg:
            return self._to_unit_float(
                task_cfg.get("question_difficulty_score"),
                default=self._to_unit_float(
                    task.get_question_difficulty_score(answer, parsed_params)
                ),
            )

        question_cfg = (
            difficulty_cfg.get("question", {})
            if isinstance(difficulty_cfg, dict)
            else {}
        )
        if not isinstance(question_cfg, dict):
            return self._to_unit_float(task.get_question_difficulty_score(answer, parsed_params))

        mode = question_cfg.get("mode", "task_default")
        if mode == "fixed":
            return self._to_unit_float(
                question_cfg.get("value"),
                default=self._to_unit_float(task.get_question_difficulty_score(answer, parsed_params)),
            )

        if mode == "answer_in_set":
            parser = str(question_cfg.get("answer_parser", "auto"))
            parsed_answer = self._parse_answer_value(answer, parser)
            configured_values = question_cfg.get("values", [])
            if not isinstance(configured_values, list):
                configured_values = []
            normalized_values = {
                self._parse_answer_value(v, parser) for v in configured_values
            }
            hit = parsed_answer in normalized_values
            hit_value = question_cfg.get("hit_value", 0.5)
            miss_value = question_cfg.get("miss_value", 1.0)
            return self._to_unit_float(hit_value if hit else miss_value)

        return self._to_unit_float(task.get_question_difficulty_score(answer, parsed_params))
    
    def _execute_rubric(
        self,
        rubric,
        task,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute rubric."""

        context = input_data.copy()
        task_type = context.get("task_type", "")
        task_cfg = self._get_task_cfg(task_type)
        visibility_floor = self._get_visibility_floor()
        requested_min = context.get("min_visibility", task_cfg.get("min_visibility", visibility_floor))
        try:
            requested_min = float(requested_min)
        except (TypeError, ValueError):
            requested_min = visibility_floor
        context["min_visibility"] = max(visibility_floor, requested_min)
        

        scene_id = context.get("scene_id")
        metadata_dir = context.get("metadata_dir")
        scene_metadata = self.scannet_loader.load_scene_metadata(scene_id, metadata_dir)
        context["scene_metadata"] = scene_metadata
        profile = self.task_support_registry.get_task_profile(task_type)
        if profile is not None and profile.input_mode == "image_pair":
            self._inject_image_pair_context(context)
        self._inject_frame_context(context)
        self._register_runtime_labels_from_context(context)
        self._prefill_entities_from_question(context)
        
        step_results = []
        parsed_params = {}
        llm_extraction_calls = 0
        context["_extra_llm_calls"] = 0
        

        for step in rubric.get_steps():
            prompt_pool_invalid_reason = self._get_prompt_pool_invalid_reason(
                task_type=context.get("task_type", ""),
                context=context,
            )
            invalid_reason = self._get_invalid_question_reason(
                task_type=context.get("task_type", ""),
                context=context
            )
            image_task_ambiguity_reason = self._get_image_task_ambiguity_reason(
                task_type=context.get("task_type", ""),
                context=context
            )
            if prompt_pool_invalid_reason:
                return {
                    "final_answer": "",
                    "parsed_params": parsed_params,
                    "step_results": step_results,
                    "context": context,
                    "invalid_reason": prompt_pool_invalid_reason
                }
            if invalid_reason:
                return {
                    "final_answer": "",
                    "parsed_params": parsed_params,
                    "step_results": step_results,
                    "context": context,
                    "invalid_reason": invalid_reason
                }
            if image_task_ambiguity_reason:
                return {
                    "final_answer": "",
                    "parsed_params": parsed_params,
                    "step_results": step_results,
                    "context": context,
                    "invalid_reason": image_task_ambiguity_reason
                }

            self._log(f"\nRunning step {step.step_id}: {step.description}")

            current_task_type = str(context.get("task_type", "")).strip()
            if (
                step.tool_name == "vlm_tool"
                and task.requires_llm_extraction()
                and self._has_prefilled_context_labels(current_task_type, context)
            ):
                prefilled = self._build_prefilled_extraction(current_task_type, context)
                context["_vlm_extraction_done"] = True
                for key, value in prefilled.items():
                    context[key] = deepcopy(value)
                    parsed_params[key] = deepcopy(value)
                step_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "params": {"question": context.get("question", "")},
                    "result": {
                        "success": True,
                        "skipped": True,
                        "source": "prefilled_context",
                        "extracted_params": deepcopy(prefilled),
                    },
                })
                self._log(f"Step {step.step_id} completed (used prefilled params)")
                continue
            

            tool = self.tools.get(step.tool_name)
            if tool is None:
                return {
                    "final_answer": "",
                    "parsed_params": parsed_params,
                    "step_results": step_results,
                    "context": context,
                    "llm_extraction_calls": llm_extraction_calls,
                    "step_error": f"Step {step.step_id} is missing tool: {step.tool_name}",
                    "step_error_code": self.ERROR_CODES["MISSING_TOOL"],
                }
            

            try:
                params = self._prepare_step_params(step, context, task)
                

                result = tool.execute(**params)
                if isinstance(result, dict) and result.get("success") is False:
                    error_msg = result.get("error", f"Step {step.step_id} failed")
                    step_results.append({
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "description": step.description,
                        "params": params,
                        "result": result
                    })
                    return {
                        "final_answer": "",
                        "parsed_params": parsed_params,
                        "step_results": step_results,
                        "context": context,
                        "llm_extraction_calls": llm_extraction_calls,
                        "step_error": f"Step {step.step_id} failed: {error_msg}",
                        "step_error_code": self.ERROR_CODES["EXECUTION_FAILED"],
                    }
                

                step_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "params": params,
                    "result": result
                })
                

                extraction_used = self._update_context_from_result(
                    step=step,
                    result=result,
                    context=context,
                    parsed_params=parsed_params,
                    task=task
                )
                if extraction_used:
                    llm_extraction_calls += 1
                
                self._log(f"Step {step.step_id} completed")
                
            except Exception as e:
                self._log(f"Step {step.step_id} failed: {e}")
                import traceback
                traceback.print_exc()
                step_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "error": str(e)
                })
                return {
                    "final_answer": "",
                    "parsed_params": parsed_params,
                    "step_results": step_results,
                    "context": context,
                    "llm_extraction_calls": llm_extraction_calls,
                    "step_error": f"Step {step.step_id} failed: {str(e)}",
                    "step_error_code": self.ERROR_CODES["EXECUTION_FAILED"],
                }
        

        final_answer = self._extract_final_answer(step_results)
        
        return {
            "final_answer": final_answer,
            "parsed_params": parsed_params,
            "step_results": step_results,
            "context": context,
            "llm_extraction_calls": llm_extraction_calls + int(context.get("_extra_llm_calls", 0))
        }

    @staticmethod
    def _is_null_like_label(value: Any) -> bool:
        if value is None:
            return True
        text = str(value).strip().lower()
        if text.startswith("error:"):
            return True
        return text in {"", "null", "none", "unknown", "n/a"}

    def _same_label(self, a: Any, b: Any) -> bool:
        if self._is_null_like_label(a) or self._is_null_like_label(b):
            return False
        a_raw = str(a).strip().lower()
        b_raw = str(b).strip().lower()
        a_std = self.map_to_standard_label(a_raw)
        b_std = self.map_to_standard_label(b_raw)
        return a_raw == b_raw or a_std == b_std

    def _get_invalid_question_reason(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Get invalid question reason."""
        return self.rule_engine.get_invalid_question_reason(
            task_type=task_type,
            context=context,
        )

    def _get_prompt_pool_invalid_reason(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        scene_counts = self._get_scene_observed_label_counts(context)
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        scene_unique_set = set(scene_unique)
        scene_observed_set = set(scene_counts.keys())

        if task_type == "object_counting":
            target_category = str(context.get("target_category", "") or "").strip().lower()
            scene_non_unique_set = set(scene_non_unique)
            if not scene_non_unique_set:
                return "object_counting invalid: no countable category has observed count >= 2 in the current input"
            if target_category and target_category not in scene_observed_set:
                return f"object_counting invalid: `target_category` `{target_category}` is not in the visible scene label pool"
            if target_category and target_category not in scene_non_unique_set:
                return f"object_counting invalid: `target_category` `{target_category}` is not in the counting label pool"
            return None

        if task_type == "object_size":
            object_label = str(context.get("object_label", "") or "").strip().lower()
            if object_label and object_label not in scene_unique_set:
                return f"object_size invalid: `object_label` `{object_label}` is not in the scene unique pool"
            return None

        if task_type == "absolute_distance":
            object1 = str(context.get("object1_label", "") or "").strip().lower()
            object2 = str(context.get("object2_label", "") or "").strip().lower()
            if object1 and object1 not in scene_unique_set:
                return f"absolute_distance invalid: `object1_label` `{object1}` is not in the scene unique pool"
            if object2 and object2 not in scene_unique_set:
                return f"absolute_distance invalid: `object2_label` `{object2}` is not in the scene unique pool"
            return None

        if task_type == "relative_distance":
            target_label = str(context.get("target_label", "") or "").strip().lower()
            candidate_labels = context.get("candidate_labels", [])
            if target_label and target_label not in scene_unique_set:
                return f"relative_distance invalid: `target_label` `{target_label}` is not in the scene unique pool"
            if isinstance(candidate_labels, list):
                for label in candidate_labels:
                    text = str(label or "").strip().lower()
                    if text and text not in scene_unique_set:
                        return f"relative_distance invalid: `candidate_label` `{text}` is not in the scene unique pool"
            return None

        if task_type == "relative_direction_hard":
            for field_name in ("positioning_label", "orienting_label", "querying_label"):
                label = str(context.get(field_name, "") or "").strip().lower()
                if label and label not in scene_unique_set:
                    return f"relative_direction_hard invalid: `{field_name}` `{label}` is not in the scene unique pool"
            return None

        if task_type == "visibility_compare":
            target_label = str(context.get("target_label", "") or "").strip().lower()
            contrast_labels = self._get_pair_visibility_contrast_labels_from_context(context)
            shared_unique = self._get_shared_unique_pair_labels_from_context(context)
            pair_non_ambiguous = self._get_pair_non_ambiguous_labels_from_context(context)
            active_pool = contrast_labels or shared_unique or pair_non_ambiguous
            if target_label and active_pool and target_label not in set(active_pool):
                return f"visibility_compare invalid: `target_label` `{target_label}` is not in the visibility pool"
            return None

        if task_type == "attribute_measurement":
            pair_non_ambiguous = set(self._get_pair_non_ambiguous_labels_from_context(context))
            for field_name in ("object1_label", "object2_label"):
                label = str(context.get(field_name, "") or "").strip().lower()
                if label and pair_non_ambiguous and label not in pair_non_ambiguous:
                    return f"attribute_measurement invalid: `{field_name}` `{label}` is not in the pair non-ambiguous pool"
            return None

        if task_type == "position_cam_obj":
            ref_counts = self._get_reference_frame_visible_label_counts(context)
            ref_unique, _ = self._split_label_counts(ref_counts)
            target_label = str(context.get("target_label", "") or "").strip().lower()
            if target_label and ref_unique and target_label not in set(ref_unique):
                return f"position_cam_obj invalid: `target_label` `{target_label}` is not in the selected reference-image unique pool"
            return None

        if task_type == "position_cam_reg":
            ref_counts = self._get_reference_frame_visible_label_counts(context)
            ref_unique, _ = self._split_label_counts(ref_counts)
            region_name = str(context.get("region_name", "") or "").strip().lower()
            if not region_name or self._is_abstract_region_phrase(region_name):
                return None
            mapped = self.map_to_standard_label(region_name)
            if mapped and mapped != "null" and ref_unique and mapped not in set(ref_unique):
                return f"position_cam_reg invalid: region anchor `{mapped}` is not in the selected reference-image unique anchor pool"
            return None

        return None

    def _question_indicates_target_in_candidates(
        self,
        question: str,
        target_label: str
    ) -> bool:
        """Question indicates target in candidates."""
        question_l = question.lower()
        target = self.map_to_standard_label(target_label)
        if not question_l or not target or target == "null":
            return False

        candidate_segments = []

        # Pattern 1: among A, B and C
        if " among " in f" {question_l} ":
            candidate_segments.append(question_l.split("among", 1)[1])

        # Pattern 2: (A, B, C)
        if "(" in question_l and ")" in question_l:
            left = question_l.find("(")
            right = question_l.find(")", left + 1)
            if left != -1 and right != -1 and right > left + 1:
                candidate_segments.append(question_l[left + 1:right])

        if not candidate_segments:
            return False

        for segment in candidate_segments:
            normalized = (
                segment.replace(" and ", ",")
                .replace(" or ", ",")
                .replace(";", ",")
            )
            raw_parts = [p.strip(" .?!:") for p in normalized.split(",")]
            mapped_parts = [self.map_to_standard_label(p) for p in raw_parts if p]
            if target in mapped_parts:
                return True

        return False

    @staticmethod
    def _count_visible_in_frame(frame_metadata: Dict[str, Any], label: str, min_visibility: float) -> int:
        objects = frame_metadata.get("objects", []) if isinstance(frame_metadata, dict) else []
        count = 0
        for obj in objects or []:
            obj_label = str(obj.get("label", "")).strip().lower()
            if obj_label != str(label).strip().lower():
                continue
            try:
                vis = float(obj.get("visibility", 0.0))
            except (TypeError, ValueError):
                vis = 0.0
            if vis >= float(min_visibility):
                count += 1
        return count

    def _build_entity_from_frame_object(
        self,
        obj: Dict[str, Any],
        fallback_name: str,
        use_camera_location: bool = False,
        camera_ref_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        loc = obj.get("camera_location") if use_camera_location else obj.get("3d_location")
        if not isinstance(loc, (list, tuple)) or len(loc) < 3:
            return None
        try:
            pos = np.array([float(loc[0]), float(loc[1]), float(loc[2])], dtype=float)
        except (TypeError, ValueError):
            return None
        name = str(obj.get("label", fallback_name)).strip().lower() or fallback_name
        entity = {
            "name": name,
            "category": name,
            "position": pos,
            "bbox": obj.get("3d_bbox"),
            "visibility": float(obj.get("visibility", 0.0)),
            "object_id": obj.get("object_id"),
            "camera_location": obj.get("camera_location"),
        }
        if use_camera_location:
            entity["position_frame"] = "camera"
            if camera_ref_idx in (1, 2):
                entity["camera_ref_idx"] = int(camera_ref_idx)
        return entity

    def _select_cam_obj_target_entity(self, context: Dict[str, Any], detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select cam obj target entity."""
        target_raw = str(context.get("target_label", "")).strip().lower()
        target = self.map_to_standard_label(target_raw)
        if self._is_null_like_label(target):
            return None

        min_visibility = float(context.get("min_visibility", self._get_visibility_floor()))

        def _visible_matches(frame_meta: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
            objs = frame_meta.get("objects", []) if isinstance(frame_meta, dict) else []
            matches = []
            for obj in objs or []:
                label = self.map_to_standard_label(str(obj.get("label", "")).strip().lower())
                if label != target:
                    continue
                try:
                    vis = float(obj.get("visibility", 0.0))
                except (TypeError, ValueError):
                    vis = 0.0
                if vis < threshold:
                    continue
                matches.append(obj)
            return matches

        ref_idx = int(context.get("camera_reference_image_idx", 2))
        primary_key = "frame_metadata_1" if ref_idx == 1 else "frame_metadata_2"

        matches_primary = _visible_matches(context.get(primary_key, {}), threshold=0.01)
        if len(matches_primary) == 1:
            return self._build_entity_from_frame_object(
                matches_primary[0],
                fallback_name=target,
                use_camera_location=True,
                camera_ref_idx=ref_idx,
            )

        matched_dets = []
        for det in detections or []:
            name = self.map_to_standard_label(str(det.get("name", "")).strip().lower())
            if name == target:
                matched_dets.append(det)
        if len(matched_dets) == 1 and isinstance(matched_dets[0].get("position"), np.ndarray):
            return matched_dets[0]

        return None

    def _select_measurement_entity(
        self,
        context: Dict[str, Any],
        target_label: str,
    ) -> Optional[Dict[str, Any]]:
        """Select measurement entity."""
        target = self.map_to_standard_label(str(target_label).strip().lower())
        if self._is_null_like_label(target):
            return None

        min_visibility = float(context.get("min_visibility", self._get_visibility_floor()))

        for frame_key in ("frame_metadata_2", "frame_metadata_1", "frame_metadata"):
            frame_meta = context.get(frame_key, {})
            objects = frame_meta.get("objects", []) if isinstance(frame_meta, dict) else []
            matches = []
            for obj in objects or []:
                label = self.map_to_standard_label(str(obj.get("label", "")).strip().lower())
                if label != target:
                    continue
                try:
                    vis = float(obj.get("visibility", 0.0))
                except (TypeError, ValueError):
                    vis = 0.0
                if vis < min_visibility:
                    continue
                matches.append(obj)
            if len(matches) == 1:
                return self._build_entity_from_frame_object(matches[0], fallback_name=target)

        return None

    @staticmethod
    def _to_camera_frame_entity(entity_world: Dict[str, Any], camera_entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """To camera frame entity."""
        if not isinstance(entity_world, dict) or not isinstance(camera_entity, dict):
            return None
        if str(entity_world.get("position_frame", "")).lower() == "camera":
            return entity_world
        pos_obj = entity_world.get("position")
        pos_cam = camera_entity.get("position")
        rot_cam = camera_entity.get("rotation")
        if not isinstance(pos_obj, np.ndarray):
            return None
        if not isinstance(pos_cam, np.ndarray):
            return None
        if not isinstance(rot_cam, np.ndarray):
            return None
        rel_world = np.array(pos_obj, dtype=float) - np.array(pos_cam, dtype=float)
        rel_cam = np.array(rot_cam, dtype=float).T @ rel_world
        out = dict(entity_world)
        out["position"] = rel_cam
        return out

    def _validate_label_in_provided_frames(
        self,
        frame_metadatas: List[Dict[str, Any]],
        label: str,
        min_visibility: float,
        role: str,
        task_type: str,
    ) -> Optional[str]:
        """Validate label in provided frames."""
        mapped = self.map_to_standard_label(label)
        if not mapped or mapped in {"null", "none"}:
            return f"{task_type} invalid: failed to extract {role}"

        counts = []
        for idx, frame_meta in enumerate(frame_metadatas, 1):
            c = self._count_visible_in_frame(frame_meta, mapped, min_visibility)
            counts.append((idx, c))

        for idx, c in counts:
            if c > 1:
                return f"{task_type} invalid: {role} `{mapped}` is not unique in provided image #{idx} ({c} matches)"

        if any(c == 1 for _, c in counts):
            return None

        return f"{task_type} invalid: {role} `{mapped}` is not visible enough in the provided images"

    def _normalize_region_phrase(self, text: Any) -> str:
        return self.task_support_registry._normalize_region_phrase(text)

    def _resolve_region_phrase_alias(self, text: Any) -> Optional[str]:
        return self.task_support_registry.resolve_region_phrase_alias(text)

    def _is_abstract_region_phrase(self, text: str) -> bool:
        t = str(text or "").strip().lower()
        if not t:
            return False
        if self._resolve_region_phrase_alias(t):
            return True
        keywords = {
            "region", "area", "zone", "room",
            "sleeping", "study", "printing", "kitchen", "bathroom",
            "living", "dining", "bedroom", "workspace", "toilet area",
        }
        return any(k in t for k in keywords)

    def _get_image_task_ambiguity_reason(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        profile = self.task_support_registry.get_task_profile(task_type)
        if profile is None:
            return None
        if not profile.ambiguity_object_fields and not profile.ambiguity_region_fields:
            return None
        min_visibility = float(context.get("min_visibility", self._get_visibility_floor()))
        frame_metadatas: List[Dict[str, Any]] = []
        if task_type in {"position_cam_obj", "position_cam_reg"}:
            ref_idx = context.get("camera_reference_image_idx")
            frame_key = "frame_metadata_1" if ref_idx == 1 else "frame_metadata_2"
            frame_meta = context.get(frame_key)
            if isinstance(frame_meta, dict):
                frame_metadatas.append(frame_meta)
        else:
            if isinstance(context.get("frame_metadata_1"), dict):
                frame_metadatas.append(context["frame_metadata_1"])
            if isinstance(context.get("frame_metadata_2"), dict):
                frame_metadatas.append(context["frame_metadata_2"])
            if not frame_metadatas and isinstance(context.get("frame_metadata"), dict):
                frame_metadatas.append(context["frame_metadata"])
        if not frame_metadatas:
            return f"{task_type} invalid: missing `frame_metadata` for the provided images"

        for field_role in profile.ambiguity_object_fields:
            label = context.get(field_role.field_name)
            if self._is_null_like_label(label):
                continue
            err = self._validate_label_in_provided_frames(
                frame_metadatas=frame_metadatas,
                label=str(label),
                min_visibility=min_visibility,
                role=field_role.role_name,
                task_type=task_type,
            )
            if err:
                return err

        visible_labels = set()
        for frame_meta in frame_metadatas:
            objects = frame_meta.get("objects", []) if isinstance(frame_meta, dict) else []
            for obj in objects or []:
                try:
                    vis = float(obj.get("visibility", 0.0))
                except (TypeError, ValueError):
                    vis = 0.0
                if vis < min_visibility:
                    continue
                label = str(obj.get("label", "")).strip().lower()
                if label:
                    visible_labels.add(label)

        for field_role in profile.ambiguity_region_fields:
            raw = context.get(field_role.field_name)
            if self._is_null_like_label(raw):
                continue
            raw_norm = str(raw).strip().lower()
            mapped = self.map_to_standard_label(raw_norm)
            candidate = None
            if mapped in visible_labels:
                candidate = mapped
            elif raw_norm in visible_labels:
                candidate = raw_norm
            if candidate is None:

                if self._is_abstract_region_phrase(raw_norm):
                    continue
                if raw_norm in self.VALID_LABELS:
                    return f"{task_type} invalid: {field_role.role_name} `{raw_norm}` is not visible enough in the provided images"
                if mapped in self.VALID_LABELS and mapped == raw_norm:
                    missing = mapped if mapped in self.VALID_LABELS else raw_norm
                    return f"{task_type} invalid: {field_role.role_name} `{missing}` is not visible enough in the provided images"

                continue
            err = self._validate_label_in_provided_frames(
                frame_metadatas=frame_metadatas,
                label=candidate,
                min_visibility=min_visibility,
                role=field_role.role_name,
                task_type=task_type,
            )
            if err:
                return err

        return None
    
    def _prepare_step_params(
        self,
        step,
        context: Dict[str, Any],
        task
    ) -> Dict[str, Any]:
        """Prepare step params."""
        return self.step_param_builder.build(
            step=step,
            context=context,
            task=task,
        )
    
    def _prepare_vlm_extraction_params(
        self,
        step,
        context: Dict[str, Any],
        task
    ) -> Dict[str, Any]:
        """Prepare vlm extraction params."""
        question = context.get("question", "")
        prompt = task.build_extraction_prompt(question)
        if not prompt:
            prompt = f"Extract key entities from question and keep compact format.\nQuestion: {question}"

        available_labels = self._collect_extraction_candidate_labels(context)
        if available_labels:
            label_text = ", ".join(available_labels)
            task_type = str(context.get("task_type", "")).strip().lower()
            if task_type == "position_cam_reg":
                prompt = (
                    f"{prompt}\n\n"
                    "Constraint:\n"
                    "- The candidate list contains visible object anchors only; it is NOT the required output space for region extraction.\n"
                    "- If the question contains an abstract region phrase such as sleeping area, bathroom area, kitchen area, or living area, output that region phrase directly.\n"
                    "- Do NOT replace the region phrase with an object label from the candidate list.\n"
                    "- Only output null if no target region phrase can be determined from the question.\n"
                    "- Keep the original required output format exactly.\n"
                    f"Visible anchor candidates (for grounding only): {label_text}\n"
                )
            else:
                prompt = (
                    f"{prompt}\n\n"
                    "Constraint:\n"
                    "- Use only labels from the candidate list when extracting object entities.\n"
                    "- If an entity cannot be determined from question or candidates, output null for that slot.\n"
                    "- Do NOT guess or replace with a similar object when uncertain.\n"
                    "- If your extracted value is null, keep it null.\n"
                    "- Keep the original required output format exactly.\n"
                    f"Candidates (from provided image(s) only): {label_text}\n"
                )

        vlm_cfg = self.config.get("vlm", {})
        extraction_use_vision = bool(vlm_cfg.get("extraction_use_vision", False))
        extraction_images = []
        if extraction_use_vision:
            if isinstance(context.get("image_paths"), list):
                extraction_images = context.get("image_paths")
            elif isinstance(context.get("image_path"), str) and context.get("image_path").strip():
                extraction_images = [context.get("image_path").strip()]
            else:
                p1 = context.get("image_path_1")
                p2 = context.get("image_path_2")
                if isinstance(p1, str) and p1.strip():
                    extraction_images.append(p1.strip())
                if isinstance(p2, str) and p2.strip():
                    extraction_images.append(p2.strip())
        return {
            "prompt": prompt,
            "max_tokens": vlm_cfg.get("max_tokens", 120),
            "temperature": vlm_cfg.get(
                "extraction_temperature",
                vlm_cfg.get("temperature", 0.0),
            ),
            "image_paths": extraction_images,
            "use_vision": extraction_use_vision,
        }

    def _split_label_counts(self, label_counts: Any) -> Tuple[List[str], List[str]]:
        if not isinstance(label_counts, dict):
            return [], []
        unique_labels: List[str] = []
        non_unique_labels: List[str] = []
        for label, count in sorted(label_counts.items()):
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            if numeric_count == 1:
                unique_labels.append(str(label).strip().lower())
            elif numeric_count > 1:
                non_unique_labels.append(str(label).strip().lower())
        return unique_labels, non_unique_labels

    def _normalize_label_count_dict(self, label_counts: Any) -> Dict[str, int]:
        if not isinstance(label_counts, dict):
            return {}
        normalized: Dict[str, int] = {}
        for label, count in label_counts.items():
            text = str(label or "").strip().lower()
            if not text or not self._is_usable_anchor_label(text):
                continue
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            if numeric_count > 0:
                normalized[text] = numeric_count
        return dict(sorted(normalized.items()))

    def _count_visible_labels_in_frame_metadata(
        self,
        frame_meta: Dict[str, Any],
        min_visibility: Optional[float] = None,
    ) -> Dict[str, int]:
        if not isinstance(frame_meta, dict):
            return {}
        if min_visibility is None:
            min_visibility = self._get_visibility_floor()
        try:
            min_visibility = float(min_visibility)
        except (TypeError, ValueError):
            min_visibility = self._get_visibility_floor()
        counts: Dict[str, int] = {}
        objects = frame_meta.get("objects", []) if isinstance(frame_meta, dict) else []
        for obj in objects or []:
            try:
                vis = float(obj.get("visibility", 0.0))
            except (TypeError, ValueError):
                vis = 0.0
            if vis < min_visibility:
                continue
            label = str(obj.get("label", "")).strip().lower()
            if not label or not self._is_usable_anchor_label(label):
                continue
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def _build_visibility_map_from_frame_metadata(self, frame_meta: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(frame_meta, dict):
            return {}
        visibility_map: Dict[str, float] = {}
        objects = frame_meta.get("objects", []) if isinstance(frame_meta, dict) else []
        for obj in objects or []:
            label = str(obj.get("label", "")).strip().lower()
            if not label or not self._is_usable_anchor_label(label):
                continue
            try:
                visibility = float(obj.get("visibility", 0.0))
            except (TypeError, ValueError):
                visibility = 0.0
            previous = visibility_map.get(label)
            if previous is None or visibility > previous:
                visibility_map[label] = visibility
        return dict(sorted(visibility_map.items()))

    def _get_context_label_counts(self, context: Dict[str, Any], key: str) -> Dict[str, int]:
        counts = context.get(key)
        return self._normalize_label_count_dict(counts)

    def _get_scene_observed_label_counts(self, context: Dict[str, Any]) -> Dict[str, int]:
        counts = self._get_context_label_counts(context, "scene_label_counts")
        if counts:
            return counts
        return self._normalize_label_count_dict(
            self._label_count_from_objects(
                context.get("scene_metadata", {}).get("objects", [])
                if isinstance(context.get("scene_metadata"), dict) else []
            )
        )

    def _get_frame_visible_label_counts(
        self,
        context: Dict[str, Any],
        *,
        frame_index: Optional[int] = None,
    ) -> Dict[str, int]:
        if frame_index == 1:
            counts = self._get_context_label_counts(context, "image1_visible_label_counts")
            if counts:
                return counts
            return self._count_visible_labels_in_frame_metadata(
                context.get("frame_metadata_1", {}),
                min_visibility=context.get("min_visibility"),
            )
        if frame_index == 2:
            counts = self._get_context_label_counts(context, "image2_visible_label_counts")
            if counts:
                return counts
            return self._count_visible_labels_in_frame_metadata(
                context.get("frame_metadata_2", {}),
                min_visibility=context.get("min_visibility"),
            )

        counts = self._get_context_label_counts(context, "frame_visible_label_counts")
        if counts:
            return counts
        return self._count_visible_labels_in_frame_metadata(
            context.get("frame_metadata", {}),
            min_visibility=context.get("min_visibility"),
        )

    def _get_reference_frame_visible_label_counts(self, context: Dict[str, Any]) -> Dict[str, int]:
        ref_idx = context.get("camera_reference_image_idx")
        if ref_idx in (1, 2):
            return self._get_frame_visible_label_counts(context, frame_index=int(ref_idx))
        counts = self._get_frame_visible_label_counts(context, frame_index=2)
        if counts:
            return counts
        return self._get_frame_visible_label_counts(context, frame_index=1)

    def _get_pair_non_ambiguous_labels_from_context(self, context: Dict[str, Any]) -> List[str]:
        image1_counts = self._get_frame_visible_label_counts(context, frame_index=1)
        image2_counts = self._get_frame_visible_label_counts(context, frame_index=2)
        labels = set(image1_counts.keys()) | set(image2_counts.keys())
        valid_labels: List[str] = []
        for label in sorted(labels):
            count1 = int(image1_counts.get(label, 0))
            count2 = int(image2_counts.get(label, 0))
            if count1 + count2 >= 1 and count1 <= 1 and count2 <= 1:
                valid_labels.append(label)
        return valid_labels

    def _get_shared_unique_pair_labels_from_context(self, context: Dict[str, Any]) -> List[str]:
        image1_counts = self._get_frame_visible_label_counts(context, frame_index=1)
        image2_counts = self._get_frame_visible_label_counts(context, frame_index=2)
        labels = set(image1_counts.keys()) | set(image2_counts.keys())
        return [
            label
            for label in sorted(labels)
            if int(image1_counts.get(label, 0)) == 1 and int(image2_counts.get(label, 0)) == 1
        ]

    def _get_pair_visibility_contrast_labels_from_context(self, context: Dict[str, Any]) -> List[str]:
        provided = context.get("pair_visibility_contrast_labels")
        if isinstance(provided, list) and provided:
            return [
                str(label).strip().lower()
                for label in provided
                if self._is_usable_anchor_label(str(label).strip().lower())
            ]

        image1_counts = self._get_frame_visible_label_counts(context, frame_index=1)
        image2_counts = self._get_frame_visible_label_counts(context, frame_index=2)
        candidate_labels = self._get_pair_non_ambiguous_labels_from_context(context)
        image1_visibility_map = self._build_visibility_map_from_frame_metadata(context.get("frame_metadata_1", {}))
        image2_visibility_map = self._build_visibility_map_from_frame_metadata(context.get("frame_metadata_2", {}))
        labels: List[str] = []
        for label in candidate_labels:
            if int(image1_counts.get(label, 0)) + int(image2_counts.get(label, 0)) < 1:
                continue
            delta = abs(float(image1_visibility_map.get(label, 0.0)) - float(image2_visibility_map.get(label, 0.0)))
            if delta >= 0.15:
                labels.append(label)
        return labels

    def _collect_available_labels_from_provided_frames(self, context: Dict[str, Any]) -> List[str]:
        """Collect available labels from provided frames."""
        labels = set()
        for counts in (
            self._get_frame_visible_label_counts(context, frame_index=1),
            self._get_frame_visible_label_counts(context, frame_index=2),
            self._get_frame_visible_label_counts(context),
        ):
            labels.update(counts.keys())
        return sorted(labels)

    def _collect_extraction_candidate_labels(self, context: Dict[str, Any]) -> List[str]:
        """Collect extraction candidate labels."""
        frame_labels = self._collect_available_labels_from_provided_frames(context)
        task_type = str(context.get("task_type", ""))
        profile = self.task_support_registry.get_task_profile(task_type)
        candidate_label_scope = (
            profile.candidate_label_scope
            if profile is not None and isinstance(profile.candidate_label_scope, str)
            else "prefer_frame_then_scene"
        )
        if candidate_label_scope == "scene_observed_counting":
            scene_counts = self._get_scene_observed_label_counts(context)
            _, non_unique_labels = self._split_label_counts(scene_counts)
            return non_unique_labels
        if candidate_label_scope == "scene_observed_unique":
            scene_counts = self._get_scene_observed_label_counts(context)
            unique_labels, _ = self._split_label_counts(scene_counts)
            return unique_labels
        if candidate_label_scope == "reference_frame_unique":
            ref_counts = self._get_reference_frame_visible_label_counts(context)
            unique_labels, _ = self._split_label_counts(ref_counts)
            return unique_labels or sorted(ref_counts.keys())
        if candidate_label_scope == "pair_non_ambiguous":
            return self._get_pair_non_ambiguous_labels_from_context(context)
        if candidate_label_scope == "pair_visibility_compare":
            contrast_labels = self._get_pair_visibility_contrast_labels_from_context(context)
            if contrast_labels:
                return contrast_labels
            shared_unique = self._get_shared_unique_pair_labels_from_context(context)
            if shared_unique:
                return shared_unique
            return self._get_pair_non_ambiguous_labels_from_context(context)
        if candidate_label_scope == "frame_only":
            return frame_labels
        if frame_labels:
            return frame_labels

        scene_metadata = context.get("scene_metadata", {})
        objects = scene_metadata.get("objects", []) if isinstance(scene_metadata, dict) else []
        labels = {
            str(obj.get("label", "")).strip().lower()
            for obj in objects or []
            if str(obj.get("label", "")).strip()
            and self._is_usable_anchor_label(str(obj.get("label", "")).strip().lower())
        }
        return sorted(labels)

    def _register_runtime_labels(self, labels: Any) -> None:
        """Register runtime labels."""
        if not isinstance(labels, (list, tuple, set)):
            return
        for label in labels:
            text = str(label or "").strip().lower()
            if text:
                self.VALID_LABELS.add(text)

    def _register_runtime_labels_from_context(self, context: Dict[str, Any]) -> None:
        """Register runtime labels from context."""
        if not isinstance(context, dict):
            return
        labels = set()

        scene_metadata = context.get("scene_metadata", {})
        objects = scene_metadata.get("objects", []) if isinstance(scene_metadata, dict) else []
        for obj in objects or []:
            label = str(obj.get("label", "")).strip().lower()
            if label:
                labels.add(label)

        for key in ("frame_metadata_1", "frame_metadata_2", "frame_metadata"):
            frame_metadata = context.get(key, {})
            frame_objects = frame_metadata.get("objects", []) if isinstance(frame_metadata, dict) else []
            for obj in frame_objects or []:
                label = str(obj.get("label", "")).strip().lower()
                if label:
                    labels.add(label)

        self._register_runtime_labels(sorted(labels))

    def _expected_extraction_defaults(self, task_type: str) -> Dict[str, Any]:
        return self.task_support_registry.get_extraction_defaults(task_type)

    def _has_prefilled_context_labels(self, task_type: str, context: Dict[str, Any]) -> bool:
        profile = self.task_support_registry.get_task_profile(task_type)
        if profile is None or not profile.context_label_fields:
            return False

        for field_name in profile.context_label_fields:
            if field_name not in context:
                return False
            value = context.get(field_name)
            if isinstance(value, list):
                normalized = [item for item in value if not self._is_null_like_label(item)]
                if not normalized:
                    return False
                continue
            if self._is_null_like_label(value):
                return False
        return True

    def _build_prefilled_extraction(self, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        profile = self.task_support_registry.get_task_profile(task_type)
        defaults = self._expected_extraction_defaults(task_type)
        extracted = deepcopy(defaults)
        if profile is None:
            return extracted

        for field_name in profile.context_label_fields:
            if field_name in context:
                extracted[field_name] = deepcopy(context[field_name])
        return self._sanitize_extracted_by_candidates(extracted, context)

    def _sanitize_extracted_by_candidates(
        self,
        extracted: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sanitize extracted by candidates."""
        if not isinstance(extracted, dict):
            return {}

        available = self._collect_extraction_candidate_labels(context)
        available_set = set(available)
        if not available:
            return extracted

        object_keys = {
            "target_category",
            "object_label",
            "object1_label",
            "object2_label",
            "target_label",
            "reference_label",
            "positioning_label",
            "orienting_label",
            "querying_label",
        }

        out = dict(extracted)
        for k in object_keys:
            if k not in out:
                continue
            v = out.get(k)
            if self._is_null_like_label(v):
                out[k] = "null"
                continue
            resolved = self._resolve_object_name_with_candidates(str(v), available)
            if resolved in available_set and self._is_usable_anchor_label(resolved):
                out[k] = resolved
            else:
                out[k] = "null"

        if isinstance(out.get("candidate_labels"), list):
            normalized = []
            for c in out["candidate_labels"]:
                if self._is_null_like_label(c):
                    continue
                mapped = self._resolve_object_name_with_candidates(str(c), available)
                if mapped in available and self._is_usable_anchor_label(mapped) and mapped not in normalized:
                    normalized.append(mapped)
            out["candidate_labels"] = normalized

        return out

    @staticmethod
    def _extract_parenthesis_segment(question: str) -> str:
        if not isinstance(question, str):
            return ""
        left = question.find("(")
        right = question.find(")", left + 1) if left >= 0 else -1
        if left >= 0 and right > left:
            return question[left + 1:right]
        return ""

    def _find_candidate_mentions_in_question(
        self,
        question: str,
        candidates: List[str],
    ) -> List[str]:
        """Find candidate mentions in question."""
        if not isinstance(question, str) or not question.strip():
            return []
        q = question.lower()
        hits = []
        for label in candidates:
            idx = q.find(label)
            if idx >= 0:
                hits.append((idx, label))

        tokens = re.findall(r"[a-zA-Z0-9]+", q)
        if tokens:
            max_label_token_len = 1
            for label in candidates:
                max_label_token_len = max(max_label_token_len, len(re.findall(r"[a-zA-Z0-9]+", label)))
            covered = {label for _, label in hits}
            for label in candidates:
                if label in covered:
                    continue
                best_score = 0.0
                best_token_idx = None

                upper = min(max_label_token_len + 1, 4)
                for n in range(1, upper + 1):
                    for i in range(0, len(tokens) - n + 1):
                        phrase = " ".join(tokens[i:i + n])
                        score = self._semantic_label_similarity(phrase, label)
                        if score > best_score:
                            best_score = score
                            best_token_idx = i
                if best_token_idx is not None and best_score >= 0.78:

                    hits.append((best_token_idx * 4, label))

        hits.sort(key=lambda x: x[0])
        ordered = []
        seen = set()
        for _, label in hits:
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        return ordered

    def _resolve_object_name_with_candidates(
        self,
        raw_name: str,
        candidates: List[str],
    ) -> str:
        """Resolve object name with candidates."""
        if self._is_null_like_label(raw_name):
            return "null"
        if not candidates:
            return self.map_to_standard_label(str(raw_name).strip().lower())

        raw = str(raw_name).strip().lower().strip(" .,:;!?")
        available = [str(x).strip().lower() for x in candidates if str(x).strip()]
        available_set = set(available)
        if raw in available_set:
            return raw

        mapped = self.map_to_standard_label(raw)
        if mapped in available_set:
            return mapped

        for name in available:
            if name in raw or raw in name:
                return name

        # token overlap
        raw_tokens = set(re.findall(r"[a-zA-Z0-9]+", raw))
        best_name = None
        best_score = 0.0
        for name in available:
            name_tokens = set(re.findall(r"[a-zA-Z0-9]+", name))
            if not name_tokens:
                continue
            score = len(raw_tokens & name_tokens) / len(name_tokens)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is not None and best_score >= 0.5:
            return best_name

        close = difflib.get_close_matches(raw, available, n=1, cutoff=0.45)
        if close:
            return close[0]
        close_mapped = difflib.get_close_matches(mapped, available, n=1, cutoff=0.45)
        if close_mapped:
            return close_mapped[0]

        best_name, best_score = self._best_semantic_match_in_candidates(raw, available)
        if best_name is not None and best_score >= 0.75:
            return best_name
        return "null"

    @staticmethod
    def _normalize_label_for_match(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return "".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> set:
        if not text:
            return set()
        if len(text) < n:
            return {text}
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def _semantic_label_similarity(self, raw_name: str, candidate: str) -> float:
        raw = str(raw_name).strip().lower()
        cand = str(candidate).strip().lower()
        if not raw or not cand:
            return 0.0
        if raw == cand:
            return 1.0

        raw_compact = self._normalize_label_for_match(raw)
        cand_compact = self._normalize_label_for_match(cand)
        if raw_compact and cand_compact and raw_compact == cand_compact:
            return 0.99

        seq_score = difflib.SequenceMatcher(None, raw, cand).ratio()
        seq_compact = difflib.SequenceMatcher(None, raw_compact, cand_compact).ratio() if raw_compact and cand_compact else 0.0

        raw_tokens = set(re.findall(r"[a-zA-Z0-9]+", raw))
        cand_tokens = set(re.findall(r"[a-zA-Z0-9]+", cand))
        token_jaccard = 0.0
        if raw_tokens and cand_tokens:
            inter = len(raw_tokens & cand_tokens)
            union = len(raw_tokens | cand_tokens)
            token_jaccard = inter / union if union else 0.0

        raw_grams = self._char_ngrams(raw_compact, 3)
        cand_grams = self._char_ngrams(cand_compact, 3)
        char_jaccard = 0.0
        if raw_grams and cand_grams:
            inter = len(raw_grams & cand_grams)
            union = len(raw_grams | cand_grams)
            char_jaccard = inter / union if union else 0.0

        base = max(
            0.55 * seq_compact + 0.25 * seq_score + 0.20 * char_jaccard,
            0.50 * seq_compact + 0.30 * char_jaccard + 0.20 * token_jaccard,
        )

        hint_boost = 0.0
        hint_map = self._region_object_ontology()
        canonical_region = self._resolve_region_phrase_alias(raw) or raw
        for key, hint_labels in hint_map.items():
            if (canonical_region == key or key in raw or key in canonical_region) and cand in hint_labels:
                hint_boost = 0.15
                break
        return min(1.0, base + hint_boost)

    def _best_semantic_match_in_candidates(self, raw_name: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        best_name: Optional[str] = None
        best_score: float = 0.0
        for cand in candidates:
            score = self._semantic_label_similarity(raw_name, cand)
            if score > best_score:
                best_score = score
                best_name = cand
        return best_name, best_score

    def _region_object_ontology(self) -> Dict[str, List[str]]:
        """Region object ontology."""
        return self.task_support_registry.get_region_object_ontology()

    def _heuristic_extract_from_question(
        self,
        task_type: str,
        question: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        """Heuristic extract from question."""
        return self.heuristic_task_parser.parse(
            task_type=task_type,
            question=question,
            candidate_labels=candidate_labels,
        )

    def _prefill_entities_from_question(self, context: Dict[str, Any]) -> None:
        """Prefill entities from question."""
        task_type = str(context.get("task_type", ""))
        question = str(context.get("question", ""))
        profile = self.task_support_registry.get_task_profile(task_type)
        has_prefilled_labels = self._has_prefilled_context_labels(task_type, context)
        has_prefilled_indices = False
        if profile is not None and profile.prefill_image_index_mode:
            if profile.prefill_image_index_mode == "pair":
                ref_idx = context.get("camera_reference_image_idx")
                tgt_idx = context.get("camera_target_image_idx")
                has_prefilled_indices = ref_idx in (1, 2) and tgt_idx in (1, 2) and int(ref_idx) != int(tgt_idx)
            else:
                ref_idx = context.get("camera_reference_image_idx")
                has_prefilled_indices = ref_idx in (1, 2)
        if has_prefilled_labels and (
            profile is None
            or not profile.prefill_image_index_mode
            or has_prefilled_indices
        ):
            return
        if profile is not None and profile.prefill_image_index_mode:
            self._prefill_image_indices_from_question(context, task_type, question)
        candidates = self._collect_extraction_candidate_labels(context)
        if not question:
            return
        allow_empty_candidates = bool(
            getattr(profile, "allow_heuristic_without_candidates", False)
        ) if profile is not None else False
        if (not candidates) and not allow_empty_candidates:
            return

        inferred = self._heuristic_extract_from_question(
            task_type=task_type,
            question=question,
            candidate_labels=candidates,
        )
        if not isinstance(inferred, dict):
            return

        for k, v in inferred.items():
            if k not in context or self._is_null_like_label(context.get(k)):
                if isinstance(v, list):
                    if len(v) > 0:
                        context[k] = v
                elif not self._is_null_like_label(v):
                    context[k] = v

    def _prefill_image_indices_from_question(self, context: Dict[str, Any], task_type: str, question: str) -> None:
        """Prefill image indices from question."""
        profile = self.task_support_registry.get_task_profile(task_type)
        image_index_mode = profile.prefill_image_index_mode if profile is not None else None
        if not image_index_mode:
            return

        existing_ref = context.get("camera_reference_image_idx")
        existing_tgt = context.get("camera_target_image_idx")
        if image_index_mode == "pair":
            if existing_ref in (1, 2) and existing_tgt in (1, 2) and int(existing_ref) != int(existing_tgt):
                context["camera_reference_image_idx"] = int(existing_ref)
                context["camera_target_image_idx"] = int(existing_tgt)
                context["question_reference_image_idx"] = int(existing_ref)
                context["question_target_image_idx"] = int(existing_tgt)
                return
        else:
            if existing_ref in (1, 2):
                context["camera_reference_image_idx"] = int(existing_ref)
                context["question_reference_image_idx"] = int(existing_ref)
                if existing_tgt in (1, 2):
                    context["question_target_image_idx"] = int(existing_tgt)
                return

        rule_ref, rule_tgt = self._parse_image_indices_rule(question, task_type)
        llm_ref, llm_tgt = self._parse_image_indices_llm(context, question, task_type)

        def pick(primary: Optional[int], fallback: Optional[int], default: int) -> int:
            if primary in (1, 2):
                return int(primary)
            if fallback in (1, 2):
                return int(fallback)
            return int(default)

        if image_index_mode == "pair":
            ref_idx = pick(llm_ref, rule_ref, 1)
            tgt_idx = pick(llm_tgt, rule_tgt, 2)
            if ref_idx == tgt_idx:

                if rule_ref in (1, 2) and rule_tgt in (1, 2) and int(rule_ref) != int(rule_tgt):
                    ref_idx, tgt_idx = int(rule_ref), int(rule_tgt)
                else:
                    ref_idx, tgt_idx = 1, 2
            context["camera_reference_image_idx"] = ref_idx
            context["camera_target_image_idx"] = tgt_idx
            context["question_reference_image_idx"] = ref_idx
            context["question_target_image_idx"] = tgt_idx
            return

        ref_default = 2
        ref_idx = pick(llm_ref, rule_ref, ref_default)
        context["camera_reference_image_idx"] = ref_idx
        context["question_reference_image_idx"] = ref_idx
        if llm_tgt in (1, 2):
            context["question_target_image_idx"] = int(llm_tgt)
        elif rule_tgt in (1, 2):
            context["question_target_image_idx"] = int(rule_tgt)

    @staticmethod
    def _parse_cam_cam_image_indices(question: str) -> Tuple[int, int]:
        """Parse cam cam image indices."""
        if not isinstance(question, str):
            return 1, 2
        q = question.lower()
        m = re.search(
            r"when\s+you\s+took\s+image\s*([12])\s*,?\s*where\s+was\s+the\s+camera\s+for\s+image\s*([12])",
            q,
        )
        if m:
            try:
                ref_idx = int(m.group(1))
                tgt_idx = int(m.group(2))
                if ref_idx in (1, 2) and tgt_idx in (1, 2) and ref_idx != tgt_idx:
                    return ref_idx, tgt_idx
            except (TypeError, ValueError):
                pass
        m2 = re.search(
            r"where\s+was\s+the\s+camera\s+for\s+image\s*([12])\s+when\s+you\s+took\s+image\s*([12])",
            q,
        )
        if m2:
            try:
                tgt_idx = int(m2.group(1))
                ref_idx = int(m2.group(2))
                if ref_idx in (1, 2) and tgt_idx in (1, 2) and ref_idx != tgt_idx:
                    return ref_idx, tgt_idx
            except (TypeError, ValueError):
                pass
        return 1, 2

    @staticmethod
    def _parse_cam_reference_image_idx(question: str, default: int = 2) -> int:
        """Parse cam reference image idx."""
        if not isinstance(question, str):
            return default
        q = question.lower()
        patterns = [
            r"when\s+i\s+am\s+taking\s+image\s*([12])",
            r"when\s+i[' ]?m\s+taking\s+image\s*([12])",
            r"when\s+you\s+are\s+taking\s+image\s*([12])",
            r"when\s+taking\s+image\s*([12])",
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx in (1, 2):
                        return idx
                except (TypeError, ValueError):
                    pass
        if "last image" in q or "image 2" in q:
            return 2
        if "first image" in q or "image 1" in q:
            return 1
        return default

    def _parse_image_indices_rule(self, question: str, task_type: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse image indices rule."""
        profile = self.task_support_registry.get_task_profile(task_type)
        parse_mode = (
            profile.image_index_parse_mode
            if profile is not None and profile.image_index_parse_mode
            else (profile.prefill_image_index_mode if profile is not None else None)
        )
        if parse_mode == "pair":
            ref, tgt = self._parse_cam_cam_image_indices(question)
            return int(ref), int(tgt)
        ref = self._parse_cam_reference_image_idx(question, default=2)
        return int(ref), None

    def _parse_image_indices_llm(
        self,
        context: Dict[str, Any],
        question: str,
        task_type: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Parse image indices llm."""
        vlm_tool = self.tools.get("vlm_tool")
        if vlm_tool is None:
            return None, None
        if not isinstance(question, str) or not question.strip():
            return None, None

        prompt = (
            "Role: Image index extractor.\n"
            "Task: extract image roles from question.\n"
            "Rules:\n"
            "- Output exactly one line in this format:\n"
            "reference_image=<1|2|null>;target_image=<1|2|null>\n"
            "- reference_image: observer image index (\"when taking Image X\").\n"
            "- target_image: queried image index if explicitly asked (e.g., \"camera for Image Y\").\n"
            "- If not explicit, output null for that slot.\n"
            f"Task type: {task_type}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        try:
            vlm_cfg = self.config.get("vlm", {})
            resp = vlm_tool.execute(
                prompt=prompt,
                image_paths=[],
                max_tokens=int(vlm_cfg.get("max_tokens", 20480)),
                temperature=float(vlm_cfg.get("extraction_temperature", vlm_cfg.get("temperature", 0.0))),
                use_vision=False,
            )
            context["_extra_llm_calls"] = int(context.get("_extra_llm_calls", 0)) + 1
            if not isinstance(resp, str) or (not resp.strip()):
                return None, None
            text = resp.strip().splitlines()[0].strip().lower()
            if text.startswith("error:"):
                return None, None
            ref = None
            tgt = None
            m_ref = re.search(r"reference_image\s*=\s*(1|2|null)", text)
            m_tgt = re.search(r"target_image\s*=\s*(1|2|null)", text)
            if m_ref and m_ref.group(1) in {"1", "2"}:
                ref = int(m_ref.group(1))
            if m_tgt and m_tgt.group(1) in {"1", "2"}:
                tgt = int(m_tgt.group(1))
            return ref, tgt
        except Exception:
            return None, None
    
    def _collect_labels_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Collect labels from context."""
        labels = []
        task_type = str(context.get("task_type", ""))
        profile = self.task_support_registry.get_task_profile(task_type)
        label_fields = (
            list(profile.context_label_fields)
            if profile is not None and profile.context_label_fields
            else self.task_support_registry.get_expected_extraction_fields(task_type)
        )

        for field_name in label_fields:
            if field_name not in context:
                continue
            value = context[field_name]
            if isinstance(value, list):
                labels.extend(value)
            else:
                labels.append(value)

        return labels

    @staticmethod
    def _extract_frame_id_from_image_path(image_path: str) -> Optional[str]:
        """Extract frame id from image path."""
        if not isinstance(image_path, str) or not image_path:
            return None
        filename = Path(image_path).name
        match = re.search(r"(\d+)(?:\.[A-Za-z0-9]+)?$", filename)
        if match:
            return match.group(1)
        return None

    def _inject_frame_context(self, context: Dict[str, Any]) -> None:
        """Inject frame context."""
        profile = self.task_support_registry.get_task_profile(str(context.get("task_type", "") or ""))
        if profile is not None and str(getattr(profile, "task_group", "") or "").strip().lower() == "scene":
            return
        if "frame_metadata" in context:
            return
        image_path = context.get("image_path")
        scene_id = context.get("scene_id")
        if not scene_id:
            return
        frame_id = context.get("frame_id")
        if frame_id is None and image_path:
            frame_id = self._extract_frame_id_from_image_path(image_path)
            if frame_id is not None:
                context["frame_id"] = frame_id
        if frame_id is None:
            return
        metadata_dir = context.get("metadata_dir")
        frame_type = context.get("frame_type", "frame_processed")
        frame_metadata = self.scannet_loader.load_frame_metadata(
            scene_id=scene_id,
            frame_id=str(frame_id),
            frame_type=frame_type,
            metadata_dir=metadata_dir,
        )
        context["frame_metadata"] = frame_metadata

    def _inject_image_pair_context(self, context: Dict[str, Any]) -> None:
        """Inject image pair context."""
        image_paths = context.get("image_paths")
        if not (isinstance(image_paths, list) and len(image_paths) >= 2):
            p1 = context.get("image_path_1")
            p2 = context.get("image_path_2")
            if isinstance(p1, str) and isinstance(p2, str):
                image_paths = [p1, p2]
                context["image_paths"] = image_paths
        if not (isinstance(image_paths, list) and len(image_paths) >= 2):
            return

        scene_id = context.get("scene_id")
        if "scene_data_path" not in context and isinstance(scene_id, str) and scene_id:
            context["scene_data_path"] = str(Path(self.scannet_loader.scannet_root) / scene_id)
        if "metadata_dir" not in context:
            context["metadata_dir"] = str(Path(self.scannet_loader.scannet_root).parent / "metadata")

        if "frame_id_1" not in context:
            frame_id_1 = self._extract_frame_id_from_image_path(image_paths[0])
            if frame_id_1 is not None:
                context["frame_id_1"] = frame_id_1
        if "frame_id_2" not in context:
            frame_id_2 = self._extract_frame_id_from_image_path(image_paths[1])
            if frame_id_2 is not None:
                context["frame_id_2"] = frame_id_2

        frame_type = context.get("frame_type", "frame_processed")
        metadata_dir = context.get("metadata_dir")
        if "frame_metadata_1" not in context and "frame_id_1" in context:
            context["frame_metadata_1"] = self.scannet_loader.load_frame_metadata(
                scene_id=scene_id,
                frame_id=str(context["frame_id_1"]),
                frame_type=frame_type,
                metadata_dir=metadata_dir,
            )
        if "frame_metadata_2" not in context and "frame_id_2" in context:
            context["frame_metadata_2"] = self.scannet_loader.load_frame_metadata(
                scene_id=scene_id,
                frame_id=str(context["frame_id_2"]),
                frame_type=frame_type,
                metadata_dir=metadata_dir,
            )
        if "frame_metadata" not in context and "frame_metadata_2" in context:
            context["frame_metadata"] = context["frame_metadata_2"]
        if context.get("task_type") == "attribute_measurement" and "measurement_type" not in context:
            context["measurement_type"] = "compare_longer"
    
    def _update_context_from_result(
        self,
        step,
        result: Any,
        context: Dict[str, Any],
        parsed_params: Dict[str, Any],
        task
    ) -> bool:
        """Update context from result."""
        extraction_used = False
        

        if step.tool_name == "vlm_tool" and isinstance(result, str):
            context["_vlm_extraction_done"] = True
            context["_vlm_extraction_raw"] = result
            parse_input = result
            transport_error = str(result).strip().lower().startswith("error:")
            if transport_error:
                parse_input = "null"
            extracted = task.parse_extracted_params(parse_input, self.map_to_standard_label)
            if not isinstance(extracted, dict):
                extracted = {}

            defaults = self._expected_extraction_defaults(str(context.get("task_type", "")))
            for key, default_val in defaults.items():
                if key not in extracted:
                    extracted[key] = default_val
            task_type = str(context.get("task_type", ""))
            fallback = self._heuristic_extract_from_question(
                task_type=task_type,
                question=str(context.get("question", "")),
                candidate_labels=self._collect_extraction_candidate_labels(context),
            )
            if isinstance(fallback, dict):
                for key, val in fallback.items():
                    current = extracted.get(key)
                    needs_fill = (
                        key not in extracted
                        or (isinstance(current, list) and len(current) == 0)
                        or (transport_error and self._is_null_like_label(current))
                    )
                    if needs_fill:
                        extracted[key] = val
            extracted = self._sanitize_extracted_by_candidates(extracted, context)

            if extracted:
                merged = {}
                for key, value in extracted.items():
                    existing = context.get(key)

                    if (key in context) and (not self._is_null_like_label(existing)) and self._is_null_like_label(value):
                        merged[key] = existing

                    elif (
                        key == "region_name"
                        and isinstance(existing, str)
                        and self._is_abstract_region_phrase(existing)
                        and isinstance(value, str)
                        and (not self._is_abstract_region_phrase(value))
                    ):
                        merged[key] = existing
                    else:
                        merged[key] = value
                context.update(merged)
                parsed_params.update(merged)
                extraction_used = True
        

        elif isinstance(result, dict):

            if "answer" in result:
                context["final_answer"] = result["answer"]
            

            for key, value in result.items():
                if key not in ["success", "error"]:
                    context[f"step_{step.step_id}_{key}"] = value
                    context[key] = value

            task_type = context.get("task_type", "")
            profile = self.task_support_registry.get_task_profile(task_type)
            detection_result_binding = profile.detection_result_binding if profile is not None else "default"
            if detection_result_binding == "pick_single_target":
                detections = result.get("detections", [])
                if isinstance(detections, list):
                    picked = self._select_cam_obj_target_entity(context=context, detections=detections)
                    if isinstance(picked, dict):
                        context["target_entity"] = picked

            if detection_result_binding == "bind_measurement_pair":
                object1_label = str(context.get("object1_label", "")).lower()
                object2_label = str(context.get("object2_label", "")).lower()
                detections = result.get("detections", [])
                if isinstance(detections, list):
                    matched_by_name = defaultdict(list)
                    for det in detections:
                        name = str(det.get("name", "")).lower()
                        if name:
                            matched_by_name[name].append(det)
                    if object1_label and len(matched_by_name.get(object1_label, [])) == 1:
                        context["entity1"] = matched_by_name[object1_label][0]
                    if object2_label and len(matched_by_name.get(object2_label, [])) == 1:
                        context["entity2"] = matched_by_name[object2_label][0]
                if "entity1" not in context and object1_label:
                    selected = self._select_measurement_entity(context, object1_label)
                    if isinstance(selected, dict):
                        context["entity1"] = selected
                if "entity2" not in context and object2_label:
                    selected = self._select_measurement_entity(context, object2_label)
                    if isinstance(selected, dict):
                        context["entity2"] = selected
        return extraction_used

    def map_to_standard_label(self, raw_text: str) -> str:
        """Map to standard label."""
        text = raw_text.lower().strip()
        exact_match_labels = self.VALID_LABELS | self.EXACT_ONLY_LABELS
        

        if text in exact_match_labels:
            return text
        if "null" in text:
            return "null"

        words = re.findall(r'\b\w+\b', text)
        stop_words = {'a', 'an', 'the', 'some', 'many', 'of', 'those', 'these', 'it', 'is'}
        clean_words = [w for w in words if w not in stop_words]

        compact_text = self._normalize_label_for_match(text)
        if compact_text:
            for label in exact_match_labels:
                if self._normalize_label_for_match(label) == compact_text:
                    return label
        

        clean_text = " ".join(clean_words)
        for label in exact_match_labels:
            if " " in label and label in clean_text:
                return label

        for word in clean_words:
            if word in exact_match_labels:
                return word
            

            stem = None
            if word.endswith('s'):
                if word.endswith('es') and word[:-2] in exact_match_labels: stem = word[:-2]
                elif word[:-1] in exact_match_labels: stem = word[:-1]
            if stem: return stem
            

            close = difflib.get_close_matches(word, list(self.VALID_LABELS), n=1, cutoff=0.8)
            if close:
                return close[0]

        matches = difflib.get_close_matches(text, list(self.VALID_LABELS), n=1, cutoff=0.78)
        return matches[0] if matches else text

    def _build_region_positions_from_context(self, context: Dict[str, Any]) -> Dict[str, List[float]]:
        """Build region positions from context."""
        scene_metadata = context.get("scene_metadata", {})
        if not isinstance(scene_metadata, dict):
            return {}

        min_visibility = float(context.get("min_visibility", self._get_visibility_floor()))
        labels = set()
        frame_keys = ("frame_metadata_1", "frame_metadata_2", "frame_metadata")
        if str(context.get("task_type", "")).strip() == "position_cam_reg":
            ref_idx = context.get("camera_reference_image_idx")
            if ref_idx == 1:
                frame_keys = ("frame_metadata_1",)
            elif ref_idx == 2:
                frame_keys = ("frame_metadata_2",)
        for key in frame_keys:
            frame_meta = context.get(key)
            objects = frame_meta.get("objects", []) if isinstance(frame_meta, dict) else []
            for obj in objects or []:
                try:
                    vis = float(obj.get("visibility", 0.0))
                except (TypeError, ValueError):
                    vis = 0.0
                if vis < min_visibility:
                    continue
                label = str(obj.get("label", "")).strip().lower()
                if label and self._is_usable_anchor_label(label):
                    labels.add(label)

        if not labels:
            return {}

        grouped: Dict[str, List[List[float]]] = {}
        for obj in scene_metadata.get("objects", []) or []:
            label = str(obj.get("label", "")).strip().lower()
            if (label not in labels) or (not self._is_usable_anchor_label(label)):
                continue
            loc = obj.get("3d_location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 3:
                continue
            try:
                xyz = [float(loc[0]), float(loc[1]), float(loc[2])]
            except (TypeError, ValueError):
                continue
            grouped.setdefault(label, []).append(xyz)

        region_positions: Dict[str, List[float]] = {}
        for label, positions in grouped.items():
            arr = np.array(positions, dtype=float)
            region_positions[label] = arr.mean(axis=0).tolist()
        return region_positions

    def _resolve_region_anchor_name(self, raw_region: str, available_names: List[str]) -> str:
        """Resolve region anchor name."""
        if not available_names:
            return str(raw_region).strip().lower()
        region = str(raw_region).strip().lower()
        canonical_region = self._resolve_region_phrase_alias(region) or region
        available = [str(x).strip().lower() for x in available_names]
        available_set = set(available)

        if region in available_set:
            return region
        if canonical_region in available_set:
            return canonical_region

        mapped = self.map_to_standard_label(region)
        if mapped in available_set:
            return mapped
        canonical_mapped = self.map_to_standard_label(canonical_region)
        if canonical_mapped in available_set:
            return canonical_mapped

        hint_map = self._region_object_ontology()
        canonical_priority = hint_map.get(canonical_region, [])
        for name in canonical_priority:
            if name in available_set:
                return name

        for key, priority_names in hint_map.items():
            if key not in region and key not in canonical_region:
                continue
            for name in priority_names:
                if name in available_set:
                    return name

        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9]+", f"{region} {canonical_region}")
            if t not in {"area", "region", "zone", "room"}
        ]
        for token in tokens:
            token_mapped = self.map_to_standard_label(token)
            if token_mapped in available_set:
                return token_mapped

        for name in available:
            if name in region or region in name:
                return name

        best_name, best_score = self._best_semantic_match_in_candidates(canonical_region, available)
        if best_name is not None and best_score >= 0.55:
            return best_name

        close = difflib.get_close_matches(canonical_region, available, n=1, cutoff=0.45)
        if close:
            return close[0]
        return "null"

    def _resolve_region_anchor_name_with_llm(
        self,
        context: Dict[str, Any],
        raw_region: str,
        available_names: List[str],
    ) -> str:
        """Resolve region anchor name with llm."""
        default_choice = self._resolve_region_anchor_name(raw_region, available_names)
        if not isinstance(raw_region, str):
            return default_choice
        raw = raw_region.strip().lower()
        if raw in {"", "null", "none"}:
            return default_choice
        canonical_region = self._resolve_region_phrase_alias(raw) or raw

        available = [str(x).strip().lower() for x in available_names]
        if raw in set(available):
            return raw

        vlm_tool = self.tools.get("vlm_tool")
        if vlm_tool is None:
            return default_choice

        question = str(context.get("question", "")).strip()
        image_paths = context.get("image_paths")
        candidates = ", ".join(sorted(available))
        prompt = (
            "Role: Spatial region-to-object mapper.\n"
            "Task: map the region phrase to ONE best object anchor from candidates.\n"
            "Rules:\n"
            "- Output exactly one object label from candidates.\n"
            "- No explanation.\n"
            f"Question: {question}\n"
            f"Region phrase: {raw}\n"
            f"Canonical region concept: {canonical_region}\n"
            f"Candidates: {candidates}\n"
            "Answer:"
        )
        try:
            vlm_cfg = self.config.get("vlm", {})
            response = vlm_tool.execute(
                prompt=prompt,
                image_paths=image_paths if isinstance(image_paths, list) else [],
                max_tokens=int(vlm_cfg.get("max_tokens", 20480)),
                temperature=float(vlm_cfg.get("extraction_temperature", vlm_cfg.get("temperature", 0.0))),
                use_vision=False,
            )
            context["_extra_llm_calls"] = int(context.get("_extra_llm_calls", 0)) + 1
            if isinstance(response, str) and response.strip():
                if response.strip().lower().startswith("error:"):
                    return default_choice
                line = response.strip().splitlines()[0].strip().lower().strip(" .,:;!?")
                if line in set(available):

                    sim = self._semantic_label_similarity(canonical_region, line)
                    if sim >= 0.60 or self._is_abstract_region_phrase(raw):
                        return line
                mapped = self._resolve_region_anchor_name(line, available_names)
                return mapped if mapped != "null" else default_choice
        except Exception:
            return default_choice

        return default_choice

    def _infer_anchor_labels_from_question(
        self,
        question: str,
        available_names: List[str],
        max_labels: int = 2,
    ) -> List[str]:
        """Infer anchor labels from question."""
        if not isinstance(question, str) or not question.strip():
            return []
        available = [str(x).strip().lower() for x in available_names]
        available_set = set(available)
        q = question.lower()
        found: List[str] = []

        for name in available:
            if name and name in q and name not in found:
                found.append(name)
                if len(found) >= max_labels:
                    return found

        tokens = re.findall(r"[a-zA-Z0-9]+", q)
        for t in tokens:
            if t in {"the", "a", "an", "area", "region", "relative", "where", "is"}:
                continue
            if self._is_abstract_region_phrase(t):
                continue
            mapped = self.map_to_standard_label(t)
            if mapped in available_set and mapped not in found:
                found.append(mapped)
                if len(found) >= max_labels:
                    return found
        if len(found) < max_labels:
            best_name, best_score = self._best_semantic_match_in_candidates(q, available)
            if best_name is not None and best_score >= 0.50 and best_name not in found:
                found.append(best_name)
        return found[:max_labels]

    def _extract_final_answer(self, step_results: List[Dict]) -> str:
        """Extract final answer."""
        if not step_results:
            return ""
        

        last_result = step_results[-1].get("result", {})
        
        if isinstance(last_result, dict) and "answer" in last_result:
            return last_result["answer"]
        elif isinstance(last_result, str):
            return last_result
        
        return ""

    @staticmethod
    def _resolve_frame_ids(
        image_paths: Optional[List[str]] = None,
        frame_ids: Optional[List[Any]] = None
    ) -> List[str]:
        """Resolve frame ids."""
        resolved: List[str] = []
        if isinstance(frame_ids, list):
            for item in frame_ids:
                if item is None:
                    continue
                frame_id = str(item).strip()
                if frame_id:
                    resolved.append(frame_id)
        if isinstance(image_paths, list):
            for path in image_paths:
                extracted = WorldSimulator._extract_frame_id_from_image_path(path)
                if extracted is not None:
                    resolved.append(extracted)
        deduped: List[str] = []
        seen = set()
        for fid in resolved:
            if fid not in seen:
                seen.add(fid)
                deduped.append(fid)
        return deduped

    @staticmethod
    def _label_count_from_objects(objects: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in objects:
            label = obj.get("label")
            if not isinstance(label, str) or not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        return counts

    def get_scene_summary(
        self,
        scene_id: str,
        metadata_dir: Optional[str] = None,
        include_objects: bool = False,
    ) -> Dict[str, Any]:
        """Get scene summary."""
        scene_metadata = self.scannet_loader.load_scene_metadata(
            scene_id=scene_id,
            metadata_dir=metadata_dir,
        )
        objects = scene_metadata.get("objects", []) or []
        label_counts = self._label_count_from_objects(objects)
        labels = sorted(label_counts.keys())
        room_size = scene_metadata.get("room_size")

        result = {
            "response_version": self.RESPONSE_VERSION,
            "success": True,
            "summary_type": "scene",
            "scene_id": scene_id,
            "total_objects": len(objects),
            "unique_label_count": len(labels),
            "labels": labels,
            "label_counts": label_counts,
            "room_size": room_size,
        }
        if include_objects:
            result["objects"] = objects
        return result

    def get_single_image_summary(
        self,
        scene_id: str,
        *,
        image_path: Optional[str] = None,
        frame_id: Optional[Any] = None,
        frame_type: str = "frame_processed",
        metadata_dir: Optional[str] = None,
        min_visibility: Optional[float] = None,
        include_objects: bool = False
    ) -> Dict[str, Any]:
        """Get single image summary."""
        resolved_frame_id = str(frame_id).strip() if frame_id is not None else None
        if not resolved_frame_id and image_path:
            resolved_frame_id = self._extract_frame_id_from_image_path(image_path)
        if not resolved_frame_id:
            return {
                "response_version": self.RESPONSE_VERSION,
                "success": False,
                "summary_type": "single_image",
                "scene_id": scene_id,
                "error_code": self.ERROR_CODES["INVALID_INPUT"],
                "error": "single_image summary requires `frame_id` or an `image_path` with a resolvable frame_id",
            }

        visibility_floor = self._get_visibility_floor()
        if min_visibility is None:
            effective_min = visibility_floor
        else:
            try:
                effective_min = max(visibility_floor, float(min_visibility))
            except (TypeError, ValueError):
                effective_min = visibility_floor

        frame_metadata = self.scannet_loader.load_frame_metadata(
            scene_id=scene_id,
            frame_id=resolved_frame_id,
            frame_type=frame_type,
            metadata_dir=metadata_dir,
        )
        all_objects = frame_metadata.get("objects", []) or []
        visible_objects = self.scannet_loader.get_visible_objects(
            frame_metadata=frame_metadata,
            min_visibility=effective_min,
        )
        visible_label_counts = self._label_count_from_objects(visible_objects)
        visible_labels = sorted(visible_label_counts.keys())

        result = {
            "response_version": self.RESPONSE_VERSION,
            "success": True,
            "summary_type": "single_image",
            "scene_id": scene_id,
            "image_path": image_path,
            "frame_id": str(resolved_frame_id),
            "frame_type": frame_type,
            "min_visibility": effective_min,
            "total_objects_in_frame": len(all_objects),
            "visible_objects_count": len(visible_objects),
            "visible_unique_label_count": len(visible_labels),
            "visible_labels": visible_labels,
            "visible_label_counts": visible_label_counts,
        }
        if include_objects:
            result["visible_objects"] = visible_objects
        return result

    def get_multi_image_summary(
        self,
        scene_id: str,
        *,
        image_paths: Optional[List[str]] = None,
        frame_ids: Optional[List[Any]] = None,
        frame_type: str = "frame_processed",
        metadata_dir: Optional[str] = None,
        min_visibility: Optional[float] = None,
        include_objects: bool = False,
    ) -> Dict[str, Any]:
        """Get multi image summary."""
        resolved_frame_ids = self._resolve_frame_ids(image_paths=image_paths, frame_ids=frame_ids)
        if not resolved_frame_ids:
            return {
                "response_version": self.RESPONSE_VERSION,
                "success": False,
                "summary_type": "multi_image",
                "scene_id": scene_id,
                "error_code": self.ERROR_CODES["INVALID_INPUT"],
                "error": "multi_image summary requires `image_paths` or `frame_ids`",
            }

        frame_summaries: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        union_counts: Dict[str, int] = {}
        union_labels = set()
        intersection_labels = None

        image_path_map: Dict[str, str] = {}
        if isinstance(image_paths, list):
            for path in image_paths:
                fid = self._extract_frame_id_from_image_path(path)
                if fid:
                    image_path_map[fid] = path

        for fid in resolved_frame_ids:
            try:
                single = self.get_single_image_summary(
                    scene_id=scene_id,
                    image_path=image_path_map.get(fid),
                    frame_id=fid,
                    frame_type=frame_type,
                    metadata_dir=metadata_dir,
                    min_visibility=min_visibility,
                    include_objects=include_objects,
                )
            except Exception as exc:
                errors.append({"frame_id": str(fid), "error": str(exc)})
                continue
            if not single.get("success", False):
                errors.append(
                    {
                        "frame_id": str(fid),
                        "error": single.get("error", "unknown error"),
                    }
                )
                continue
            frame_summaries.append(single)
            labels = set(single.get("visible_labels", []))
            union_labels |= labels
            if intersection_labels is None:
                intersection_labels = set(labels)
            else:
                intersection_labels &= labels
            for label, cnt in single.get("visible_label_counts", {}).items():
                union_counts[label] = union_counts.get(label, 0) + int(cnt)

        if intersection_labels is None:
            intersection_labels = set()

        return {
            "response_version": self.RESPONSE_VERSION,
            "success": len(frame_summaries) > 0,
            "summary_type": "multi_image",
            "scene_id": scene_id,
            "requested_frames": [str(fid) for fid in resolved_frame_ids],
            "num_frames_requested": len(resolved_frame_ids),
            "num_frames_succeeded": len(frame_summaries),
            "num_frames_failed": len(errors),
            "frame_summaries": frame_summaries,
            "aggregate": {
                "union_visible_labels": sorted(union_labels),
                "intersection_visible_labels": sorted(intersection_labels),
                "sum_visible_label_counts": dict(sorted(union_counts.items())),
            },
            "errors": errors,
        }

    def get_environment_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get environment summary."""
        if not isinstance(input_data, dict):
            return {
                "response_version": self.RESPONSE_VERSION,
                "success": False,
                "error_code": self.ERROR_CODES["INVALID_INPUT"],
                "error": "`input_data` must be a dict",
            }

        scene_id = input_data.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            return {
                "response_version": self.RESPONSE_VERSION,
                "success": False,
                "error_code": self.ERROR_CODES["INVALID_INPUT"],
                "error": "Missing `scene_id`",
            }

        summary_type_raw = input_data.get("summary_type", "scene")
        summary_resolution = self.summary_type_resolver.resolve(summary_type_raw)
        summary_type = summary_resolution.summary_type
        if summary_type is None:
            return {
                "response_version": self.RESPONSE_VERSION,
                "success": False,
                "scene_id": scene_id,
                "error_code": self.ERROR_CODES["INVALID_SUMMARY_TYPE"],
                "error": summary_resolution.error,
            }

        try:
            if summary_type == "scene":
                return self.get_scene_summary(
                    scene_id=scene_id,
                    metadata_dir=input_data.get("metadata_dir"),
                    include_objects=bool(input_data.get("include_objects", False)),
                )
            if summary_type == "single_image":
                return self.get_single_image_summary(
                    scene_id=scene_id,
                    image_path=input_data.get("image_path"),
                    frame_id=input_data.get("frame_id"),
                    frame_type=input_data.get("frame_type", "frame_processed"),
                    metadata_dir=input_data.get("metadata_dir"),
                    min_visibility=input_data.get("min_visibility"),
                    include_objects=bool(input_data.get("include_objects", False)),
                )
            return self.get_multi_image_summary(
                scene_id=scene_id,
                image_paths=input_data.get("image_paths"),
                frame_ids=input_data.get("frame_ids"),
                frame_type=input_data.get("frame_type", "frame_processed"),
                metadata_dir=input_data.get("metadata_dir"),
                min_visibility=input_data.get("min_visibility"),
                include_objects=bool(input_data.get("include_objects", False)),
            )
        except Exception as exc:
            return {
                "response_version": self.RESPONSE_VERSION,
                "success": False,
                "summary_type": summary_type,
                "scene_id": scene_id,
                "error_code": self.ERROR_CODES["EXECUTION_FAILED"],
                "error": str(exc),
            }
    
    def get_supported_tasks(self) -> List[str]:
        """Get supported tasks."""
        return sorted(self.task_registry.keys())

    def get_supported_summary_types(self, include_aliases: bool = False) -> Any:
        """Get supported summary types."""
        if include_aliases:
            return dict(self.summary_type_resolver.get_alias_mapping())
        return sorted(set(self.summary_type_resolver.get_alias_mapping().values()))

    def get_supported_task_groups(self, include_aliases: bool = False) -> Any:
        """Get supported task groups."""
        if include_aliases:
            return {
                alias: task_group
                for alias, task_group in self.task_group_resolver.get_alias_mapping().items()
                if not is_legacy_benchmark_name(alias)
            }
        return self.task_support_registry.get_supported_task_groups()

    def resolve_task_type_name(self, input_type: Any) -> Dict[str, Any]:
        """Resolve task type name."""
        resolution = self._resolve_task_type(input_type)
        return {
            "input": input_type,
            "normalized_input": resolution.normalized_input,
            "supported": resolution.task_type is not None,
            "task_type": resolution.task_type,
            "match_type": resolution.match_type,
            "rejection_reason": resolution.rejection_reason,
        }

    def resolve_task_group_name(self, task_group: Any) -> Dict[str, Any]:
        """Resolve task group name."""
        resolution = self.task_group_resolver.resolve(task_group)
        return {
            "input": task_group,
            "normalized_input": resolution.normalized_input,
            "supported": resolution.task_group is not None,
            "task_group": resolution.task_group,
            "match_type": resolution.match_type,
            "error": resolution.error,
        }

    def resolve_summary_type_name(self, summary_type: Any) -> Dict[str, Any]:
        """Resolve summary type name."""
        resolution = self.summary_type_resolver.resolve(summary_type)
        return {
            "input": summary_type,
            "normalized_input": resolution.normalized_input,
            "supported": resolution.summary_type is not None,
            "summary_type": resolution.summary_type,
            "match_type": resolution.match_type,
            "error": resolution.error,
        }

    def supports_task_type(self, input_type: Any) -> bool:
        """Supports task type."""
        resolution = self._resolve_task_type(input_type)
        return resolution.task_type is not None

    def get_task_alias_mapping(self) -> Dict[str, str]:
        """Get task alias mapping."""
        return {
            alias: task_type
            for alias, task_type in self.task_resolver.get_alias_mapping().items()
            if not is_legacy_benchmark_name(alias)
        }

    def get_unsupported_task_aliases(self) -> Dict[str, str]:
        """Get unsupported task aliases."""
        return dict(self.task_resolver.get_rejection_rules())

    def get_task_profile(self, task_type: Any) -> Dict[str, Any]:
        """Get task profile."""
        resolution = self._resolve_task_type(task_type)
        canonical = resolution.task_type if resolution.task_type is not None else task_type
        profile = self.task_support_registry.get_task_profile(canonical)
        if profile is None:
            return {
                "supported": False,
                "input": task_type,
                "error": resolution.rejection_reason or f"Unsupported task type: {task_type}",
            }

        return {
            "supported": True,
            "task_type": profile.task_type,
            "task_group": self.task_support_registry.get_task_group(profile.task_type),
            "task_family": self.task_support_registry.get_task_group(profile.task_type),
            "family": self.task_support_registry.get_task_group(profile.task_type),
            "input_mode": profile.input_mode,
            "aliases": sorted([
                alias for alias in profile.aliases
                if not is_legacy_benchmark_name(alias)
            ]),
            "extraction_defaults": deepcopy(profile.extraction_defaults),
            "expected_extraction_fields": list(profile.expected_extraction_fields),
            "context_label_fields": list(profile.context_label_fields),
            "candidate_label_scope": profile.candidate_label_scope,
            "allow_heuristic_without_candidates": profile.allow_heuristic_without_candidates,
            "has_heuristic_extractor": callable(profile.heuristic_extractor),
            "heuristic_policy": (
                {
                    "strategy": profile.heuristic_policy.strategy,
                    "target_field": profile.heuristic_policy.target_field,
                    "enable_relative_to_me_fallback": profile.heuristic_policy.enable_relative_to_me_fallback,
                    "allow_abstract_region_subject": profile.heuristic_policy.allow_abstract_region_subject,
                }
                if profile.heuristic_policy is not None else None
            ),
            "same_label_constraints": [
                {
                    "left_field": constraint.left_field,
                    "right_field": constraint.right_field,
                    "error_message": constraint.error_message,
                }
                for constraint in profile.same_label_constraints
            ],
            "distinct_field_group_constraints": [
                {
                    "field_names": list(constraint.field_names),
                    "error_message": constraint.error_message,
                }
                for constraint in profile.distinct_field_group_constraints
            ],
            "ambiguity_object_fields": [
                {"field_name": field_role.field_name, "role_name": field_role.role_name}
                for field_role in profile.ambiguity_object_fields
            ],
            "ambiguity_region_fields": [
                {"field_name": field_role.field_name, "role_name": field_role.role_name}
                for field_role in profile.ambiguity_region_fields
            ],
            "object_detection_policy": (
                {
                    "frame_source": profile.object_detection_policy.frame_source,
                    "include_scene_metadata": profile.object_detection_policy.include_scene_metadata,
                    "use_camera_location": profile.object_detection_policy.use_camera_location,
                    "target_label_fields": list(profile.object_detection_policy.target_label_fields),
                }
                if profile.object_detection_policy is not None else None
            ),
            "detection_result_binding": profile.detection_result_binding,
            "spatial_relation_policy": (
                {
                    "binding": profile.spatial_relation_policy.binding,
                    "default_reference_frame": profile.spatial_relation_policy.default_reference_frame,
                }
                if profile.spatial_relation_policy is not None else None
            ),
            "measurement_default": profile.measurement_default,
            "camera_pair_answer_mode": profile.camera_pair_answer_mode,
            "prefill_image_index_mode": profile.prefill_image_index_mode,
            "image_index_parse_mode": profile.image_index_parse_mode,
            "region_anchor_infer_max_labels": profile.region_anchor_infer_max_labels,
        }

    def get_task_info(self, task_type: Any) -> Dict[str, Any]:
        """Get task info."""
        resolution = self._resolve_task_type(task_type)
        if resolution.task_type is None:
            return {
                "supported": False,
                "input": task_type,
                "normalized_input": resolution.normalized_input,
                "match_type": resolution.match_type,
                "error": resolution.rejection_reason or f"Unsupported task type: {task_type}",
            }

        canonical = resolution.task_type
        task_class = self.task_registry.get(canonical)
        rubric_class = self.rubric_registry.get(canonical)
        task = task_class() if callable(task_class) else task_class
        rubric = rubric_class() if callable(rubric_class) else rubric_class
        task_cfg = self.config.get("tasks", {}).get(canonical, {})
        raw_task_info = task.get_info() if task is not None else {}
        public_task_info = dict(raw_task_info)
        public_task_info["task_id"] = canonical
        public_task_info["task_alias_ids"] = [
            alias for alias in raw_task_info.get("task_alias_ids", [])
            if not is_legacy_benchmark_name(alias)
        ]
        public_task_info["task_group"] = self.task_support_registry.get_task_group(canonical) or "custom"
        alias_mapping = self.get_task_alias_mapping()
        aliases = sorted([
            alias for alias, mapped_task in alias_mapping.items()
            if mapped_task == canonical
        ])
        task_group = self.task_support_registry.get_task_group(canonical) or "custom"

        rubric_steps = []
        raw_rubric_info = rubric.get_info() if rubric is not None else {}
        public_rubric_info = dict(raw_rubric_info)
        if rubric is not None:
            public_rubric_info["rubric_id"] = raw_rubric_info.get("rubric_id", f"rubric_{canonical}")
            public_rubric_info["task_id"] = canonical
            public_rubric_info["rubric_alias_ids"] = [
                alias for alias in raw_rubric_info.get("rubric_alias_ids", [])
                if not is_legacy_benchmark_name(alias)
            ]
            public_rubric_info["task_alias_ids"] = [
                alias for alias in raw_rubric_info.get("task_alias_ids", [])
                if not is_legacy_benchmark_name(alias)
            ]
        if rubric is not None:
            for step in rubric.get_steps():
                rubric_steps.append(
                    {
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "description": step.description,
                        "required_params": list(step.required_params),
                        "optional_params": list(step.optional_params or []),
                    }
                )

        return {
            "supported": True,
            "input": task_type,
            "resolved_task_type": canonical,
            "match_type": resolution.match_type,
            "task_group": task_group,
            "task_family": task_group,
            "enabled": bool(task_cfg.get("enabled", True)),
            "task_profile": self.get_task_profile(canonical),
            "task_info": public_task_info,
            "rubric_info": public_rubric_info,
            "aliases": aliases,
            "required_tools": task.get_required_tools() if task is not None else [],
            "rubric_steps": rubric_steps,
            "extraction_defaults": self.task_support_registry.get_extraction_defaults(canonical),
            "expected_extraction_fields": self.task_support_registry.get_expected_extraction_fields(canonical),
            "difficulty_config": task_cfg.get("difficulty", {}),
        }

    def get_task_catalog(
        self,
        task_group: Optional[str] = None,
        enabled_only: bool = False,
        task_family: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get task catalog."""
        selected_group = task_group if task_group is not None else task_family
        if isinstance(selected_group, str) and selected_group.strip():
            group_resolution = self.task_group_resolver.resolve(selected_group)
            if group_resolution.task_group is None:
                return []
            normalized_group = group_resolution.task_group
        else:
            normalized_group = None
        catalog = []
        for task_type in self.get_supported_tasks():
            info = self.get_task_info(task_type)
            if not info.get("supported", False):
                continue
            if enabled_only and (not info.get("enabled", True)):
                continue
            if normalized_group and info.get("task_group") != normalized_group:
                continue
            catalog.append(info)
        return catalog

    def get_interface_overview(self) -> Dict[str, Any]:
        """Get interface overview."""
        return {
            "core_interfaces": {
                "validate_and_answer": {
                    "description": "Validate a task instance and return validity, answer, and execution details from scene, single-image, image-pair, or sequence inputs",
                    "required_inputs": ["task_type", "scene_id", "question"],
                    "optional_inputs": [
                        "image_path",
                        "image_paths",
                        "image_path_1",
                        "image_path_2",
                        "frame_ids",
                        "video_path",
                        "start_frame",
                        "end_frame",
                        "metadata_dir",
                    ],
                },
                "get_environment_summary": {
                    "description": "Return a scene / single_image / multi_image environment summary (accepts aliases such as `image_pair`)",
                    "required_inputs": ["scene_id"],
                    "optional_inputs": ["include_objects"],
                    "summary_types": self.get_supported_summary_types(include_aliases=False),
                },
            },
            "task_management_interfaces": {
                "get_supported_tasks": "Return the canonical task_type list",
                "get_supported_task_groups": "Return task groups (`scene` / `single_image` / `image_pair`)",
                "get_task_catalog": "Return the task catalog and task/rubric/support metadata",
                "get_task_info": "Return details for one task",
                "get_task_profile": "Return the TaskProfile view for one task",
                "resolve_task_type_name": "Map a noisy task name to a supported task",
                "resolve_task_group_name": "Map a noisy task group name to the canonical task_group",
                "get_task_alias_mapping": "Show the alias -> task_type mapping",
                "get_region_phrase_aliases": "Show the abstract-region-phrase -> canonical-region mapping",
                "resolve_region_phrase_alias": "Map a natural-language region phrase to the canonical region",
                "get_unsupported_task_aliases": "Show explicitly rejected task aliases and reasons",
                "register_task_alias": "Register a task alias at runtime",
                "register_task_group_alias": "Register a task-group alias at runtime",
                "register_unsupported_task_alias": "Explicitly reject a noisy task name",
                "register_region_ontology": "Extend the region -> object-anchor prior",
                "register_region_phrase_alias": "Extend the abstract region phrase alias mapping",
                "register_task_profile": "Register or update task support from a TaskProfile",
                "register_task": "Register a new task at runtime",
                "unregister_task": "Remove a task at runtime",
            },
            "summary_management_interfaces": {
                "get_supported_summary_types": "Return supported environment summary types",
                "resolve_summary_type_name": "Map a noisy summary type to the canonical summary_type",
            },
        }

    def register_tool(self, tool_name: str, tool: Any) -> None:
        """Register tool."""
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("`tool_name` must be a non-empty string")
        primary_name = tool_name.strip()
        self.tools[primary_name] = tool

        intrinsic_name = getattr(tool, "name", None)
        if isinstance(intrinsic_name, str) and intrinsic_name.strip():
            self.tools[intrinsic_name.strip()] = tool

        for alias in getattr(tool, "aliases", ()) or ():
            if isinstance(alias, str) and alias.strip():
                self.tools[alias.strip()] = tool

    def register_task_alias(self, alias: str, task_type: str) -> None:
        """Register task alias."""
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("`alias` must be a non-empty string")
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("`task_type` must be a non-empty string")
        task_type = task_type.strip()
        alias = alias.strip()
        self.task_resolver.task_aliases[self._normalize(alias)] = task_type

        profile = self.task_support_registry.get_task_profile(task_type)
        if profile is not None and alias not in profile.aliases:
            updated_profile = deepcopy(profile)
            updated_profile.aliases = tuple(list(updated_profile.aliases) + [alias])
            self.task_support_registry.register_task_profile(updated_profile)

    def register_summary_type_alias(self, alias: str, canonical: str) -> None:
        """Register summary type alias."""
        self.summary_type_resolver.register_alias(alias, canonical)

    def register_task_group_alias(self, alias: str, canonical: str) -> None:
        """Register task group alias."""
        self.task_group_resolver.register_alias(alias, canonical)

    def register_region_ontology(self, key: str, candidate_labels: List[str]) -> None:
        """Register region ontology."""
        self.task_support_registry.register_region_ontology(key, candidate_labels)

    def get_region_phrase_aliases(self) -> Dict[str, List[str]]:
        """Get region phrase aliases."""
        return self.task_support_registry.get_region_phrase_aliases()

    def resolve_region_phrase_alias(self, phrase: str) -> Optional[str]:
        """Resolve region phrase alias."""
        return self.task_support_registry.resolve_region_phrase_alias(phrase)

    def register_region_phrase_alias(self, alias: str, canonical: str) -> None:
        """Register region phrase alias."""
        self.task_support_registry.register_region_phrase_alias(canonical, [alias])

    def register_task_profile(
        self,
        task_profile: TaskProfile,
        *,
        task_class: Any = None,
        rubric_class: Any = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register task profile."""
        if not isinstance(task_profile, TaskProfile):
            raise ValueError("`task_profile` must be a `TaskProfile` instance")
        if not isinstance(task_profile.task_type, str) or not task_profile.task_type.strip():
            raise ValueError("`task_profile.task_type` must be a non-empty string")

        task_type = task_profile.task_type.strip()
        normalized_profile = deepcopy(task_profile)
        normalized_profile.task_type = task_type
        normalized_profile.task_group = normalize_task_group_name(
            normalized_profile.task_group or normalized_profile.family,
            input_mode=normalized_profile.input_mode,
        )
        normalized_profile.family = normalized_profile.task_group
        normalized_profile.aliases = tuple(dict.fromkeys([
            alias.strip()
            for alias in normalized_profile.aliases
            if isinstance(alias, str) and alias.strip()
        ]))

        if task_type not in self.task_registry and task_class is None:
            raise ValueError(f"New task {task_type} requires `task_class`")
        if task_type not in self.rubric_registry and rubric_class is None:
            raise ValueError(f"New task {task_type} requires `rubric_class`")

        if task_class is not None:
            self.task_registry[task_type] = task_class
        if rubric_class is not None:
            self.rubric_registry[task_type] = rubric_class
        self.all_task_names.add(task_type)

        if isinstance(tools, dict):
            for tool_name, tool in tools.items():
                self.register_tool(tool_name, tool)

        self.task_resolver.remove_aliases_for_task(task_type)
        self.task_support_registry.register_task_profile(normalized_profile)

        for alias in normalized_profile.aliases:
            if isinstance(alias, str) and alias.strip():
                self.task_resolver.task_aliases[self._normalize(alias)] = task_type

        self.heuristic_task_parser.unregister_task_extractor(task_type)

    def register_task(
        self,
        task_type: str,
        task_class: Any,
        rubric_class: Any,
        *,
        task_profile: Optional[TaskProfile] = None,
        aliases: Optional[List[str]] = None,
        tools: Optional[Dict[str, Any]] = None,
        extraction_defaults: Optional[Dict[str, Any]] = None,
        expected_extraction_fields: Optional[List[str]] = None,
        heuristic_extractor: Optional[Any] = None,
        task_group: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> None:
        """Register task."""
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("`task_type` must be a non-empty string")

        task_type = task_type.strip()
        profile = self.task_support_registry.build_task_profile(
            task_type,
            task_profile=task_profile,
            aliases=aliases,
            extraction_defaults=extraction_defaults,
            expected_extraction_fields=expected_extraction_fields,
            heuristic_extractor=heuristic_extractor,
            task_group=task_group,
            task_family=task_family,
        )

        self.register_task_profile(
            profile,
            task_class=task_class,
            rubric_class=rubric_class,
            tools=tools,
        )

    def register_unsupported_task_alias(self, alias: str, reason: str) -> None:
        """Register unsupported task alias."""
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("`alias` must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("`reason` must be a non-empty string")
        self.task_resolver.task_rejection_reasons[self._normalize(alias)] = reason.strip()

    def unregister_task(self, task_type: Any) -> bool:
        """Unregister task."""
        resolution = self._resolve_task_type(task_type)
        canonical = resolution.task_type if resolution.task_type is not None else (
            str(task_type).strip() if isinstance(task_type, str) else None
        )
        if not canonical or canonical not in self.task_registry:
            return False

        self.task_registry.pop(canonical, None)
        self.rubric_registry.pop(canonical, None)
        self.all_task_names.discard(canonical)
        self.task_resolver.remove_aliases_for_task(canonical)
        self.task_support_registry.remove_task_support(canonical)
        self.heuristic_task_parser.unregister_task_extractor(canonical)
        return True
    
    def get_task_rubric(self, task_type: str):
        """Get task rubric."""
        resolution = self._resolve_task_type(task_type)
        canonical = resolution.task_type if resolution.task_type is not None else task_type
        rubric_class = self.rubric_registry.get(canonical)
        if rubric_class:
            return rubric_class() if callable(rubric_class) else rubric_class
        return None
    
    def get_execution_history(self) -> List[Dict]:
        """Get execution history."""
        return self.execution_history
    
    def clear_history(self):
        """Clear history."""
        self.execution_history = []
