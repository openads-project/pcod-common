# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Sequence

import yaml

SCHEMA_VERSION = "2.0"

_ROOT_KEYS = {"schema_version", "artifact", "frozen_contract", "runtime_defaults"}
_ARTIFACT_KEYS = {
    "bundle_name",
    "export_format",
    "backend",
    "precision",
    "device",
    "head_name",
    "export_timestamp_utc",
    "files",
    "triton",
    "inputs",
    "outputs",
    "size_priors",
    "size_priors_source",
}
_ARTIFACT_FILE_KEYS = {
    "model",
    "checkpoint",
    "resolved_training_config",
    "triton_repository",
    "triton_config",
    "triton_model",
}
_TRITON_KEYS = {"enabled", "model_name", "model_version"}
_TENSOR_KEYS = {"name", "dtype", "shape", "description"}
_FROZEN_CONTRACT_KEYS = {"preprocessing", "postprocessing", "model"}
_PREPROCESSING_KEYS = {
    "max_num_points",
    "num_point_features",
    "point_cloud_range",
    "voxel_size",
    "point_feature_normalization",
}
_POINT_CLOUD_RANGE_KEYS = {"x", "y", "z"}
_VOXEL_SIZE_KEYS = {"x", "y", "z"}
_POINT_FEATURE_NORMALIZATION_KEYS = {"type", "min_value", "max_value", "epsilon"}
_POSTPROCESSING_KEYS = {"grid_size", "num_classes", "class_names"}
_GRID_SIZE_KEYS = {"x", "y"}
_MODEL_KEYS = {"stride", "up_stride", "first_up_stride", "pillar_map_size", "pillar_map_range"}
_RUNTIME_DEFAULTS_KEYS = {"preprocessing", "postprocessing"}
_RUNTIME_PREPROCESSING_KEYS = {"point_feature"}
_POINT_FEATURE_KEYS = {"value_threshold"}
_RUNTIME_POSTPROCESSING_KEYS = {"class_score_threshold", "nms"}
_NMS_KEYS = {"score_threshold", "iou_threshold", "max_num_objects"}


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load and validate a model manifest from a YAML file."""
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be a YAML mapping")
    validate_manifest(payload)
    return payload


def score_threshold_list(value: Any) -> List[float]:
    """Normalize a scalar or list score threshold to a list."""
    if isinstance(value, list):
        return [float(v) for v in value]
    if value is None:
        return []
    return [float(value)]


def _require_mapping(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest key '{key}' must be a mapping")
    return value


def _require_keys(payload: Dict[str, Any], keys: Iterable[str], *, scope: str) -> None:
    for key in keys:
        if key not in payload:
            raise ValueError(f"Manifest {scope} missing required key '{key}'")


def _reject_unknown_keys(payload: Dict[str, Any], *, allowed: set[str], scope: str) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Manifest {scope} contains unsupported key(s): {', '.join(unknown)}")


def _require_probability(value: Any, *, field: str) -> float:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"Manifest field '{field}' must be within [0.0, 1.0]")
    return numeric


def _require_non_negative_int(value: Any, *, field: str, allow_zero: bool = True) -> int:
    numeric = int(value)
    if allow_zero:
        if numeric < 0:
            raise ValueError(f"Manifest field '{field}' must be zero or positive")
    elif numeric <= 0:
        raise ValueError(f"Manifest field '{field}' must be positive")
    return numeric


def _require_non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"Manifest field '{field}' must be a scalar string")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Manifest field '{field}' must not be empty")
    return text


def _require_bundle_relative_path(value: Any, *, field: str) -> str:
    text = _require_non_empty_string(value, field=field)
    path = PurePosixPath(text)
    if path.is_absolute():
        raise ValueError(f"Manifest field '{field}' must be a bundle-relative path")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"Manifest field '{field}' must not escape the bundle root")
    return text


def _require_range2(value: Any, *, field: str) -> List[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Manifest field '{field}' must be a 2-element list")
    return [float(value[0]), float(value[1])]


def _require_non_empty_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest field '{field}' must be a non-empty list")
    return value


def _validate_tensor_list(value: Any, *, field: str) -> None:
    entries = _require_non_empty_sequence(value, field=field)
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest field '{field}[{idx}]' must be a mapping")
        _reject_unknown_keys(entry, allowed=_TENSOR_KEYS, scope=f"{field}[{idx}]")
        _require_keys(entry, ["name", "dtype", "shape"], scope=f"{field}[{idx}]")
        _require_non_empty_string(entry["name"], field=f"{field}[{idx}].name")
        _require_non_empty_string(entry["dtype"], field=f"{field}[{idx}].dtype")
        shape = _require_non_empty_sequence(entry["shape"], field=f"{field}[{idx}].shape")
        for dim_idx, dim in enumerate(shape):
            if not isinstance(dim, (str, int)):
                raise ValueError(f"Manifest field '{field}[{idx}].shape[{dim_idx}]' must be a string or integer")
        if "description" in entry:
            _require_non_empty_string(entry["description"], field=f"{field}[{idx}].description")


def _validate_size_priors(value: Any, *, field: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"Manifest field '{field}' must be a list")
    for idx, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"Manifest field '{field}[{idx}]' must be a 3-element list")
        for element_idx, element in enumerate(row):
            try:
                float(element)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Manifest field '{field}[{idx}][{element_idx}]' must be numeric") from exc


def _validate_artifact(payload: Dict[str, Any]) -> None:
    _reject_unknown_keys(payload, allowed=_ARTIFACT_KEYS, scope="artifact")
    _require_keys(
        payload,
        [
            "bundle_name",
            "export_format",
            "backend",
            "precision",
            "device",
            "files",
            "triton",
            "inputs",
            "outputs",
        ],
        scope="artifact",
    )

    _require_non_empty_string(payload["bundle_name"], field="artifact.bundle_name")
    _require_non_empty_string(payload["export_format"], field="artifact.export_format")
    _require_non_empty_string(payload["backend"], field="artifact.backend")
    _require_non_empty_string(payload["precision"], field="artifact.precision")
    _require_non_empty_string(payload["device"], field="artifact.device")

    if "head_name" in payload:
        _require_non_empty_string(payload["head_name"], field="artifact.head_name")
    if "export_timestamp_utc" in payload:
        _require_non_empty_string(payload["export_timestamp_utc"], field="artifact.export_timestamp_utc")

    files = _require_mapping(payload, "files")
    _reject_unknown_keys(files, allowed=_ARTIFACT_FILE_KEYS, scope="artifact.files")
    _require_keys(files, ["checkpoint", "resolved_training_config"], scope="artifact.files")
    _require_bundle_relative_path(files["checkpoint"], field="artifact.files.checkpoint")
    _require_bundle_relative_path(files["resolved_training_config"], field="artifact.files.resolved_training_config")
    for optional_file_key in ("model", "triton_repository", "triton_config", "triton_model"):
        if optional_file_key in files:
            _require_bundle_relative_path(files[optional_file_key], field=f"artifact.files.{optional_file_key}")

    triton = _require_mapping(payload, "triton")
    _reject_unknown_keys(triton, allowed=_TRITON_KEYS, scope="artifact.triton")
    _require_keys(triton, ["enabled"], scope="artifact.triton")
    if not isinstance(triton["enabled"], bool):
        raise ValueError("Manifest field 'artifact.triton.enabled' must be a boolean")
    if triton["enabled"]:
        _require_keys(triton, ["model_name", "model_version"], scope="artifact.triton")
        _require_non_empty_string(triton["model_name"], field="artifact.triton.model_name")
        _require_non_empty_string(triton["model_version"], field="artifact.triton.model_version")
        for required in ("triton_repository", "triton_config", "triton_model"):
            if required not in files:
                raise ValueError(f"Manifest artifact.files missing required key '{required}' for Triton exports")
    elif "model" not in files:
        raise ValueError("Manifest artifact.files must define 'model' for non-Triton exports")

    _validate_tensor_list(payload["inputs"], field="artifact.inputs")
    _validate_tensor_list(payload["outputs"], field="artifact.outputs")

    if "size_priors" in payload:
        _validate_size_priors(payload["size_priors"], field="artifact.size_priors")
    if "size_priors_source" in payload:
        _require_non_empty_string(payload["size_priors_source"], field="artifact.size_priors_source")


def _validate_frozen_contract(payload: Dict[str, Any]) -> None:
    _reject_unknown_keys(payload, allowed=_FROZEN_CONTRACT_KEYS, scope="frozen_contract")
    _require_keys(payload, ["preprocessing", "postprocessing", "model"], scope="frozen_contract")

    preprocessing = _require_mapping(payload, "preprocessing")
    _reject_unknown_keys(preprocessing, allowed=_PREPROCESSING_KEYS, scope="frozen_contract.preprocessing")
    _require_keys(
        preprocessing,
        [
            "max_num_points",
            "num_point_features",
            "point_cloud_range",
            "voxel_size",
            "point_feature_normalization",
        ],
        scope="frozen_contract.preprocessing",
    )
    _require_non_negative_int(
        preprocessing["max_num_points"],
        field="frozen_contract.preprocessing.max_num_points",
        allow_zero=False,
    )
    _require_non_negative_int(
        preprocessing["num_point_features"],
        field="frozen_contract.preprocessing.num_point_features",
        allow_zero=False,
    )

    point_cloud_range = _require_mapping(preprocessing, "point_cloud_range")
    _reject_unknown_keys(
        point_cloud_range, allowed=_POINT_CLOUD_RANGE_KEYS, scope="frozen_contract.preprocessing.point_cloud_range"
    )
    _require_keys(point_cloud_range, ["x", "y", "z"], scope="frozen_contract.preprocessing.point_cloud_range")
    for axis in ("x", "y", "z"):
        _require_range2(point_cloud_range[axis], field=f"frozen_contract.preprocessing.point_cloud_range.{axis}")

    voxel_size = _require_mapping(preprocessing, "voxel_size")
    _reject_unknown_keys(voxel_size, allowed=_VOXEL_SIZE_KEYS, scope="frozen_contract.preprocessing.voxel_size")
    _require_keys(voxel_size, ["x", "y", "z"], scope="frozen_contract.preprocessing.voxel_size")
    for axis in ("x", "y", "z"):
        float(voxel_size[axis])

    norm = _require_mapping(preprocessing, "point_feature_normalization")
    _reject_unknown_keys(
        norm,
        allowed=_POINT_FEATURE_NORMALIZATION_KEYS,
        scope="frozen_contract.preprocessing.point_feature_normalization",
    )
    _require_keys(norm, ["type", "epsilon"], scope="frozen_contract.preprocessing.point_feature_normalization")
    epsilon = float(norm["epsilon"])
    if epsilon <= 0.0:
        raise ValueError("Manifest field 'frozen_contract.preprocessing.point_feature_normalization.epsilon' must be > 0")
    norm_type = str(norm["type"])
    if norm_type == "min_max":
        if "min_value" not in norm or "max_value" not in norm:
            raise ValueError("Manifest frozen_contract preprocessing min_max normalization requires min_value and max_value")
        if float(norm["min_value"]) >= float(norm["max_value"]):
            raise ValueError("Manifest frozen_contract preprocessing min_max normalization requires min_value < max_value")
    elif norm_type not in {"none", "value_threshold", "min_max", "z_score"}:
        raise ValueError("Manifest field 'frozen_contract.preprocessing.point_feature_normalization.type' is invalid")

    postprocessing = _require_mapping(payload, "postprocessing")
    _reject_unknown_keys(postprocessing, allowed=_POSTPROCESSING_KEYS, scope="frozen_contract.postprocessing")
    _require_keys(
        postprocessing,
        ["grid_size", "num_classes", "class_names"],
        scope="frozen_contract.postprocessing",
    )
    grid_size = _require_mapping(postprocessing, "grid_size")
    _reject_unknown_keys(grid_size, allowed=_GRID_SIZE_KEYS, scope="frozen_contract.postprocessing.grid_size")
    _require_keys(grid_size, ["x", "y"], scope="frozen_contract.postprocessing.grid_size")
    _require_non_negative_int(grid_size["x"], field="frozen_contract.postprocessing.grid_size.x", allow_zero=False)
    _require_non_negative_int(grid_size["y"], field="frozen_contract.postprocessing.grid_size.y", allow_zero=False)
    num_classes = _require_non_negative_int(
        postprocessing["num_classes"],
        field="frozen_contract.postprocessing.num_classes",
        allow_zero=False,
    )
    class_names = postprocessing["class_names"]
    if not isinstance(class_names, list) or len(class_names) != num_classes:
        raise ValueError("Manifest field 'frozen_contract.postprocessing.class_names' must contain one entry per class")
    for idx, class_name in enumerate(class_names):
        _require_non_empty_string(class_name, field=f"frozen_contract.postprocessing.class_names[{idx}]")

    model = _require_mapping(payload, "model")
    _reject_unknown_keys(model, allowed=_MODEL_KEYS, scope="frozen_contract.model")
    _require_keys(
        model,
        [
            "stride",
            "up_stride",
            "first_up_stride",
            "pillar_map_size",
            "pillar_map_range",
        ],
        scope="frozen_contract.model",
    )
    for field_name in ("stride", "up_stride"):
        values = _require_non_empty_sequence(model[field_name], field=f"frozen_contract.model.{field_name}")
        for idx, value in enumerate(values):
            _require_non_negative_int(value, field=f"frozen_contract.model.{field_name}[{idx}]", allow_zero=False)
    _require_non_negative_int(
        model["first_up_stride"],
        field="frozen_contract.model.first_up_stride",
        allow_zero=False,
    )
    pillar_map_size = _require_range2(model["pillar_map_size"], field="frozen_contract.model.pillar_map_size")
    for idx, value in enumerate(pillar_map_size):
        if int(value) <= 0:
            raise ValueError(f"Manifest field 'frozen_contract.model.pillar_map_size[{idx}]' must be positive")
    pillar_map_range = model["pillar_map_range"]
    if not isinstance(pillar_map_range, list) or len(pillar_map_range) != 3:
        raise ValueError("Manifest field 'frozen_contract.model.pillar_map_range' must contain three ranges")
    for idx, value in enumerate(pillar_map_range):
        _require_range2(value, field=f"frozen_contract.model.pillar_map_range[{idx}]")


def _validate_runtime_defaults(payload: Dict[str, Any], *, norm_type: str, num_classes: int) -> None:
    _reject_unknown_keys(payload, allowed=_RUNTIME_DEFAULTS_KEYS, scope="runtime_defaults")
    _require_keys(payload, ["preprocessing", "postprocessing"], scope="runtime_defaults")

    preprocessing = _require_mapping(payload, "preprocessing")
    _reject_unknown_keys(preprocessing, allowed=_RUNTIME_PREPROCESSING_KEYS, scope="runtime_defaults.preprocessing")
    _require_keys(preprocessing, ["point_feature"], scope="runtime_defaults.preprocessing")
    point_feature = _require_mapping(preprocessing, "point_feature")
    _reject_unknown_keys(point_feature, allowed=_POINT_FEATURE_KEYS, scope="runtime_defaults.preprocessing.point_feature")
    if norm_type == "value_threshold":
        value_threshold = float(point_feature.get("value_threshold", 0.0))
        if value_threshold <= 0.0:
            raise ValueError("Manifest field 'runtime_defaults.preprocessing.point_feature.value_threshold' must be > 0")
    elif "value_threshold" in point_feature:
        float(point_feature["value_threshold"])

    postprocessing = _require_mapping(payload, "postprocessing")
    _reject_unknown_keys(postprocessing, allowed=_RUNTIME_POSTPROCESSING_KEYS, scope="runtime_defaults.postprocessing")
    _require_keys(postprocessing, ["class_score_threshold", "nms"], scope="runtime_defaults.postprocessing")
    _require_probability(
        postprocessing.get("class_score_threshold", 0.0),
        field="runtime_defaults.postprocessing.class_score_threshold",
    )
    nms = _require_mapping(postprocessing, "nms")
    _reject_unknown_keys(nms, allowed=_NMS_KEYS, scope="runtime_defaults.postprocessing.nms")
    _require_keys(
        nms,
        ["score_threshold", "iou_threshold", "max_num_objects"],
        scope="runtime_defaults.postprocessing.nms",
    )
    _require_probability(nms["iou_threshold"], field="runtime_defaults.postprocessing.nms.iou_threshold")
    _require_non_negative_int(
        nms["max_num_objects"],
        field="runtime_defaults.postprocessing.nms.max_num_objects",
    )
    score_thresholds = score_threshold_list(nms["score_threshold"])
    if not score_thresholds:
        raise ValueError("Manifest field 'runtime_defaults.postprocessing.nms.score_threshold' must not be empty")
    if len(score_thresholds) not in {1, num_classes}:
        raise ValueError(
            "Manifest field 'runtime_defaults.postprocessing.nms.score_threshold' must contain one value or one value per class"
        )
    for value in score_thresholds:
        _require_probability(value, field="runtime_defaults.postprocessing.nms.score_threshold")


def validate_manifest(payload: Dict[str, Any]) -> None:
    """Validate the structure and values of a model manifest."""
    _reject_unknown_keys(payload, allowed=_ROOT_KEYS, scope="root")
    _require_keys(payload, ["schema_version", "artifact", "frozen_contract", "runtime_defaults"], scope="root")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema_version '{payload['schema_version']}', expected '{SCHEMA_VERSION}'")

    artifact = _require_mapping(payload, "artifact")
    frozen_contract = _require_mapping(payload, "frozen_contract")
    runtime_defaults = _require_mapping(payload, "runtime_defaults")

    _validate_artifact(artifact)
    _validate_frozen_contract(frozen_contract)

    norm = _require_mapping(_require_mapping(frozen_contract, "preprocessing"), "point_feature_normalization")
    postprocessing = _require_mapping(frozen_contract, "postprocessing")
    _validate_runtime_defaults(
        runtime_defaults,
        norm_type=str(norm["type"]),
        num_classes=int(postprocessing["num_classes"]),
    )
