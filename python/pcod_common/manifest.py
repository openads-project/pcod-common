from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

SCHEMA_VERSION = "1.0"


@dataclass
class ModelManifest:
    schema_version: str
    model_name: str
    precision: str
    device: str
    preprocessing: Dict[str, Any]
    postprocessing: Dict[str, Any]
    model: Dict[str, Any]
    triton: Dict[str, Any] | None = None


def validate_manifest(payload: Dict[str, Any]) -> None:
    required = ["schema_version", "model_name", "precision", "device", "preprocessing", "postprocessing", "model"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Manifest missing required key '{key}'")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version '{payload['schema_version']}', expected '{SCHEMA_VERSION}'"
        )


def score_threshold_list(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if value is None:
        return []
    return [float(value)]
