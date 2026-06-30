# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from pcod_common import manifest


def _minimal_payload() -> dict:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "artifact": {
            "bundle_name": "pbod_fp16_gpu_onnx_test",
            "export_format": "onnx_fp16_gpu",
            "backend": "onnx",
            "precision": "fp16",
            "device": "cuda",
            "files": {
                "model": "model.onnx",
                "checkpoint": "checkpoints/best.pt",
                "resolved_training_config": "config/resolved_training_config.yml",
            },
            "triton": {
                "enabled": False,
            },
            "inputs": [
                {
                    "name": "point_features",
                    "dtype": "float16",
                    "shape": ["batch", 100, 1],
                }
            ],
            "outputs": [
                {
                    "name": "reg_logits",
                    "dtype": "float16",
                    "shape": ["batch", 400, 7],
                }
            ],
        },
        "frozen_contract": {
            "preprocessing": {
                "max_num_points": 100,
                "num_point_features": 1,
                "point_cloud_range": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                },
                "voxel_size": {
                    "x": 0.1,
                    "y": 0.1,
                    "z": 0.1,
                },
                "point_feature_normalization": {
                    "type": "value_threshold",
                    "epsilon": 1e-6,
                },
            },
            "postprocessing": {
                "grid_size": {"x": 10, "y": 10},
                "num_classes": 1,
                "class_names": ["car"],
            },
            "model": {
                "stride": [2, 1, 2],
                "up_stride": [1, 1, 2],
                "first_up_stride": 1,
                "pillar_map_size": [10, 10],
                "pillar_map_range": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            },
        },
        "runtime_defaults": {
            "preprocessing": {
                "point_feature": {
                    "value_threshold": 1.0,
                }
            },
            "postprocessing": {
                "class_score_threshold": 0.0,
                "nms": {
                    "score_threshold": 0.2,
                    "iou_threshold": 0.5,
                    "max_num_objects": 10,
                },
            },
        },
    }


def test_validate_manifest_accepts_minimal_payload():
    manifest.validate_manifest(_minimal_payload())


def test_load_manifest_reads_yaml(tmp_path: Path):
    path = tmp_path / "model_manifest.yml"
    path.write_text(
        """
schema_version: "2.0"
artifact:
  bundle_name: pbod
  export_format: onnx_fp32_gpu
  backend: onnx
  precision: fp32
  device: cuda
  files:
    model: model.onnx
    checkpoint: checkpoints/best.pt
    resolved_training_config: config/resolved_training_config.yml
  triton:
    enabled: false
  inputs:
    - name: point_features
      dtype: float32
      shape: [batch, 4, 1]
  outputs:
    - name: reg_logits
      dtype: float32
      shape: [batch, 16, 7]
frozen_contract:
  preprocessing:
    max_num_points: 4
    num_point_features: 1
    point_cloud_range:
      x: [-1.0, 1.0]
      y: [-1.0, 1.0]
      z: [-1.0, 1.0]
    voxel_size:
      x: 0.5
      y: 0.5
      z: 0.5
    point_feature_normalization:
      type: value_threshold
      epsilon: 1.0e-6
  postprocessing:
    grid_size:
      x: 2
      y: 2
    num_classes: 1
    class_names: [car]
  model:
    stride: [2]
    up_stride: [1]
    first_up_stride: 1
    pillar_map_size: [4, 4]
    pillar_map_range: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
runtime_defaults:
  preprocessing:
    point_feature:
      value_threshold: 12.0
  postprocessing:
    class_score_threshold: 0.1
    nms:
      score_threshold: [0.2]
      iou_threshold: 0.5
      max_num_objects: 16
"""
    )
    payload = manifest.load_manifest(path)
    assert payload["artifact"]["files"]["checkpoint"] == "checkpoints/best.pt"


def test_validate_manifest_rejects_wrong_schema_version():
    payload = _minimal_payload()
    payload["schema_version"] = "1.0"
    with pytest.raises(ValueError):
        manifest.validate_manifest(payload)


@pytest.mark.parametrize("missing_key", ["artifact", "frozen_contract", "runtime_defaults"])
def test_validate_manifest_rejects_missing_required_key(missing_key):
    payload = _minimal_payload()
    payload.pop(missing_key)
    with pytest.raises(ValueError, match="missing required key"):
        manifest.validate_manifest(payload)


def test_validate_manifest_requires_runtime_value_threshold_for_value_threshold_norm():
    payload = _minimal_payload()
    payload["runtime_defaults"]["preprocessing"]["point_feature"]["value_threshold"] = 0.0
    with pytest.raises(ValueError, match="value_threshold"):
        manifest.validate_manifest(payload)


def test_validate_manifest_rejects_frozen_value_threshold_key():
    payload = _minimal_payload()
    payload["frozen_contract"]["preprocessing"]["point_feature_normalization"]["value_threshold"] = 1.0
    with pytest.raises(ValueError, match="unsupported key"):
        manifest.validate_manifest(payload)


def test_validate_manifest_accepts_per_class_nms_thresholds():
    payload = _minimal_payload()
    payload["frozen_contract"]["postprocessing"]["num_classes"] = 2
    payload["frozen_contract"]["postprocessing"]["class_names"] = ["car", "pedestrian"]
    payload["runtime_defaults"]["postprocessing"]["nms"]["score_threshold"] = [0.1, 0.2]
    manifest.validate_manifest(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        (0.25, [0.25]),
        ([0.1, 0.2], [0.1, 0.2]),
    ],
)
def test_score_threshold_list(value, expected):
    assert manifest.score_threshold_list(value) == expected


def test_validate_manifest_rejects_unknown_artifact_key():
    payload = _minimal_payload()
    payload["artifact"]["hardware_compatible"] = True
    with pytest.raises(ValueError, match="unsupported key"):
        manifest.validate_manifest(payload)


def test_validate_manifest_rejects_bundle_escaping_file_path():
    payload = _minimal_payload()
    payload["artifact"]["files"]["checkpoint"] = "../best.pt"
    with pytest.raises(ValueError, match="bundle root"):
        manifest.validate_manifest(payload)
