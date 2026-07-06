// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <string>
#include <vector>

namespace pcod_common {

/** Immutable preprocessing contract embedded in an exported model bundle. */
struct FrozenPreprocessConfig {
  /** Contract for transforming the additional scalar point feature. */
  struct PointFeatureNormalizationContract {
    std::string type;        ///< Normalization strategy name.
    float min_value = 0.0f;  ///< Training-time minimum.
    float max_value = 0.0f;  ///< Training-time maximum.
    float epsilon = 1e-6f;   ///< Numerical stability term.
  };

  int max_num_points = 0;                                         ///< Maximum input point count.
  int num_point_features = 0;                                     ///< Features expected per point.
  std::array<float, 2> x_range{};                                 ///< Accepted X range.
  std::array<float, 2> y_range{};                                 ///< Accepted Y range.
  std::array<float, 2> z_range{};                                 ///< Accepted Z range.
  float voxel_x = 0.0f;                                           ///< Voxel width along X.
  float voxel_y = 0.0f;                                           ///< Voxel width along Y.
  float voxel_z = 0.0f;                                           ///< Voxel width along Z.
  PointFeatureNormalizationContract point_feature_normalization;  ///< Scalar feature contract.
};

/** Immutable postprocessing contract embedded in an exported model bundle. */
struct FrozenPostprocessConfig {
  int grid_x = 0;                        ///< Output grid size along X.
  int grid_y = 0;                        ///< Output grid size along Y.
  int num_classes = 0;                   ///< Number of semantic classes.
  std::vector<std::string> class_names;  ///< Class names in output order.
};

/** Immutable model architecture values needed during decoding. */
struct FrozenModelConfig {
  std::vector<int> stride;                                 ///< Backbone strides.
  std::vector<int> up_stride;                              ///< Decoder upsampling strides.
  int first_up_stride = 1;                                 ///< First decoder upsampling factor.
  std::array<int, 2> pillar_map_size{};                    ///< Base pillar-map dimensions.
  std::array<std::array<float, 2>, 3> pillar_map_range{};  ///< XYZ pillar-map ranges.
};

/** Runtime-adjustable defaults supplied by the exported bundle. */
struct RuntimeDefaults {
  /** Preprocessing defaults. */
  struct Preprocessing {
    /** Additional point-feature defaults. */
    struct PointFeature {
      float value_threshold = 0.0f;  ///< Value-threshold normalization divisor.
    };

    PointFeature point_feature;  ///< Additional point-feature defaults.
  };

  /** Postprocessing defaults. */
  struct Postprocessing {
    float class_score_threshold = 0.0f;       ///< Decoder confidence threshold.
    std::vector<float> nms_score_thresholds;  ///< NMS class score thresholds.
    float nms_iou_threshold = 0.1f;           ///< Rotated NMS IoU threshold.
    int max_detections = 0;                   ///< Maximum final detection count.
  };

  Preprocessing preprocessing;    ///< Preprocessing defaults.
  Postprocessing postprocessing;  ///< Postprocessing defaults.
};

/** Export artifact metadata and deployment file references. */
struct ArtifactConfig {
  /** Bundle-relative artifact paths. */
  struct Files {
    std::string model;                     ///< Exported model path.
    std::string checkpoint;                ///< Source checkpoint path.
    std::string resolved_training_config;  ///< Resolved training configuration path.
    std::string triton_repository;         ///< Triton repository path.
    std::string triton_config;             ///< Triton model configuration path.
    std::string triton_model;              ///< Triton model artifact path.
  };

  /** Optional Triton deployment metadata. */
  struct TritonDeployment {
    bool enabled = false;       ///< Whether Triton artifacts were exported.
    std::string model_name;     ///< Triton model name.
    std::string model_version;  ///< Triton model version.
  };

  /** Named model tensor contract. */
  struct Tensor {
    std::string name;                ///< Tensor name.
    std::string dtype;               ///< Triton datatype name.
    std::vector<std::string> shape;  ///< Dimensions, including symbolic values.
  };

  std::string bundle_name;                        ///< Human-readable bundle name.
  std::string export_format;                      ///< Model serialization format.
  std::string backend;                            ///< Runtime backend.
  std::string precision;                          ///< Numeric precision.
  std::string device;                             ///< Export target device.
  std::string head_name;                          ///< Detection head identifier.
  std::string export_timestamp_utc;               ///< UTC export timestamp.
  Files files;                                    ///< Bundle-relative files.
  TritonDeployment triton;                        ///< Triton deployment metadata.
  std::vector<Tensor> inputs;                     ///< Ordered input tensor contracts.
  std::vector<Tensor> outputs;                    ///< Ordered output tensor contracts.
  std::vector<std::array<float, 3>> size_priors;  ///< Per-class length, width, and height priors.
  std::string size_priors_source;                 ///< Provenance of the size priors.
};

/** Complete immutable inference contract. */
struct FrozenContract {
  FrozenPreprocessConfig preprocessing;    ///< Preprocessing contract.
  FrozenPostprocessConfig postprocessing;  ///< Postprocessing contract.
  FrozenModelConfig model;                 ///< Model architecture contract.
};

/** Canonical parsed representation of `model_manifest.yml`. */
struct ModelManifest {
  std::string schema_version;        ///< Manifest schema version.
  ArtifactConfig artifact;           ///< Artifact metadata.
  FrozenContract frozen_contract;    ///< Non-overridable inference contract.
  RuntimeDefaults runtime_defaults;  ///< Overridable runtime defaults.
};

/** @param path Manifest YAML path. @return Parsed and validated manifest. */
ModelManifest LoadModelManifest(const std::string& path);
/** @param manifest Manifest to validate. @throws std::runtime_error for contract violations. */
void ValidateModelManifest(const ModelManifest& manifest);

}  // namespace pcod_common
