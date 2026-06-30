// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <string>
#include <vector>

namespace pcod_common {

struct FrozenPreprocessConfig {
  struct PointFeatureNormalizationContract {
    std::string type;
    float min_value = 0.0f;
    float max_value = 0.0f;
    float epsilon = 1e-6f;
  };

  int max_num_points = 0;
  int num_point_features = 0;
  std::array<float, 2> x_range{};
  std::array<float, 2> y_range{};
  std::array<float, 2> z_range{};
  float voxel_x = 0.0f;
  float voxel_y = 0.0f;
  float voxel_z = 0.0f;
  PointFeatureNormalizationContract point_feature_normalization;
};

struct FrozenPostprocessConfig {
  int grid_x = 0;
  int grid_y = 0;
  int num_classes = 0;
  std::vector<std::string> class_names;
};

struct FrozenModelConfig {
  std::vector<int> stride;
  std::vector<int> up_stride;
  int first_up_stride = 1;
  std::array<int, 2> pillar_map_size{};
  std::array<std::array<float, 2>, 3> pillar_map_range{};
};

struct RuntimeDefaults {
  struct Preprocessing {
    struct PointFeature {
      float value_threshold = 0.0f;
    };

    PointFeature point_feature;
  };

  struct Postprocessing {
    float class_score_threshold = 0.0f;
    std::vector<float> nms_score_thresholds;
    float nms_iou_threshold = 0.1f;
    int max_detections = 0;
  };

  Preprocessing preprocessing;
  Postprocessing postprocessing;
};

struct ArtifactConfig {
  struct Files {
    std::string model;
    std::string checkpoint;
    std::string resolved_training_config;
    std::string triton_repository;
    std::string triton_config;
    std::string triton_model;
  };

  struct TritonDeployment {
    bool enabled = false;
    std::string model_name;
    std::string model_version;
  };

  struct Tensor {
    std::string name;
    std::string dtype;
    std::vector<std::string> shape;
  };

  std::string bundle_name;
  std::string export_format;
  std::string backend;
  std::string precision;
  std::string device;
  std::string head_name;
  std::string export_timestamp_utc;
  Files files;
  TritonDeployment triton;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<std::array<float, 3>> size_priors;
  std::string size_priors_source;
};

struct FrozenContract {
  FrozenPreprocessConfig preprocessing;
  FrozenPostprocessConfig postprocessing;
  FrozenModelConfig model;
};

struct ModelManifest {
  std::string schema_version;
  ArtifactConfig artifact;
  FrozenContract frozen_contract;
  RuntimeDefaults runtime_defaults;
};

ModelManifest LoadModelManifest(const std::string& path);
void ValidateModelManifest(const ModelManifest& manifest);

}  // namespace pcod_common
