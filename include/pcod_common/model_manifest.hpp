#pragma once

#include <array>
#include <string>
#include <vector>

namespace pcod_common {

struct PreprocessConfig {
  struct PointFeatureNormalization {
    std::string type;
    float intensity_threshold = 0.0f;
    float min_intensity = 0.0f;
    float max_intensity = 0.0f;
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
  PointFeatureNormalization point_features_normalization;
};

struct PostprocessConfig {
  int grid_x = 0;
  int grid_y = 0;
  int num_classes = 0;
  std::vector<std::string> class_names;
  std::vector<float> score_thresholds;
  float nms_iou_threshold = 0.1f;
  int max_detections = 0;
};

struct ModelConfig {
  std::vector<int> stride;
  std::vector<int> up_stride;
  int first_up_stride = 1;
  std::array<int, 2> pillar_map_size{};
  std::array<std::array<float, 2>, 3> pillar_map_range{};
  bool mask_is_bool = true;
  bool zero_intensity = false;
};

struct TritonConfig {
  std::string model_name;
  std::string model_version;
  std::string precision;
};

struct ModelManifest {
  std::string schema_version;
  std::string model_name;
  std::string precision;
  std::string device;
  PreprocessConfig preprocessing;
  PostprocessConfig postprocessing;
  ModelConfig model;
  TritonConfig triton;
};

ModelManifest LoadModelManifest(const std::string& path);
void ValidateModelManifest(const ModelManifest& manifest);

}  // namespace pcod_common
