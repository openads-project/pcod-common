#pragma once

#include <cmath>
#include <string>

namespace pcod_common {

enum class PointFeatureNormalizationType { kNone, kValueThreshold, kMinMax, kZScore };

PointFeatureNormalizationType ParsePointFeatureNormalizationType(const std::string& value);

struct PointPreprocessConfig {
  float x_min = 0.0f;
  float x_max = 0.0f;
  float y_min = 0.0f;
  float y_max = 0.0f;
  float z_min = 0.0f;
  float z_max = 0.0f;
  PointFeatureNormalizationType normalization_type = PointFeatureNormalizationType::kNone;
  float value_threshold = 1.0f;
  float min_value = 0.0f;
  float max_value = 1.0f;
  float epsilon = 1e-6f;

  bool remove_points_in_zone = false;
  float nd_x_min = 0.0f;
  float nd_x_max = 0.0f;
  float nd_y_min = 0.0f;
  float nd_y_max = 0.0f;

  bool det_area_remove_outside = false;
  float da_cx = 0.0f;
  float da_cy = 0.0f;
  float da_radius = 0.0f;
  float da_bearing_rad = 0.0f;
  float da_fov_rad = 0.0f;
};

class PointPreprocessor {
 public:
  explicit PointPreprocessor(const PointPreprocessConfig& config) : config_(config) {}

  bool IsPointValid(float x, float y, float z) const;
  void SetZScoreStats(float mean, float stddev);
  float NormalizePointFeature(float intensity) const;

 private:
  PointPreprocessConfig config_;
  float z_score_mean_ = 0.0f;
  float z_score_std_ = 1.0f;
};

}  // namespace pcod_common
