#pragma once

#include <cmath>

namespace pcod_common {

struct PointPreprocessConfig {
  float x_min = 0.0f;
  float x_max = 0.0f;
  float y_min = 0.0f;
  float y_max = 0.0f;
  float z_min = 0.0f;
  float z_max = 0.0f;
  float intensity_threshold = 1.0f;
  bool zero_intensity = false;

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
  float NormalizeIntensity(float intensity) const;

 private:
  PointPreprocessConfig config_;
};

}  // namespace pcod_common
