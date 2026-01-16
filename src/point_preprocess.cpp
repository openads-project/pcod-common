#include "pcod_common/point_preprocess.hpp"

namespace pcod_common {

bool PointPreprocessor::IsPointValid(float x, float y, float z) const {
  const bool in_range =
      (x >= config_.x_min && x < config_.x_max && !std::isnan(x) && y >= config_.y_min && y < config_.y_max &&
       !std::isnan(y) && z >= config_.z_min && z < config_.z_max && !std::isnan(z));
  if (!in_range) {
    return false;
  }

  if (config_.det_area_remove_outside) {
    const float delta_x = x - config_.da_cx;
    const float delta_y = y - config_.da_cy;
    const float squared_distance_to_center = delta_x * delta_x + delta_y * delta_y;
    if (squared_distance_to_center > config_.da_radius * config_.da_radius) {
      return false;
    }
    float angle_to_center = std::atan2(delta_y, delta_x);
    float angle_offset = angle_to_center - config_.da_bearing_rad;
    while (angle_offset > static_cast<float>(M_PI)) {
      angle_offset -= static_cast<float>(2.0 * M_PI);
    }
    while (angle_offset < static_cast<float>(-M_PI)) {
      angle_offset += static_cast<float>(2.0 * M_PI);
    }
    if (std::abs(angle_offset) > config_.da_fov_rad * 0.5f + 1e-6f) {
      return false;
    }
  }

  if (config_.remove_points_in_zone) {
    const bool in_nd_zone =
        (x >= config_.nd_x_min && x <= config_.nd_x_max && y >= config_.nd_y_min && y <= config_.nd_y_max);
    if (in_nd_zone) {
      return false;
    }
  }

  return true;
}

float PointPreprocessor::NormalizeIntensity(float intensity) const {
  if (config_.zero_intensity) {
    return 0.0f;
  }
  if (config_.intensity_threshold <= 0.0f) {
    return intensity;
  }
  const float scaled = intensity / config_.intensity_threshold;
  if (scaled < 0.0f) {
    return 0.0f;
  }
  if (scaled > 1.0f) {
    return 1.0f;
  }
  return scaled;
}

}  // namespace pcod_common
