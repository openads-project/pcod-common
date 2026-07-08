// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/point_preprocess.hpp"

#include <stdexcept>

namespace pcod_common {

PointFeatureNormalizationType ParsePointFeatureNormalizationType(const std::string& value) {
  if (value == "none") {
    return PointFeatureNormalizationType::kNone;
  }
  if (value == "value_threshold") {
    return PointFeatureNormalizationType::kValueThreshold;
  }
  if (value == "min_max") {
    return PointFeatureNormalizationType::kMinMax;
  }
  if (value == "z_score") {
    return PointFeatureNormalizationType::kZScore;
  }
  throw std::runtime_error("Unsupported point feature normalization type: " + value);
}

bool PointPreprocessor::IsPointValid(float x, float y, float z) const {
  const bool in_range = (x >= config_.x_min && x < config_.x_max && !std::isnan(x) && y >= config_.y_min && y < config_.y_max &&
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
    if (std::abs(angle_offset) > config_.da_fov_rad * 0.5F + 1e-6F) {
      return false;
    }
  }

  if (config_.remove_points_in_zone) {
    const bool in_nd_zone = (x >= config_.nd_x_min && x <= config_.nd_x_max && y >= config_.nd_y_min && y <= config_.nd_y_max);
    if (in_nd_zone) {
      return false;
    }
  }

  return true;
}

void PointPreprocessor::SetZScoreStats(float mean, float stddev) {
  z_score_mean_ = mean;
  z_score_std_ = std::abs(stddev) > config_.epsilon ? stddev : config_.epsilon;
}

float PointPreprocessor::NormalizePointFeature(float intensity) const {
  if (config_.normalization_type == PointFeatureNormalizationType::kNone) {
    return intensity;
  }
  if (config_.normalization_type == PointFeatureNormalizationType::kValueThreshold) {
    if (config_.value_threshold <= 0.0F) {
      return intensity;
    }
    const float clipped = std::min(std::max(intensity, 0.0F), config_.value_threshold);
    return clipped / std::max(config_.value_threshold, config_.epsilon);
  }
  if (config_.normalization_type == PointFeatureNormalizationType::kMinMax) {
    const float denom = std::max(config_.max_value - config_.min_value, config_.epsilon);
    const float scaled = (intensity - config_.min_value) / denom;
    return std::min(std::max(scaled, 0.0F), 1.0F);
  }
  if (config_.normalization_type == PointFeatureNormalizationType::kZScore) {
    return (intensity - z_score_mean_) / std::max(z_score_std_, config_.epsilon);
  }
  return intensity;
}

}  // namespace pcod_common
