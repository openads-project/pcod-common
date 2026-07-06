// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <string>

namespace pcod_common {

/** Available normalization strategies for the additional point feature. */
enum class PointFeatureNormalizationType { kNone, kValueThreshold, kMinMax, kZScore };

/** @param value Manifest normalization name. @return Parsed strategy. @throws std::invalid_argument for unknown names. */
PointFeatureNormalizationType ParsePointFeatureNormalizationType(const std::string& value);

/** Geometric filtering and point-feature normalization configuration. */
struct PointPreprocessConfig {
  float x_min = 0.0f;                                                                       ///< Minimum accepted X coordinate.
  float x_max = 0.0f;                                                                       ///< Maximum accepted X coordinate.
  float y_min = 0.0f;                                                                       ///< Minimum accepted Y coordinate.
  float y_max = 0.0f;                                                                       ///< Maximum accepted Y coordinate.
  float z_min = 0.0f;                                                                       ///< Minimum accepted Z coordinate.
  float z_max = 0.0f;                                                                       ///< Maximum accepted Z coordinate.
  PointFeatureNormalizationType normalization_type = PointFeatureNormalizationType::kNone;  ///< Feature transform.
  float value_threshold = 1.0f;  ///< Divisor for value-threshold normalization.
  float min_value = 0.0f;        ///< Minimum for min-max normalization.
  float max_value = 1.0f;        ///< Maximum for min-max normalization.
  float epsilon = 1e-6f;         ///< Lower bound for unsafe normalization divisors.

  bool remove_points_in_zone = false;  ///< Enable rectangular no-detection-zone filtering.
  float nd_x_min = 0.0f;               ///< No-detection-zone minimum X.
  float nd_x_max = 0.0f;               ///< No-detection-zone maximum X.
  float nd_y_min = 0.0f;               ///< No-detection-zone minimum Y.
  float nd_y_max = 0.0f;               ///< No-detection-zone maximum Y.

  bool det_area_remove_outside = false;  ///< Enable radial field-of-view filtering.
  float da_cx = 0.0f;                    ///< Detection-area center X.
  float da_cy = 0.0f;                    ///< Detection-area center Y.
  float da_radius = 0.0f;                ///< Detection-area radius.
  float da_bearing_rad = 0.0f;           ///< Detection-area center bearing in radians.
  float da_fov_rad = 0.0f;               ///< Detection-area angular width in radians.
};

/** Apply the configured geometric filters and scalar feature normalization. */
class PointPreprocessor {
 public:
  /** @param config Initial preprocessing configuration. */
  explicit PointPreprocessor(const PointPreprocessConfig& config) : config_(config) {}

  /** @param x Point X. @param y Point Y. @param z Point Z. @return Whether the point passes all filters. */
  bool IsPointValid(float x, float y, float z) const;
  /** @param mean Training mean. @param stddev Training standard deviation. */
  void SetZScoreStats(float mean, float stddev);
  /** @param intensity Raw additional feature. @return Normalized and clamped value. */
  float NormalizePointFeature(float intensity) const;

 private:
  PointPreprocessConfig config_;
  float z_score_mean_ = 0.0f;
  float z_score_std_ = 1.0f;
};

}  // namespace pcod_common
