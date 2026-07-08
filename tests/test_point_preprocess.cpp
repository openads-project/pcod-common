// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/point_preprocess.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

/** Run point preprocessing regression checks. */
int main() {
  pcod_common::PointPreprocessConfig config;
  config.x_min = -1.0f;
  config.x_max = 1.0f;
  config.y_min = -1.0f;
  config.y_max = 1.0f;
  config.z_min = -1.0f;
  config.z_max = 1.0f;
  config.normalization_type = pcod_common::PointFeatureNormalizationType::kValueThreshold;
  config.value_threshold = 10.0f;
  config.remove_points_in_zone = true;
  config.nd_x_min = -0.2f;
  config.nd_x_max = 0.2f;
  config.nd_y_min = -0.2f;
  config.nd_y_max = 0.2f;
  config.det_area_remove_outside = true;
  config.da_cx = 0.0f;
  config.da_cy = 0.0f;
  config.da_radius = 1.0f;
  config.da_bearing_rad = 0.0f;
  config.da_fov_rad = 1.5707963267948966f;

  pcod_common::PointPreprocessor preprocessor(config);

  assert(!preprocessor.IsPointValid(0.0f, 0.0f, 0.0f));
  assert(preprocessor.IsPointValid(0.6f, 0.0f, 0.0f));
  assert(!preprocessor.IsPointValid(0.0f, 0.6f, 0.0f));
  assert(!preprocessor.IsPointValid(2.0f, 0.0f, 0.0f));

  assert(std::abs(preprocessor.NormalizePointFeature(5.0f) - 0.5f) < 1e-6f);
  assert(std::abs(preprocessor.NormalizePointFeature(20.0f) - 1.0f) < 1e-6f);
  assert(std::abs(preprocessor.NormalizePointFeature(-1.0f) - 0.0f) < 1e-6f);

  {
    pcod_common::PointPreprocessConfig mm_cfg = config;
    mm_cfg.normalization_type = pcod_common::PointFeatureNormalizationType::kMinMax;
    mm_cfg.min_value = 10.0f;
    mm_cfg.max_value = 20.0f;
    pcod_common::PointPreprocessor minmax(mm_cfg);
    assert(std::abs(minmax.NormalizePointFeature(5.0f) - 0.0f) < 1e-6f);
    assert(std::abs(minmax.NormalizePointFeature(15.0f) - 0.5f) < 1e-6f);
    assert(std::abs(minmax.NormalizePointFeature(25.0f) - 1.0f) < 1e-6f);
  }

  {
    pcod_common::PointPreprocessConfig z_cfg = config;
    z_cfg.normalization_type = pcod_common::PointFeatureNormalizationType::kZScore;
    z_cfg.epsilon = 1e-3f;
    pcod_common::PointPreprocessor zscore(z_cfg);
    zscore.SetZScoreStats(10.0f, 2.0f);
    assert(std::abs(zscore.NormalizePointFeature(14.0f) - 2.0f) < 1e-6f);
  }

  {
    assert(pcod_common::ParsePointFeatureNormalizationType("none") == pcod_common::PointFeatureNormalizationType::kNone);
    assert(pcod_common::ParsePointFeatureNormalizationType("value_threshold") ==
           pcod_common::PointFeatureNormalizationType::kValueThreshold);
    assert(pcod_common::ParsePointFeatureNormalizationType("min_max") == pcod_common::PointFeatureNormalizationType::kMinMax);
    assert(pcod_common::ParsePointFeatureNormalizationType("z_score") == pcod_common::PointFeatureNormalizationType::kZScore);
    bool threw = false;
    try {
      (void)pcod_common::ParsePointFeatureNormalizationType("unsupported");
    } catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  return 0;
}
