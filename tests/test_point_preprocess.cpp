#include "pcod_common/point_preprocess.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main() {
  pcod_common::PointPreprocessConfig config;
  config.x_min = -1.0f;
  config.x_max = 1.0f;
  config.y_min = -1.0f;
  config.y_max = 1.0f;
  config.z_min = -1.0f;
  config.z_max = 1.0f;
  config.normalization_type = pcod_common::PointFeatureNormalizationType::kIntensityThreshold;
  config.intensity_threshold = 10.0f;
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

  assert(std::abs(preprocessor.NormalizeIntensity(5.0f) - 0.5f) < 1e-6f);
  assert(std::abs(preprocessor.NormalizeIntensity(20.0f) - 1.0f) < 1e-6f);
  assert(std::abs(preprocessor.NormalizeIntensity(-1.0f) - 0.0f) < 1e-6f);

  config.zero_intensity = true;
  pcod_common::PointPreprocessor zeroed(config);
  assert(std::abs(zeroed.NormalizeIntensity(5.0f)) < 1e-6f);

  {
    pcod_common::PointPreprocessConfig mm_cfg = config;
    mm_cfg.zero_intensity = false;
    mm_cfg.normalization_type = pcod_common::PointFeatureNormalizationType::kMinMax;
    mm_cfg.min_intensity = 10.0f;
    mm_cfg.max_intensity = 20.0f;
    pcod_common::PointPreprocessor minmax(mm_cfg);
    assert(std::abs(minmax.NormalizeIntensity(5.0f) - 0.0f) < 1e-6f);
    assert(std::abs(minmax.NormalizeIntensity(15.0f) - 0.5f) < 1e-6f);
    assert(std::abs(minmax.NormalizeIntensity(25.0f) - 1.0f) < 1e-6f);
  }

  {
    pcod_common::PointPreprocessConfig z_cfg = config;
    z_cfg.zero_intensity = false;
    z_cfg.normalization_type = pcod_common::PointFeatureNormalizationType::kZScore;
    z_cfg.epsilon = 1e-3f;
    pcod_common::PointPreprocessor zscore(z_cfg);
    zscore.SetZScoreStats(10.0f, 2.0f);
    assert(std::abs(zscore.NormalizeIntensity(14.0f) - 2.0f) < 1e-6f);
  }

  {
    assert(pcod_common::ParsePointFeatureNormalizationType("none") ==
           pcod_common::PointFeatureNormalizationType::kNone);
    assert(pcod_common::ParsePointFeatureNormalizationType("intensity_threshold") ==
           pcod_common::PointFeatureNormalizationType::kIntensityThreshold);
    assert(pcod_common::ParsePointFeatureNormalizationType("min_max") ==
           pcod_common::PointFeatureNormalizationType::kMinMax);
    assert(pcod_common::ParsePointFeatureNormalizationType("z_score") ==
           pcod_common::PointFeatureNormalizationType::kZScore);
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
