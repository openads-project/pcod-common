#include "pcod_common/point_preprocess.hpp"

#include <cassert>
#include <cmath>

int main() {
  pcod_common::PointPreprocessConfig config;
  config.x_min = -1.0f;
  config.x_max = 1.0f;
  config.y_min = -1.0f;
  config.y_max = 1.0f;
  config.z_min = -1.0f;
  config.z_max = 1.0f;
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
  return 0;
}
