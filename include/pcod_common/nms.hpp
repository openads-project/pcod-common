#pragma once

#include <vector>

#include "pcod_common/bounding_box.hpp"

namespace pcod_common {

struct NmsConfig {
  std::vector<float> score_thresholds;
  float iou_threshold = 0.1f;
  int max_detections = 64;
  float internal_score_threshold = 0.5f;
};

void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config);

}  // namespace pcod_common
