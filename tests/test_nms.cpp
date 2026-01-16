#include "pcod_common/nms.hpp"

#include <cassert>

int main() {
  pcod_common::BoundingBox box_a;
  box_a.center = {0.0f, 0.0f};
  box_a.length = 4.0f;
  box_a.width = 2.0f;
  box_a.existence_probability = 0.9f;
  box_a.classification.push_back({0, 1.0f});

  pcod_common::BoundingBox box_b = box_a;
  box_b.center = {0.5f, 0.0f};
  box_b.existence_probability = 0.8f;

  std::vector<pcod_common::BoundingBox> boxes{box_a, box_b};

  pcod_common::NmsConfig config;
  config.score_thresholds = {0.1f};
  config.iou_threshold = 0.1f;
  config.max_detections = 1;

  pcod_common::ApplyRotatedNms(boxes, config);
  assert(boxes.size() == 1);
  return 0;
}
