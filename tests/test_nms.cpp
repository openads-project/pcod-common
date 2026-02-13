#include "pcod_common/nms.hpp"

#include <cassert>
#include <cmath>

int main() {
  {
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
  }

  {
    pcod_common::BoundingBox keep_box;
    keep_box.center = {0.0f, 0.0f};
    keep_box.length = 1.0f;
    keep_box.width = 1.0f;
    keep_box.existence_probability = 0.6f;
    keep_box.classification.push_back({0, 0.9f});
    keep_box.classification.push_back({1, 0.1f});

    pcod_common::BoundingBox drop_box;
    drop_box.center = {10.0f, 10.0f};  // ensure no IoU suppression interaction
    drop_box.length = 1.0f;
    drop_box.width = 1.0f;
    drop_box.existence_probability = 0.4f;
    drop_box.classification.push_back({0, 0.1f});
    drop_box.classification.push_back({1, 0.9f});

    std::vector<pcod_common::BoundingBox> boxes{keep_box, drop_box};
    pcod_common::NmsConfig config;
    config.score_thresholds = {0.1f, 0.9f};
    config.iou_threshold = 0.1f;
    config.max_detections = 10;
    config.internal_score_threshold = 0.5f;

    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 1);
    assert(std::abs(boxes[0].center[0]) < 1e-6f);
    assert(std::abs(boxes[0].center[1]) < 1e-6f);
  }

  return 0;
}
