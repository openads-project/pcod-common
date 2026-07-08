// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/nms.hpp"

#include <algorithm>

#include "pcod_common/math.hpp"

namespace pcod_common {

void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config) {
  if (bboxes.empty()) {
    return;
  }

  std::vector<std::pair<float, BoundingBox*>> scored;
  scored.reserve(bboxes.size());

  for (auto& bbox : bboxes) {
    float max_class_score = 0.0F;
    std::size_t max_class_idx = 0;
    for (const auto& entry : bbox.classification) {
      if (entry.score > max_class_score) {
        max_class_score = entry.score;
        max_class_idx = entry.class_idx;
      }
    }
    float class_thresh = config.internal_score_threshold;
    if (max_class_idx < config.score_thresholds.size()) {
      class_thresh = config.score_thresholds[max_class_idx];
    }
    float score = scale_score(bbox.existence_probability, class_thresh, config.internal_score_threshold);
    scored.emplace_back(score, &bbox);
  }

  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<BoundingBox> kept;
  kept.reserve(bboxes.size());

  for (const auto& candidate : scored) {
    if (candidate.first < config.internal_score_threshold) {
      break;
    }
    bool keep = true;
    for (const auto& kept_box : kept) {
      if (candidate.second->overlaps(kept_box, config.iou_threshold)) {
        keep = false;
        break;
      }
    }
    if (keep) {
      kept.push_back(*candidate.second);
      if (static_cast<int>(kept.size()) >= config.max_detections) {
        break;
      }
    }
  }

  bboxes = std::move(kept);
}

}  // namespace pcod_common
