// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/nms.hpp"

#include <algorithm>
#include <limits>

namespace pcod_common {

void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config) {
  if (config.max_detections <= 0) {
    bboxes.clear();
    return;
  }
  if (bboxes.empty()) {
    return;
  }

  auto best_class = [](const BoundingBox& box) {
    float best = -std::numeric_limits<float>::infinity();
    std::size_t index = 0;
    for (const auto& entry : box.classification) {
      if (entry.score > best) {
        best = entry.score;
        index = entry.class_idx;
      }
    }
    return index;
  };
  std::vector<std::pair<float, BoundingBox*>> scored;
  scored.reserve(bboxes.size());

  for (auto& bbox : bboxes) {
    const std::size_t max_class_idx = best_class(bbox);
    float class_thresh = config.internal_score_threshold;
    if (!config.score_thresholds.empty()) {
      class_thresh = config.score_thresholds[std::min(max_class_idx, config.score_thresholds.size() - 1)];
    }
    const float score = bbox.existence_probability;
    if (score < class_thresh) {
      continue;
    }
    scored.emplace_back(score, &bbox);
  }

  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<BoundingBox> kept;
  kept.reserve(bboxes.size());

  for (const auto& candidate : scored) {
    bool keep = true;
    for (const auto& kept_box : kept) {
      if (best_class(*candidate.second) == best_class(kept_box) && candidate.second->overlaps(kept_box, config.iou_threshold)) {
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
