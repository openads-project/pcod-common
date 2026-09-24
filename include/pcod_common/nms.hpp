// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "pcod_common/bounding_box.hpp"

namespace pcod_common {

/** Configuration for class-aware rotated non-maximum suppression. */
struct NmsConfig {
  std::vector<float> score_thresholds;    ///< Per-class thresholds, or one threshold shared by all classes.
  float iou_threshold = 0.1f;             ///< Rotated 3D IoU above which a lower-scored box is suppressed.
  int max_detections = 64;                ///< Maximum number of boxes retained.
  float internal_score_threshold = 0.5f;  ///< Fallback score threshold when no per-class thresholds are supplied.
};

/** Filter and reorder boxes in place.
 * @param bboxes Boxes to process.
 * @param config Suppression configuration.
 */
void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config);

}  // namespace pcod_common
