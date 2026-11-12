// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "pcod_common/bounding_box.hpp"

namespace pcod_common {

/** Configuration for class-aware rotated NMS and optional class-independent duplicate merging. */
struct NmsConfig {
  std::vector<float> score_thresholds;    ///< Per-class thresholds, or one threshold shared by all classes.
  float iou_threshold = 0.1f;             ///< Rotated 3D IoU above which a lower-scored box is suppressed.
  int max_detections = 64;                ///< Maximum number of boxes retained.
  float internal_score_threshold = 0.5f;  ///< Fallback score threshold when no per-class thresholds are supplied.
  bool merge_enabled = false;             ///< Merge high-overlap candidates independently of their winning class.
  float merge_score_threshold =
      0.5f;  ///< Strict rotated 3D intersection/geometric-mean-volume score threshold for merging; must be in [0, 1].
};

/** Counts collected during one NMS/merging call, reset even for empty input. */
struct NmsStats {
  std::size_t candidates = 0;       ///< Candidates passing the score threshold.
  std::size_t iou_evaluations = 0;  ///< Exact polygon intersections after inexpensive overlap rejection.
  std::size_t merged_boxes = 0;     ///< Candidates merged into a retained representative.
};

/** Filter and reorder boxes in place.
 * Merging retains the strongest representative's geometry, score, existence and velocity.
 * Full class distributions are averaged using detection scores as weights (uniform if all are zero).
 * Classification scores remain logits: merged distributions are stored as log probabilities.
 * Membership and ordinary NMS use the original representative, preventing transitive merging
 * and changes in suppression caused by a fused winning class. Candidates are consumed past
 * the output limit when merging is enabled, so late members can still contribute.
 * Boxes without positive heights can be suppressed in BEV, but are never merged.
 * @param bboxes Boxes to process.
 * @param config Suppression and merging configuration.
 * @param stats Optional operation counts.
 * @throws std::invalid_argument If merging is enabled with an invalid threshold or class logits.
 */
void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config, NmsStats* stats);

/** Filter boxes without collecting operation counts, preserving the original two-argument API.
 * @param bboxes Boxes to process.
 * @param config Suppression and merging configuration.
 */
void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config);

}  // namespace pcod_common
