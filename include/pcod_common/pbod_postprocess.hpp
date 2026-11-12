// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "pcod_common/bounding_box.hpp"
#include "pcod_common/pillar_grid.hpp"

namespace pcod_common {

/** Confidence factors used for PBOD score filtering and NMS ranking. */
enum class PbodScoreMode { Existence, ExistenceQuality, ExistenceQualityClass };

/** Parse a runtime score mode.
 * @param value Runtime confidence selection name.
 * @return Parsed confidence selection.
 * @throws std::invalid_argument for unsupported values.
 */
PbodScoreMode ParsePbodScoreMode(const std::string& value);

/** Semantic configuration for PBOD output decoding. */
struct PbodPostprocessConfig {
  std::vector<std::string> class_names;                        ///< Class name in model-output order.
  std::vector<float> score_thresholds;                         ///< Per-class thresholds, or one shared threshold.
  PbodScoreMode score_mode = PbodScoreMode::ExistenceQuality;  ///< Confidence factors used for detection scores.
};

/** Non-owning view over the dense tensors emitted by a PBOD head. */
struct PbodOutputsView {
  const float* focal_logits = nullptr;       ///< Quality-weighted presence logits, shape `[num_pillars]`.
  const float* objectness_logits = nullptr;  ///< Presence logits, shape `[num_pillars]`.
  const float* size_posterior = nullptr;     ///< Shape `[num_pillars, num_classes * 3]`.
  const float* class_logits = nullptr;       ///< Shape `[num_pillars, num_classes]`.
  const float* reg_logits = nullptr;         ///< Shape `[num_pillars, num_classes * reg_dim]`.
  int num_pillars = 0;                       ///< Number of spatial pillars.
  int num_classes = 0;                       ///< Number of semantic classes.
  int reg_dim = 0;                           ///< Regression values per pillar and class.
};

/** Decode PBOD tensors into oriented boxes.
 * @param outputs Non-owning model-output tensors.
 * @param grid Spatial pillar centers.
 * @param config Class names, thresholds, and confidence selection.
 * @return Decoded oriented boxes.
 */
std::vector<BoundingBox> DecodePbod(const PbodOutputsView& outputs, const PillarGrid& grid, const PbodPostprocessConfig& config);

}  // namespace pcod_common
