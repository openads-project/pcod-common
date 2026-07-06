// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "pcod_common/bounding_box.hpp"
#include "pcod_common/pillar_grid.hpp"

namespace pcod_common {

/** Semantic configuration for PBOD output decoding. */
struct PbodPostprocessConfig {
  std::vector<std::string> class_names;  ///< Class name in model-output order.
  std::vector<float> score_thresholds;   ///< Per-class thresholds, or one shared threshold.
};

/** Non-owning view over the dense tensors emitted by a PBOD head. */
struct PbodOutputsView {
  const float* focal_logits = nullptr;    ///< Shape `[num_pillars]`.
  const float* size_posterior = nullptr;  ///< Shape `[num_pillars, num_classes * 3]`.
  const float* class_logits = nullptr;    ///< Shape `[num_pillars, num_classes]`.
  const float* reg_logits = nullptr;      ///< Shape `[num_pillars, num_classes * reg_dim]`.
  int num_pillars = 0;                    ///< Number of spatial pillars.
  int num_classes = 0;                    ///< Number of semantic classes.
  int reg_dim = 0;                        ///< Regression values per pillar and class.
};

/** Decode PBOD tensors into oriented boxes.
 * @param outputs Non-owning model-output tensors.
 * @param grid Spatial pillar centers.
 * @param config Class names and thresholds.
 * @return Decoded oriented boxes.
 */
std::vector<BoundingBox> DecodePbod(const PbodOutputsView& outputs, const PillarGrid& grid, const PbodPostprocessConfig& config);

}  // namespace pcod_common
