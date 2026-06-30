// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "pcod_common/bounding_box.hpp"
#include "pcod_common/pillar_grid.hpp"

namespace pcod_common {

struct PbodPostprocessConfig {
  std::vector<std::string> class_names;
  std::vector<float> score_thresholds;
};

struct PbodOutputsView {
  const float* focal_logits = nullptr;       // [num_pillars]
  const float* size_posterior = nullptr;     // [num_pillars, num_classes * 3]
  const float* class_logits = nullptr;       // [num_pillars, num_classes]
  const float* reg_logits = nullptr;         // [num_pillars, num_classes * reg_dim]
  int num_pillars = 0;
  int num_classes = 0;
  int reg_dim = 0;
};

std::vector<BoundingBox> DecodePbod(
    const PbodOutputsView& outputs,
    const PillarGrid& grid,
    const PbodPostprocessConfig& config);

}  // namespace pcod_common
