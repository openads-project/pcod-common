// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pbod_postprocess.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "pcod_common/pillar_grid.hpp"

int main() {
  pcod_common::PillarGrid grid = pcod_common::BuildPillarGrid(
      {1, 1}, {{{0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}}}, 1, 1);

  const int num_pillars = 1;
  const int num_classes = 2;
  float focal_logits[num_pillars] = {0.0f};
  std::vector<float> size_posterior(static_cast<std::size_t>(num_pillars * num_classes * 3), 1.0f);
  float class_logits[num_pillars * num_classes] = {0.1f, 0.9f};
  std::vector<float> reg_logits(static_cast<std::size_t>(num_pillars * num_classes * 7), 0.0f);

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits;
  view.size_posterior = size_posterior.data();
  view.class_logits = class_logits;
  view.reg_logits = reg_logits.data();
  view.num_pillars = num_pillars;
  view.num_classes = num_classes;
  view.reg_dim = 7;

  pcod_common::PbodPostprocessConfig config;
  config.class_names = {"car", "pedestrian"};

  auto boxes = pcod_common::DecodePbod(view, grid, config);
  assert(boxes.size() == 1);
  const auto& box = boxes[0];
  assert(box.classification.size() == 2);
  assert(!box.has_velocity);
  assert(std::abs(box.length - 1.0f) < 1e-6f);
  assert(std::abs(box.width - 1.0f) < 1e-6f);
  assert(std::abs(box.height - 1.0f) < 1e-6f);
  assert(std::abs(box.center[0] - 0.5f) < 1e-6f);
  assert(std::abs(box.center[1] - 0.5f) < 1e-6f);
  return 0;
}
