// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pbod_postprocess.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "pcod_common/pillar_grid.hpp"

/** Run PBOD decoder regression checks. */
int main() {
  {
    pcod_common::PillarGrid grid = pcod_common::BuildPillarGrid({1, 1}, {{{0.0F, 1.0F}, {0.0F, 1.0F}, {0.0F, 1.0F}}}, 1, 1);

    const int num_pillars = 1;
    const int num_classes = 2;
    float focal_logits[num_pillars] = {0.0F};
    std::vector<float> size_posterior(static_cast<std::size_t>(num_pillars * num_classes * 3), 1.0F);
    float class_logits[num_pillars * num_classes] = {0.1F, 0.9F};
    std::vector<float> reg_logits(static_cast<std::size_t>(num_pillars * num_classes * 7), 0.0F);

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
    assert(std::abs(box.length - 1.0F) < 1e-6F);
    assert(std::abs(box.width - 1.0F) < 1e-6F);
    assert(std::abs(box.height - 1.0F) < 1e-6F);
    assert(std::abs(box.center[0] - 0.5F) < 1e-6F);
    assert(std::abs(box.center[1] - 0.5F) < 1e-6F);
  }

  {
    pcod_common::PillarGrid grid = pcod_common::BuildPillarGrid({2, 1}, {{{0.0F, 2.0F}, {0.0F, 1.0F}, {0.0F, 2.0F}}}, 1, 1);

    const int num_pillars = 2;
    const int num_classes = 2;
    float focal_logits[num_pillars] = {0.0F, 2.0F};
    std::vector<float> size_posterior = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    std::vector<float> class_logits = {0.7F, 0.2F, 0.1F, 1.2F};
    std::vector<float> reg_logits(static_cast<std::size_t>(num_pillars * num_classes * 7), 0.0F);
    reg_logits[0] = 0.25F;
    reg_logits[1] = -0.5F;
    reg_logits[2] = 0.75F;
    reg_logits[3] = std::log(2.0F);
    reg_logits[4] = std::log(0.5F);
    reg_logits[5] = std::log(1.5F);
    reg_logits[6] = 3.5F;

    const std::size_t second_pillar_class_one = static_cast<std::size_t>((1 * num_classes + 1) * 7);
    reg_logits[second_pillar_class_one + 0] = -0.2F;
    reg_logits[second_pillar_class_one + 1] = 0.3F;
    reg_logits[second_pillar_class_one + 2] = -0.4F;
    reg_logits[second_pillar_class_one + 3] = std::log(0.5F);
    reg_logits[second_pillar_class_one + 4] = std::log(2.0F);
    reg_logits[second_pillar_class_one + 5] = std::log(1.0F);
    reg_logits[second_pillar_class_one + 6] = -3.5F;

    pcod_common::PbodOutputsView view;
    view.focal_logits = focal_logits;
    view.size_posterior = size_posterior.data();
    view.class_logits = class_logits.data();
    view.reg_logits = reg_logits.data();
    view.num_pillars = num_pillars;
    view.num_classes = num_classes;
    view.reg_dim = 7;

    pcod_common::PbodPostprocessConfig config;
    config.class_names = {"car", "pedestrian"};
    config.score_thresholds = {0.4F, 0.8F};

    auto boxes = pcod_common::DecodePbod(view, grid, config);
    assert(boxes.size() == 2);

    const auto& first = boxes[0];
    assert(first.classification[0].class_idx == 0);
    assert(std::abs(first.existence_probability - 0.5F) < 1e-6F);
    assert(std::abs(first.length - 2.0F) < 1e-6F);
    assert(std::abs(first.width - 1.0F) < 1e-6F);
    assert(std::abs(first.height - 4.5F) < 1e-6F);
    assert(std::abs(first.center[0] - 0.75F) < 1e-6F);
    assert(std::abs(first.center[1] + 0.5F) < 1e-6F);
    assert(std::abs(first.z - 2.25F) < 1e-6F);
    assert(std::abs(first.yaw + 2.7831855F) < 1e-5F);

    const auto& second = boxes[1];
    assert(second.classification[1].class_idx == 1);
    assert(std::abs(second.existence_probability - 0.880797F) < 1e-5F);
    assert(std::abs(second.length - 5.0F) < 1e-6F);
    assert(std::abs(second.width - 22.0F) < 1e-6F);
    assert(std::abs(second.height - 12.0F) < 1e-6F);
    assert(std::abs(second.center[0] + 0.5F) < 1e-6F);
    assert(std::abs(second.center[1] - 3.8F) < 1e-6F);
    assert(std::abs(second.z + 4.8F) < 1e-6F);
    assert(std::abs(second.yaw - 2.7831855F) < 1e-5F);
  }
  return 0;
}
