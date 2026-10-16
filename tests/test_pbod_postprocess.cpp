// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pbod_postprocess.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "pcod_common/nms.hpp"
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
    view.objectness_logits = focal_logits;
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
    assert(box.detection_score.has_value());
    assert(std::abs(*box.detection_score - 0.5F) < 1e-6F);
    assert(!box.has_velocity);
    assert(std::abs(box.length - 1.0F) < 1e-6F);
    assert(std::abs(box.width - 1.0F) < 1e-6F);
    assert(std::abs(box.height - 1.0F) < 1e-6F);
    assert(std::abs(box.center[0] - 0.5F) < 1e-6F);
    assert(std::abs(box.center[1] - 0.5F) < 1e-6F);
    // Masked dense cells stay absent even with a zero confidence threshold.
    focal_logits[0] = -10000.0F;
    for (auto mode : {pcod_common::PbodScoreMode::Existence, pcod_common::PbodScoreMode::ExistenceQuality,
                      pcod_common::PbodScoreMode::ExistenceQualityClass}) {
      config.score_mode = mode;
      assert(pcod_common::DecodePbod(view, grid, config).empty());
    }
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
    float objectness_logits[num_pillars] = {2.0F, 0.0F};
    view.objectness_logits = objectness_logits;
    view.size_posterior = size_posterior.data();
    view.class_logits = class_logits.data();
    view.reg_logits = reg_logits.data();
    view.num_pillars = num_pillars;
    view.num_classes = num_classes;
    view.reg_dim = 7;

    pcod_common::PbodPostprocessConfig config;
    config.class_names = {"car", "pedestrian"};
    config.score_thresholds = {0.3F, 0.6F};
    config.score_mode = pcod_common::PbodScoreMode::ExistenceQualityClass;

    auto boxes = pcod_common::DecodePbod(view, grid, config);
    assert(boxes.size() == 2);

    const auto& first = boxes[0];
    assert(first.classification[0].class_idx == 0);
    assert(std::abs(first.existence_probability - 0.880797F) < 1e-5F);
    assert(first.detection_score.has_value());
    assert(std::abs(*first.detection_score - 0.5F / (1.0F + std::exp(-0.5F))) < 1e-6F);
    assert(std::abs(first.length - 2.0F) < 1e-6F);
    assert(std::abs(first.width - 1.0F) < 1e-6F);
    assert(std::abs(first.height - 4.5F) < 1e-6F);
    assert(std::abs(first.center[0] - 0.75F) < 1e-6F);
    assert(std::abs(first.center[1] + 0.5F) < 1e-6F);
    assert(std::abs(first.z - 2.25F) < 1e-6F);
    assert(std::abs(first.yaw + 2.7831855F) < 1e-5F);

    const auto& second = boxes[1];
    assert(second.classification[1].class_idx == 1);
    assert(std::abs(second.existence_probability - 0.5F) < 1e-6F);
    assert(second.detection_score.has_value());
    assert(std::abs(*second.detection_score - 0.880797F / (1.0F + std::exp(-1.1F))) < 1e-5F);
    assert(std::abs(second.length - 5.0F) < 1e-6F);
    assert(std::abs(second.width - 22.0F) < 1e-6F);
    assert(std::abs(second.height - 12.0F) < 1e-6F);
    assert(std::abs(second.center[0] + 0.5F) < 1e-6F);
    assert(std::abs(second.center[1] - 3.8F) < 1e-6F);
    assert(std::abs(second.z + 4.8F) < 1e-6F);
    assert(std::abs(second.yaw - 2.7831855F) < 1e-5F);
  }
  {
    // Overlapping boxes have different winners for each confidence combination.
    auto grid = pcod_common::BuildPillarGrid({3, 1}, {{{0.0F, 3.0F}, {0.0F, 1.0F}, {0.0F, 1.0F}}}, 1, 1);
    const auto logit = [](float probability) { return std::log(probability / (1.0F - probability)); };
    float focal_logits[] = {logit(0.4F), logit(0.8F), logit(0.7F)};
    float objectness_logits[] = {logit(0.9F), logit(0.7F), logit(0.6F)};
    float class_logits[] = {logit(0.99F), 0.0F, logit(0.55F), 0.0F, logit(0.95F), 0.0F};
    std::vector<float> sizes(18, 1.0F);
    std::vector<float> regression(42, 0.0F);
    for (int i = 0; i < 3; ++i) {
      regression[static_cast<std::size_t>(i * 14)] = -static_cast<float>(i);
    }
    pcod_common::PbodOutputsView view;
    view.focal_logits = focal_logits;
    view.objectness_logits = objectness_logits;
    view.class_logits = class_logits;
    view.size_posterior = sizes.data();
    view.reg_logits = regression.data();
    view.num_pillars = 3;
    view.num_classes = 2;
    view.reg_dim = 7;

    pcod_common::PbodPostprocessConfig config;
    assert(config.score_mode == pcod_common::PbodScoreMode::ExistenceQuality);
    const std::vector<pcod_common::PbodScoreMode> modes = {pcod_common::PbodScoreMode::Existence,
                                                           pcod_common::PbodScoreMode::ExistenceQuality,
                                                           pcod_common::PbodScoreMode::ExistenceQualityClass};
    const float expected_scores[3][3] = {{0.9F, 0.7F, 0.6F}, {0.4F, 0.8F, 0.7F}, {0.396F, 0.44F, 0.665F}};
    const std::size_t expected_filtered_counts[] = {2, 2, 1};
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
      config.score_mode = modes[mode];
      config.score_thresholds = {0.0F};
      auto boxes = pcod_common::DecodePbod(view, grid, config);
      assert(boxes.size() == 3);
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        assert(std::abs(*boxes[i].detection_score - expected_scores[mode][i]) < 1e-6F);
        assert(std::abs(boxes[i].existence_probability - 1.0F / (1.0F + std::exp(-objectness_logits[i]))) < 1e-6F);
        assert(boxes[i].classification[0].score == class_logits[i * 2]);
      }
      pcod_common::NmsConfig nms_config;
      nms_config.score_thresholds = {0.0F};
      nms_config.iou_threshold = 0.1F;
      pcod_common::ApplyRotatedNms(boxes, nms_config);
      assert(boxes.size() == 1);
      assert(std::abs(*boxes[0].detection_score - expected_scores[mode][mode]) < 1e-6F);

      config.score_thresholds = {0.65F};
      boxes = pcod_common::DecodePbod(view, grid, config);
      assert(boxes.size() == expected_filtered_counts[mode]);
      // NMS independently applies the same selected score threshold.
      config.score_thresholds = {0.0F};
      boxes = pcod_common::DecodePbod(view, grid, config);
      nms_config.score_thresholds = {0.65F};
      nms_config.iou_threshold = 1.0F;
      pcod_common::ApplyRotatedNms(boxes, nms_config);
      assert(boxes.size() == expected_filtered_counts[mode]);
    }
  }
  return 0;
}
