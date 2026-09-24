// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pbod_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pcod_common/math.hpp"

namespace pcod_common {

namespace {
inline float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

std::vector<float> CopyTensorView(const float* values, std::size_t count) {
  std::vector<float> copy(count);
  std::copy_n(values, count, copy.begin());
  return copy;
}
}  // namespace

std::vector<BoundingBox> DecodePbod(const PbodOutputsView& outputs, const PillarGrid& grid, const PbodPostprocessConfig& config) {
  std::vector<BoundingBox> objects;
  if (outputs.focal_logits == nullptr || outputs.objectness_logits == nullptr || outputs.size_posterior == nullptr ||
      outputs.class_logits == nullptr || outputs.reg_logits == nullptr) {
    throw std::invalid_argument("DecodePbod requires non-null output tensor pointers.");
  }
  if (outputs.num_pillars <= 0 || outputs.num_classes <= 0) {
    return objects;
  }
  if (outputs.reg_dim > 0 && outputs.reg_dim < 7) {
    throw std::invalid_argument("DecodePbod requires reg_dim >= 7.");
  }
  if (grid.centers.size() < static_cast<std::size_t>(outputs.num_pillars) * 3) {
    throw std::invalid_argument("DecodePbod received fewer pillar centers than num_pillars.");
  }

  const int reg_dim = outputs.reg_dim > 0 ? outputs.reg_dim : 7;
  const std::size_t num_pillars = static_cast<std::size_t>(outputs.num_pillars);
  const std::size_t num_classes = static_cast<std::size_t>(outputs.num_classes);
  const std::size_t reg_dim_size = static_cast<std::size_t>(reg_dim);

  const std::vector<float> focal_logits = CopyTensorView(outputs.focal_logits, num_pillars);
  const std::vector<float> objectness_logits = CopyTensorView(outputs.objectness_logits, num_pillars);
  const std::vector<float> size_posterior = CopyTensorView(outputs.size_posterior, num_pillars * num_classes * 3U);
  const std::vector<float> class_logits = CopyTensorView(outputs.class_logits, num_pillars * num_classes);
  const std::vector<float> reg_logits = CopyTensorView(outputs.reg_logits, num_pillars * num_classes * reg_dim_size);

  objects.reserve(static_cast<std::size_t>(outputs.num_pillars));
  for (int idx = 0; idx < outputs.num_pillars; ++idx) {
    const std::size_t pillar_idx = static_cast<std::size_t>(idx);
    const float quality_weighted_presence = sigmoid(focal_logits[pillar_idx]);
    const float objectness = sigmoid(objectness_logits[pillar_idx]);

    int best_class = 0;
    const std::size_t class_base = pillar_idx * num_classes;
    float best_logit = class_logits[class_base];
    for (int c = 1; c < outputs.num_classes; ++c) {
      float logit = class_logits[class_base + static_cast<std::size_t>(c)];
      if (logit > best_logit) {
        best_logit = logit;
        best_class = c;
      }
    }

    float class_denom = 0.0F;
    for (int c = 0; c < outputs.num_classes; ++c) {
      class_denom += std::exp(class_logits[class_base + static_cast<std::size_t>(c)] - best_logit);
    }
    const float score = quality_weighted_presence / class_denom;
    if (!std::isfinite(score) || !std::isfinite(objectness)) {
      continue;
    }
    float score_thresh = 0.0F;
    if (!config.score_thresholds.empty()) {
      const std::size_t class_idx = static_cast<std::size_t>(best_class);
      score_thresh =
          class_idx < config.score_thresholds.size() ? config.score_thresholds[class_idx] : config.score_thresholds.front();
    }

    if (score < score_thresh) {
      continue;
    }

    BoundingBox box;
    const std::size_t class_idx = static_cast<std::size_t>(best_class);
    const std::size_t size_offset = (pillar_idx * num_classes + class_idx) * 3U;
    const std::size_t reg_offset = (pillar_idx * num_classes + class_idx) * reg_dim_size;
    const std::size_t center_offset = pillar_idx * 3U;

    box.length = std::exp(std::clamp(reg_logits[reg_offset + 3U], -10.0F, 10.0F)) * size_posterior[size_offset + 0U];
    box.width = std::exp(std::clamp(reg_logits[reg_offset + 4U], -10.0F, 10.0F)) * size_posterior[size_offset + 1U];
    box.height = std::exp(std::clamp(reg_logits[reg_offset + 5U], -10.0F, 10.0F)) * size_posterior[size_offset + 2U];
    box.center[0] = reg_logits[reg_offset + 0U] * size_posterior[size_offset + 0U] + grid.centers[center_offset + 0U];
    box.center[1] = reg_logits[reg_offset + 1U] * size_posterior[size_offset + 1U] + grid.centers[center_offset + 1U];
    box.z = reg_logits[reg_offset + 2U] * size_posterior[size_offset + 2U] + grid.centers[center_offset + 2U];
    box.yaw = wrap_to_range(reg_logits[reg_offset + 6U], -static_cast<float>(M_PI), static_cast<float>(M_PI));
    if (!std::isfinite(box.length) || !std::isfinite(box.width) || !std::isfinite(box.height) || !std::isfinite(box.yaw) ||
        !std::isfinite(box.center[0]) || !std::isfinite(box.center[1]) || !std::isfinite(box.z)) {
      continue;
    }

    for (int c = 0; c < outputs.num_classes; ++c) {
      const std::size_t current_class = static_cast<std::size_t>(c);
      box.classification.push_back({current_class, class_logits[class_base + current_class]});
    }
    box.existence_probability = objectness;
    box.detection_score = score;

    objects.push_back(std::move(box));
  }

  return objects;
}

}  // namespace pcod_common
