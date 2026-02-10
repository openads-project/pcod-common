#include "pcod_common/pbod_postprocess.hpp"

#include <cmath>

#include "pcod_common/math.hpp"

namespace pcod_common {

namespace {
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}

std::vector<BoundingBox> DecodePbod(
    const PbodOutputsView& outputs,
    const PillarGrid& grid,
    const PbodPostprocessConfig& config) {
  std::vector<BoundingBox> objects;
  if (outputs.num_pillars <= 0 || outputs.num_classes <= 0) {
    return objects;
  }

  const int reg_dim = outputs.reg_dim > 0 ? outputs.reg_dim : 7;

  objects.reserve(static_cast<std::size_t>(outputs.num_pillars));
  for (int idx = 0; idx < outputs.num_pillars; ++idx) {
    const float score = sigmoid(outputs.focal_logits[idx]);

    int best_class = 0;
    float best_logit = outputs.class_logits[idx * outputs.num_classes];
    for (int c = 1; c < outputs.num_classes; ++c) {
      float logit = outputs.class_logits[idx * outputs.num_classes + c];
      if (logit > best_logit) {
        best_logit = logit;
        best_class = c;
      }
    }

    float score_thresh = 0.0f;
    if (!config.score_thresholds.empty()) {
      const std::size_t class_idx = static_cast<std::size_t>(best_class);
      score_thresh = class_idx < config.score_thresholds.size() ? config.score_thresholds[class_idx]
                                                                 : config.score_thresholds.front();
    }

    if (score < score_thresh) {
      continue;
    }

    BoundingBox box;
    const int class_offset = best_class * reg_dim;
    const int size_offset = best_class * 3;
    const float* size_ptr = outputs.size_posterior + idx * outputs.num_classes * 3 + size_offset;
    const float* reg_ptr = outputs.reg_logits + idx * outputs.num_classes * reg_dim + class_offset;
    const float* center = grid.center_at(idx);

    box.length = std::exp(reg_ptr[3]) * size_ptr[0];
    box.width = std::exp(reg_ptr[4]) * size_ptr[1];
    box.height = std::exp(reg_ptr[5]) * size_ptr[2];
    box.center[0] = reg_ptr[0] * size_ptr[0] + center[0];
    box.center[1] = reg_ptr[1] * size_ptr[1] + center[1];
    box.z = reg_ptr[2] * size_ptr[2] + center[2];
    box.yaw = wrap_to_range(reg_ptr[6], -static_cast<float>(M_PI), static_cast<float>(M_PI));
    if (std::isnan(box.length) || std::isnan(box.width) || std::isnan(box.height) || std::isnan(box.yaw)) {
      continue;
    }

    for (int c = 0; c < outputs.num_classes; ++c) {
      box.classification.push_back({static_cast<std::size_t>(c), outputs.class_logits[idx * outputs.num_classes + c]});
    }
    box.existence_probability = score;

    objects.push_back(std::move(box));
  }

  return objects;
}

}  // namespace pcod_common
