#include "pcod_common/pbod_postprocess.hpp"
#include "pcod_common/pillar_grid.hpp"
#include "pcod_common/point_preprocess.hpp"

#include <cassert>

int main() {
  pcod_common::PointPreprocessConfig pre_cfg;
  pre_cfg.x_min = -1.0f;
  pre_cfg.x_max = 1.0f;
  pre_cfg.y_min = -1.0f;
  pre_cfg.y_max = 1.0f;
  pre_cfg.z_min = -1.0f;
  pre_cfg.z_max = 1.0f;
  pre_cfg.normalization_type = pcod_common::PointFeatureNormalizationType::kIntensityThreshold;
  pre_cfg.intensity_threshold = 10.0f;

  pcod_common::PointPreprocessor preprocessor(pre_cfg);
  assert(preprocessor.IsPointValid(0.5f, 0.1f, 0.0f));

  pcod_common::PillarGrid grid = pcod_common::BuildPillarGrid(
      {2, 2}, {{{0.0f, 2.0f}, {0.0f, 2.0f}, {0.0f, 1.0f}}}, 1, 1);

  float focal_logits[4] = {2.0f, -2.0f, 2.0f, -2.0f};
  float size_posterior[12] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float class_logits[8] = {0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f};
  float reg_logits[28] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits;
  view.size_posterior = size_posterior;
  view.class_logits = class_logits;
  view.reg_logits = reg_logits;
  const int num_pillars = 4;
  view.num_pillars = num_pillars;
  view.num_classes = 2;
  view.reg_dim = 7;

  pcod_common::PbodPostprocessConfig post_cfg;
  post_cfg.class_names = {"car", "pedestrian"};
  post_cfg.score_thresholds = {0.5f};

  auto boxes = pcod_common::DecodePbod(view, grid, post_cfg);
  const std::size_t expected_boxes = 2;
  assert(boxes.size() == expected_boxes);
  return 0;
}
