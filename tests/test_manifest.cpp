// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/model_manifest.hpp"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>

/** Run manifest parsing and unknown-key validation checks. */
int main() {
  const std::string path = "./test_manifest.yml";
  std::ofstream out(path);
  out << "schema_version: '2.0'\n";
  out << "artifact:\n";
  out << "  bundle_name: 'pbod_fp16_gpu_onnx_test'\n";
  out << "  export_format: 'onnx_fp16_gpu'\n";
  out << "  backend: 'onnx'\n";
  out << "  precision: 'fp16'\n";
  out << "  device: 'cuda'\n";
  out << "  files:\n";
  out << "    model: 'model.onnx'\n";
  out << "    checkpoint: 'checkpoints/best.pt'\n";
  out << "    resolved_training_config: 'config/resolved_training_config.yml'\n";
  out << "  triton:\n";
  out << "    enabled: false\n";
  out << "  inputs:\n";
  out << "    - name: 'point_features'\n";
  out << "      dtype: 'float16'\n";
  out << "      shape: ['batch', 100, 1]\n";
  out << "  outputs:\n";
  out << "    - name: 'reg_logits'\n";
  out << "      dtype: 'float16'\n";
  out << "      shape: ['batch', 10, 7]\n";
  out << "frozen_contract:\n";
  out << "  preprocessing:\n";
  out << "    max_num_points: 100\n";
  out << "    num_point_features: 1\n";
  out << "    point_cloud_range:\n";
  out << "      x: [-1.0, 1.0]\n";
  out << "      y: [-1.0, 1.0]\n";
  out << "      z: [-1.0, 1.0]\n";
  out << "    voxel_size:\n";
  out << "      x: 0.1\n";
  out << "      y: 0.1\n";
  out << "      z: 0.1\n";
  out << "    point_feature_normalization:\n";
  out << "      type: value_threshold\n";
  out << "      epsilon: 1e-6\n";
  out << "  postprocessing:\n";
  out << "    grid_size:\n";
  out << "      x: 10\n";
  out << "      y: 10\n";
  out << "    num_classes: 1\n";
  out << "    class_names: ['car']\n";
  out << "  model:\n";
  out << "    stride: [2, 1, 2]\n";
  out << "    up_stride: [1, 1, 2]\n";
  out << "    first_up_stride: 1\n";
  out << "    pillar_map_size: [10, 10]\n";
  out << "    pillar_map_range: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]\n";
  out << "runtime_defaults:\n";
  out << "  preprocessing:\n";
  out << "    point_feature:\n";
  out << "      value_threshold: 1.0\n";
  out << "  postprocessing:\n";
  out << "    class_score_threshold: 0.0\n";
  out << "    nms:\n";
  out << "      score_threshold: 0.2\n";
  out << "      iou_threshold: 0.5\n";
  out << "      max_num_objects: 10\n";
  out.close();

  auto manifest = pcod_common::LoadModelManifest(path);
  pcod_common::ValidateModelManifest(manifest);
  assert(manifest.frozen_contract.preprocessing.max_num_points == 100);
  assert(manifest.runtime_defaults.preprocessing.point_feature.value_threshold == 1.0f);

  const std::string invalid_path = "./test_manifest_invalid.yml";
  std::ofstream invalid(invalid_path);
  invalid << "schema_version: '2.0'\n";
  invalid << "artifact:\n";
  invalid << "  bundle_name: 'pbod_fp16_gpu_onnx_test'\n";
  invalid << "  export_format: 'onnx_fp16_gpu'\n";
  invalid << "  backend: 'onnx'\n";
  invalid << "  precision: 'fp16'\n";
  invalid << "  device: 'cuda'\n";
  invalid << "  hardware_compatible: true\n";
  invalid << "  files:\n";
  invalid << "    model: 'model.onnx'\n";
  invalid << "    checkpoint: 'checkpoints/best.pt'\n";
  invalid << "    resolved_training_config: 'config/resolved_training_config.yml'\n";
  invalid << "  triton:\n";
  invalid << "    enabled: false\n";
  invalid << "  inputs:\n";
  invalid << "    - name: 'point_features'\n";
  invalid << "      dtype: 'float16'\n";
  invalid << "      shape: ['batch', 100, 1]\n";
  invalid << "  outputs:\n";
  invalid << "    - name: 'reg_logits'\n";
  invalid << "      dtype: 'float16'\n";
  invalid << "      shape: ['batch', 10, 7]\n";
  invalid << "frozen_contract:\n";
  invalid << "  preprocessing:\n";
  invalid << "    max_num_points: 100\n";
  invalid << "    num_point_features: 1\n";
  invalid << "    point_cloud_range:\n";
  invalid << "      x: [-1.0, 1.0]\n";
  invalid << "      y: [-1.0, 1.0]\n";
  invalid << "      z: [-1.0, 1.0]\n";
  invalid << "    voxel_size:\n";
  invalid << "      x: 0.1\n";
  invalid << "      y: 0.1\n";
  invalid << "      z: 0.1\n";
  invalid << "    point_feature_normalization:\n";
  invalid << "      type: value_threshold\n";
  invalid << "      epsilon: 1e-6\n";
  invalid << "  postprocessing:\n";
  invalid << "    grid_size:\n";
  invalid << "      x: 10\n";
  invalid << "      y: 10\n";
  invalid << "    num_classes: 1\n";
  invalid << "    class_names: ['car']\n";
  invalid << "  model:\n";
  invalid << "    stride: [2, 1, 2]\n";
  invalid << "    up_stride: [1, 1, 2]\n";
  invalid << "    first_up_stride: 1\n";
  invalid << "    pillar_map_size: [10, 10]\n";
  invalid << "    pillar_map_range: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]\n";
  invalid << "runtime_defaults:\n";
  invalid << "  preprocessing:\n";
  invalid << "    point_feature:\n";
  invalid << "      value_threshold: 1.0\n";
  invalid << "  postprocessing:\n";
  invalid << "    class_score_threshold: 0.0\n";
  invalid << "    nms:\n";
  invalid << "      score_threshold: 0.2\n";
  invalid << "      iou_threshold: 0.5\n";
  invalid << "      max_num_objects: 10\n";
  invalid.close();

  bool rejected_unknown_key = false;
  try {
    (void)pcod_common::LoadModelManifest(invalid_path);
  } catch (const std::runtime_error&) {
    rejected_unknown_key = true;
  }
  assert(rejected_unknown_key);
  return 0;
}
