#include "pcod_common/model_manifest.hpp"

#include <cassert>
#include <fstream>
#include <string>

int main() {
  const std::string path = "./test_manifest.yml";
  std::ofstream out(path);
  out << "schema_version: '1.0'\n";
  out << "model_name: 'pbod'\n";
  out << "precision: 'fp16'\n";
  out << "device: 'cuda'\n";
  out << "preprocessing:\n";
  out << "  max_num_points: 100\n";
  out << "  num_point_features: 1\n";
  out << "  point_cloud_range:\n";
  out << "    x: [-1.0, 1.0]\n";
  out << "    y: [-1.0, 1.0]\n";
  out << "    z: [-1.0, 1.0]\n";
  out << "  voxel_size:\n";
  out << "    x: 0.1\n";
  out << "    y: 0.1\n";
  out << "    z: 0.1\n";
  out << "  point_features_normalization:\n";
  out << "    type: intensity_threshold\n";
  out << "    intensity_threshold: 1.0\n";
  out << "    epsilon: 1e-6\n";
  out << "postprocessing:\n";
  out << "  grid_size:\n";
  out << "    x: 10\n";
  out << "    y: 10\n";
  out << "  num_classes: 1\n";
  out << "  class_names: ['car']\n";
  out << "  nms_iou_threshold: 0.5\n";
  out << "  score_threshold: 0.2\n";
  out << "  max_detections: 10\n";
  out << "model:\n";
  out << "  stride: [2, 1, 2]\n";
  out << "  up_stride: [1, 1, 2]\n";
  out << "  first_up_stride: 1\n";
  out << "  pillar_map_size: [10, 10]\n";
  out << "  pillar_map_range: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]\n";
  out << "  mask_is_bool: true\n";
  out << "  zero_intensity: false\n";
  out.close();

  auto manifest = pcod_common::LoadModelManifest(path);
  pcod_common::ValidateModelManifest(manifest);
  assert(manifest.preprocessing.max_num_points == 100);
  return 0;
}
