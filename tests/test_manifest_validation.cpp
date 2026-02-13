#include "pcod_common/model_manifest.hpp"
#include "pcod_common/version.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

pcod_common::ModelManifest MakeValidManifest() {
  pcod_common::ModelManifest manifest;
  manifest.schema_version = pcod_common::kManifestSchemaVersion;
  manifest.model_name = "pbod";
  manifest.precision = "fp32";
  manifest.device = "cuda";

  manifest.preprocessing.max_num_points = 100;
  manifest.preprocessing.num_point_features = 1;
  manifest.preprocessing.x_range = {-1.0f, 1.0f};
  manifest.preprocessing.y_range = {-1.0f, 1.0f};
  manifest.preprocessing.z_range = {-1.0f, 1.0f};
  manifest.preprocessing.voxel_x = 0.1f;
  manifest.preprocessing.voxel_y = 0.1f;
  manifest.preprocessing.voxel_z = 0.1f;
  manifest.preprocessing.point_features_normalization.type = "intensity_threshold";
  manifest.preprocessing.point_features_normalization.intensity_threshold = 1.0f;
  manifest.preprocessing.point_features_normalization.epsilon = 1e-6f;

  manifest.postprocessing.grid_x = 10;
  manifest.postprocessing.grid_y = 10;
  manifest.postprocessing.num_classes = 2;
  manifest.postprocessing.class_names = {"car", "pedestrian"};
  manifest.postprocessing.score_thresholds = {0.2f, 0.3f};
  manifest.postprocessing.nms_iou_threshold = 0.5f;
  manifest.postprocessing.max_detections = 10;

  manifest.model.stride = {2, 1, 2};
  manifest.model.up_stride = {1, 1, 2};
  manifest.model.first_up_stride = 1;
  manifest.model.pillar_map_size = {10, 10};
  manifest.model.pillar_map_range = {{{-1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, 1.0f}}};
  manifest.model.mask_is_bool = true;
  manifest.model.zero_intensity = false;
  return manifest;
}

bool ExpectRuntimeError(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

void WriteManifestWithScoreThreshold(const std::string& path, const std::string& score_threshold_value) {
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
  out << "  num_classes: 3\n";
  out << "  class_names: ['car', 'pedestrian', 'truck']\n";
  out << "  nms_iou_threshold: 0.5\n";
  out << "  score_threshold: " << score_threshold_value << "\n";
  out << "  max_detections: 10\n";
  out << "model:\n";
  out << "  stride: [2, 1, 2]\n";
  out << "  up_stride: [1, 1, 2]\n";
  out << "  first_up_stride: 1\n";
  out << "  pillar_map_size: [10, 10]\n";
  out << "  pillar_map_range: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]\n";
  out << "  mask_is_bool: true\n";
  out << "  zero_intensity: false\n";
}

}  // namespace

int main() {
  {
    auto manifest = MakeValidManifest();
    pcod_common::ValidateModelManifest(manifest);
  }

  {
    auto bad = MakeValidManifest();
    bad.schema_version = "0.9";
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.preprocessing.max_num_points = 0;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.postprocessing.num_classes = 0;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.postprocessing.max_detections = 0;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.preprocessing.point_features_normalization.type = "min_max";
    bad.preprocessing.point_features_normalization.min_intensity = 5.0f;
    bad.preprocessing.point_features_normalization.max_intensity = 5.0f;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.preprocessing.point_features_normalization.type = "unsupported";
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    const std::string scalar_path = "./test_manifest_score_scalar.yml";
    WriteManifestWithScoreThreshold(scalar_path, "0.2");
    auto scalar_manifest = pcod_common::LoadModelManifest(scalar_path);
    assert(scalar_manifest.postprocessing.score_thresholds.size() == 1);
    assert(std::abs(scalar_manifest.postprocessing.score_thresholds[0] - 0.2f) < 1e-6f);
  }

  {
    const std::string list_path = "./test_manifest_score_list.yml";
    WriteManifestWithScoreThreshold(list_path, "[0.1, 0.2, 0.3]");
    auto list_manifest = pcod_common::LoadModelManifest(list_path);
    assert(list_manifest.postprocessing.score_thresholds.size() == 3);
    assert(std::abs(list_manifest.postprocessing.score_thresholds[0] - 0.1f) < 1e-6f);
    assert(std::abs(list_manifest.postprocessing.score_thresholds[1] - 0.2f) < 1e-6f);
    assert(std::abs(list_manifest.postprocessing.score_thresholds[2] - 0.3f) < 1e-6f);
  }

  return 0;
}
