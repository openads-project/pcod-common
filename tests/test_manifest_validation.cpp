#include "pcod_common/model_manifest.hpp"
#include "pcod_common/version.hpp"

#include <cassert>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

pcod_common::ModelManifest MakeValidManifest() {
  pcod_common::ModelManifest manifest;
  manifest.schema_version = pcod_common::kManifestSchemaVersion;

  manifest.artifact.bundle_name = "pbod_fp32_gpu_triton_test";
  manifest.artifact.export_format = "triton_fp32_gpu";
  manifest.artifact.backend = "triton_onnx";
  manifest.artifact.precision = "fp32";
  manifest.artifact.device = "cuda";
  manifest.artifact.files.checkpoint = "checkpoints/best.pt";
  manifest.artifact.files.resolved_training_config = "config/resolved_training_config.yml";
  manifest.artifact.files.triton_repository = ".";
  manifest.artifact.files.triton_config = "config.pbtxt";
  manifest.artifact.files.triton_model = "1/model.onnx";
  manifest.artifact.triton.enabled = true;
  manifest.artifact.triton.model_name = "pbod_repo";
  manifest.artifact.triton.model_version = "1";
  manifest.artifact.inputs.push_back({"point_features", "float32", {"batch", "100", "1"}});
  manifest.artifact.outputs.push_back({"reg_logits", "float32", {"batch", "200", "14"}});

  manifest.frozen_contract.preprocessing.max_num_points = 100;
  manifest.frozen_contract.preprocessing.num_point_features = 1;
  manifest.frozen_contract.preprocessing.x_range = {-1.0f, 1.0f};
  manifest.frozen_contract.preprocessing.y_range = {-1.0f, 1.0f};
  manifest.frozen_contract.preprocessing.z_range = {-1.0f, 1.0f};
  manifest.frozen_contract.preprocessing.voxel_x = 0.1f;
  manifest.frozen_contract.preprocessing.voxel_y = 0.1f;
  manifest.frozen_contract.preprocessing.voxel_z = 0.1f;
  manifest.frozen_contract.preprocessing.point_feature_normalization.type = "value_threshold";
  manifest.frozen_contract.preprocessing.point_feature_normalization.epsilon = 1e-6f;

  manifest.frozen_contract.postprocessing.grid_x = 10;
  manifest.frozen_contract.postprocessing.grid_y = 10;
  manifest.frozen_contract.postprocessing.num_classes = 2;
  manifest.frozen_contract.postprocessing.class_names = {"car", "pedestrian"};

  manifest.frozen_contract.model.stride = {2, 1, 2};
  manifest.frozen_contract.model.up_stride = {1, 1, 2};
  manifest.frozen_contract.model.first_up_stride = 1;
  manifest.frozen_contract.model.pillar_map_size = {10, 10};
  manifest.frozen_contract.model.pillar_map_range = {{{-1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, 1.0f}}};

  manifest.runtime_defaults.preprocessing.point_feature.value_threshold = 1.0f;
  manifest.runtime_defaults.postprocessing.class_score_threshold = 0.0f;
  manifest.runtime_defaults.postprocessing.nms_score_thresholds = {0.2f, 0.3f};
  manifest.runtime_defaults.postprocessing.nms_iou_threshold = 0.5f;
  manifest.runtime_defaults.postprocessing.max_detections = 10;

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

}  // namespace

int main() {
  {
    auto manifest = MakeValidManifest();
    pcod_common::ValidateModelManifest(manifest);
  }

  {
    auto bad = MakeValidManifest();
    bad.schema_version = "1.0";
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.artifact.files.checkpoint.clear();
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.artifact.files.checkpoint = "../best.pt";
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.frozen_contract.preprocessing.max_num_points = 0;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.frozen_contract.postprocessing.class_names = {"car"};
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.runtime_defaults.postprocessing.nms_score_thresholds = {};
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.artifact.outputs.clear();
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.runtime_defaults.preprocessing.point_feature.value_threshold = 0.0f;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.frozen_contract.preprocessing.point_feature_normalization.type = "min_max";
    bad.frozen_contract.preprocessing.point_feature_normalization.min_value = 5.0f;
    bad.frozen_contract.preprocessing.point_feature_normalization.max_value = 5.0f;
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  {
    auto bad = MakeValidManifest();
    bad.frozen_contract.preprocessing.point_feature_normalization.type = "unsupported";
    assert(ExpectRuntimeError([&]() { pcod_common::ValidateModelManifest(bad); }));
  }

  return 0;
}
