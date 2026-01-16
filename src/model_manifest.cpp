#include "pcod_common/model_manifest.hpp"
#include "pcod_common/version.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace pcod_common {

namespace {

template <typename T>
T require_scalar(const YAML::Node& node, const std::string& name) {
  if (!node || !node.IsScalar()) {
    throw std::runtime_error("Manifest field '" + name + "' is missing or not a scalar");
  }
  return node.as<T>();
}

std::array<float, 2> require_range(const YAML::Node& node, const std::string& name) {
  if (!node || !node.IsSequence() || node.size() != 2) {
    throw std::runtime_error("Manifest field '" + name + "' must be a 2-element sequence");
  }
  return {node[0].as<float>(), node[1].as<float>()};
}

std::vector<float> load_score_thresholds(const YAML::Node& node) {
  std::vector<float> values;
  if (!node) {
    return values;
  }
  if (node.IsSequence()) {
    for (const auto& entry : node) {
      values.push_back(entry.as<float>());
    }
    return values;
  }
  if (node.IsScalar()) {
    values.push_back(node.as<float>());
  }
  return values;
}

}  // namespace

ModelManifest LoadModelManifest(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  ModelManifest manifest;

  manifest.schema_version = require_scalar<std::string>(root["schema_version"], "schema_version");
  manifest.model_name = require_scalar<std::string>(root["model_name"], "model_name");
  manifest.precision = require_scalar<std::string>(root["precision"], "precision");
  manifest.device = require_scalar<std::string>(root["device"], "device");

  const auto preprocessing = root["preprocessing"];
  manifest.preprocessing.max_num_points = require_scalar<int>(preprocessing["max_num_points"],
                                                              "preprocessing.max_num_points");
  manifest.preprocessing.num_point_features = require_scalar<int>(preprocessing["num_point_features"],
                                                                  "preprocessing.num_point_features");
  const auto pc_range = preprocessing["point_cloud_range"];
  manifest.preprocessing.x_range = require_range(pc_range["x"], "preprocessing.point_cloud_range.x");
  manifest.preprocessing.y_range = require_range(pc_range["y"], "preprocessing.point_cloud_range.y");
  manifest.preprocessing.z_range = require_range(pc_range["z"], "preprocessing.point_cloud_range.z");
  const auto voxel = preprocessing["voxel_size"];
  manifest.preprocessing.voxel_x = require_scalar<float>(voxel["x"], "preprocessing.voxel_size.x");
  manifest.preprocessing.voxel_y = require_scalar<float>(voxel["y"], "preprocessing.voxel_size.y");
  manifest.preprocessing.voxel_z = require_scalar<float>(voxel["z"], "preprocessing.voxel_size.z");

  const auto postprocessing = root["postprocessing"];
  const auto grid = postprocessing["grid_size"];
  manifest.postprocessing.grid_x = require_scalar<int>(grid["x"], "postprocessing.grid_size.x");
  manifest.postprocessing.grid_y = require_scalar<int>(grid["y"], "postprocessing.grid_size.y");
  manifest.postprocessing.num_classes = require_scalar<int>(postprocessing["num_classes"],
                                                           "postprocessing.num_classes");
  if (postprocessing["class_names"] && postprocessing["class_names"].IsSequence()) {
    for (const auto& entry : postprocessing["class_names"]) {
      manifest.postprocessing.class_names.push_back(entry.as<std::string>());
    }
  }
  manifest.postprocessing.score_thresholds = load_score_thresholds(postprocessing["score_threshold"]);
  manifest.postprocessing.nms_iou_threshold = require_scalar<float>(postprocessing["nms_iou_threshold"],
                                                                    "postprocessing.nms_iou_threshold");
  manifest.postprocessing.max_detections = require_scalar<int>(postprocessing["max_detections"],
                                                               "postprocessing.max_detections");
  if (postprocessing["with_velocity"]) {
    manifest.postprocessing.with_velocity = postprocessing["with_velocity"].as<bool>();
  }

  const auto model = root["model"];
  if (model["stride"] && model["stride"].IsSequence()) {
    for (const auto& entry : model["stride"]) {
      manifest.model.stride.push_back(entry.as<int>());
    }
  }
  if (model["up_stride"] && model["up_stride"].IsSequence()) {
    for (const auto& entry : model["up_stride"]) {
      manifest.model.up_stride.push_back(entry.as<int>());
    }
  }
  manifest.model.first_up_stride = require_scalar<int>(model["first_up_stride"], "model.first_up_stride");
  if (!model["pillar_map_size"] || !model["pillar_map_size"].IsSequence() || model["pillar_map_size"].size() != 2) {
    throw std::runtime_error("Manifest field 'model.pillar_map_size' must be [x, y]");
  }
  manifest.model.pillar_map_size = {model["pillar_map_size"][0].as<int>(), model["pillar_map_size"][1].as<int>()};
  if (!model["pillar_map_range"] || !model["pillar_map_range"].IsSequence() || model["pillar_map_range"].size() != 3) {
    throw std::runtime_error("Manifest field 'model.pillar_map_range' must be [[x_min,x_max],[y_min,y_max],[z_min,z_max]]");
  }
  manifest.model.pillar_map_range = {
      require_range(model["pillar_map_range"][0], "model.pillar_map_range[0]"),
      require_range(model["pillar_map_range"][1], "model.pillar_map_range[1]"),
      require_range(model["pillar_map_range"][2], "model.pillar_map_range[2]")};
  manifest.model.mask_is_bool = require_scalar<bool>(model["mask_is_bool"], "model.mask_is_bool");
  manifest.model.zero_intensity = require_scalar<bool>(model["zero_intensity"], "model.zero_intensity");

  const auto triton = root["triton"];
  if (triton) {
    if (triton["model_name"]) {
      manifest.triton.model_name = triton["model_name"].as<std::string>();
    }
    if (triton["model_version"]) {
      manifest.triton.model_version = triton["model_version"].as<std::string>();
    }
    if (triton["precision"]) {
      manifest.triton.precision = triton["precision"].as<std::string>();
    }
  }

  return manifest;
}

void ValidateModelManifest(const ModelManifest& manifest) {
  if (manifest.schema_version.empty()) {
    throw std::runtime_error("Manifest schema_version is required");
  }
  if (manifest.schema_version != pcod_common::kManifestSchemaVersion) {
    throw std::runtime_error("Unsupported manifest schema_version: " + manifest.schema_version);
  }
  if (manifest.preprocessing.max_num_points <= 0) {
    throw std::runtime_error("preprocessing.max_num_points must be > 0");
  }
  if (manifest.preprocessing.num_point_features <= 0) {
    throw std::runtime_error("preprocessing.num_point_features must be > 0");
  }
  if (manifest.postprocessing.num_classes <= 0) {
    throw std::runtime_error("postprocessing.num_classes must be > 0");
  }
  if (manifest.postprocessing.max_detections <= 0) {
    throw std::runtime_error("postprocessing.max_detections must be > 0");
  }
  if (manifest.model.pillar_map_size[0] <= 0 || manifest.model.pillar_map_size[1] <= 0) {
    throw std::runtime_error("model.pillar_map_size must be positive");
  }
}

}  // namespace pcod_common
