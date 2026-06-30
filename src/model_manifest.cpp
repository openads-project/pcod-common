// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/model_manifest.hpp"
#include "pcod_common/version.hpp"

#include <filesystem>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace pcod_common {

namespace {

void EnsureMap(const YAML::Node& node, const std::string& field) {
  if (!node || !node.IsMap()) {
    throw std::runtime_error("Manifest key '" + field + "' must be a mapping");
  }
}

void EnsureSequence(const YAML::Node& node, const std::string& field) {
  if (!node || !node.IsSequence()) {
    throw std::runtime_error("Manifest field '" + field + "' must be a sequence");
  }
}

void EnsureKnownKeys(const YAML::Node& node, const std::string& scope, std::initializer_list<const char*> allowed) {
  EnsureMap(node, scope);
  for (const auto& entry : node) {
    const std::string key = entry.first.as<std::string>();
    bool supported = false;
    for (const char* candidate : allowed) {
      if (key == candidate) {
        supported = true;
        break;
      }
    }
    if (!supported) {
      throw std::runtime_error("Manifest " + scope + " contains unsupported key '" + key + "'");
    }
  }
}

YAML::Node RequireChild(const YAML::Node& parent, const std::string& key, const std::string& field) {
  EnsureMap(parent, field);
  YAML::Node child = parent[key];
  if (!child) {
    throw std::runtime_error("Manifest field '" + field + "' is missing");
  }
  return child;
}

YAML::Node LoadOptionalChild(const YAML::Node& parent, const std::string& key) {
  if (!parent || !parent.IsMap()) {
    return {};
  }
  return parent[key];
}

std::string RequireScalarText(const YAML::Node& node, const std::string& field) {
  if (!node || !node.IsScalar()) {
    throw std::runtime_error("Manifest field '" + field + "' must be a scalar");
  }
  return node.Scalar();
}

template <typename T>
T RequireScalarValue(const YAML::Node& node, const std::string& field) {
  if (!node || !node.IsScalar()) {
    throw std::runtime_error("Manifest field '" + field + "' must be a scalar");
  }
  try {
    return node.as<T>();
  } catch (const YAML::Exception& exc) {
    throw std::runtime_error("Manifest field '" + field + "' is invalid: " + std::string(exc.what()));
  }
}

template <typename T>
T LoadOptionalScalarValue(const YAML::Node& node, const std::string& field, const T& default_value) {
  if (!node) {
    return default_value;
  }
  return RequireScalarValue<T>(node, field);
}

std::string LoadOptionalText(const YAML::Node& node, const std::string& field) {
  if (!node) {
    return "";
  }
  return RequireScalarText(node, field);
}

std::string RequireNonEmptyString(const std::string& value, const std::string& field) {
  if (value.empty()) {
    throw std::runtime_error("Manifest field '" + field + "' must not be empty");
  }
  return value;
}

std::string RequireBundleRelativePath(const std::string& value, const std::string& field) {
  RequireNonEmptyString(value, field);
  const std::filesystem::path bundle_path(value);
  if (bundle_path.is_absolute()) {
    throw std::runtime_error("Manifest field '" + field + "' must be a bundle-relative path");
  }
  for (const auto& part : bundle_path) {
    if (part == "..") {
      throw std::runtime_error("Manifest field '" + field + "' must not escape the bundle root");
    }
  }
  return value;
}

std::array<float, 2> RequireRange2(const YAML::Node& node, const std::string& field) {
  EnsureSequence(node, field);
  if (node.size() != 2) {
    throw std::runtime_error("Manifest field '" + field + "' must be a 2-element sequence");
  }
  return {node[0].as<float>(), node[1].as<float>()};
}

std::vector<int> RequirePositiveIntSequence(const YAML::Node& node, const std::string& field) {
  EnsureSequence(node, field);
  if (node.size() == 0) {
    throw std::runtime_error("Manifest field '" + field + "' must be a non-empty sequence");
  }
  std::vector<int> values;
  values.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    const int value = node[i].as<int>();
    if (value <= 0) {
      throw std::runtime_error("Manifest field '" + field + "[" + std::to_string(i) + "]' must be positive");
    }
    values.push_back(value);
  }
  return values;
}

std::vector<std::string> RequireStringSequence(const YAML::Node& node, const std::string& field) {
  EnsureSequence(node, field);
  if (node.size() == 0) {
    throw std::runtime_error("Manifest field '" + field + "' must be a non-empty sequence");
  }
  std::vector<std::string> values;
  values.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    values.push_back(RequireNonEmptyString(RequireScalarText(node[i], field + "[" + std::to_string(i) + "]"),
                                           field + "[" + std::to_string(i) + "]"));
  }
  return values;
}

std::vector<float> RequireScoreThresholds(const YAML::Node& node, const std::string& field) {
  std::vector<float> values;
  if (node.IsSequence()) {
    if (node.size() == 0) {
      throw std::runtime_error("Manifest field '" + field + "' must not be empty");
    }
    values.reserve(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
      values.push_back(node[i].as<float>());
    }
    return values;
  }
  if (!node.IsScalar()) {
    throw std::runtime_error("Manifest field '" + field + "' must be a scalar or sequence");
  }
  values.push_back(node.as<float>());
  return values;
}

std::vector<ArtifactConfig::Tensor> RequireTensorList(const YAML::Node& node, const std::string& field) {
  EnsureSequence(node, field);
  if (node.size() == 0) {
    throw std::runtime_error("Manifest field '" + field + "' must be a non-empty sequence");
  }

  std::vector<ArtifactConfig::Tensor> tensors;
  tensors.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string item_scope = field + "[" + std::to_string(i) + "]";
    YAML::Node entry = node[i];
    EnsureKnownKeys(entry, item_scope, {"name", "dtype", "shape", "description"});
    YAML::Node name = RequireChild(entry, "name", item_scope + ".name");
    YAML::Node dtype = RequireChild(entry, "dtype", item_scope + ".dtype");
    YAML::Node shape = RequireChild(entry, "shape", item_scope + ".shape");

    ArtifactConfig::Tensor tensor;
    tensor.name = RequireNonEmptyString(RequireScalarText(name, item_scope + ".name"), item_scope + ".name");
    tensor.dtype = RequireNonEmptyString(RequireScalarText(dtype, item_scope + ".dtype"), item_scope + ".dtype");

    EnsureSequence(shape, item_scope + ".shape");
    if (shape.size() == 0) {
      throw std::runtime_error("Manifest field '" + item_scope + ".shape' must be a non-empty sequence");
    }
    tensor.shape.reserve(shape.size());
    for (std::size_t dim_idx = 0; dim_idx < shape.size(); ++dim_idx) {
      tensor.shape.push_back(RequireScalarText(shape[dim_idx], item_scope + ".shape[" + std::to_string(dim_idx) + "]"));
    }

    if (YAML::Node description = LoadOptionalChild(entry, "description")) {
      RequireNonEmptyString(RequireScalarText(description, item_scope + ".description"), item_scope + ".description");
    }
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

std::vector<std::array<float, 3>> LoadOptionalSizePriors(const YAML::Node& node, const std::string& field) {
  if (!node) {
    return {};
  }
  EnsureSequence(node, field);
  std::vector<std::array<float, 3>> priors;
  priors.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string row_field = field + "[" + std::to_string(i) + "]";
    EnsureSequence(node[i], row_field);
    if (node[i].size() != 3) {
      throw std::runtime_error("Manifest field '" + row_field + "' must be a 3-element sequence");
    }
    priors.push_back({node[i][0].as<float>(), node[i][1].as<float>(), node[i][2].as<float>()});
  }
  return priors;
}

}  // namespace

ModelManifest LoadModelManifest(const std::string& path) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    EnsureKnownKeys(root, "root", {"schema_version", "artifact", "frozen_contract", "runtime_defaults"});

    const YAML::Node artifact = RequireChild(root, "artifact", "artifact");
    EnsureKnownKeys(
        artifact,
        "artifact",
        {"bundle_name", "export_format", "backend", "precision", "device", "head_name", "export_timestamp_utc",
         "files", "triton", "inputs", "outputs", "size_priors", "size_priors_source"});

    const YAML::Node files = RequireChild(artifact, "files", "artifact.files");
    EnsureKnownKeys(files, "artifact.files",
                    {"model", "checkpoint", "resolved_training_config", "triton_repository", "triton_config",
                     "triton_model"});

    const YAML::Node triton = RequireChild(artifact, "triton", "artifact.triton");
    EnsureKnownKeys(triton, "artifact.triton", {"enabled", "model_name", "model_version"});

    const YAML::Node inputs = RequireChild(artifact, "inputs", "artifact.inputs");
    const YAML::Node outputs = RequireChild(artifact, "outputs", "artifact.outputs");

    const YAML::Node frozen_contract = RequireChild(root, "frozen_contract", "frozen_contract");
    EnsureKnownKeys(frozen_contract, "frozen_contract", {"preprocessing", "postprocessing", "model"});

    const YAML::Node preprocessing = RequireChild(frozen_contract, "preprocessing", "frozen_contract.preprocessing");
    EnsureKnownKeys(preprocessing, "frozen_contract.preprocessing",
                    {"max_num_points", "num_point_features", "point_cloud_range", "voxel_size",
                     "point_feature_normalization"});

    const YAML::Node point_cloud_range =
        RequireChild(preprocessing, "point_cloud_range", "frozen_contract.preprocessing.point_cloud_range");
    EnsureKnownKeys(point_cloud_range, "frozen_contract.preprocessing.point_cloud_range", {"x", "y", "z"});

    const YAML::Node voxel_size = RequireChild(preprocessing, "voxel_size", "frozen_contract.preprocessing.voxel_size");
    EnsureKnownKeys(voxel_size, "frozen_contract.preprocessing.voxel_size", {"x", "y", "z"});

    const YAML::Node normalization = RequireChild(
        preprocessing, "point_feature_normalization", "frozen_contract.preprocessing.point_feature_normalization");
    EnsureKnownKeys(normalization, "frozen_contract.preprocessing.point_feature_normalization",
                    {"type", "min_value", "max_value", "epsilon"});

    const YAML::Node postprocessing =
        RequireChild(frozen_contract, "postprocessing", "frozen_contract.postprocessing");
    EnsureKnownKeys(postprocessing, "frozen_contract.postprocessing", {"grid_size", "num_classes", "class_names"});

    const YAML::Node grid_size = RequireChild(postprocessing, "grid_size", "frozen_contract.postprocessing.grid_size");
    EnsureKnownKeys(grid_size, "frozen_contract.postprocessing.grid_size", {"x", "y"});

    const YAML::Node model = RequireChild(frozen_contract, "model", "frozen_contract.model");
    EnsureKnownKeys(model, "frozen_contract.model",
                    {"stride", "up_stride", "first_up_stride", "pillar_map_size", "pillar_map_range"});

    const YAML::Node runtime_defaults = RequireChild(root, "runtime_defaults", "runtime_defaults");
    EnsureKnownKeys(runtime_defaults, "runtime_defaults", {"preprocessing", "postprocessing"});

    const YAML::Node runtime_preprocessing =
        RequireChild(runtime_defaults, "preprocessing", "runtime_defaults.preprocessing");
    EnsureKnownKeys(runtime_preprocessing, "runtime_defaults.preprocessing", {"point_feature"});

    const YAML::Node point_feature =
        RequireChild(runtime_preprocessing, "point_feature", "runtime_defaults.preprocessing.point_feature");
    EnsureKnownKeys(point_feature, "runtime_defaults.preprocessing.point_feature", {"value_threshold"});

    const YAML::Node runtime_postprocessing =
        RequireChild(runtime_defaults, "postprocessing", "runtime_defaults.postprocessing");
    EnsureKnownKeys(runtime_postprocessing, "runtime_defaults.postprocessing", {"class_score_threshold", "nms"});

    const YAML::Node nms = RequireChild(runtime_postprocessing, "nms", "runtime_defaults.postprocessing.nms");
    EnsureKnownKeys(nms, "runtime_defaults.postprocessing.nms", {"score_threshold", "iou_threshold", "max_num_objects"});

    ModelManifest manifest;

    manifest.schema_version = RequireScalarText(RequireChild(root, "schema_version", "schema_version"), "schema_version");

    manifest.artifact.bundle_name =
        RequireScalarText(RequireChild(artifact, "bundle_name", "artifact.bundle_name"), "artifact.bundle_name");
    manifest.artifact.export_format =
        RequireScalarText(RequireChild(artifact, "export_format", "artifact.export_format"), "artifact.export_format");
    manifest.artifact.backend =
        RequireScalarText(RequireChild(artifact, "backend", "artifact.backend"), "artifact.backend");
    manifest.artifact.precision =
        RequireScalarText(RequireChild(artifact, "precision", "artifact.precision"), "artifact.precision");
    manifest.artifact.device =
        RequireScalarText(RequireChild(artifact, "device", "artifact.device"), "artifact.device");
    manifest.artifact.head_name = LoadOptionalText(LoadOptionalChild(artifact, "head_name"), "artifact.head_name");
    manifest.artifact.export_timestamp_utc =
        LoadOptionalText(LoadOptionalChild(artifact, "export_timestamp_utc"), "artifact.export_timestamp_utc");

    manifest.artifact.files.model = LoadOptionalText(LoadOptionalChild(files, "model"), "artifact.files.model");
    manifest.artifact.files.checkpoint =
        RequireScalarText(RequireChild(files, "checkpoint", "artifact.files.checkpoint"), "artifact.files.checkpoint");
    manifest.artifact.files.resolved_training_config = RequireScalarText(
        RequireChild(files, "resolved_training_config", "artifact.files.resolved_training_config"),
        "artifact.files.resolved_training_config");
    manifest.artifact.files.triton_repository =
        LoadOptionalText(LoadOptionalChild(files, "triton_repository"), "artifact.files.triton_repository");
    manifest.artifact.files.triton_config =
        LoadOptionalText(LoadOptionalChild(files, "triton_config"), "artifact.files.triton_config");
    manifest.artifact.files.triton_model =
        LoadOptionalText(LoadOptionalChild(files, "triton_model"), "artifact.files.triton_model");

    manifest.artifact.triton.enabled =
        RequireScalarValue<bool>(RequireChild(triton, "enabled", "artifact.triton.enabled"), "artifact.triton.enabled");
    manifest.artifact.triton.model_name =
        LoadOptionalText(LoadOptionalChild(triton, "model_name"), "artifact.triton.model_name");
    manifest.artifact.triton.model_version =
        LoadOptionalText(LoadOptionalChild(triton, "model_version"), "artifact.triton.model_version");

    manifest.artifact.inputs = RequireTensorList(inputs, "artifact.inputs");
    manifest.artifact.outputs = RequireTensorList(outputs, "artifact.outputs");
    manifest.artifact.size_priors = LoadOptionalSizePriors(LoadOptionalChild(artifact, "size_priors"), "artifact.size_priors");
    manifest.artifact.size_priors_source =
        LoadOptionalText(LoadOptionalChild(artifact, "size_priors_source"), "artifact.size_priors_source");

    manifest.frozen_contract.preprocessing.max_num_points = RequireScalarValue<int>(
        RequireChild(preprocessing, "max_num_points", "frozen_contract.preprocessing.max_num_points"),
        "frozen_contract.preprocessing.max_num_points");
    manifest.frozen_contract.preprocessing.num_point_features = RequireScalarValue<int>(
        RequireChild(preprocessing, "num_point_features", "frozen_contract.preprocessing.num_point_features"),
        "frozen_contract.preprocessing.num_point_features");
    manifest.frozen_contract.preprocessing.x_range =
        RequireRange2(RequireChild(point_cloud_range, "x", "frozen_contract.preprocessing.point_cloud_range.x"),
                      "frozen_contract.preprocessing.point_cloud_range.x");
    manifest.frozen_contract.preprocessing.y_range =
        RequireRange2(RequireChild(point_cloud_range, "y", "frozen_contract.preprocessing.point_cloud_range.y"),
                      "frozen_contract.preprocessing.point_cloud_range.y");
    manifest.frozen_contract.preprocessing.z_range =
        RequireRange2(RequireChild(point_cloud_range, "z", "frozen_contract.preprocessing.point_cloud_range.z"),
                      "frozen_contract.preprocessing.point_cloud_range.z");
    manifest.frozen_contract.preprocessing.voxel_x =
        RequireScalarValue<float>(RequireChild(voxel_size, "x", "frozen_contract.preprocessing.voxel_size.x"),
                                  "frozen_contract.preprocessing.voxel_size.x");
    manifest.frozen_contract.preprocessing.voxel_y =
        RequireScalarValue<float>(RequireChild(voxel_size, "y", "frozen_contract.preprocessing.voxel_size.y"),
                                  "frozen_contract.preprocessing.voxel_size.y");
    manifest.frozen_contract.preprocessing.voxel_z =
        RequireScalarValue<float>(RequireChild(voxel_size, "z", "frozen_contract.preprocessing.voxel_size.z"),
                                  "frozen_contract.preprocessing.voxel_size.z");
    manifest.frozen_contract.preprocessing.point_feature_normalization.type = RequireScalarText(
        RequireChild(normalization, "type", "frozen_contract.preprocessing.point_feature_normalization.type"),
        "frozen_contract.preprocessing.point_feature_normalization.type");
    manifest.frozen_contract.preprocessing.point_feature_normalization.epsilon = RequireScalarValue<float>(
        RequireChild(normalization, "epsilon", "frozen_contract.preprocessing.point_feature_normalization.epsilon"),
        "frozen_contract.preprocessing.point_feature_normalization.epsilon");
    if (manifest.frozen_contract.preprocessing.point_feature_normalization.type == "min_max") {
      manifest.frozen_contract.preprocessing.point_feature_normalization.min_value = RequireScalarValue<float>(
          RequireChild(normalization, "min_value", "frozen_contract.preprocessing.point_feature_normalization.min_value"),
          "frozen_contract.preprocessing.point_feature_normalization.min_value");
      manifest.frozen_contract.preprocessing.point_feature_normalization.max_value = RequireScalarValue<float>(
          RequireChild(normalization, "max_value", "frozen_contract.preprocessing.point_feature_normalization.max_value"),
          "frozen_contract.preprocessing.point_feature_normalization.max_value");
    }

    manifest.frozen_contract.postprocessing.grid_x =
        RequireScalarValue<int>(RequireChild(grid_size, "x", "frozen_contract.postprocessing.grid_size.x"),
                                "frozen_contract.postprocessing.grid_size.x");
    manifest.frozen_contract.postprocessing.grid_y =
        RequireScalarValue<int>(RequireChild(grid_size, "y", "frozen_contract.postprocessing.grid_size.y"),
                                "frozen_contract.postprocessing.grid_size.y");
    manifest.frozen_contract.postprocessing.num_classes = RequireScalarValue<int>(
        RequireChild(postprocessing, "num_classes", "frozen_contract.postprocessing.num_classes"),
        "frozen_contract.postprocessing.num_classes");
    manifest.frozen_contract.postprocessing.class_names =
        RequireStringSequence(RequireChild(postprocessing, "class_names", "frozen_contract.postprocessing.class_names"),
                              "frozen_contract.postprocessing.class_names");

    manifest.frozen_contract.model.stride =
        RequirePositiveIntSequence(RequireChild(model, "stride", "frozen_contract.model.stride"),
                                   "frozen_contract.model.stride");
    manifest.frozen_contract.model.up_stride =
        RequirePositiveIntSequence(RequireChild(model, "up_stride", "frozen_contract.model.up_stride"),
                                   "frozen_contract.model.up_stride");
    manifest.frozen_contract.model.first_up_stride = RequireScalarValue<int>(
        RequireChild(model, "first_up_stride", "frozen_contract.model.first_up_stride"),
        "frozen_contract.model.first_up_stride");

    {
      const YAML::Node pillar_map_size =
          RequireChild(model, "pillar_map_size", "frozen_contract.model.pillar_map_size");
      EnsureSequence(pillar_map_size, "frozen_contract.model.pillar_map_size");
      if (pillar_map_size.size() != 2) {
        throw std::runtime_error("Manifest field 'frozen_contract.model.pillar_map_size' must be a 2-element sequence");
      }
      manifest.frozen_contract.model.pillar_map_size = {pillar_map_size[0].as<int>(), pillar_map_size[1].as<int>()};
    }

    {
      const YAML::Node pillar_map_range =
          RequireChild(model, "pillar_map_range", "frozen_contract.model.pillar_map_range");
      EnsureSequence(pillar_map_range, "frozen_contract.model.pillar_map_range");
      if (pillar_map_range.size() != 3) {
        throw std::runtime_error("Manifest field 'frozen_contract.model.pillar_map_range' must contain three ranges");
      }
      manifest.frozen_contract.model.pillar_map_range = {
          RequireRange2(pillar_map_range[0], "frozen_contract.model.pillar_map_range[0]"),
          RequireRange2(pillar_map_range[1], "frozen_contract.model.pillar_map_range[1]"),
          RequireRange2(pillar_map_range[2], "frozen_contract.model.pillar_map_range[2]")};
    }

    manifest.runtime_defaults.preprocessing.point_feature.value_threshold = LoadOptionalScalarValue<float>(
        LoadOptionalChild(point_feature, "value_threshold"), "runtime_defaults.preprocessing.point_feature.value_threshold",
        0.0f);
    manifest.runtime_defaults.postprocessing.class_score_threshold = RequireScalarValue<float>(
        RequireChild(runtime_postprocessing, "class_score_threshold", "runtime_defaults.postprocessing.class_score_threshold"),
        "runtime_defaults.postprocessing.class_score_threshold");
    manifest.runtime_defaults.postprocessing.nms_score_thresholds = RequireScoreThresholds(
        RequireChild(nms, "score_threshold", "runtime_defaults.postprocessing.nms.score_threshold"),
        "runtime_defaults.postprocessing.nms.score_threshold");
    manifest.runtime_defaults.postprocessing.nms_iou_threshold = RequireScalarValue<float>(
        RequireChild(nms, "iou_threshold", "runtime_defaults.postprocessing.nms.iou_threshold"),
        "runtime_defaults.postprocessing.nms.iou_threshold");
    manifest.runtime_defaults.postprocessing.max_detections = RequireScalarValue<int>(
        RequireChild(nms, "max_num_objects", "runtime_defaults.postprocessing.nms.max_num_objects"),
        "runtime_defaults.postprocessing.nms.max_num_objects");

    ValidateModelManifest(manifest);
    return manifest;
  } catch (const YAML::Exception& exc) {
    throw std::runtime_error("Failed to parse manifest '" + path + "': " + std::string(exc.what()));
  }
}

void ValidateModelManifest(const ModelManifest& manifest) {
  if (manifest.schema_version.empty()) {
    throw std::runtime_error("Manifest schema_version is required");
  }
  if (manifest.schema_version != pcod_common::kManifestSchemaVersion) {
    throw std::runtime_error("Unsupported manifest schema_version: " + manifest.schema_version);
  }

  RequireNonEmptyString(manifest.artifact.bundle_name, "artifact.bundle_name");
  RequireNonEmptyString(manifest.artifact.export_format, "artifact.export_format");
  RequireNonEmptyString(manifest.artifact.backend, "artifact.backend");
  RequireNonEmptyString(manifest.artifact.precision, "artifact.precision");
  RequireNonEmptyString(manifest.artifact.device, "artifact.device");
  if (!manifest.artifact.head_name.empty()) {
    RequireNonEmptyString(manifest.artifact.head_name, "artifact.head_name");
  }
  if (!manifest.artifact.export_timestamp_utc.empty()) {
    RequireNonEmptyString(manifest.artifact.export_timestamp_utc, "artifact.export_timestamp_utc");
  }

  if (!manifest.artifact.files.model.empty()) {
    RequireBundleRelativePath(manifest.artifact.files.model, "artifact.files.model");
  }
  RequireBundleRelativePath(manifest.artifact.files.checkpoint, "artifact.files.checkpoint");
  RequireBundleRelativePath(manifest.artifact.files.resolved_training_config,
                            "artifact.files.resolved_training_config");
  if (!manifest.artifact.files.triton_repository.empty()) {
    RequireBundleRelativePath(manifest.artifact.files.triton_repository, "artifact.files.triton_repository");
  }
  if (!manifest.artifact.files.triton_config.empty()) {
    RequireBundleRelativePath(manifest.artifact.files.triton_config, "artifact.files.triton_config");
  }
  if (!manifest.artifact.files.triton_model.empty()) {
    RequireBundleRelativePath(manifest.artifact.files.triton_model, "artifact.files.triton_model");
  }

  if (manifest.artifact.triton.enabled) {
    RequireNonEmptyString(manifest.artifact.triton.model_name, "artifact.triton.model_name");
    RequireNonEmptyString(manifest.artifact.triton.model_version, "artifact.triton.model_version");
    if (manifest.artifact.files.triton_repository.empty() || manifest.artifact.files.triton_config.empty() ||
        manifest.artifact.files.triton_model.empty()) {
      throw std::runtime_error("artifact.files must declare Triton repository, config, and model paths");
    }
  } else if (manifest.artifact.files.model.empty()) {
    throw std::runtime_error("artifact.files.model is required for non-Triton exports");
  }

  if (manifest.artifact.inputs.empty()) {
    throw std::runtime_error("artifact.inputs must not be empty");
  }
  for (std::size_t i = 0; i < manifest.artifact.inputs.size(); ++i) {
    const auto& tensor = manifest.artifact.inputs[i];
    if (tensor.name.empty() || tensor.dtype.empty() || tensor.shape.empty()) {
      throw std::runtime_error("artifact.inputs[" + std::to_string(i) + "] must define name, dtype, and shape");
    }
  }
  if (manifest.artifact.outputs.empty()) {
    throw std::runtime_error("artifact.outputs must not be empty");
  }
  for (std::size_t i = 0; i < manifest.artifact.outputs.size(); ++i) {
    const auto& tensor = manifest.artifact.outputs[i];
    if (tensor.name.empty() || tensor.dtype.empty() || tensor.shape.empty()) {
      throw std::runtime_error("artifact.outputs[" + std::to_string(i) + "] must define name, dtype, and shape");
    }
  }

  if (manifest.frozen_contract.preprocessing.max_num_points <= 0) {
    throw std::runtime_error("frozen_contract.preprocessing.max_num_points must be > 0");
  }
  if (manifest.frozen_contract.preprocessing.num_point_features <= 0) {
    throw std::runtime_error("frozen_contract.preprocessing.num_point_features must be > 0");
  }
  if (manifest.frozen_contract.postprocessing.num_classes <= 0) {
    throw std::runtime_error("frozen_contract.postprocessing.num_classes must be > 0");
  }
  if (manifest.frozen_contract.postprocessing.class_names.size() !=
      static_cast<std::size_t>(manifest.frozen_contract.postprocessing.num_classes)) {
    throw std::runtime_error("frozen_contract.postprocessing.class_names must contain one entry per class");
  }
  if (manifest.frozen_contract.model.pillar_map_size[0] <= 0 ||
      manifest.frozen_contract.model.pillar_map_size[1] <= 0) {
    throw std::runtime_error("frozen_contract.model.pillar_map_size must be positive");
  }
  if (manifest.frozen_contract.model.stride.empty()) {
    throw std::runtime_error("frozen_contract.model.stride must not be empty");
  }
  if (manifest.frozen_contract.model.up_stride.empty()) {
    throw std::runtime_error("frozen_contract.model.up_stride must not be empty");
  }
  if (manifest.frozen_contract.model.first_up_stride <= 0) {
    throw std::runtime_error("frozen_contract.model.first_up_stride must be > 0");
  }
  if (manifest.frozen_contract.preprocessing.point_feature_normalization.epsilon <= 0.0f) {
    throw std::runtime_error("frozen_contract.preprocessing.point_feature_normalization.epsilon must be > 0");
  }

  const auto& norm = manifest.frozen_contract.preprocessing.point_feature_normalization;
  if (norm.type == "value_threshold") {
    if (manifest.runtime_defaults.preprocessing.point_feature.value_threshold <= 0.0f) {
      throw std::runtime_error(
          "runtime_defaults.preprocessing.point_feature.value_threshold must be > 0 when value_threshold normalization is used");
    }
  } else if (norm.type == "min_max") {
    if (!(norm.min_value < norm.max_value)) {
      throw std::runtime_error(
          "frozen_contract.preprocessing.point_feature_normalization requires min_value < max_value");
    }
  } else if (norm.type == "z_score") {
    // ok
  } else if (norm.type != "none") {
    throw std::runtime_error("frozen_contract.preprocessing.point_feature_normalization.type is invalid");
  }

  if (manifest.runtime_defaults.postprocessing.class_score_threshold < 0.0f ||
      manifest.runtime_defaults.postprocessing.class_score_threshold > 1.0f) {
    throw std::runtime_error("runtime_defaults.postprocessing.class_score_threshold must be within [0.0, 1.0]");
  }
  if (manifest.runtime_defaults.postprocessing.nms_iou_threshold < 0.0f ||
      manifest.runtime_defaults.postprocessing.nms_iou_threshold > 1.0f) {
    throw std::runtime_error("runtime_defaults.postprocessing.nms.iou_threshold must be within [0.0, 1.0]");
  }
  if (manifest.runtime_defaults.postprocessing.max_detections < 0) {
    throw std::runtime_error("runtime_defaults.postprocessing.nms.max_num_objects must be zero or positive");
  }
  if (manifest.runtime_defaults.postprocessing.nms_score_thresholds.empty()) {
    throw std::runtime_error("runtime_defaults.postprocessing.nms.score_threshold must not be empty");
  }
  const std::size_t score_count = manifest.runtime_defaults.postprocessing.nms_score_thresholds.size();
  if (score_count != 1 &&
      score_count != static_cast<std::size_t>(manifest.frozen_contract.postprocessing.num_classes)) {
    throw std::runtime_error(
        "runtime_defaults.postprocessing.nms.score_threshold must contain one value or one value per class");
  }
  for (float threshold : manifest.runtime_defaults.postprocessing.nms_score_thresholds) {
    if (threshold < 0.0f || threshold > 1.0f) {
      throw std::runtime_error("runtime_defaults.postprocessing.nms.score_threshold entries must be within [0.0, 1.0]");
    }
  }
}

}  // namespace pcod_common
