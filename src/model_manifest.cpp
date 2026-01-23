#include "pcod_common/model_manifest.hpp"
#include "pcod_common/version.hpp"

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace pcod_common {

namespace {

struct YamlNode {
  enum class Kind { kMap, kSeq, kScalar };
  Kind kind = Kind::kMap;
  std::map<std::string, YamlNode> map;
  std::vector<YamlNode> seq;
  std::string scalar;
};

struct Line {
  int indent = 0;
  std::string content;
};

std::string trim(const std::string& value) {
  size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
    ++start;
  }
  size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(start, end - start);
}

std::string rtrim(const std::string& value) {
  size_t end = value.size();
  while (end > 0 && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(0, end);
}

std::vector<Line> load_lines(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open manifest: " + path);
  }
  std::vector<Line> lines;
  std::string raw;
  while (std::getline(file, raw)) {
    const auto comment = raw.find('#');
    if (comment != std::string::npos) {
      raw = raw.substr(0, comment);
    }
    raw = rtrim(raw);
    if (raw.empty() || raw.find_first_not_of(' ') == std::string::npos) {
      continue;
    }
    int indent = 0;
    while (indent < static_cast<int>(raw.size()) && raw[indent] == ' ') {
      ++indent;
    }
    std::string content = trim(raw.substr(indent));
    lines.push_back({indent, content});
  }
  return lines;
}

YamlNode parse_value(const std::vector<Line>& lines, size_t& index, int indent);

YamlNode parse_sequence(const std::vector<Line>& lines, size_t& index, int indent) {
  YamlNode node;
  node.kind = YamlNode::Kind::kSeq;
  while (index < lines.size()) {
    const auto& line = lines[index];
    if (line.indent != indent || line.content.rfind("- ", 0) != 0) {
      break;
    }
    std::string rest = trim(line.content.substr(2));
    if (!rest.empty() && rest.rfind("- ", 0) == 0) {
      YamlNode nested;
      nested.kind = YamlNode::Kind::kSeq;
      YamlNode first;
      first.kind = YamlNode::Kind::kScalar;
      first.scalar = trim(rest.substr(2));
      nested.seq.push_back(first);
      ++index;
      while (index < lines.size() && lines[index].indent > indent) {
        if (lines[index].indent != indent + 2 || lines[index].content.rfind("- ", 0) != 0) {
          break;
        }
        YamlNode item;
        item.kind = YamlNode::Kind::kScalar;
        item.scalar = trim(lines[index].content.substr(2));
        nested.seq.push_back(item);
        ++index;
      }
      node.seq.push_back(nested);
      continue;
    }
    if (!rest.empty()) {
      YamlNode item;
      item.kind = YamlNode::Kind::kScalar;
      item.scalar = rest;
      node.seq.push_back(item);
      ++index;
      continue;
    }
    ++index;
    if (index < lines.size() && lines[index].indent > indent) {
      node.seq.push_back(parse_value(lines, index, lines[index].indent));
    } else {
      YamlNode empty;
      empty.kind = YamlNode::Kind::kMap;
      node.seq.push_back(empty);
    }
  }
  return node;
}

YamlNode parse_map(const std::vector<Line>& lines, size_t& index, int indent) {
  YamlNode node;
  node.kind = YamlNode::Kind::kMap;
  while (index < lines.size()) {
    const auto& line = lines[index];
    if (line.indent < indent) {
      break;
    }
    if (line.indent > indent) {
      throw std::runtime_error("Malformed manifest indentation");
    }
    const auto colon = line.content.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("Malformed manifest line: " + line.content);
    }
    std::string key = trim(line.content.substr(0, colon));
    std::string value = trim(line.content.substr(colon + 1));
    ++index;
    if (!value.empty()) {
      YamlNode scalar;
      scalar.kind = YamlNode::Kind::kScalar;
      scalar.scalar = value;
      node.map[key] = scalar;
      continue;
    }
    if (index < lines.size() && lines[index].indent > indent) {
      node.map[key] = parse_value(lines, index, lines[index].indent);
    } else {
      YamlNode empty;
      empty.kind = YamlNode::Kind::kMap;
      node.map[key] = empty;
    }
  }
  return node;
}

YamlNode parse_value(const std::vector<Line>& lines, size_t& index, int indent) {
  if (lines[index].content.rfind("- ", 0) == 0) {
    return parse_sequence(lines, index, indent);
  }
  return parse_map(lines, index, indent);
}

YamlNode load_yaml(const std::string& path) {
  auto lines = load_lines(path);
  size_t index = 0;
  return parse_map(lines, index, 0);
}

const YamlNode& require_node(const YamlNode& root, const std::vector<std::string>& path) {
  const YamlNode* node = &root;
  for (const auto& key : path) {
    if (node->kind != YamlNode::Kind::kMap) {
      throw std::runtime_error("Manifest field '" + key + "' is not a map");
    }
    auto it = node->map.find(key);
    if (it == node->map.end()) {
      throw std::runtime_error("Manifest field '" + key + "' is missing");
    }
    node = &it->second;
  }
  return *node;
}

const YamlNode* find_node(const YamlNode& root, const std::vector<std::string>& path) {
  const YamlNode* node = &root;
  for (const auto& key : path) {
    if (node->kind != YamlNode::Kind::kMap) {
      return nullptr;
    }
    auto it = node->map.find(key);
    if (it == node->map.end()) {
      return nullptr;
    }
    node = &it->second;
  }
  return node;
}

std::string require_scalar_string(const YamlNode& root, const std::vector<std::string>& path) {
  const auto& node = require_node(root, path);
  if (node.kind != YamlNode::Kind::kScalar) {
    throw std::runtime_error("Manifest field '" + path.back() + "' is not a scalar");
  }
  return node.scalar;
}

template <typename T>
T parse_scalar(const std::string& value);

template <>
int parse_scalar<int>(const std::string& value) {
  return std::stoi(value);
}

template <>
float parse_scalar<float>(const std::string& value) {
  return std::stof(value);
}

template <>
bool parse_scalar<bool>(const std::string& value) {
  if (value == "true" || value == "True" || value == "1") {
    return true;
  }
  if (value == "false" || value == "False" || value == "0") {
    return false;
  }
  throw std::runtime_error("Invalid boolean value: " + value);
}

template <>
std::string parse_scalar<std::string>(const std::string& value) {
  if (value.size() >= 2) {
    if ((value.front() == '\'' && value.back() == '\'') || (value.front() == '"' && value.back() == '"')) {
      return value.substr(1, value.size() - 2);
    }
  }
  return value;
}

template <typename T>
T require_scalar(const YamlNode& root, const std::vector<std::string>& path) {
  return parse_scalar<T>(require_scalar_string(root, path));
}

std::vector<std::string> load_sequence_strings(const YamlNode& root, const std::vector<std::string>& path) {
  const auto& node = require_node(root, path);
  std::vector<std::string> values;
  if (node.kind == YamlNode::Kind::kSeq) {
    for (const auto& item : node.seq) {
      if (item.kind != YamlNode::Kind::kScalar) {
        throw std::runtime_error("Manifest sequence '" + path.back() + "' must be scalar entries");
      }
      values.push_back(item.scalar);
    }
    return values;
  }
  if (node.kind == YamlNode::Kind::kScalar) {
    std::string scalar = node.scalar;
    if (scalar.size() >= 2 && scalar.front() == '[' && scalar.back() == ']') {
      scalar = scalar.substr(1, scalar.size() - 2);
      std::stringstream ss(scalar);
      std::string item;
      while (std::getline(ss, item, ',')) {
        values.push_back(trim(item));
      }
      return values;
    }
    values.push_back(node.scalar);
  }
  return values;
}

std::vector<std::string> split_inline_list(const std::string& value) {
  if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
    return {};
  }
  std::string inner = value.substr(1, value.size() - 2);
  std::vector<std::string> items;
  std::string current;
  int depth = 0;
  for (char ch : inner) {
    if (ch == '[') {
      ++depth;
      current.push_back(ch);
      continue;
    }
    if (ch == ']') {
      --depth;
      current.push_back(ch);
      continue;
    }
    if (ch == ',' && depth == 0) {
      items.push_back(trim(current));
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    items.push_back(trim(current));
  }
  return items;
}

std::array<float, 2> require_range(const YamlNode& root, const std::vector<std::string>& path) {
  const auto& node = require_node(root, path);
  if (node.kind == YamlNode::Kind::kSeq && node.seq.size() == 2) {
    return {parse_scalar<float>(node.seq[0].scalar), parse_scalar<float>(node.seq[1].scalar)};
  }
  if (node.kind == YamlNode::Kind::kScalar) {
    auto items = split_inline_list(node.scalar);
    if (items.size() == 2) {
      return {parse_scalar<float>(items[0]), parse_scalar<float>(items[1])};
    }
  }
  throw std::runtime_error("Manifest field '" + path.back() + "' must be a 2-element sequence");
}

std::array<float, 2> require_range_node(const YamlNode& node, const std::string& name) {
  if (node.kind == YamlNode::Kind::kSeq && node.seq.size() == 2) {
    return {parse_scalar<float>(node.seq[0].scalar), parse_scalar<float>(node.seq[1].scalar)};
  }
  if (node.kind == YamlNode::Kind::kScalar) {
    auto items = split_inline_list(node.scalar);
    if (items.size() == 2) {
      return {parse_scalar<float>(items[0]), parse_scalar<float>(items[1])};
    }
  }
  throw std::runtime_error("Manifest field '" + name + "' must be a 2-element sequence");
}

std::vector<float> load_score_thresholds(const YamlNode& root, const std::vector<std::string>& path) {
  std::vector<float> values;
  const auto& node = require_node(root, path);
  if (node.kind == YamlNode::Kind::kSeq) {
    for (const auto& entry : node.seq) {
      values.push_back(parse_scalar<float>(entry.scalar));
    }
    return values;
  }
  if (node.kind == YamlNode::Kind::kScalar) {
    std::string scalar = node.scalar;
    if (scalar.size() >= 2 && scalar.front() == '[' && scalar.back() == ']') {
      scalar = scalar.substr(1, scalar.size() - 2);
      std::stringstream ss(scalar);
      std::string item;
      while (std::getline(ss, item, ',')) {
        values.push_back(parse_scalar<float>(trim(item)));
      }
      return values;
    }
    values.push_back(parse_scalar<float>(node.scalar));
  }
  return values;
}

}  // namespace

ModelManifest LoadModelManifest(const std::string& path) {
  YamlNode root = load_yaml(path);
  ModelManifest manifest;

  manifest.schema_version = require_scalar<std::string>(root, {"schema_version"});
  manifest.model_name = require_scalar<std::string>(root, {"model_name"});
  manifest.precision = require_scalar<std::string>(root, {"precision"});
  manifest.device = require_scalar<std::string>(root, {"device"});

  manifest.preprocessing.max_num_points = require_scalar<int>(root, {"preprocessing", "max_num_points"});
  manifest.preprocessing.num_point_features = require_scalar<int>(root, {"preprocessing", "num_point_features"});
  manifest.preprocessing.x_range = require_range(root, {"preprocessing", "point_cloud_range", "x"});
  manifest.preprocessing.y_range = require_range(root, {"preprocessing", "point_cloud_range", "y"});
  manifest.preprocessing.z_range = require_range(root, {"preprocessing", "point_cloud_range", "z"});
  manifest.preprocessing.voxel_x = require_scalar<float>(root, {"preprocessing", "voxel_size", "x"});
  manifest.preprocessing.voxel_y = require_scalar<float>(root, {"preprocessing", "voxel_size", "y"});
  manifest.preprocessing.voxel_z = require_scalar<float>(root, {"preprocessing", "voxel_size", "z"});
  manifest.preprocessing.point_features_normalization.type =
      require_scalar<std::string>(root, {"preprocessing", "point_features_normalization", "type"});
  manifest.preprocessing.point_features_normalization.epsilon =
      require_scalar<float>(root, {"preprocessing", "point_features_normalization", "epsilon"});
  if (manifest.preprocessing.point_features_normalization.type == "intensity_threshold") {
    manifest.preprocessing.point_features_normalization.intensity_threshold =
        require_scalar<float>(root, {"preprocessing", "point_features_normalization", "intensity_threshold"});
  } else if (manifest.preprocessing.point_features_normalization.type == "min_max") {
    manifest.preprocessing.point_features_normalization.min_intensity =
        require_scalar<float>(root, {"preprocessing", "point_features_normalization", "min_intensity"});
    manifest.preprocessing.point_features_normalization.max_intensity =
        require_scalar<float>(root, {"preprocessing", "point_features_normalization", "max_intensity"});
  } else if (manifest.preprocessing.point_features_normalization.type == "z_score") {
    // No additional required fields.
  } else if (manifest.preprocessing.point_features_normalization.type != "none") {
    throw std::runtime_error("Unsupported preprocessing.point_features_normalization.type");
  }

  manifest.postprocessing.grid_x = require_scalar<int>(root, {"postprocessing", "grid_size", "x"});
  manifest.postprocessing.grid_y = require_scalar<int>(root, {"postprocessing", "grid_size", "y"});
  manifest.postprocessing.num_classes = require_scalar<int>(root, {"postprocessing", "num_classes"});
  manifest.postprocessing.class_names =
      load_sequence_strings(root, {"postprocessing", "class_names"});
  if (manifest.postprocessing.class_names.empty()) {
    manifest.postprocessing.class_names.resize(manifest.postprocessing.num_classes, "");
  }
  manifest.postprocessing.score_thresholds =
      load_score_thresholds(root, {"postprocessing", "score_threshold"});
  manifest.postprocessing.nms_iou_threshold =
      require_scalar<float>(root, {"postprocessing", "nms_iou_threshold"});
  manifest.postprocessing.max_detections =
      require_scalar<int>(root, {"postprocessing", "max_detections"});

  auto stride_vals = load_sequence_strings(root, {"model", "stride"});
  for (const auto& entry : stride_vals) {
    manifest.model.stride.push_back(parse_scalar<int>(entry));
  }
  auto up_stride_vals = load_sequence_strings(root, {"model", "up_stride"});
  for (const auto& entry : up_stride_vals) {
    manifest.model.up_stride.push_back(parse_scalar<int>(entry));
  }
  manifest.model.first_up_stride = require_scalar<int>(root, {"model", "first_up_stride"});

  const auto& pillar_map_size_node = require_node(root, {"model", "pillar_map_size"});
  if (pillar_map_size_node.kind == YamlNode::Kind::kSeq && pillar_map_size_node.seq.size() == 2) {
    manifest.model.pillar_map_size = {parse_scalar<int>(pillar_map_size_node.seq[0].scalar),
                                      parse_scalar<int>(pillar_map_size_node.seq[1].scalar)};
  } else if (pillar_map_size_node.kind == YamlNode::Kind::kScalar) {
    auto items = split_inline_list(pillar_map_size_node.scalar);
    if (items.size() == 2) {
      manifest.model.pillar_map_size = {parse_scalar<int>(items[0]), parse_scalar<int>(items[1])};
    } else {
      throw std::runtime_error("Manifest field 'model.pillar_map_size' must be [x, y]");
    }
  } else {
    throw std::runtime_error("Manifest field 'model.pillar_map_size' must be [x, y]");
  }

  const auto& pillar_map_range_node = require_node(root, {"model", "pillar_map_range"});
  if (pillar_map_range_node.kind == YamlNode::Kind::kSeq && pillar_map_range_node.seq.size() == 3) {
    manifest.model.pillar_map_range = {
        require_range_node(pillar_map_range_node.seq[0], "model.pillar_map_range[0]"),
        require_range_node(pillar_map_range_node.seq[1], "model.pillar_map_range[1]"),
        require_range_node(pillar_map_range_node.seq[2], "model.pillar_map_range[2]")};
  } else if (pillar_map_range_node.kind == YamlNode::Kind::kScalar) {
    auto rows = split_inline_list(pillar_map_range_node.scalar);
    if (rows.size() != 3) {
      throw std::runtime_error("Manifest field 'model.pillar_map_range' must be [[x_min,x_max],[y_min,y_max],[z_min,z_max]]");
    }
    manifest.model.pillar_map_range = {
        require_range_node(YamlNode{YamlNode::Kind::kScalar, {}, {}, rows[0]}, "model.pillar_map_range[0]"),
        require_range_node(YamlNode{YamlNode::Kind::kScalar, {}, {}, rows[1]}, "model.pillar_map_range[1]"),
        require_range_node(YamlNode{YamlNode::Kind::kScalar, {}, {}, rows[2]}, "model.pillar_map_range[2]")};
  } else {
    throw std::runtime_error("Manifest field 'model.pillar_map_range' must be [[x_min,x_max],[y_min,y_max],[z_min,z_max]]");
  }

  manifest.model.mask_is_bool = require_scalar<bool>(root, {"model", "mask_is_bool"});
  manifest.model.zero_intensity = require_scalar<bool>(root, {"model", "zero_intensity"});

  auto triton_it = root.map.find("triton");
  if (triton_it != root.map.end() && triton_it->second.kind == YamlNode::Kind::kMap) {
    const auto& triton_node = triton_it->second;
    auto it = triton_node.map.find("model_name");
    if (it != triton_node.map.end() && it->second.kind == YamlNode::Kind::kScalar) {
      manifest.triton.model_name = it->second.scalar;
    }
    it = triton_node.map.find("model_version");
    if (it != triton_node.map.end() && it->second.kind == YamlNode::Kind::kScalar) {
      manifest.triton.model_version = it->second.scalar;
    }
    it = triton_node.map.find("precision");
    if (it != triton_node.map.end() && it->second.kind == YamlNode::Kind::kScalar) {
      manifest.triton.precision = it->second.scalar;
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
  if (manifest.preprocessing.point_features_normalization.epsilon <= 0.0f) {
    throw std::runtime_error("preprocessing.point_features_normalization.epsilon must be > 0");
  }
  const auto& norm = manifest.preprocessing.point_features_normalization;
  if (norm.type == "intensity_threshold") {
    if (norm.intensity_threshold <= 0.0f) {
      throw std::runtime_error("preprocessing.point_features_normalization.intensity_threshold must be > 0");
    }
  } else if (norm.type == "min_max") {
    if (!(norm.min_intensity < norm.max_intensity)) {
      throw std::runtime_error("preprocessing.point_features_normalization requires min_intensity < max_intensity");
    }
  } else if (norm.type == "z_score") {
    // ok
  } else if (norm.type != "none") {
    throw std::runtime_error("preprocessing.point_features_normalization.type is invalid");
  }
}

}  // namespace pcod_common
