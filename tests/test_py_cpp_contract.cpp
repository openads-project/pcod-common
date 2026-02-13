#include "pcod_common/model_manifest.hpp"
#include "pcod_common/nms.hpp"
#include "pcod_common/version.hpp"

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct CommandResult {
  int exit_code = -1;
  std::string stdout_text;
};

std::string Trim(const std::string& value) {
  std::size_t first = 0;
  while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) {
    --last;
  }
  return value.substr(first, last - first);
}

CommandResult RunCommandCapture(const std::string& command) {
  CommandResult result;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return result;
  }
  char buffer[256];
  while (fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) {
    result.stdout_text += buffer;
  }
  result.exit_code = pclose(pipe);
  return result;
}

std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    lines.push_back(Trim(line));
  }
  return lines;
}

std::vector<float> ParseCsvFloats(const std::string& csv) {
  std::vector<float> out;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const std::string trimmed = Trim(item);
    if (!trimmed.empty()) {
      out.push_back(std::stof(trimmed));
    }
  }
  return out;
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
  const std::string python_dir = PCOD_COMMON_PYTHON_DIR;
  const std::string py_prefix = "PYTHONPATH='" + python_dir + "' python3 -c \"";

  {
    const auto result = RunCommandCapture(
        py_prefix + "from pcod_common.manifest import SCHEMA_VERSION; print(SCHEMA_VERSION)\"");
    assert(result.exit_code == 0);
    assert(Trim(result.stdout_text) == pcod_common::kManifestSchemaVersion);
  }

  {
    const auto result = RunCommandCapture(
        py_prefix +
        "from pcod_common.manifest import score_threshold_list as s; "
        "print(','.join(str(v) for v in s(None))); "
        "print(','.join(str(v) for v in s(0.25))); "
        "print(','.join(str(v) for v in s([0.1, 0.2])))\"");
    assert(result.exit_code == 0);
    const auto lines = SplitLines(result.stdout_text);
    assert(lines.size() == 3);
    assert(lines[0].empty());
    assert(lines[1] == "0.25");
    assert(lines[2] == "0.1,0.2");
  }

  {
    const std::string scalar_path = "./test_py_cpp_score_scalar.yml";
    const std::string list_path = "./test_py_cpp_score_list.yml";
    WriteManifestWithScoreThreshold(scalar_path, "0.2");
    WriteManifestWithScoreThreshold(list_path, "[0.1, 0.2, 0.3]");

    auto scalar_manifest = pcod_common::LoadModelManifest(scalar_path);
    auto list_manifest = pcod_common::LoadModelManifest(list_path);

    assert(scalar_manifest.postprocessing.score_thresholds.size() == 1);
    assert(std::abs(scalar_manifest.postprocessing.score_thresholds[0] - 0.2f) < 1e-6f);
    assert(list_manifest.postprocessing.score_thresholds.size() == 3);
    assert(std::abs(list_manifest.postprocessing.score_thresholds[0] - 0.1f) < 1e-6f);
    assert(std::abs(list_manifest.postprocessing.score_thresholds[1] - 0.2f) < 1e-6f);
    assert(std::abs(list_manifest.postprocessing.score_thresholds[2] - 0.3f) < 1e-6f);
  }

  {
    const auto torch_check = RunCommandCapture(
        py_prefix + "import torch, torchvision; print('ok')\"");
    if (torch_check.exit_code == 0) {
      pcod_common::BoundingBox a;
      a.center = {0.0f, 0.0f};
      a.length = 1.0f;
      a.width = 1.0f;
      a.existence_probability = 0.9f;
      a.classification.push_back({0, 1.0f});

      pcod_common::BoundingBox b = a;
      b.center = {0.1f, 0.0f};
      b.existence_probability = 0.8f;

      pcod_common::BoundingBox c = a;
      c.center = {10.0f, 0.0f};
      c.existence_probability = 0.7f;

      std::vector<pcod_common::BoundingBox> boxes = {a, b, c};
      pcod_common::NmsConfig cfg;
      cfg.score_thresholds = {0.5f};
      cfg.iou_threshold = 0.1f;
      cfg.max_detections = 10;
      cfg.internal_score_threshold = 0.5f;
      pcod_common::ApplyRotatedNms(boxes, cfg);

      std::vector<float> cpp_kept_x;
      for (const auto& box : boxes) {
        cpp_kept_x.push_back(box.center[0]);
      }

      const auto py_nms = RunCommandCapture(
          py_prefix +
          "from pcod_common.postprocess import apply_nms; "
          "import torch; "
          "boxes=torch.tensor([[0.0,0.0,0.0,1.0,1.0,1.0,0.0],[0.1,0.0,0.0,1.0,1.0,1.0,0.0],[10.0,0.0,0.0,1.0,1.0,1.0,0.0]],dtype=torch.float32); "
          "scores=torch.tensor([0.9,0.8,0.7],dtype=torch.float32); "
          "labels=torch.tensor([0,0,0],dtype=torch.long); "
          "kept_boxes,_,_=apply_nms(boxes,scores,labels,[0.5],0.1,10,use_rotated=False); "
          "print(','.join(str(float(v)) for v in kept_boxes[:,0].tolist()))\"");
      assert(py_nms.exit_code == 0);

      const auto py_kept_x = ParseCsvFloats(py_nms.stdout_text);
      assert(py_kept_x.size() == cpp_kept_x.size());
      for (std::size_t i = 0; i < py_kept_x.size(); ++i) {
        assert(std::abs(py_kept_x[i] - cpp_kept_x[i]) < 1e-5f);
      }
    }
  }

  return 0;
}
