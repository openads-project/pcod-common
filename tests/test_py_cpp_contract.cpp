// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/model_manifest.hpp"
#include "pcod_common/nms.hpp"
#include "pcod_common/pbod_postprocess.hpp"
#include "pcod_common/pillar_grid.hpp"
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

/** Captured result of a shell command used by contract tests. */
struct CommandResult {
  int exit_code = -1;       ///< Process exit code returned by pclose.
  std::string stdout_text;  ///< Captured stdout text.
};

/** Trim leading and trailing whitespace.
 * @param value Input text.
 * @return Trimmed text.
 */
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

/** Run a shell command and capture stdout.
 * @param command Command line passed to the shell.
 * @return Exit code and captured stdout.
 */
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

/** Split text into trimmed lines.
 * @param text Input text.
 * @return Trimmed lines.
 */
std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    lines.push_back(Trim(line));
  }
  return lines;
}

/** Parse comma-separated floats.
 * @param csv Comma-separated scalar values.
 * @return Parsed float values.
 */
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

/** Write a manifest fixture with configurable score-threshold syntax.
 * @param path Destination YAML path.
 * @param score_threshold_value YAML value to write for the score threshold.
 */
void WriteManifestWithScoreThreshold(const std::string& path, const std::string& score_threshold_value) {
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
  out << "      shape: ['batch', 300, 21]\n";
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
  out << "    num_classes: 3\n";
  out << "    class_names: ['car', 'pedestrian', 'truck']\n";
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
  out << "      score_threshold: " << score_threshold_value << "\n";
  out << "      iou_threshold: 0.5\n";
  out << "      max_num_objects: 10\n";
}

}  // namespace

/** Run C++ and Python contract parity checks. */
int main() {
  const std::string python_dir = PCOD_COMMON_PYTHON_DIR;
  const std::string py_prefix = "PYTHONPATH='" + python_dir + "' python3 -c \"";

  {
    const auto result = RunCommandCapture(py_prefix + "from pcod_common.manifest import SCHEMA_VERSION; print(SCHEMA_VERSION)\"");
    assert(result.exit_code == 0);
    assert(Trim(result.stdout_text) == pcod_common::kManifestSchemaVersion);
  }

  {
    const auto result = RunCommandCapture(py_prefix +
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

    assert(scalar_manifest.runtime_defaults.postprocessing.nms_score_thresholds.size() == 1);
    assert(std::abs(scalar_manifest.runtime_defaults.postprocessing.nms_score_thresholds[0] - 0.2f) < 1e-6f);
    assert(list_manifest.runtime_defaults.postprocessing.nms_score_thresholds.size() == 3);
    assert(std::abs(list_manifest.runtime_defaults.postprocessing.nms_score_thresholds[0] - 0.1f) < 1e-6f);
    assert(std::abs(list_manifest.runtime_defaults.postprocessing.nms_score_thresholds[1] - 0.2f) < 1e-6f);
    assert(std::abs(list_manifest.runtime_defaults.postprocessing.nms_score_thresholds[2] - 0.3f) < 1e-6f);
  }

  {
    const auto torch_check = RunCommandCapture(py_prefix + "import torch, torchvision; print('ok')\"");
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

      const auto py_nms = RunCommandCapture(py_prefix +
                                            "from pcod_common.postprocess import apply_nms; "
                                            "import torch; "
                                            "boxes=torch.tensor([[0.0,0.0,0.0,1.0,1.0,1.0,0.0],[0.1,0.0,0.0,1.0,1.0,1.0,0.0],[10."
                                            "0,0.0,0.0,1.0,1.0,1.0,0.0]],dtype=torch.float32); "
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

  {
    pcod_common::PillarGrid grid;
    grid.centers = {0.5F, 0.5F, 0.0F};
    float reg[14] = {};
    reg[7] = 0.25F;
    reg[10] = 0.5F;
    float focal[1] = {2.0F};
    float classes[2] = {-2.0F, -1.0F};
    float sizes[6] = {1.0F, 1.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    pcod_common::PbodOutputsView view;
    view.reg_logits = reg;
    view.focal_logits = focal;
    view.class_logits = classes;
    view.size_posterior = sizes;
    view.num_pillars = 1;
    view.num_classes = 2;
    view.reg_dim = 7;
    pcod_common::PbodPostprocessConfig cfg;
    auto decoded = pcod_common::DecodePbod(view, grid, cfg);
    assert(decoded.size() == 1);
    const auto torch_available =
        RunCommandCapture(py_prefix + "import importlib.util; print(importlib.util.find_spec('torch') is not None)\"");
    if (Trim(torch_available.stdout_text) == "True") {
      const auto python = RunCommandCapture(py_prefix +
                                            "import torch; from pcod_common.box_ops import decode_pbod; "
                                            "r=torch.zeros(1,1,14); r[0,0,7]=.25; r[0,0,10]=.5; "
                                            "_,b,s=decode_pbod(r,torch.tensor([[[2.]]]),torch.tensor([[[-2.,-1.]]]),"
                                            "torch.tensor([[[1.,1.,1.,2.,3.,4.]]]),torch.tensor([[[.5,.5,0.]]])); "
                                            "print(','.join(str(float(v)) for v in [b[0,0],b[0,3],b[0,4],b[0,5],s[0]]))\"");
      assert(python.exit_code == 0);
      const auto values = ParseCsvFloats(python.stdout_text);
      const auto& box = decoded.front();
      const std::vector<float> expected{box.center[0], box.length, box.width, box.height, box.existence_probability};
      assert(values.size() == expected.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        assert(std::abs(values[i] - expected[i]) < 1e-5F);
      }
    }
  }
  {
    std::vector<pcod_common::BoundingBox> boxes;
    for (int i = 0; i < 4; ++i) {
      pcod_common::BoundingBox box;
      box.center = {0.0F, 0.0F};
      box.z = i == 1 ? 3.0F : 0.0F;
      box.length = 2.0F;
      box.width = 1.0F;
      box.height = 1.0F;
      box.existence_probability = 0.9F - 0.1F * i;
      box.classification = {{0, i == 2 ? -2.0F : -1.0F}, {1, i == 2 ? -1.0F : -2.0F}};
      boxes.push_back(box);
    }
    pcod_common::NmsConfig cfg;
    cfg.score_thresholds = {0.1F, 0.1F};
    cfg.iou_threshold = 0.5F;
    pcod_common::ApplyRotatedNms(boxes, cfg);
    assert(boxes.size() == 3);
    assert(std::abs(boxes[0].existence_probability - .9F) < 1e-6F);
    assert(std::abs(boxes[1].z - 3.0F) < 1e-6F);
    assert(std::abs(boxes[2].existence_probability - .7F) < 1e-6F);
  }

  return 0;
}
