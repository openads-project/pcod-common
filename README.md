# pcod-common

Shared preprocessing and postprocessing library for Point Cloud Object Detection.

This repo is the common runtime layer used by both the training/export pipeline and the ROS inference node.
It centralizes geometry math, decoding, NMS, and manifest parsing so those two repos stay aligned.

What you can do with pcod-common:
- Build a small C++ library for PBOD decoding, rotated NMS, and point filtering.
- Consume the same model manifest schema in Python and C++.
- Use PyTorch CUDA extensions for pillarization and rotated NMS during training/inference.

This repository provides:
- C++ core utilities for postprocessing (PBOD decoding + rotated NMS).
- CUDA kernels for pillarization and rotated NMS (built via PyTorch extensions when needed).
- A lightweight Python package for training/inference utilities.
- A shared model manifest schema used by both training export and ROS inference.

## Layout

- `include/pcod_common/`: public C++ headers.
- `src/`: C++ implementations.
- `csrc/`: CUDA/C++ kernels for torch extensions.
- `python/pcod_common/`: Python package sources.
- `schemas/`: JSON schema for the model manifest.
- `tests/`: C++ smoke tests.
- `python/tests/`: Python unit tests.

## Build (C++)

```sh
apt-get update && apt-get install -y cmake g++ pkg-config libyaml-cpp-dev
cmake -S . -B build -DPCOD_COMMON_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

CUDA kernels are optional and are built at runtime via the PyTorch extension loaders in `python/pcod_common/torch_extensions/`.
Some C++ tests validate Python/C++ contract parity and require `python3` to be available on `PATH`.

## Tests (Python)

```sh
pytest
```

`python/tests/test_postprocess.py` requires `torch` and `torchvision`.
If those packages are not installed, those postprocess tests are skipped and manifest tests still run.

## Devcontainer

A basic devcontainer definition is provided in `.devcontainer/`. In a container, install Python deps (including torch) and run:

```sh
pip install -e .[dev]
cmake -S . -B build -DPCOD_COMMON_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
pytest
```

## Python Usage

Install from the repo root (editable for development):

```sh
pip install -e .
```

Example usage (training):

```python
from pcod_common.preprocessing.pillars import PillarPreprocessor, PillarPreprocessorConfig
from pcod_common.postprocess import apply_nms

cfg = PillarPreprocessorConfig(
    x_min=-50.0, x_max=50.0,
    y_min=-50.0, y_max=50.0,
    z_min=-2.0, z_max=3.0,
    voxel_x=0.2, voxel_y=0.2,
    point_feature_dim=1,
)
preprocessor = PillarPreprocessor(cfg)
```

## End-to-End Example (C++)

This minimal example shows point filtering + PBOD decoding using the C++ API.
It keeps the tensors tiny (four pillars, two classes) so the control flow is easy to follow.

```cpp
#include "pcod_common/pbod_postprocess.hpp"
#include "pcod_common/pillar_grid.hpp"
#include "pcod_common/point_preprocess.hpp"

#include <vector>

int main() {
  pcod_common::PointPreprocessConfig pre_cfg;
  pre_cfg.x_min = -1.0f;
  pre_cfg.x_max = 1.0f;
  pre_cfg.y_min = -1.0f;
  pre_cfg.y_max = 1.0f;
  pre_cfg.z_min = -1.0f;
  pre_cfg.z_max = 1.0f;
  pre_cfg.normalization_type = pcod_common::PointFeatureNormalizationType::kValueThreshold;
  pre_cfg.value_threshold = 10.0f;

  // 1) Basic point filtering (range checks + optional masks).
  pcod_common::PointPreprocessor preprocessor(pre_cfg);
  if (!preprocessor.IsPointValid(0.5f, 0.1f, 0.0f)) {
    return 1;
  }

  // 2) Build a 2x2 pillar grid so the decoder has a center location.
  pcod_common::PillarGrid grid = pcod_common::BuildPillarGrid(
      {2, 2}, {{{0.0f, 2.0f}, {0.0f, 2.0f}, {0.0f, 1.0f}}}, 1, 1);

  // 3) Dummy model outputs for four pillars and two classes.
  //    Two pillars will be filtered out by the score threshold below.
  const int num_pillars = 4;
  const int num_classes = 2;
  float focal_logits[num_pillars] = {2.0f, -2.0f, 2.0f, -2.0f};
  float class_logits[num_pillars * num_classes] = {
      0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f};
  std::vector<float> size_posterior(num_pillars * num_classes * 3, 1.0f);
  std::vector<float> reg_logits(num_pillars * num_classes * 7, 0.0f);

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits;
  view.size_posterior = size_posterior.data();
  view.class_logits = class_logits;
  view.reg_logits = reg_logits.data();
  view.num_pillars = num_pillars;
  view.num_classes = num_classes;
  view.reg_dim = 7;

  // 4) Decode into bounding boxes (class list sets the output metadata).
  pcod_common::PbodPostprocessConfig post_cfg;
  post_cfg.class_names = {"car", "pedestrian"};
  post_cfg.score_thresholds = {0.5f};

  auto boxes = pcod_common::DecodePbod(view, grid, post_cfg);
  const std::size_t expected_boxes = 2;
  return boxes.size() == expected_boxes ? 0 : 1;
}
```

## Model Manifest

Training export emits one canonical YAML file named `model_manifest.yml` inside every exported bundle.
The manifest is split into three sections:

- `artifact`: bundle metadata and file references that always point to files inside the bundle
- `frozen_contract`: non-overridable inference contract that must match the exported model exactly
- `runtime_defaults`: exported defaults for inference-time behavior that may be overridden by the inference user

ROS inference treats `frozen_contract` as mandatory source-of-truth model configuration and uses `runtime_defaults` as the initial values for overridable ROS parameters such as `preprocessing.point_feature.value_threshold` and NMS thresholds.
The schema lives in `schemas/model_manifest.schema.json`.
Both the Python and C++ loaders validate the same canonical structure, reject unsupported keys, and require bundle-relative file references.
The C++ implementation parses YAML via `yaml-cpp` rather than a custom YAML subset parser.

## Integration Notes

- Training repo should include pcod-common as a submodule and add it to the Python environment (e.g., `pip install -e pcod-common`).
- ROS repo should include pcod-common as a submodule and link against the C++ library.

## Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE).
