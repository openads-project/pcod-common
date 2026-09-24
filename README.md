# pcod-common

<p align="center">
  <a href="https://github.com/openads-project"><img src="https://img.shields.io/badge/OpenADS-f5ff01"/></a>
  <a href="https://github.com/openads-project/pcod-common/releases/latest"><img src="https://img.shields.io/github/v/release/openads-project/pcod-common"/></a>
  <a href="https://github.com/openads-project/pcod-common/blob/main/LICENSE"><img src="https://img.shields.io/github/license/openads-project/pcod-common"/></a>
  <br>
  <a href="https://github.com/openads-project/pcod-common/actions/workflows/ci.yml"><img src="https://github.com/openads-project/pcod-common/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://openads-project.github.io/pcod-common"><img alt="Documentation" src="https://github.com/openads-project/pcod-common/actions/workflows/docs.yml/badge.svg?branch=main"/></a>
</p>

**Shared Preprocessing and Postprocessing Library for Point Cloud Object Detection**

This repository provides shared C++ and Python components for point cloud object detection training, model export, and ROS 2 inference. Using the same geometry, decoding, non-maximum suppression (NMS), and model-manifest implementations keeps the training and inference pipelines consistent.

The library includes PBOD decoding, rotated NMS, point filtering, CUDA kernels for pillarization and rotated NMS, and a model-manifest schema shared by the C++ and Python APIs.

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>

> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).

## 🚀 Quick Start

### Requirements

- CMake 3.16 or newer
- A C++17 compiler
- yaml-cpp
- Python 3.12 or newer for the Python package
- Optional: a CUDA toolkit compatible with the installed PyTorch build and a CUDA-capable GPU for the CUDA extensions

### C++ Installation

Clone, build, and install the C++ library:

```sh
git clone https://github.com/openads-project/pcod-common.git
cmake -S pcod-common -B build/pcod-common -DPCOD_COMMON_BUILD_TESTS=OFF
cmake --build build/pcod-common
cmake --install build/pcod-common --prefix /path/to/prefix
```

After installation, link your CMake target to the package:

```cmake
find_package(pcod_common CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE pcod_common::pcod_common)
```

When using a custom installation prefix, add it to `CMAKE_PREFIX_PATH` when configuring the consuming project, for example with `-DCMAKE_PREFIX_PATH=/path/to/prefix`.

Alternatively, add this repository directly to your CMake project:

```cmake
add_subdirectory(pcod-common)
target_link_libraries(my_target PRIVATE pcod_common)
```

### Python Installation

Install the Python package from the repository root:

```sh
pip install .
```

### Python Usage Example

```python
from pcod_common.preprocessing.pillars import PillarPreprocessor, PillarPreprocessorConfig

config = PillarPreprocessorConfig(
    x_min=-50.0,
    x_max=50.0,
    y_min=-50.0,
    y_max=50.0,
    z_min=-2.0,
    z_max=3.0,
    voxel_x=0.2,
    voxel_y=0.2,
    point_feature_dim=1,
)
preprocessor = PillarPreprocessor(config)
```

### Detection Postprocessing

`pcod_common.box_ops.decode_pbod` decodes PBOD outputs into sample indices, boxes, and scores.
Its tensor inputs have a leading sample dimension; use size 1 for a single sample.
Each output box has eight values: `(x, y, z, length, width, height, yaw, class_index)`.
The decoder selects the best class per cell and computes its score as
`sigmoid(focal_logit) * softmax(class_logits)[class_index]`. Here `focal_logit`
contains presence multiplied by predicted localization quality.
C++ `DecodePbod` handles one set of pillars and selects one class per pillar. It requires
separate `objectness_logits`, stores their sigmoid in `existence_probability`, and stores the
quality- and class-weighted ranking score in `detection_score`.
NMS filters and ranks boxes by `detection_score` when it is available.

`pcod_common.box_ops.aligned_box_iou` computes differentiable rotated IoU for corresponding box pairs
in `(x, y, z, length, width, height, yaw)` format. It uses 3D overlap by default; pass `three_d=False`
for bird's-eye-view IoU. Set `distance_penalty=True` for DIoU.

Rotated NMS in Python and C++ suppresses lower-scored boxes of the same class using 3D IoU when both
boxes have positive height. It uses bird's-eye-view IoU when either box has zero or negative height.
Score thresholds apply directly to the decoded scores; the C++ `NmsConfig::internal_score_threshold`
is used only when no class thresholds are supplied.

## 💻 Development

### Repository Layout

- `include/pcod_common/`: public C++ headers
- `src/`: C++ implementations
- `csrc/`: CUDA/C++ kernels for PyTorch extensions
- `python/pcod_common/`: Python package sources
- `schemas/`: JSON schema for the model manifest
- `tests/`: C++ tests
- `python/tests/`: Python tests

### C++ Tests

On Debian or Ubuntu, install the required build dependencies and run the test suite:

```sh
apt-get update && apt-get install -y cmake g++ pkg-config libyaml-cpp-dev python3 python3-yaml
cmake -S . -B build -DPCOD_COMMON_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Some C++ tests compare the Python and C++ contracts and require `python3` to be available on `PATH`.

### Python Tests

Install the package in editable mode with its development dependencies and run the test suite:

```sh
pip install -e ".[dev]"
pytest
```

`python/tests/test_postprocess.py` and `python/tests/test_box_ops.py` require PyTorch and TorchVision. Tests that need the rotated NMS extension skip when its build toolchain is unavailable, and CUDA tests skip when CUDA is unavailable. The manifest tests do not require PyTorch.

### Build Python Distributions

Build and validate the wheel and source distribution from the repository root:

```sh
python3 -m pip install build twine
python3 -m build
python3 -m twine check dist/*
```

Published distributions include the model-manifest schema and the C++/CUDA sources required to build the optional PyTorch extensions at runtime. Validating a distribution does not require a GPU. Compiling the extensions requires Ninja and a CUDA toolkit compatible with the installed PyTorch build; running them requires a CUDA-capable GPU.

### Development Container

A basic development container configuration is provided in `.devcontainer/`. In the container, install the system and Python dependencies, including PyTorch, and run:

```sh
sudo apt-get update
sudo apt-get install -y pkg-config libyaml-cpp-dev
pip install -e ".[dev]"
cmake -S . -B build -DPCOD_COMMON_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
pytest
```

CUDA kernels are built on demand by the PyTorch extension loaders in `python/pcod_common/torch_extensions/`.

## C++ Usage Example

This example demonstrates point filtering and PBOD decoding with the C++ API. It uses four pillars and two classes to keep the control flow easy to follow.

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
  float objectness_logits[num_pillars] = {3.0f, 0.0f, 3.0f, 0.0f};
  float class_logits[num_pillars * num_classes] = {
      0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f};
  std::vector<float> size_posterior(num_pillars * num_classes * 3, 1.0f);
  std::vector<float> reg_logits(num_pillars * num_classes * 7, 0.0f);

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits;
  view.objectness_logits = objectness_logits;
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

Each exported model bundle contains a `model_manifest.yml` file with three sections:

- `artifact`: bundle metadata and references to files within the bundle
- `frozen_contract`: model settings that inference applications cannot override and that must match the exported model
- `runtime_defaults`: default inference settings that applications may override

The ROS 2 inference node uses `frozen_contract` as the authoritative model configuration and initializes overridable ROS parameters, such as `preprocessing.point_feature.value_threshold` and the NMS thresholds, from `runtime_defaults`. The schema is defined in `schemas/model_manifest.schema.json`.

## Integration Notes

- For training and model export, include `pcod-common` as a Git submodule and add it to the Python environment, for example with `pip install -e pcod-common`.
- For ROS 2 inference, include `pcod-common` as a Git submodule and link against the C++ library.

For a complete ROS 2 integration example, see [point_cloud_object_detection](https://github.com/openads-project/point_cloud_object_detection), which includes `pcod-common` as a Git submodule and links against its C++ library.

## 📝 Documentation

Implementation details are available in the [Source Code Documentation](https://openads-project.github.io/pcod-common).

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE).

## 🙏 Acknowledgements

Development and maintenance of this repository are supported by the following projects. We acknowledge the funding of the respective institutions.

| Project | Funding Institution | Grant Number |
| --- | --- | --- |
| [AIGGREGATE](https://aiggregate.eu/) | 🇪🇺 European Union | 101202457 |

<p>
  <img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_funded_en.jpg" height=70>
</p>

<sup><sub>Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor CINEA can be held responsible for them.</sub></sup>
