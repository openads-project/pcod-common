# pcod-common

Shared preprocessing and postprocessing library for Point Cloud Object Detection.

This repository provides:
- C++ core utilities for postprocessing (PBOD decoding + rotated NMS).
- CUDA kernels for pillarization and rotated NMS (used via torch extensions).
- A lightweight Python package for training/inference utilities.
- A shared model manifest schema used by both training export and ROS inference.

## Layout

- `include/pcod_common/`: public C++ headers.
- `src/`: C++ implementations.
- `csrc/`: CUDA/C++ kernels for torch extensions.
- `python/pcod_common/`: Python package sources.
- `schemas/`: JSON schema for the model manifest.
- `tests/`: C++/Python smoke tests.

## Build (C++)

```sh
cmake -S . -B build -DPCOD_COMMON_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

CUDA kernels are optional; enable with `-DPCOD_COMMON_ENABLE_CUDA=ON` and a CUDA-capable toolchain.

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

## Model Manifest

Training export emits a `model_manifest.yml` file alongside the model artifacts. ROS inference loads this manifest to validate preprocessing/postprocessing expectations. The JSON schema is in `schemas/model_manifest.schema.json`.

## Integration Notes

- Training repo should include pcod-common as a submodule and add it to the Python environment (e.g., `pip install -e pcod-common`).
- ROS repo should include pcod-common as a submodule and link against the C++ library.

