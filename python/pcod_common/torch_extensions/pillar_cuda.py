from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from torch.utils.cpp_extension import load


def load_pillar_cuda_extension():
    try:
        return importlib.import_module("pcod_common._pillar_cuda")
    except Exception:
        try:
            return importlib.import_module("pcod_common__pillar_cuda")
        except Exception:
            pass

    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        ext_dir = Path.cwd() / ".torch_extensions"
        ext_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    bin_path = str(Path(sys.executable).parent)
    if bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

    root_dir = Path(__file__).resolve().parents[3]
    src_cpp = root_dir / "csrc" / "pillar_cuda.cpp"
    src_cuda = root_dir / "csrc" / "pillar_cuda.cu"

    ext = load(
        name="pcod_common__pillar_cuda",
        sources=[str(src_cpp), str(src_cuda)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    sys.modules.setdefault("pcod_common._pillar_cuda", ext)
    return ext


def pillar_preprocess(
    points_mask,
    points_xyz,
    points_feature,
    x_min: float,
    y_min: float,
    z_min: float,
    x_max: float,
    y_max: float,
    z_max: float,
    voxel_x: float,
    voxel_y: float,
    grid_x: int,
    grid_y: int,
):
    ext = load_pillar_cuda_extension()
    return ext.pillar_preprocess(
        points_mask,
        points_xyz,
        points_feature,
        x_min,
        y_min,
        z_min,
        x_max,
        y_max,
        z_max,
        voxel_x,
        voxel_y,
        grid_x,
        grid_y,
    )


__all__ = ["load_pillar_cuda_extension", "pillar_preprocess"]
