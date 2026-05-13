from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from pcod_common.torch_extensions.build import import_first_available, load_cached_torch_extension

_REGISTERED_PILLAR_OPS: dict[str, Any] = {}


def _resolve_csrc_dir() -> Path:
    """Locate pcod-common CUDA/C++ sources for both editable and non-editable installs."""
    env_override = os.getenv("PCOD_COMMON_CSRC_DIR")
    candidates = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend(
        [
            # Editable install from /workspace/pcod-common
            Path(__file__).resolve().parents[3] / "csrc",
            # Wheel/sdist containing package-local sources
            Path(__file__).resolve().parents[1] / "csrc",
            # Typical workspace checkout fallback
            Path.cwd() / "pcod-common" / "csrc",
            Path("/workspace/pcod-common/csrc"),
        ]
    )

    for candidate in candidates:
        if (candidate / "pillar_cuda.cpp").exists() and (candidate / "pillar_cuda.cu").exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate pcod-common CUDA sources (pillar_cuda.cpp/pillar_cuda.cu). "
        f"Searched: {searched}. Set PCOD_COMMON_CSRC_DIR to the csrc directory."
    )


def load_pillar_cuda_extension():
    imported = import_first_available(("pcod_common._pillar_cuda", "pcod_common__pillar_cuda"))
    if imported is not None:
        return imported

    csrc_dir = _resolve_csrc_dir()
    src_cpp = csrc_dir / "pillar_cuda.cpp"
    src_cuda = csrc_dir / "pillar_cuda.cu"

    return load_cached_torch_extension(
        primary_module_name="pcod_common._pillar_cuda",
        import_module_names=("pcod_common._pillar_cuda", "pcod_common__pillar_cuda"),
        extension_name="pcod_common__pillar_cuda",
        sources=(src_cpp, src_cuda),
        lock_name="pcod_common__pillar_cuda.lock",
        torch_extensions_dir=Path.cwd() / ".torch_extensions",
    )


def _pillar_preprocess_eager(
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


def _pillar_preprocess_compile_op(
    *,
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
    import torch

    constants = (
        float(x_min),
        float(y_min),
        float(z_min),
        float(x_max),
        float(y_max),
        float(z_max),
        float(voxel_x),
        float(voxel_y),
        int(grid_x),
        int(grid_y),
    )
    digest = hashlib.sha1(repr(constants).encode("utf-8")).hexdigest()[:16]
    registered = _REGISTERED_PILLAR_OPS.get(digest)
    if registered is not None:
        return registered

    op_name = f"pcod_common::pillar_preprocess_{digest}"

    @torch.library.custom_op(op_name, mutates_args=(), schema="(Tensor[] inputs) -> Tensor[]")
    def pillar_preprocess_op(inputs):
        return list(_pillar_preprocess_eager(*inputs, *constants))

    @pillar_preprocess_op.register_fake
    def fake(inputs):
        points_mask, points_xyz, points_feature = inputs
        batch_size = points_mask.shape[0]
        max_points = points_mask.shape[1]
        extra_feature_dim = points_feature.shape[2]
        num_pillars = int(grid_x) * int(grid_y)
        feature_dim = 11 + extra_feature_dim + 6
        device = points_xyz.device
        return [
            torch.empty((batch_size, max_points), dtype=torch.int64, device=device),
            torch.empty((batch_size, num_pillars), dtype=torch.int32, device=device),
            torch.empty((batch_size, num_pillars, 3), dtype=torch.float32, device=device),
            torch.empty((batch_size, max_points, feature_dim), dtype=torch.float32, device=device),
        ]

    _REGISTERED_PILLAR_OPS[digest] = pillar_preprocess_op
    return pillar_preprocess_op


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
    try:
        import torch

        is_compiling = bool(getattr(torch.compiler, "is_compiling", lambda: False)())
    except Exception:
        is_compiling = False

    if is_compiling:
        op = _pillar_preprocess_compile_op(
            x_min=x_min,
            y_min=y_min,
            z_min=z_min,
            x_max=x_max,
            y_max=y_max,
            z_max=z_max,
            voxel_x=voxel_x,
            voxel_y=voxel_y,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        result = op([points_mask, points_xyz, points_feature])
        return tuple(result)

    return _pillar_preprocess_eager(
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
