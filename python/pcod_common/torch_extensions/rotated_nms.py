# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""C++/CUDA rotated NMS extension loader.

Always attempts to import or build the extension; raises on failure.
"""

from __future__ import annotations

import os
from pathlib import Path

from pcod_common.torch_extensions.build import import_first_available, load_cached_torch_extension


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
        if (candidate / "rotated_nms.cpp").exists() and (candidate / "rotated_nms_cuda.cu").exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate pcod-common CUDA sources (rotated_nms.cpp/rotated_nms_cuda.cu). "
        f"Searched: {searched}. Set PCOD_COMMON_CSRC_DIR to the csrc directory."
    )


def load_rotated_nms_extension(force_build: bool | None = None):
    """Import, reuse, or build the rotated-NMS CUDA extension."""
    imported = import_first_available(("pcod_common._rotated_nms", "pcod_common__rotated_nms"))
    if imported is not None:
        return imported

    build_flag = True if force_build is None else bool(force_build)
    if os.getenv("PCODT_BUILD_ROTATED_NMS") == "1":
        build_flag = True
    if not build_flag:
        raise RuntimeError("Rotated NMS extension import failed and build is disabled.")

    csrc_dir = _resolve_csrc_dir()
    src_cpp = csrc_dir / "rotated_nms.cpp"
    src_cuda = csrc_dir / "rotated_nms_cuda.cu"

    return load_cached_torch_extension(
        primary_module_name="pcod_common._rotated_nms",
        import_module_names=("pcod_common._rotated_nms", "pcod_common__rotated_nms"),
        extension_name="pcod_common__rotated_nms",
        sources=(src_cpp, src_cuda),
        lock_name="pcod_common__rotated_nms.lock",
        verbose=False,
        force_build=build_flag,
        torch_extensions_dir=Path.cwd() / ".torch_extensions",
    )


__all__ = ["load_rotated_nms_extension"]
