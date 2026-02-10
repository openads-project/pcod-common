"""C++/CUDA rotated NMS extension loader.

Always attempts to import or build the extension; raises on failure.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import time

from torch.utils.cpp_extension import load


def _ddp_rank_info() -> tuple[int, int]:
    try:
        rank = int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0)
    except (TypeError, ValueError):
        rank = 0
    try:
        world_size = int(os.environ.get('WORLD_SIZE') or 1)
    except (TypeError, ValueError):
        world_size = 1
    return rank, world_size


def _resolve_csrc_dir() -> Path:
    """Locate pcod-common CUDA/C++ sources for both editable and non-editable installs."""
    env_override = os.getenv('PCOD_COMMON_CSRC_DIR')
    candidates = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend(
        [
            # Editable install from /workspace/pcod-common
            Path(__file__).resolve().parents[3] / 'csrc',
            # Wheel/sdist containing package-local sources
            Path(__file__).resolve().parents[1] / 'csrc',
            # Typical workspace checkout fallback
            Path.cwd() / 'pcod-common' / 'csrc',
            Path('/workspace/pcod-common/csrc'),
        ]
    )

    for candidate in candidates:
        if (candidate / 'rotated_nms.cpp').exists() and (candidate / 'rotated_nms_cuda.cu').exists():
            return candidate

    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        'Could not locate pcod-common CUDA sources (rotated_nms.cpp/rotated_nms_cuda.cu). '
        f'Searched: {searched}. Set PCOD_COMMON_CSRC_DIR to the csrc directory.'
    )


def load_rotated_nms_extension(force_build: bool | None = None):
    if 'TORCH_EXTENSIONS_DIR' not in os.environ:
        ext_dir = Path.cwd() / '.torch_extensions'
        ext_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TORCH_EXTENSIONS_DIR'] = str(ext_dir)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')

    try:
        return importlib.import_module('pcod_common._rotated_nms')
    except Exception:
        try:
            return importlib.import_module('pcod_common__rotated_nms')
        except Exception:
            pass

    bin_path = str(Path(sys.executable).parent)
    if bin_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')

    rank, world_size = _ddp_rank_info()
    wait_for_rank0 = world_size > 1 and rank > 0

    build_flag = True if force_build is None else bool(force_build)
    if os.getenv('PCODT_BUILD_ROTATED_NMS') == '1':
        build_flag = True
    if not build_flag:
        raise RuntimeError('Rotated NMS extension import failed and build is disabled.')

    if wait_for_rank0:
        timeout = float(os.getenv('PCODT_ROTATED_NMS_WAIT_TIMEOUT', 600))
        interval = 2.0
        deadline = time.time() + timeout
        print(
            f'[rotated_nms][rank {rank}] Waiting up to {timeout:.0f}s for rank0 to build extension...',
            flush=True,
        )
        while time.time() < deadline:
            try:
                return importlib.import_module('pcod_common._rotated_nms')
            except Exception:
                try:
                    return importlib.import_module('pcod_common__rotated_nms')
                except Exception:
                    time.sleep(interval)
        raise RuntimeError('[rotated_nms] Timeout waiting for primary rank to build extension.')

    csrc_dir = _resolve_csrc_dir()
    src_cpp = csrc_dir / 'rotated_nms.cpp'
    src_cuda = csrc_dir / 'rotated_nms_cuda.cu'
    ext = load(
        name='pcod_common__rotated_nms',
        sources=[str(src_cpp), str(src_cuda)],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        verbose=True,
    )
    sys.modules.setdefault('pcod_common._rotated_nms', ext)
    return ext


__all__ = ['load_rotated_nms_extension']
