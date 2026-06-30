# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Sequence

from torch.utils.cpp_extension import get_default_build_root, load


def ensure_torch_extension_environment(
    *,
    torch_extensions_dir: Path | None = None,
    default_cuda_arch_list: str | None = "8.0",
) -> None:
    """Prepare environment variables needed by torch's JIT extension builder."""

    if torch_extensions_dir is not None and "TORCH_EXTENSIONS_DIR" not in os.environ:
        torch_extensions_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)

    if default_cuda_arch_list is not None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", default_cuda_arch_list)

    bin_path = str(Path(sys.executable).parent)
    if bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")


def import_first_available(module_names: Sequence[str]) -> ModuleType | None:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def load_cached_extension_module(
    module_name: str,
    extension_name: str,
    sources: Sequence[Path],
    *,
    build_root: Path | None = None,
) -> ModuleType | None:
    """Import a fresh cached torch extension artifact if one exists."""

    resolved_build_root = (
        Path(build_root) if build_root is not None else Path(get_default_build_root())
    )
    candidates = sorted(
        resolved_build_root.glob(f"py*_cu*/{extension_name}/{extension_name}.so"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    newest_source_mtime = max(Path(source).stat().st_mtime for source in sources)
    for candidate in candidates:
        try:
            if candidate.stat().st_mtime < newest_source_mtime:
                continue
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module
            return module
        except Exception:
            continue
    return None


@contextmanager
def exclusive_build_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_cached_torch_extension(
    *,
    primary_module_name: str,
    import_module_names: Sequence[str],
    extension_name: str,
    sources: Sequence[Path],
    lock_name: str | None = None,
    extra_cflags: Sequence[str] = ("-O3",),
    extra_cuda_cflags: Sequence[str] = ("-O3",),
    verbose: bool = False,
    force_build: bool = True,
    torch_extensions_dir: Path | None = None,
    default_cuda_arch_list: str | None = "8.0",
) -> ModuleType:
    """Import, reuse, or build a torch C++/CUDA extension.

    The function keeps extension loading deterministic across local runs and DDP
    ranks: a fresh cached artifact is reused when possible, and only one process
    may run torch's JIT build for a given extension at a time.
    """

    ensure_torch_extension_environment(
        torch_extensions_dir=torch_extensions_dir,
        default_cuda_arch_list=default_cuda_arch_list,
    )

    imported = import_first_available(import_module_names)
    if imported is not None:
        return imported

    source_paths = [Path(source) for source in sources]
    cached_module = load_cached_extension_module(primary_module_name, extension_name, source_paths)
    if cached_module is not None:
        return cached_module

    if not force_build:
        raise RuntimeError(f"{extension_name} import failed and build is disabled.")

    lock_path = Path(get_default_build_root()) / (lock_name or f"{extension_name}.lock")
    with exclusive_build_lock(lock_path):
        imported = import_first_available(import_module_names)
        if imported is not None:
            return imported
        cached_module = load_cached_extension_module(
            primary_module_name, extension_name, source_paths
        )
        if cached_module is not None:
            return cached_module
        ext = load(
            name=extension_name,
            sources=[str(source) for source in source_paths],
            extra_cflags=list(extra_cflags),
            extra_cuda_cflags=list(extra_cuda_cflags),
            verbose=verbose,
        )
        sys.modules.setdefault(primary_module_name, ext)
        return ext
