# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

from pcod_common.torch_extensions.build import load_cached_torch_extension
from pcod_common.torch_extensions.rotated_nms import load_rotated_nms_extension


def test_cached_extension_uses_build_name_and_custom_cache(tmp_path: Path, monkeypatch) -> None:
    """Load a custom-cache artifact using its compiled native-module name."""
    build_root = tmp_path / "torch_extensions"
    primary_module_name = "pcod_common._test_extension"
    extension_name = "pcod_common__test_extension"
    extension_dir = build_root / extension_name
    extension_dir.mkdir(parents=True)
    extension_path = extension_dir / f"{extension_name}.so"
    extension_path.write_bytes(b"not-a-real-extension")
    source = tmp_path / "test_extension.cpp"
    source.write_text("// source\n")
    source.touch()
    extension_path.touch()

    loaded_module = ModuleType(extension_name)
    requested_module_names = []

    class FakeLoader:
        def exec_module(self, module: ModuleType) -> None:
            module.loaded_from_cache = True

    def fake_spec_from_file_location(module_name, location):
        requested_module_names.append(module_name)
        return SimpleNamespace(loader=FakeLoader())

    def fail_build(**kwargs):
        raise AssertionError("unexpected extension build")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec_from_file_location)
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: loaded_module)
    monkeypatch.setattr("pcod_common.torch_extensions.build.load", fail_build)
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)

    try:
        loaded = load_cached_torch_extension(
            primary_module_name=primary_module_name,
            import_module_names=(primary_module_name, extension_name),
            extension_name=extension_name,
            sources=(source,),
            torch_extensions_dir=build_root,
        )

        assert loaded is loaded_module
        assert loaded.loaded_from_cache is True
        assert requested_module_names == [extension_name]
        assert sys.modules[primary_module_name] is loaded_module
    finally:
        sys.modules.pop(primary_module_name, None)


def test_extension_lock_follows_custom_cache_root(tmp_path: Path, monkeypatch) -> None:
    """Place the inter-process build lock beside a custom extension cache."""
    build_root = tmp_path / "torch_extensions"
    source = tmp_path / "test_extension.cpp"
    source.write_text("// source\n")
    extension_name = "pcod_common__lock_test_extension"
    loaded_module = ModuleType(extension_name)
    lock_paths = []

    @contextmanager
    def capture_lock(path: Path):
        lock_paths.append(path)
        yield

    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.setattr("pcod_common.torch_extensions.build.exclusive_build_lock", capture_lock)
    monkeypatch.setattr("pcod_common.torch_extensions.build.load", lambda **kwargs: loaded_module)

    try:
        loaded = load_cached_torch_extension(
            primary_module_name=extension_name,
            import_module_names=(extension_name,),
            extension_name=extension_name,
            sources=(source,),
            lock_name="test-extension.lock",
            torch_extensions_dir=build_root,
        )

        assert loaded is loaded_module
        assert lock_paths == [build_root / "test-extension.lock"]
    finally:
        sys.modules.pop(extension_name, None)


def test_rotated_nms_loader_disables_verbose_build_output(tmp_path: Path, monkeypatch) -> None:
    """Keep successful Ninja no-op messages out of application output."""
    (tmp_path / "rotated_nms.cpp").write_text("// source\n")
    (tmp_path / "rotated_nms_cuda.cu").write_text("// source\n")
    loaded_module = ModuleType("pcod_common._rotated_nms")
    loader_arguments = {}

    monkeypatch.setenv("PCOD_COMMON_CSRC_DIR", str(tmp_path))
    monkeypatch.setattr(
        "pcod_common.torch_extensions.rotated_nms.import_first_available",
        lambda module_names: None,
    )

    def fake_load_cached_torch_extension(**kwargs):
        loader_arguments.update(kwargs)
        return loaded_module

    monkeypatch.setattr(
        "pcod_common.torch_extensions.rotated_nms.load_cached_torch_extension",
        fake_load_cached_torch_extension,
    )

    assert load_rotated_nms_extension() is loaded_module
    assert loader_arguments["verbose"] is False
