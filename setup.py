# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Setuptools build hooks for package runtime resources."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithRuntimeResources(build_py):
    """Include JIT extension sources and schemas in installed packages."""

    resource_directories = ("csrc", "schemas")

    def _resource_files(self) -> list[tuple[Path, Path]]:
        repository = Path(__file__).resolve().parent
        package_build_directory = Path(self.build_lib) / "pcod_common"
        return [
            (source, package_build_directory / source.relative_to(repository))
            for directory in self.resource_directories
            for source in sorted((repository / directory).glob("*"))
            if source.is_file()
        ]

    def run(self) -> None:
        """Build Python modules and copy package runtime resources."""
        super().run()
        for source, destination in self._resource_files():
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        """Report runtime resources as build outputs for wheel installation."""
        outputs = super().get_outputs(include_bytecode)
        return outputs + [str(destination) for _, destination in self._resource_files()]

    def get_source_files(self) -> list[str]:
        """Report runtime resources as source-distribution inputs."""
        sources = super().get_source_files()
        repository = Path(__file__).resolve().parent
        return sources + [source.relative_to(repository).as_posix() for source, _ in self._resource_files()]


setup(cmdclass={"build_py": BuildPyWithRuntimeResources})
