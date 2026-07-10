# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Check that all declared project versions match."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path


def python_version(path: Path) -> str:
    """Read the static ``__version__`` assignment from a Python module."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError(f"No static __version__ assignment found in {path}")


def main() -> int:
    """Compare the CMake, package metadata, and Python versions."""
    repository = Path(__file__).resolve().parents[1]
    versions = {
        "VERSION": (repository / "VERSION").read_text(encoding="utf-8").strip(),
        "pyproject.toml": tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"],
        "python/pcod_common/version.py": python_version(repository / "python/pcod_common/version.py"),
    }

    if len(set(versions.values())) != 1:
        print("Project versions do not match:")
        for source, version in versions.items():
            print(f"  {source}: {version}")
        return 1

    print(f"All project versions match: {next(iter(versions.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
