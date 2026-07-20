# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Verify an installed pcod-common distribution outside its checkout."""

from __future__ import annotations

import json
import sys
from importlib import metadata, resources
from pathlib import Path

import pcod_common
from pcod_common.manifest import score_threshold_list

EXPECTED_RESOURCES = (
    "csrc/pillar_cuda.cpp",
    "csrc/pillar_cuda.cu",
    "csrc/rotated_nms.cpp",
    "csrc/rotated_nms_cuda.cu",
    "schemas/model_manifest.schema.json",
)


def main() -> int:
    """Validate metadata, imports, schema contents, and extension sources."""
    expected_version = sys.argv[1] if len(sys.argv) > 1 else "1.0.0"
    installed_version = metadata.version("pcod-common")
    if installed_version != expected_version or pcod_common.__version__ != expected_version:
        raise RuntimeError(
            f"Version mismatch: metadata={installed_version}, package={pcod_common.__version__}, expected={expected_version}"
        )

    package_root = resources.files("pcod_common")
    missing = [resource for resource in EXPECTED_RESOURCES if not package_root.joinpath(resource).is_file()]
    if missing:
        raise RuntimeError(f"Installed package is missing runtime resources: {', '.join(missing)}")

    schema = json.loads(package_root.joinpath("schemas/model_manifest.schema.json").read_text(encoding="utf-8"))
    if schema.get("title") != "PCOD Model Bundle Manifest":
        raise RuntimeError("Installed model manifest schema is invalid")

    if score_threshold_list(0.5) != [0.5]:
        raise RuntimeError("Manifest helper smoke test failed")

    package_path = Path(pcod_common.__file__).resolve()
    print(f"Validated pcod-common {installed_version} at {package_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
