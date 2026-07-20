# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

_version_path = Path(__file__).resolve().parents[2] / "VERSION"
if _version_path.exists():
    __version__ = _version_path.read_text(encoding="utf-8").strip()
