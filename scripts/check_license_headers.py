# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COPYRIGHT = "Copyright Institute for Automotive Engineering (ika), RWTH Aachen University"
SPDX = "SPDX-License-Identifier: Apache-2.0"
COMMENT_PREFIXES = {
    ".c": "//",
    ".cc": "//",
    ".cpp": "//",
    ".cuh": "//",
    ".cu": "//",
    ".cxx": "//",
    ".h": "//",
    ".hpp": "//",
    ".py": "#",
}


def tracked_source_files(repository: Path) -> list[Path]:
    """Return source files tracked by Git or newly added to the repository."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    paths = result.stdout.decode().split("\0")
    return sorted(repository / path for path in paths if path and Path(path).suffix in COMMENT_PREFIXES)


def main() -> int:
    """Check every source file and report missing license headers."""
    repository = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in tracked_source_files(repository):
        prefix = COMMENT_PREFIXES[path.suffix]
        expected = [f"{prefix} {COPYRIGHT}", f"{prefix} {SPDX}"]
        actual = path.read_text(encoding="utf-8").splitlines()[:2]
        if actual != expected:
            failures.append(str(path.relative_to(repository)))

    if failures:
        print("The following source files do not begin with the required license header:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All tracked source files contain the required Apache-2.0 license header.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
