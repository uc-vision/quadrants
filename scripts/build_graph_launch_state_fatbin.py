#!/usr/bin/env python3
"""Build the pre-compiled CUDA graph launch-state upload kernel fatbins."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fatbin_common import build_fatbin_header  # noqa: E402

repo_root = Path(__file__).resolve().parent.parent
source = repo_root / "quadrants" / "runtime" / "cuda" / "graph_launch_state.cu"
output_header = repo_root / "quadrants" / "runtime" / "cuda" / "graph_launch_state_fatbin.h"

sm_versions = [60, 70, 80, 90, 100, 110, 120]


def main() -> None:
    build_fatbin_header(
        script_name=Path(__file__).name,
        src=source,
        out_header=output_header,
        sm_versions=sm_versions,
        base_name="kGraphLaunchStateFatbin",
    )


if __name__ == "__main__":
    main()
