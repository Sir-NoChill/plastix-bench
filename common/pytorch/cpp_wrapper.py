"""Reusable sentinel for C++ implementations.

A per-impl `run_benchmark.py` is just two lines:

    from common/pytorch.cpp_wrapper import main
    main()

The orchestrator finds the sentinel by globbing `*/run_benchmark.py`, then
invokes it with `--build-dir <build>` and whatever pass-through flags the
benchmark accepts. The wrapper computes the binary path from its own
location:

    <bench>/<impl>/run_benchmark.py            ← this file's symlink/copy
    <build_dir>/<bench>/<impl>/run_benchmark   ← binary

We `execvp` rather than `subprocess.run` so signals (Ctrl-C, time limits)
propagate cleanly and there is no double-fork overhead in tight sweeps.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _find_binary(here: Path, build_dir: Path) -> Path:
    """Mirror the source layout under build_dir: `<build>/<bench>/<impl>/run_benchmark`."""
    # `here` is .../<bench>/<impl>; the CMake helper deposits the binary at
    # the same <bench>/<impl> path under the build directory.
    if len(here.parts) < 2:
        raise SystemExit(f"cpp_wrapper: cannot derive <bench>/<impl> from {here}")
    rel = Path(*here.parts[-2:])     # <bench>/<impl>
    return build_dir / rel / "run_benchmark"


def main() -> None:
    here = Path(sys.argv[0]).resolve().parent

    # We split argv into wrapper-only flags (consumed here) and pass-through
    # flags (forwarded to the C++ binary). `--build-dir` is the only flag we
    # peel off; everything else flows through unchanged.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path,
                        default=Path("build-host"))
    parser.add_argument("--print-binary", action="store_true",
                        help="print the resolved binary path and exit")
    wrapper_args, passthrough = parser.parse_known_args()

    binary = _find_binary(here, wrapper_args.build_dir.resolve())
    if wrapper_args.print_binary:
        print(binary)
        return
    if not binary.exists():
        raise SystemExit(
            f"cpp_wrapper: binary not found at {binary}\n"
            f"  build it first: cmake --build {wrapper_args.build_dir} -j")
    # exec replaces the Python process; output streams go straight to the
    # parent terminal / pipe.
    os.execvp(str(binary), [str(binary), *passthrough])


if __name__ == "__main__":
    main()
