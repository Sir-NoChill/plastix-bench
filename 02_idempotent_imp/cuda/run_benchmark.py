"""CUDA implementation sentinel — discovers and execs the compiled binary.

Mirrors the cpp/ and plastix/ sentinels: the orchestrator finds this via the
`*/run_benchmark.py` glob, and the shared wrapper execs the binary built at
`<build_dir>/<bench>/cuda/run_benchmark`. Only present in CUDA-enabled builds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))

from cpp_wrapper import main  # noqa: E402


if __name__ == "__main__":
    main()
