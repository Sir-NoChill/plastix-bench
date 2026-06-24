"""C++ implementation sentinel — discovers and execs the compiled binary.

The wrapper logic is shared via `common/pytorch/cpp_wrapper.py`; see the
other `cpp/` benches for the convention.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))

from cpp_wrapper import main  # noqa: E402


if __name__ == "__main__":
    main()
