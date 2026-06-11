"""C++ implementation sentinel — discovers and execs the compiled binary.

Tells the orchestrator: "this dir holds a C++ benchmark; the source lives
here and the built binary lives under `<build_dir>/<bench>/
<impl>/run_benchmark`". All wrapper logic is shared via
`common/pytorch/cpp_wrapper.py`.
"""
import sys
from pathlib import Path

# Add common/pytorch to sys.path so the shared wrapper is
# importable regardless of where the orchestrator was invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))

from cpp_wrapper import main  # noqa: E402


if __name__ == "__main__":
    main()
