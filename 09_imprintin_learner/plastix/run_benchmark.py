"""Plastix implementation sentinel — discovers and execs the compiled binary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared_python"))

from cpp_wrapper import main  # noqa: E402


if __name__ == "__main__":
    main()
