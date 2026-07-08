"""Baseline: evosax — JAX evolution strategies (CMA-ES, PGPE, OpenES, ...).

Why here: a gradient-free, population-based optimiser over a FIXED network's
weights — the evolutionary counterpart to Plastix's local learning, and a natural
pairing with the TensorNEAT (topology-evolving) baseline. Shows the cost of
weight optimisation without backprop on GPU.

Mapped task: evolve the weights of a small fixed MLP on a control/regression
fitness for a few generations. Report best fitness, wall time, peak RSS, param
count.

STATUS: scaffold. Fill in the TODO to run a real evosax strategy loop.
    uv sync --extra evosax
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_baseline_args, emit, try_import  # noqa: E402

FRAMEWORK = "evosax"
TASK = "weight-ES on a fixed MLP (control)"
PARADIGM = "evolution strategies (JAX)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_baseline_args(p)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--pop-size", type=int, default=256)
    args = p.parse_args()

    mod = try_import("evosax")
    if mod is None:
        emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
              "status": "skipped",
              "notes": "not installed — `uv sync --extra evosax`"}, args.out)
        return

    # TODO: instantiate a strategy (e.g. evosax.Strategies["CMA_ES"]) over the
    # flat MLP params, ask/eval/tell for --generations against the fitness fn,
    # track best fitness. Wrap the loop for wall time; peak RSS polled by
    # baselines.py.
    emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
          "metric": "", "metric_kind": "fitness", "n_params": "",
          "status": "stub",
          "notes": f"evosax {getattr(mod, '__version__', '?')} present; "
                   f"ES loop TODO (gens={args.generations}, pop={args.pop_size})"},
         args.out)


if __name__ == "__main__":
    main()
