"""Baseline: Nengo — Neural Engineering Framework (neuromorphic).

Why here: Nengo builds function-computing networks out of populations of spiking
neurons via the NEF (represent → transform → dynamics). It's a "neuron-first"
framework like Plastix but with a very different construction philosophy
(principled decoders vs. learned per-connection rules), so it's a good
neuromorphic point of comparison.

Mapped task: an NEF version of a simple suite task — e.g. Mackey-Glass one-step
prediction (cf. 05/07) built as an NEF network. Report test error, wall time,
peak RSS, neuron count.

STATUS: scaffold. Fill in the TODO to build + simulate an NEF model.
    uv sync --extra nengo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_baseline_args, emit, try_import  # noqa: E402

FRAMEWORK = "nengo"
TASK = "Mackey-Glass 1-step (NEF)"
PARADIGM = "neuromorphic NEF"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_baseline_args(p)
    p.add_argument("--neurons", type=int, default=1000)
    args = p.parse_args()

    mod = try_import("nengo")
    if mod is None:
        emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
              "status": "skipped",
              "notes": "not installed — `uv sync --extra nengo`"}, args.out)
        return

    # TODO: with nengo.Network() build Ensembles (n_neurons=--neurons) + a
    # Connection with a learned/solved decoder for the target function; run
    # nengo.Simulator over the series; score prediction error. Wrap sim for wall
    # time; peak RSS polled by baselines.py.
    emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
          "metric": "", "metric_kind": "mse", "n_params": args.neurons,
          "status": "stub",
          "notes": f"nengo {getattr(mod, '__version__', '?')} present; "
                   f"NEF model TODO (neurons={args.neurons})"}, args.out)


if __name__ == "__main__":
    main()
