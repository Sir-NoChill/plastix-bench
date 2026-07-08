"""Baseline: spiking neural network framework (Norse, falling back to snnTorch).

Why here: a dedicated SNN library is the natural external comparison for bench 08
(Spiking Heidelberg Digits). Plastix implements the SNN with hand-written LIF +
e-prop policies; Norse/snnTorch provide batteries-included spiking layers +
surrogate-gradient training. Shows the accuracy/throughput/memory of the same
task through a purpose-built spiking stack.

Mapped task: 08_snn_shd (SHD digit classification). Report test accuracy, wall
time, peak RSS, and parameter count.

STATUS: scaffold. snnTorch is already a base dependency; Norse is the optional
richer alternative. Fill in the TODO to train + score on SHD.
    uv sync --extra snn        # for Norse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_baseline_args, emit, try_import  # noqa: E402

FRAMEWORK = "snn"
TASK = "08_snn_shd (SHD classification)"
PARADIGM = "spiking NN (surrogate-grad / e-prop)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_baseline_args(p)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    mod = try_import("norse") or try_import("snntorch")
    if mod is None:
        emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
              "status": "skipped",
              "notes": "no norse/snntorch — `uv sync --extra snn`"}, args.out)
        return

    backend = mod.__name__
    # TODO: load SHD (reuse 08_snn_shd/pytorch/data loaders), build a LIF network
    # in norse/snntorch, train --epochs with a surrogate gradient, score test acc.
    # Wrap training for wall time; peak RSS polled by baselines.py.
    emit({"framework": f"{FRAMEWORK}:{backend}", "task": TASK, "paradigm": PARADIGM,
          "metric": "", "metric_kind": "acc", "n_params": "",
          "status": "stub",
          "notes": f"{backend} present; SHD train/score TODO (epochs={args.epochs})"},
         args.out)


if __name__ == "__main__":
    main()
