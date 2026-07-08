"""Baseline: TensorNEAT — JAX-accelerated NEAT (evolving topologies on GPU).

Why here: NEAT evolves network *structure* (nodes + connections) under a fitness
signal — the closest external analogue to Plastix's runtime structural
adaptation, but via population search rather than per-step local rules. Shows the
cost/behaviour of "grow a topology" done evolutionarily.

Mapped task: a small control / regression problem (e.g. the imprinting-style
scaling target or a standard CartPole-ish fitness), evolved for a few
generations; report best fitness (as the metric), wall time, peak RSS and the
best genome's param count.

STATUS: scaffold. Fill in the TODO to run a real TensorNEAT pipeline.
TensorNEAT is GitHub-only (not on PyPI):
    uv pip install "tensorneat @ git+https://github.com/EMI-Group/tensorneat.git" jax
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_baseline_args, emit, try_import  # noqa: E402

FRAMEWORK = "tensorneat"
TASK = "structural-adaptation (control/regression)"
PARADIGM = "evolutionary NEAT (JAX)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_baseline_args(p)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--pop-size", type=int, default=256)
    args = p.parse_args()

    mod = try_import("tensorneat")
    if mod is None:
        emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
              "status": "skipped",
              "notes": "not installed — pip install git+https://github.com/EMI-Group/tensorneat.git"},
             args.out)
        return

    # TODO: build a tensorneat Pipeline (algorithm=NEAT, a Problem, common.State),
    # run `pipeline.auto_run(state)` for --generations, then read best fitness +
    # genome size. Wrap the evolve loop for wall time; peak RSS is polled by
    # baselines.py. Emit the row below with real numbers.
    emit({"framework": FRAMEWORK, "task": TASK, "paradigm": PARADIGM,
          "metric": "", "metric_kind": "fitness", "n_params": "",
          "status": "stub",
          "notes": f"tensorneat {getattr(mod, '__version__', '?')} present; "
                   f"model TODO (gens={args.generations}, pop={args.pop_size})"},
         args.out)


if __name__ == "__main__":
    main()
