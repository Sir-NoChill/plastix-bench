"""TEMPLATE — JAX benchmark skeleton (NOT auto-discovered).

Copy this to `<bench>/jax/run_benchmark.py` and fill in the three TODOs to add a
real JAX impl of a bench. The orchestrator globs `*/<impl>/run_benchmark.py`, so
this file's `_template` suffix keeps it out of discovery until it is copied.

The contract every impl follows (so runs.csv has one schema across frameworks):
  * parse the shared args (`add_common_args`) + any bench-specific ones,
  * probe memory at three milestones (overhead / after-data / after-model),
  * wrap the per-step loop in a PhaseTimer (JAX-async-safe: pass each phase's
    result to mark_* so timing is real, not just dispatch),
  * emit a one-row summary via `write_summary_csv` carrying at least
    wall_seconds, a test_* metric, n_units/n_edges, plus **timer.summary_fields
    and **probe.summary_fields.

Once a real JAX bench exists, uncomment the ("jax", ...) line in
bench_meta.FRAMEWORKS to fold a jax column into the phase/memory tables, and
`jax` is already registered in orchestrator IMPL_ORDER/COLOUR/LABEL.

Run (after copying + filling in):
    uv sync --extra jax
    uv run python <bench>/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# common/jax is two levels up from <bench>/jax/ ; adjust parents[N] if you nest
# the template elsewhere. From <bench>/jax/run_benchmark.py it is parents[2].
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/jax"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    add_common_args,
    output_paths,
    write_summary_csv,
)

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402


def run(args) -> dict:
    probe = MemoryProbe()
    probe.start()

    # 1) TODO: build/load the dataset -----------------------------------------
    key = jax.random.PRNGKey(args.seed)
    # x_train, y_train = ...
    probe.end_dataset()

    # 2) TODO: build the model params -----------------------------------------
    # params = init_params(key, ...)
    n_units, n_edges = 0, 0
    probe.end_weights()

    # 3) TODO: the timed training loop ----------------------------------------
    timer = PhaseTimer()
    steps = 100 if args.quick else 1000
    sse = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        timer.tick()
        # fwd = forward(params, x)               ; timer.mark_forward(fwd)
        # loss = loss_fn(fwd, y)                  ; timer.mark_loss(loss)
        # grads = jax.grad(loss_fn)(params, ...)  ; timer.mark_backward(grads)
        # params = sgd(params, grads)            ; timer.mark_update(params)
        timer.mark_forward(); timer.mark_loss(); timer.mark_backward()
        timer.mark_update(); timer.mark_prune(); timer.mark_grow(); timer.mark_reset()
        timer.step_done()
    wall = time.perf_counter() - t0
    mse = sse / max(1, steps)

    hist_path, summary_path, _ = output_paths(args, "jax_template")
    summary = {
        "workload": "jax_template",
        "wall_seconds": round(wall, 6),
        "test_mse": round(mse, 8),
        "metric_kind": "mse",
        "n_units": n_units,
        "n_edges": n_edges,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.3f}s (TEMPLATE — fill in the TODOs)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot
    # p.add_argument("--hidden", type=int, default=256)   # bench-specific args
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
