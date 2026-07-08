"""Workload 11 — imprinting-style scaling sweep, PyTorch.

Counterpart to 11_scaling_imprint/plastix. Builds a sparse imprinting-style DAG
of EXACTLY `--neurons N` units (N swept ~10k..10M by scaling.py), runs a fixed
number of pipeline steps, and reports per-step wall time + the RSS memory
breakdown. Same per-step compute as the Plastix impl (one sparse layer-advance
forward + a TD(λ) edge update), so the plastix-vs-pytorch scaling is
apples-to-apples on CPU.

Topology is generated in-process (no dataset file): each non-input unit gets
`--fanin` incoming edges from random earlier ids (src < dst), built vectorised
so 10M neurons stays tractable. Duplicate (src,dst) pairs are allowed — for a
scaling micro-benchmark they just add parallel edges; size still scales as
N*fanin.

Usage:
    uv run python 11_scaling_imprint/pytorch/run_benchmark.py \
        --neurons 100000 --steps 200 --device cpu --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    resolve_device,
    write_summary_csv,
)

LR = 0.01
DECAY = 0.9


def run(args) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(resolve_device(getattr(args, "device", "cpu")))

    probe = MemoryProbe()
    probe.start()

    neurons = max(int(args.neurons), args.inputs + 2)
    n_in = int(args.inputs)
    fanin = int(args.fanin)
    steps = int(args.steps)
    if args.quick:
        neurons = min(neurons, 20000)
        steps = min(steps, 100)
    output_id = neurons - 1

    probe.end_dataset()  # no external dataset; stream is generated per step

    # --- build the sparse DAG (vectorised) ---------------------------------
    # dst repeats each non-input id `fanin` times; src is a random earlier id
    # (src < dst) so the pipeline advances one layer per step.
    dst_np = np.repeat(np.arange(n_in, neurons, dtype=np.int64), fanin)
    e = dst_np.shape[0]
    src_np = (rng.random(e) * dst_np).astype(np.int64)  # in [0, dst)

    src = torch.from_numpy(src_np).to(dev)
    dst = torch.from_numpy(dst_np).to(dev)
    w = (torch.rand(e, device=dev) * 0.02 - 0.01)
    elig = torch.zeros(e, device=dev)
    act = torch.zeros(neurons, device=dev)
    hidden_mask = torch.ones(neurons, dtype=torch.bool, device=dev)
    hidden_mask[:n_in] = False
    probe.end_weights()

    log = StructuralLog(output_paths(args, "scaling_imprint")[0])

    timer = PhaseTimer()
    sse = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            # sparse binary input (~1/8 active); target = active fraction
            bits = (torch.rand(n_in, device=dev) < 0.125).float()
            target = bits.mean()

            timer.tick()
            contrib = w * act[src]
            pre = torch.zeros(neurons, device=dev).index_add_(0, dst, contrib)
            act_new = torch.tanh(pre)
            act_new[:n_in] = bits
            act = act_new
            timer.mark_forward()

            delta = target - act[output_id]
            timer.mark_loss()
            timer.mark_backward()

            elig.mul_(DECAY).add_(act[src])
            w.add_(elig, alpha=LR * float(delta))
            timer.mark_update()
            timer.mark_prune()
            timer.mark_grow()
            timer.mark_reset()
            timer.step_done()

            sse += float(delta) ** 2
    wall = time.perf_counter() - t0
    mse = sse / max(1, steps)

    hist_path, summary_path, _ = output_paths(args, "scaling_imprint")
    log.log(1, n_units=neurons, n_edges=int(e), edges=None,
            val_loss=mse, test_mse=mse, step=steps)
    log.flush()

    summary = {
        "workload": "11_scaling_imprint",
        "neurons": neurons,
        "max_steps": steps,
        "wall_seconds": round(wall, 6),
        "test_mse": round(mse, 8),
        "metric_kind": "mse",
        "n_units": neurons,
        "n_edges": int(e),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] neurons={neurons} edges={e} wall={wall:.3f}s "
          f"step_ns={wall * 1e9 / max(1, steps):.1f}  test_mse={mse:.6f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--neurons", type=int, default=10000)
    p.add_argument("--inputs", type=int, default=64)
    p.add_argument("--fanin", type=int, default=4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
