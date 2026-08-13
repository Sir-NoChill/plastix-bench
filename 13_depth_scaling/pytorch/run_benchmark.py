"""Workload 13 - depth scaling sweep, PyTorch.

Counterpart to 13_depth_scaling/plastix. Holds N fixed and sweeps `--depth L`:
the (N - inputs) non-input units are split into L sequential layers, each unit
wired from `--fanin` random units in the previous layer. The forward pass runs
L sequential index_add scatters (one per level, in dependency order), mirroring
the Plastix Pipeline model's level-by-level advance, so per-step cost reflects
the depth of the network. Fixed topology, no runtime growth.

Usage:
    python 13_depth_scaling/pytorch/run_benchmark.py \
        --neurons 200000 --depth 8 --steps 100 --device cpu --no-plot
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


def _layer_bounds(n_in: int, neurons: int, depth: int):
    """Contiguous [begin, end) unit ranges for each of `depth` layers,
    matching the Plastix DepthBuilder's partitioning of the hidden units."""
    hidden = neurons - n_in
    bounds = []
    made = 0
    for lay in range(1, depth + 1):
        remaining = hidden - made
        layers_left = depth - lay + 1
        size = remaining if lay == depth else max(1, remaining // layers_left)
        begin = n_in + made
        end = begin + size
        bounds.append((begin, end))
        made += end - begin
    return bounds


def run(args) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(resolve_device(getattr(args, "device", "cpu")))

    probe = MemoryProbe()
    probe.start()

    n_in = int(args.inputs)
    fanin = int(args.fanin)
    depth = max(1, int(args.depth))
    neurons = max(int(args.neurons), n_in + depth + 1)
    steps = int(args.steps)
    if args.quick:
        neurons = min(neurons, 20000)
        steps = min(steps, 100)
    output_id = neurons - 1
    probe.end_dataset()

    # Per-layer edge tensors: layer l draws `fanin` sources from layer l-1
    # (inputs are the level-0 "layer 0").
    prev = (0, n_in)
    layers = []  # each: (src, dst, w, elig)
    total_edges = 0
    for (begin, end) in _layer_bounds(n_in, neurons, depth):
        dst_np = np.repeat(np.arange(begin, end, dtype=np.int64), fanin)
        psize = prev[1] - prev[0]
        src_np = (prev[0] + (rng.random(dst_np.shape[0]) * psize)).astype(np.int64)
        src = torch.from_numpy(src_np).to(dev)
        dst = torch.from_numpy(dst_np).to(dev)
        w = (torch.rand(dst_np.shape[0], device=dev) * 0.02 - 0.01)
        elig = torch.zeros(dst_np.shape[0], device=dev)
        layers.append((begin, end, src, dst, w, elig))
        total_edges += int(dst_np.shape[0])
        prev = (begin, end)

    act = torch.zeros(neurons, device=dev)
    probe.end_weights()

    log = StructuralLog(output_paths(args, "depth_scaling")[0])
    timer = PhaseTimer()
    sse = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            bits = (torch.rand(n_in, device=dev) < 0.125).float()
            target = bits.mean()

            timer.tick()
            act = torch.zeros(neurons, device=dev)
            act[:n_in] = bits
            # sequential level-by-level advance (the depth cost): each layer's
            # units are the contiguous range [begin, end), so scatter into a
            # scratch and slice-assign the activated block.
            for (begin, end, src, dst, w, _elig) in layers:
                pre = torch.zeros(neurons, device=dev).index_add_(
                    0, dst, w * act[src])
                act[begin:end] = torch.tanh(pre[begin:end])
            timer.mark_forward()

            delta = target - act[output_id]
            timer.mark_loss()
            timer.mark_backward()

            for (_begin, _end, src, _dst, w, elig) in layers:
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

    hist_path, summary_path, _ = output_paths(args, "depth_scaling")
    log.log(n_units=neurons, n_edges=total_edges, edges=None,
            val_loss=mse, test_mse=mse, step=steps)
    log.flush()

    summary = {
        "workload": "13_depth_scaling",
        "neurons": neurons,
        "depth": depth,
        "max_steps": steps,
        "wall_seconds": round(wall, 6),
        "test_mse": round(mse, 8),
        "metric_kind": "mse",
        "n_units": neurons,
        "n_edges": total_edges,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] neurons={neurons} depth={depth} edges={total_edges} "
          f"wall={wall:.3f}s step_ns={wall * 1e9 / max(1, steps):.1f}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--neurons", type=int, default=200000)
    p.add_argument("--inputs", type=int, default=64)
    p.add_argument("--fanin", type=int, default=4)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
