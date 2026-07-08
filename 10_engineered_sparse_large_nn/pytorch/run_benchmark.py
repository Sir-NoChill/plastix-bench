"""
Workload 10 -- engineered sparse large NN.

PyTorch implementation of a streaming-regression learner on a deep
(~1000-layer), sparse, irregular DAG that grows and shrinks while running in
*pipeline* propagation (the signal advances exactly one layer per step). This
mirrors what the C++ and Plastix implementations compute, loading the same
`topology.bin` so the initial network and data stream are identical, and
driving structural churn from the shared LCG so the topology evolution matches.

Per-step algorithm (see README):
  1. Inputs:   act[i] = x_t[i]  for i in [0, n_in)
  2. Forward:  pre = scatter_add(w * act[src], dst); one synchronous sparse
               mat-vec over the *previous* activations (one layer advance, NOT
               a full topological forward). act = tanh(pre) for hidden units,
               linear for the output unit.
  3. Loss:     delta = target_t - act[output_id]; accumulate delta^2.
  4. Update:   elig = DECAY*elig + act[src]; w += LR * delta * elig   (TD(λ)).

Metric: test_mse = mean delta^2 over the final 10% of steps.

Usage:
    uv run python 10_engineered_sparse_large_nn/pytorch/run_benchmark.py \
        --quick --device cpu --no-plot
"""

from __future__ import annotations

import argparse
import struct
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
    plot_run,
    plot_test_curve,
    resolve_device,
    test_plot_path,
    write_summary_csv,
)


# --- learner constants (identical across impls) ---
LR = 0.01
DECAY = 0.9

# --- growth / shrink schedule (shared with C++/Plastix) ---
WARMUP = 1000
GROW_EVERY = 500
GROW_UNITS = 4
FANIN = 4
PRUNE_EDGES = 8
MAX_UNITS = 6000
LCG_SEED = 0x9E3779B97F4A7C15

MAGIC = b"ESLN"
U64_MASK = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Shared 64-bit LCG (identical generator across impls)
# ---------------------------------------------------------------------------

class LCG:
    """state = state*6364136223846793005 + 1442695040888963407 (wrapping u64);
    lcg() returns (state >> 33) (31-bit)."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & U64_MASK

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & U64_MASK
        return self.state >> 33


# ---------------------------------------------------------------------------
# topology.bin reader
# ---------------------------------------------------------------------------

def _resolve_topology(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "topology.bin",
        Path("10_engineered_sparse_large_nn/topology.bin"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_topology(path: Path):
    """Returns (n_in, n_units, output_id, layer[n_units] int64,
    edges (E,2) int64, X (n_steps, n_in) f32, Y (n_steps,) f32)."""
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise RuntimeError(f"Not an ESLN file (bad magic): {path}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != 1:
            raise RuntimeError(f"Unsupported ESLN version {version}")
        n_in, n_units, n_edges, n_steps, output_id = struct.unpack("<IIIII", f.read(20))
        layer = np.frombuffer(f.read(n_units * 4), dtype=np.uint32).astype(np.int64)
        edges = np.frombuffer(f.read(n_edges * 8), dtype=np.uint32).reshape(n_edges, 2).astype(np.int64)
        rec = np.frombuffer(f.read(n_steps * (n_in + 1) * 4), dtype=np.float32).reshape(n_steps, n_in + 1)
    X = np.ascontiguousarray(rec[:, :n_in])
    Y = np.ascontiguousarray(rec[:, n_in])
    return n_in, n_units, output_id, layer, edges.copy(), X, Y


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_loop(
    n_in: int, n_units: int, output_id: int,
    layer: np.ndarray, edges: np.ndarray,
    X: np.ndarray, Y: np.ndarray,
    *, n_steps: int, log_every: int, log: StructuralLog,
    device: str, timer: PhaseTimer, probe: MemoryProbe | None = None,
):
    dev = torch.device(device)

    # Activations and per-unit masks. We keep layer as a Python list so growth
    # can extend it cheaply, plus a device tensor for hidden/output masks.
    layer_list = layer.tolist()
    max_hidden_layer = max(layer_list[n_in:output_id]) if output_id > n_in else 1

    act = torch.zeros(n_units, device=dev)
    # hidden = non-input, non-output units
    hidden_mask = torch.zeros(n_units, dtype=torch.bool, device=dev)
    if output_id > n_in:
        hidden_mask[n_in:output_id] = True
    # (units after output_id, if any, are also hidden; output sits at output_id)
    if output_id + 1 < n_units:
        hidden_mask[output_id + 1:] = True

    # Edge tensors in insertion order.
    src = torch.as_tensor(edges[:, 0], dtype=torch.long, device=dev)
    dst = torch.as_tensor(edges[:, 1], dtype=torch.long, device=dev)
    w = torch.zeros(edges.shape[0], device=dev)
    elig = torch.zeros(edges.shape[0], device=dev)

    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    # Edge/activation state (weights, traces, on-device stream copies) is the
    # constructed footprint here; there's no separate nn.Module to measure.
    if probe is not None:
        probe.end_weights()

    lcg = LCG(LCG_SEED)

    window_sse = 0.0
    window_cnt = 0
    epoch = 0

    tail_start = n_steps - max(1, n_steps // 10)
    tail_sse = 0.0
    tail_cnt = 0

    cur_units = n_units

    with torch.no_grad():
        for t in range(n_steps):
            x_t = Xt[t]
            target = Yt[t]

            timer.tick()
            # 1. inputs + 2. one-layer-advance pipeline forward over previous act
            contrib = w * act[src]
            pre = torch.zeros(cur_units, device=dev).index_add_(0, dst, contrib)
            act_new = act.clone()
            act_new[hidden_mask] = torch.tanh(pre[hidden_mask])
            act_new[output_id] = torch.tanh(pre[output_id])  # bounded output (finite over long horizons)
            act_new[:n_in] = x_t
            act = act_new
            timer.mark_forward()

            # 3. loss
            delta_t = target - act[output_id]
            timer.mark_loss()

            # 4. TD(λ)-style update, vectorized over edges
            elig.mul_(DECAY).add_(act[src])
            w.add_(elig, alpha=LR * float(delta_t))
            timer.mark_update()

            # structural: grow / shrink
            if t >= WARMUP and (t - WARMUP) % GROW_EVERY == 0:
                (cur_units, act, hidden_mask, src, dst, w, elig,
                 layer_list, max_hidden_layer) = _grow(
                    cur_units, act, hidden_mask, src, dst, w, elig,
                    layer_list, max_hidden_layer, output_id, lcg, dev)
                timer.mark_grow()
                src, dst, w, elig = _shrink(src, dst, w, elig, output_id, lcg)
                timer.mark_prune()
            else:
                # keep both phases sampled every step (≈0 when no structural op)
                timer.mark_grow()
                timer.mark_prune()
            timer.step_done()

            d2 = float(delta_t) ** 2
            window_sse += d2
            window_cnt += 1
            if t >= tail_start:
                tail_sse += d2
                tail_cnt += 1

            if window_cnt >= log_every or t + 1 == n_steps:
                window_mse = window_sse / window_cnt
                epoch += 1
                n_edges_live = int(src.numel())
                log.log(epoch, cur_units, n_edges_live, edges=None,
                        val_loss=window_mse, train_loss=window_mse,
                        train_step=t + 1, test_mse=window_mse)
                print(f"[ep {epoch}] step={t+1} window_mse={window_mse:.6f} "
                      f"n_units={cur_units} n_edges={n_edges_live}")
                window_sse = 0.0
                window_cnt = 0

    test_mse = tail_sse / max(tail_cnt, 1)
    return test_mse, cur_units, int(src.numel())


def _grow(cur_units, act, hidden_mask, src, dst, w, elig,
          layer_list, max_hidden_layer, output_id, lcg, dev):
    """Append GROW_UNITS hidden units (each with FANIN incoming edges from
    distinct earlier units + one edge to output) while cur_units < MAX_UNITS."""
    new_units = 0
    new_layers = []
    new_src = []
    new_dst = []
    for _ in range(GROW_UNITS):
        if cur_units >= MAX_UNITS:
            break
        new_id = cur_units + new_units
        newlayer = 1 + (lcg.next() % max(max_hidden_layer, 1))
        new_layers.append(newlayer)
        chosen = set()
        attempts = 0
        for _f in range(FANIN):
            s = -1
            while attempts < 10 * FANIN:
                attempts += 1
                cand = lcg.next() % output_id
                if cand in chosen:
                    continue
                if layer_list[cand] >= newlayer:
                    continue
                s = cand
                break
            if s < 0:
                continue
            chosen.add(s)
            new_src.append(s)
            new_dst.append(new_id)
        # edge new_unit -> output
        new_src.append(new_id)
        new_dst.append(output_id)
        new_units += 1

    if new_units == 0:
        return (cur_units, act, hidden_mask, src, dst, w, elig,
                layer_list, max_hidden_layer)

    # extend per-unit state
    layer_list.extend(new_layers)
    max_hidden_layer = max(max_hidden_layer, max(new_layers))
    act = torch.cat([act, torch.zeros(new_units, device=dev)])
    hidden_mask = torch.cat(
        [hidden_mask, torch.ones(new_units, dtype=torch.bool, device=dev)])
    cur_units += new_units

    # extend edge state (new weights/traces start at 0)
    if new_src:
        src = torch.cat([src, torch.tensor(new_src, dtype=torch.long, device=dev)])
        dst = torch.cat([dst, torch.tensor(new_dst, dtype=torch.long, device=dev)])
        n_new = len(new_src)
        w = torch.cat([w, torch.zeros(n_new, device=dev)])
        elig = torch.cat([elig, torch.zeros(n_new, device=dev)])

    return (cur_units, act, hidden_mask, src, dst, w, elig,
            layer_list, max_hidden_layer)


def _shrink(src, dst, w, elig, output_id, lcg):
    """Prune PRUNE_EDGES edges chosen by lcg()%live_edge_count, skipping edges
    into the output unit. Removes by rebuilding the edge tensors (insertion
    order preserved among survivors)."""
    live = src.numel()
    if live == 0:
        return src, dst, w, elig
    dst_cpu = dst.tolist()
    to_remove = set()
    for _ in range(PRUNE_EDGES):
        idx = lcg.next() % live
        if dst_cpu[idx] == output_id:
            continue
        to_remove.add(idx)
    if not to_remove:
        return src, dst, w, elig
    keep = torch.ones(live, dtype=torch.bool, device=src.device)
    keep[torch.tensor(sorted(to_remove), dtype=torch.long, device=src.device)] = False
    return src[keep], dst[keep], w[keep], elig[keep]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    probe = MemoryProbe()
    probe.start()

    path = _resolve_topology(args.data_dir)
    if path is None:
        raise SystemExit(
            f"[err] topology.bin not found under {args.data_dir}.\n"
            f"      Generate via 10_engineered_sparse_large_nn/gen.py, or pass "
            f"--data-dir <dir-with-topology.bin>."
        )

    n_in, n_units, output_id, layer, edges, X, Y = _load_topology(path)
    probe.end_dataset()
    n_steps = X.shape[0]
    if args.max_steps > 0:
        n_steps = min(n_steps, args.max_steps)
    if args.quick:
        n_steps = min(n_steps, 2000)

    log_every = args.log_every
    if args.quick:
        log_every = min(log_every, 200)

    print(f"[info] device={device}  topology={path}  steps={n_steps}  "
          f"n_in={n_in}  n_units={n_units}  n_edges={edges.shape[0]}  "
          f"output_id={output_id}  quick={int(args.quick)}")

    hist_path, summary_path, plot_path = output_paths(args, "engineered_sparse_nn")
    log = StructuralLog(hist_path)

    timer = PhaseTimer()
    t0 = time.perf_counter()
    test_mse, final_units, final_edges = run_loop(
        n_in, n_units, output_id, layer, edges, X, Y,
        n_steps=n_steps, log_every=log_every, log=log,
        device=device, timer=timer, probe=probe,
    )
    wall = time.perf_counter() - t0

    log.flush()

    summary = {
        "workload": "10_engineered_sparse_large_nn",
        "dataset": "engineered_sparse_nn",
        "max_steps": n_steps,
        "wall_seconds": round(wall, 3),
        "test_mse": round(test_mse, 6),
        "metric_kind": "mse",
        "n_units": int(final_units),
        "n_edges": int(final_edges),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.2f}s  test_mse={test_mse:.6f}  "
          f"n_units={final_units}  n_edges={final_edges}  "
          f"(step_count={timer.step_count})")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title="Engineered sparse large NN -- pipeline TD(λ)")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "engineered_sparse_nn")
        plot_test_curve(
            log.records, tpath,
            title="Engineered sparse large NN test MSE",
            metric_key="test_mse", ylabel="window MSE",
            higher_is_better=False,
        )
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--max-steps", type=int, default=10000,
                   help="cap the number of timesteps to run (0 = all)")
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
