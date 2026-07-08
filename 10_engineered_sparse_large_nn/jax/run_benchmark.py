"""Workload 10 -- engineered sparse large NN, JAX port.

Mirrors 10_engineered_sparse_large_nn/pytorch: a streaming-regression learner on
a deep (~1000-layer), sparse, irregular DAG that grows and shrinks while running
in *pipeline* propagation (the signal advances exactly one layer per step). It
loads the same `topology.bin` so the initial network and data stream are
identical, and drives structural churn from the shared LCG so the topology
evolution matches the pytorch/C++/Plastix impls. This is the JAX reference impl
-- same summary schema (metric + phase + memory columns) as those impls.

Per-step algorithm (see README):
  1. Inputs:   act[i] = x_t[i]  for i in [0, n_in)
  2. Forward:  pre = scatter_add(w * act[src], dst); one synchronous sparse
               mat-vec over the *previous* activations (one layer advance, NOT a
               full topological forward). act = tanh(pre) for hidden/output.
  3. Loss:     delta = target_t - act[output_id]; accumulate delta^2.
  4. Update:   elig = DECAY*elig + act[src]; w += LR * delta * elig   (TD(λ)).

JAX timing notes:
  * The per-step compute (sparse scatter-add forward + TD(λ) edge update) is one
    jitted fn. The scatter-add is `zeros(N).at[dst].add(w*act[src])` -- JAX's
    functional scatter-add, the analogue of torch.index_add_.
  * Structural ops (grow/shrink) change the edge-array shapes, so they are done
    in host numpy -- exactly as the pytorch impl rebuilds its edge tensors -- and
    the jax arrays are rebuilt. A shape change triggers an XLA recompile; that is
    accepted for this port (faithfulness first). This bench is prune-dominant, so
    most structural cost is the shrink (mark_prune).
  * There is no backward phase (TD is a direct forward-driven update), so
    mark_backward is called no-arg -- ~0ns -- matching the pytorch/Plastix impls.
  * ONE warmup step compiles the jitted fn off the clock before the timed loop.

Usage:
    uv run python 10_engineered_sparse_large_nn/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/jax"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    plot_run,
    plot_test_curve,
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
    next() returns (state >> 33) (31-bit)."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & U64_MASK

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & U64_MASK
        return self.state >> 33


# ---------------------------------------------------------------------------
# topology.bin reader (numpy; identical to the pytorch loader)
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
# jitted per-step compute
# ---------------------------------------------------------------------------
#
# One jitted fn does the whole numeric step: a sparse scatter-add forward (one
# layer advance over the previous activations), then the vectorized TD(λ) edge
# update. `cur_units`, `output_id`, and `n_in` are static (they only change on a
# structural op, which forces a rebuild + recompile anyway).

@partial(jax.jit, static_argnums=(8, 9, 10))
def _step(act, src, dst, w, elig, hidden_mask, x_t, target,
          cur_units, output_id, n_in):
    # 2. one-layer-advance pipeline forward over previous activations.
    contrib = w * act[src]
    pre = jnp.zeros(cur_units).at[dst].add(contrib)     # scatter-add (index_add)
    new_act = jnp.tanh(pre)
    # hidden + output get tanh(pre); inputs get overwritten with x_t; the rest
    # (any stale non-hidden slot) keep their previous activation.
    act_new = jnp.where(hidden_mask, new_act, act)
    act_new = act_new.at[output_id].set(jnp.tanh(pre[output_id]))
    act_new = act_new.at[:n_in].set(x_t)

    # 3. loss
    delta_t = target - act_new[output_id]

    # 4. TD(λ)-style update, vectorized over edges (uses the just-computed act).
    elig = DECAY * elig + act_new[src]
    w = w + (LR * delta_t) * elig
    return act_new, w, elig, delta_t


# ---------------------------------------------------------------------------
# structural surgery (host numpy; rebuilds the edge arrays)
# ---------------------------------------------------------------------------

def _grow(cur_units, layer_list, max_hidden_layer, output_id, lcg):
    """Append GROW_UNITS hidden units (each with FANIN incoming edges from
    distinct earlier units + one edge to output) while cur_units < MAX_UNITS.
    Returns (new_units, new_layers, new_src, new_dst, max_hidden_layer)."""
    new_units = 0
    new_layers: list[int] = []
    new_src: list[int] = []
    new_dst: list[int] = []
    for _ in range(GROW_UNITS):
        if cur_units + new_units >= MAX_UNITS:
            break
        new_id = cur_units + new_units
        newlayer = 1 + (lcg.next() % max(max_hidden_layer, 1))
        new_layers.append(newlayer)
        chosen: set[int] = set()
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
    return new_units, new_layers, new_src, new_dst, max_hidden_layer


def _shrink_indices(dst_np, live, output_id, lcg):
    """Pick PRUNE_EDGES edge indices to remove (lcg()%live), skipping edges into
    the output unit. Returns a sorted list of unique indices to drop."""
    to_remove: set[int] = set()
    for _ in range(PRUNE_EDGES):
        idx = lcg.next() % live
        if dst_np[idx] == output_id:
            continue
        to_remove.add(idx)
    return sorted(to_remove)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_loop(
    n_in: int, n_units: int, output_id: int,
    layer: np.ndarray, edges: np.ndarray,
    X: np.ndarray, Y: np.ndarray,
    *, n_steps: int, log_every: int, log: StructuralLog,
    timer: PhaseTimer, probe: MemoryProbe | None = None,
):
    layer_list = layer.tolist()
    max_hidden_layer = max(layer_list[n_in:output_id]) if output_id > n_in else 1

    # hidden = non-input, non-output units (mirrors the pytorch mask exactly).
    hidden_mask_np = np.zeros(n_units, dtype=bool)
    if output_id > n_in:
        hidden_mask_np[n_in:output_id] = True
    if output_id + 1 < n_units:
        hidden_mask_np[output_id + 1:] = True

    # Host-side edge arrays (numpy) drive the structural surgery; jax arrays are
    # rebuilt from them whenever a shape changes.
    src_np = np.ascontiguousarray(edges[:, 0]).astype(np.int64)
    dst_np = np.ascontiguousarray(edges[:, 1]).astype(np.int64)
    w_np = np.zeros(edges.shape[0], dtype=np.float32)
    elig_np = np.zeros(edges.shape[0], dtype=np.float32)

    cur_units = n_units
    act = jnp.zeros(cur_units, dtype=jnp.float32)
    hidden_mask = jnp.asarray(hidden_mask_np)
    src = jnp.asarray(src_np)
    dst = jnp.asarray(dst_np)
    w = jnp.asarray(w_np)
    elig = jnp.asarray(elig_np)

    Xj = jnp.asarray(X)
    Yj = jnp.asarray(Y)
    # Edge/activation state (weights, traces, on-device stream copies) is the
    # constructed footprint here; there's no separate module to measure.
    if probe is not None:
        probe.end_weights()

    lcg = LCG(LCG_SEED)

    # Warmup: compile _step off the clock (one step on the initial shapes).
    a0, w0, e0, d0 = _step(act, src, dst, w, elig, hidden_mask,
                           Xj[0], Yj[0], cur_units, output_id, n_in)
    jax.block_until_ready((a0, w0, e0, d0))

    window_sse = 0.0
    window_cnt = 0
    epoch = 0

    tail_start = n_steps - max(1, n_steps // 10)
    tail_sse = 0.0
    tail_cnt = 0

    for t in range(n_steps):
        x_t = Xj[t]
        target = Yj[t]

        timer.tick()
        act, w, elig, delta_t = _step(
            act, src, dst, w, elig, hidden_mask,
            x_t, target, cur_units, output_id, n_in)
        timer.mark_forward(act)
        # loss (delta already computed inside the fused step; sync it).
        timer.mark_loss(delta_t)
        # no backward phase in this bench (direct TD update) -- mark ~0.
        timer.mark_backward()
        # the TD weight update is fused into _step; sync w for the update phase.
        timer.mark_update(w)

        d_val = float(delta_t)

        # structural: grow / shrink (host numpy surgery -> rebuild jax arrays).
        if t >= WARMUP and (t - WARMUP) % GROW_EVERY == 0:
            # --- grow (minor; append units + their fan-in/out edges) ---
            timer.tick()
            (new_units, new_layers, new_src, new_dst,
             max_hidden_layer) = _grow(cur_units, layer_list,
                                       max_hidden_layer, output_id, lcg)
            grew = new_units > 0
            if grew:
                layer_list.extend(new_layers)
                max_hidden_layer = max(max_hidden_layer, max(new_layers))
                cur_units += new_units
                # extend per-unit host state (new units are hidden).
                hidden_mask_np = np.concatenate(
                    [hidden_mask_np, np.ones(new_units, dtype=bool)])
                if new_src:
                    src_np = np.concatenate(
                        [src_np, np.asarray(new_src, dtype=np.int64)])
                    dst_np = np.concatenate(
                        [dst_np, np.asarray(new_dst, dtype=np.int64)])
                    n_new = len(new_src)
                    w_np = np.concatenate(
                        [w_np, np.zeros(n_new, dtype=np.float32)])
                    elig_np = np.concatenate(
                        [elig_np, np.zeros(n_new, dtype=np.float32)])
                # rebuild the grown jax arrays.
                act = jnp.concatenate(
                    [act, jnp.zeros(new_units, dtype=jnp.float32)])
                hidden_mask = jnp.asarray(hidden_mask_np)
                src = jnp.asarray(src_np)
                dst = jnp.asarray(dst_np)
                w = jnp.asarray(w_np)
                elig = jnp.asarray(elig_np)
            timer.mark_grow((w, act) if grew else None)

            # --- shrink (prune-dominant; remove edges) ---
            timer.tick()
            live = src_np.shape[0]
            drop = _shrink_indices(dst_np, live, output_id, lcg) if live else []
            pruned = len(drop) > 0
            if pruned:
                keep = np.ones(live, dtype=bool)
                keep[np.asarray(drop, dtype=np.int64)] = False
                src_np = src_np[keep]
                dst_np = dst_np[keep]
                w_np = w_np[keep]
                elig_np = elig_np[keep]
                src = jnp.asarray(src_np)
                dst = jnp.asarray(dst_np)
                w = jnp.asarray(w_np)
                elig = jnp.asarray(elig_np)
            timer.mark_prune((w, src) if pruned else None)
        else:
            # keep both phases sampled every step (≈0 when no structural op).
            timer.mark_grow()
            timer.mark_prune()
        timer.mark_reset()
        timer.step_done()

        d2 = d_val * d_val
        window_sse += d2
        window_cnt += 1
        if t >= tail_start:
            tail_sse += d2
            tail_cnt += 1

        if window_cnt >= log_every or t + 1 == n_steps:
            window_mse = window_sse / window_cnt
            epoch += 1
            n_edges_live = int(src_np.shape[0])
            log.log(epoch, cur_units, n_edges_live, edges=None,
                    val_loss=window_mse, train_loss=window_mse,
                    train_step=t + 1, test_mse=window_mse)
            print(f"[ep {epoch}] step={t+1} window_mse={window_mse:.6f} "
                  f"n_units={cur_units} n_edges={n_edges_live}")
            window_sse = 0.0
            window_cnt = 0

    test_mse = tail_sse / max(tail_cnt, 1)
    return test_mse, cur_units, int(src_np.shape[0])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> dict:
    np.random.seed(args.seed)

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

    print(f"[info] jax devices={jax.devices()}  topology={path}  steps={n_steps}  "
          f"n_in={n_in}  n_units={n_units}  n_edges={edges.shape[0]}  "
          f"output_id={output_id}  quick={int(args.quick)}")

    hist_path, summary_path, plot_path = output_paths(args, "engineered_sparse_nn")
    log = StructuralLog(hist_path)

    timer = PhaseTimer()
    t0 = time.perf_counter()
    test_mse, final_units, final_edges = run_loop(
        n_in, n_units, output_id, layer, edges, X, Y,
        n_steps=n_steps, log_every=log_every, log=log,
        timer=timer, probe=probe,
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
                 title="Engineered sparse large NN -- pipeline TD(λ) [jax]")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "engineered_sparse_nn")
        plot_test_curve(
            log.records, tpath,
            title="Engineered sparse large NN test MSE [jax]",
            metric_key="test_mse", ylabel="window MSE",
            higher_is_better=False,
        )
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick, etc.
    p.add_argument("--max-steps", type=int, default=10000,
                   help="cap the number of timesteps to run (0 = all)")
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
