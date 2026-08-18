"""Workload 12 — connection-growth ADD-cost microbench, **plastax** port.

Mirrors 12_churn/plastix (the C++ oracle): build a layered feed-forward DAG,
call `DoAddConnections` ONCE, and time it. No forward/loss/training — it is a
pure structural-growth cost probe. The oracle grows `GrowFanout` edges *per
source unit*, each a random pick from the next level (level L → L+1 only).

**The headline finding this bench exposes.** The native growth engine samples
`GrowFanout` candidates per source — O(N·k). plastax's `AddConn` trait instead
scores the FULL num_units² (src,dst) candidate grid and takes the global
top-`max_candidates` via `lax.top_k` — **O(N²)**. So plastax reproduces the
level-window + random-init growth semantics, but its candidate enumeration does
NOT scale to the oracle's default N=100 000 (10¹⁰ candidates). This port runs at
feasible N and measures the O(N²) wall directly (`--sweep`). plastax's *retrace
contract* is not the bottleneck here (adds are level-preserving into a pre-sized
arena — no resort, no recompile); the candidate scan is.

Growth trait: `score` returns a per-(src,dst) pseudo-random value, but only for
level-preserving pairs (level[dst]==level[src]+1), else -inf (mirrors the
oracle's `ShouldAddOutgoing`); `init` sets weight 0 / eligibility 0 (mirrors
`InitConnection`).

Usage:
    uv run python 12_churn/plastax/run_benchmark.py --no-plot --neurons 512
    uv run python 12_churn/plastax/run_benchmark.py --no-plot --sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "plastax"))
from common import (  # noqa: E402
    MemoryProbe,
    add_common_args,
    build_phase_runners,
    build_pipeline_state,
    device_bytes,
    output_paths,
    write_summary_csv,
)

import plastax as px  # noqa: E402
from plastax._types import LEVEL  # noqa: E402

Elig = px.FieldSpec.f32("elig")  # eligibility trace column (unused in 12, needed by 14)
IsOutput = px.FieldSpec.f32("is_output")


def _hash01(a, b, seed):
    """Per-(src,dst,seed) pseudo-random float in [0,1) — the plastax stand-in
    for the oracle's splitmix64 candidate index (dispatch_cpu.hpp:798-806)."""
    a = a.astype(jnp.uint32)
    b = b.astype(jnp.uint32)
    s = seed.astype(jnp.uint32)
    x = a * jnp.uint32(2654435761) + b * jnp.uint32(2246822519) + s * jnp.uint32(3266489917)
    x = (x ^ (x >> 16)) * jnp.uint32(2246822519)
    x = x ^ (x >> 13)
    return (x >> 8).astype(jnp.float32) / jnp.float32(1 << 24)


class ChurnForward(px.ForwardPass):
    """ReLU forward (oracle Forward). Defined because Network requires it;
    never timed by this add-only microbench (the oracle's forward is likewise
    'cheap, unused by the add microbench')."""

    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(self, u, i, g, acc):
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jnp.maximum(acc, jnp.float32(0.0))))


class ChurnAddConn(px.AddConn):
    """Level-preserving random growth. score: random in [0,1) for
    level[dst]==level[src]+1 pairs, -inf otherwise (oracle ShouldAddOutgoing +
    Neighbourhood=1 window). init: weight 0, eligibility 0 (InitConnection)."""

    def __init__(self, max_candidates: int) -> None:
        self.max_candidates = int(max_candidates)

    def score(self, u, src, dst, g):
        eligible = u[LEVEL, dst] == (u[LEVEL, src] + jnp.int32(1))
        rnd = _hash01(src, dst, g["grow_seed"])
        return jnp.where(eligible, rnd, jnp.float32(-jnp.inf))

    def init(self, u, src, dst, g):
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.0)), (Elig, jnp.float32(0.0)))


def make_net(max_candidates: int, neighbourhood: int = 1):
    class ChurnNet(px.Network[dict]):
        forward_pass = ChurnForward()
        add_conn = ChurnAddConn(max_candidates)
        extra_unit_fields = (IsOutput,)
        extra_conn_fields = (Elig,)
        propagation = px.Propagation.PIPELINE

    ChurnNet.neighbourhood = neighbourhood
    return ChurnNet


def layer_sizes(n: int, n_in: int, depth: int) -> list[int]:
    """Even split of the N-NIn hidden units across `depth` layers, last layer
    taking the remainder (oracle DepthBuilder)."""
    remaining = n - n_in
    layers = max(1, depth)
    sizes = []
    for lay in range(1, layers + 1):
        sz = remaining if lay == layers else remaining // (layers - lay + 1)
        sizes.append(sz)
        remaining -= sz
    return sizes


def build_churn_dag(net, n, n_in, depth, fanin, seed, capacity):
    rng = np.random.default_rng(seed)
    sizes = layer_sizes(n, n_in, depth)
    levels = np.zeros(n, dtype=np.int32)
    layer_units = [list(range(n_in))]
    idx = n_in
    for lay, sz in enumerate(sizes, start=1):
        us = list(range(idx, idx + sz))
        idx += sz
        for u in us:
            levels[u] = lay
        layer_units.append(us)

    froms: list[int] = []
    tos: list[int] = []
    weights: list[float] = []
    for lay in range(1, len(layer_units)):
        prev = layer_units[lay - 1]
        for u in layer_units[lay]:
            want = min(min(fanin, 16), len(prev))
            for s in rng.choice(prev, size=want, replace=False):
                froms.append(int(s))
                tos.append(int(u))
                weights.append(float(rng.uniform(-0.01, 0.01)))

    is_output = np.zeros(n, dtype=np.float32)
    is_output[n - 1] = 1.0
    n_src = n - len(layer_units[-1])  # units NOT in the last level (can grow out)

    static, state = build_pipeline_state(
        net,
        num_units=n,
        input_ids=tuple(range(n_in)),
        output_ids=(n - 1,),
        from_ids=froms,
        to_ids=tos,
        weights=weights,
        level_of=levels,
        extra_unit_cols={IsOutput.name: is_output},
        globals_={"grow_seed": jnp.uint32(0x1234 ^ (seed & 0xFFFFFFFF))},
        capacity=capacity,
    )
    return static, state, n_src, len(froms)


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


def measure_add(n, n_in, depth, fanin, grow_k, seed):
    """Build the DAG and time ONE add_conn phase (grow_k edges/source target)."""
    n_src = n - layer_sizes(n, n_in, depth)[-1]
    max_candidates = min(grow_k * max(1, n_src), n * n)
    initial_edges_est = fanin * (n - n_in)
    capacity = _next_pow2(2 * (initial_edges_est + max_candidates) + 64)

    net = make_net(max_candidates)
    static, state, n_src, n_edges0 = build_churn_dag(
        net, n, n_in, depth, fanin, seed, capacity
    )
    grow = build_phase_runners(net, static, donate=False)["grow"]
    dummy = px.StepInputs(inputs=jnp.zeros((n_in,), jnp.float32), targets=None)

    # warmup / compile off the clock
    s2, _ = grow(state, dummy)
    jax.block_until_ready(s2)
    added = int((~s2.conns[0][px.DEAD.name]).sum()) - n_edges0

    reps = 20
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        s2, _ = grow(state, dummy)  # same input each time (non-donating)
        jax.block_until_ready(s2)
    add_ns = (time.perf_counter_ns() - t0) / reps
    return {
        "n": n, "n_edges0": n_edges0, "capacity": capacity,
        "max_candidates": max_candidates, "added": added, "add_ns": add_ns,
    }


def run(args) -> dict:
    probe = MemoryProbe()
    probe.start()
    probe.end_dataset()

    sizes = (
        [128, 256, 512, 1024, 2048] if args.sweep else [args.neurons]
    )
    print(f"[info] plastax devices={jax.devices()} churn add-cost probe "
          f"(grow_k={args.grow_k}, depth={args.depth}, fanin={args.fanin})")
    print(f"[info] plastax AddConn scores the FULL n² candidate grid → O(n²); "
          f"native GrowFanout samples grow_k/source → O(n·k)")

    rows = []
    for n in sizes:
        r = measure_add(n, args.inputs, args.depth, args.fanin, args.grow_k, args.seed)
        rows.append(r)
        print(f"[add] n={r['n']:>6d} edges0={r['n_edges0']:>8d} "
              f"added={r['added']:>8d} cand_grid={r['n'] ** 2:>12d} "
              f"add_ns={r['add_ns']:>14,.0f} ({r['add_ns'] / 1e6:.2f} ms)")
    probe.end_weights()

    # O(n²) check: ns per candidate-grid-cell should be roughly flat across n.
    if len(rows) > 1:
        print("[scaling] add_ns / n²  (flat ⇒ O(n²) candidate scan dominates):")
        for r in rows:
            print(f"          n={r['n']:>6d}  {r['add_ns'] / (r['n'] ** 2):.3f} ns/cell")

    last = rows[-1]
    summary = {
        "workload": "12_churn",
        "dataset": "none-structural",
        "neurons": last["n"], "inputs": args.inputs, "depth": args.depth,
        "fanin": args.fanin, "grow_k": args.grow_k,
        "n_units": last["n"], "n_edges": last["n_edges0"],
        "edges_added": last["added"], "candidate_grid": last["n"] ** 2,
        "add_ns_mean": round(last["add_ns"], 1),
        "add_ns_per_cell": round(last["add_ns"] / (last["n"] ** 2), 5),
        "seed": args.seed,
        "peak_vram_bytes": int(device_bytes()),
        "sweep": ";".join(f"{r['n']}:{r['add_ns']:.0f}" for r in rows),
        **probe.summary_fields(),
    }
    _, summary_path, _ = output_paths(args, "churn")
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] n={last['n']} add={last['add_ns'] / 1e6:.2f}ms added={last['added']} "
          f"vram={summary['peak_vram_bytes'] / 1e6:.0f}MB")
    print(f"[done] wrote {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--neurons", type=int, default=512,
                   help="total units (NOT the oracle's 100000 — plastax AddConn is O(n²))")
    p.add_argument("--inputs", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fanin", type=int, default=4)
    p.add_argument("--grow-k", type=int, default=8, help="target edges added per source unit")
    p.add_argument("--sweep", action="store_true", help="scaling sweep over n to expose O(n²)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
