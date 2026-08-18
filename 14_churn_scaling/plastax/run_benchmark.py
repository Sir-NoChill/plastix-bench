"""Workload 14 — full per-step connection churn, **plastax** port.

Mirrors 14_churn_scaling/plastix (the C++ oracle): the same layered feed-forward
DAG as bench 12, driven for N steps under **PIPELINE**, with the full churn loop
per step — forward(tanh) → loss(δ=target-out) → update(eligibility delta rule)
→ prune(~½ of edges by a stateless hash) → grow(GrowFanout/source) → reset. Live
edge count oscillates at a low steady state; the bench measures per-step and
per-phase cost, not accuracy (with a one-hop PIPELINE forward over i.i.d. inputs
there is no learnable signal — it is a structural-throughput microbench, as the
oracle intends).

Per-step trait rules (oracle):
  * forward:  a = tanh(Σ w·a_src)  (one level-advance/step; output a = prediction)
  * loss:     δ = target - out, staged onto the output unit (native G.Delta)
  * update:   e = decay·e + a_src ;  w += lr·δ·e     (reward-modulated Hebbian)
  * prune:    Mix64(cid) % 2 == 0   (~50% of slots, stateless hash — native)
  * grow:     GrowFanout random level→level+1 edges/source (see bench 12 header
              for the O(n²)-vs-O(n·k) AddConn scaling caveat)
  * reset:    advance the growth seed so each step samples a fresh candidate set

Adds are level-preserving into a pre-sized arena, so no resort / no overflow /
no retrace after the first trace — plastax's retrace contract holds; the
`AddConn` candidate scan (O(n²)) is the cost that does not scale, not retracing.

Usage:
    uv run python 14_churn_scaling/plastax/run_benchmark.py --no-plot --neurons 512 --steps 60
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
    PhaseTimer,
    add_common_args,
    build_phase_runners,
    build_pipeline_state,
    device_bytes,
    measure_fused_step_ns,
    output_paths,
    run_phase_timed_step,
    write_summary_csv,
)

import plastax as px  # noqa: E402
from plastax._types import ACTIVATION, LEVEL  # noqa: E402
from plastax.state import live_conn_count  # noqa: E402

Elig = px.FieldSpec.f32("elig")  # eligibility trace e_ij
Delta = px.FieldSpec.f32("delta")  # δ = target-out, staged on the output unit
IsOutput = px.FieldSpec.f32("is_output")


def _hash01(a, b, seed):
    a = a.astype(jnp.uint32)
    b = b.astype(jnp.uint32)
    s = seed.astype(jnp.uint32)
    x = a * jnp.uint32(2654435761) + b * jnp.uint32(2246822519) + s * jnp.uint32(3266489917)
    x = (x ^ (x >> 16)) * jnp.uint32(2246822519)
    x = x ^ (x >> 13)
    return (x >> 8).astype(jnp.float32) / jnp.float32(1 << 24)


def _mix64(cid):
    x = cid.astype(jnp.uint32)
    x = (x ^ (x >> 16)) * jnp.uint32(2246822519)
    x = (x ^ (x >> 13)) * jnp.uint32(3266489917)
    return x ^ (x >> 16)


class ChurnForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[ACTIVATION, src]

    def apply(self, u, i, g, acc):
        del u, i, g
        return px.UnitWrite.of((ACTIVATION, jnp.tanh(acc)))


class ChurnLoss(px.Loss):
    def per_output(self, u, i, target, g):
        del g
        out = u[ACTIVATION, i]
        diff = target - out  # oracle G.Delta = T - V
        return jnp.float32(0.5) * diff * diff, px.UnitWrite.of((Delta, diff))


class ChurnUpdate(px.UpdateConn):
    """e = decay·e + a_src ;  w += lr·δ·e   (δ read from the single output unit,
    the plastax stand-in for the oracle's global G.Delta)."""

    def __init__(self, lr, decay, out_id) -> None:
        self._lr = jnp.float32(lr)
        self._decay = jnp.float32(decay)
        self._out = int(out_id)

    def incoming(self, u, dst, src, c, cid, g):
        del dst, g
        e_new = self._decay * c[Elig, cid] + u[ACTIVATION, src]
        w_new = c[px.WEIGHT, cid] + self._lr * u[Delta, self._out] * e_new
        return px.ConnWrite.of((Elig, e_new), (px.WEIGHT, w_new))

    def outgoing(self, u, src, dst, c, cid, g):
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


class ChurnPrune(px.PruneConn):
    def predicate(self, u, c, cid, g):
        del u, c, g
        return (_mix64(cid) % jnp.uint32(2)) == jnp.uint32(0)


class ChurnAddConn(px.AddConn):
    def __init__(self, max_candidates: int) -> None:
        self.max_candidates = int(max_candidates)

    def score(self, u, src, dst, g):
        eligible = u[LEVEL, dst] == (u[LEVEL, src] + jnp.int32(1))
        rnd = _hash01(src, dst, g["grow_seed"])
        return jnp.where(eligible, rnd, jnp.float32(-jnp.inf))

    def init(self, u, src, dst, g):
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.0)), (Elig, jnp.float32(0.0)))


class ChurnReset(px.ResetGlobal):
    def reset(self, g):
        return {**g, "grow_seed": g["grow_seed"] + jnp.uint32(0x9E37)}


def make_net(max_candidates, lr, decay, out_id, neighbourhood=1):
    class ChurnNet(px.Network[dict]):
        forward_pass = ChurnForward()
        loss = ChurnLoss()
        update_conn = ChurnUpdate(lr, decay, out_id)
        prune_conn = ChurnPrune()
        add_conn = ChurnAddConn(max_candidates)
        reset_global = ChurnReset()
        extra_unit_fields = (IsOutput, Delta)
        extra_conn_fields = (Elig,)
        propagation = px.Propagation.PIPELINE

    ChurnNet.neighbourhood = neighbourhood
    return ChurnNet


def layer_sizes(n, n_in, depth):
    remaining = n - n_in
    layers = max(1, depth)
    sizes = []
    for lay in range(1, layers + 1):
        sz = remaining if lay == layers else remaining // (layers - lay + 1)
        sizes.append(sz)
        remaining -= sz
    return sizes


def _next_pow2(x):
    return 1 << max(0, (x - 1)).bit_length()


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

    froms, tos, weights = [], [], []
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
    n_src = n - len(layer_units[-1])

    static, state = build_pipeline_state(
        net, num_units=n, input_ids=tuple(range(n_in)), output_ids=(n - 1,),
        from_ids=froms, to_ids=tos, weights=weights, level_of=levels,
        extra_unit_cols={IsOutput.name: is_output},
        globals_={"grow_seed": jnp.uint32(0x1234 ^ (seed & 0xFFFFFFFF))},
        capacity=capacity,
    )
    return static, state, n_src, len(froms)


def run(args) -> dict:
    probe = MemoryProbe()
    probe.start()

    n, n_in, depth, fanin = args.neurons, args.inputs, args.depth, args.fanin
    n_src = n - layer_sizes(n, n_in, depth)[-1]
    max_candidates = min(args.grow_k * max(1, n_src), n * n)
    initial_edges_est = fanin * (n - n_in)
    # peak live ≈ (survivors of ~½ prune) + one grow; size well above it.
    capacity = _next_pow2(4 * initial_edges_est + 4 * max_candidates + 64)

    net = make_net(max_candidates, args.lr, args.decay, out_id=n - 1)
    static, state, n_src, n_edges0 = build_churn_dag(net, n, n_in, depth, fanin, args.seed, capacity)
    probe.end_dataset()
    probe.end_weights()

    # streaming data: 64 Bernoulli(1/8) features; target = active fraction.
    rng = np.random.default_rng(0xABCDEF ^ args.seed)
    feats = (rng.random((args.steps + 1, n_in)) < 0.125).astype(np.float32)
    targets = feats.mean(axis=1).astype(np.float32)

    runners = build_phase_runners(net, static, donate=True)
    w0 = np.asarray(state.conns[0][px.WEIGHT.name]).copy()

    print(f"[info] plastax devices={jax.devices()} churn loop n={n} edges0={n_edges0} "
          f"cap={capacity} max_cand={max_candidates} steps={args.steps}")
    # Adds are level-preserving by construction (score returns -inf unless
    # level[dst]==level[src]+1), so needs_resort stays False; overflow is
    # ruled out post-hoc by edges_max < capacity (checked after the loop).

    # warmup/compile off the clock
    si0 = px.StepInputs(inputs=jnp.asarray(feats[0]), targets=jnp.asarray(targets[0:1]))
    state, _ = run_phase_timed_step(runners, state, si0, PhaseTimer())
    jax.block_until_ready(state)

    timer = PhaseTimer()
    edges_min, edges_max = n_edges0, n_edges0
    t0 = time.perf_counter()
    for t in range(1, args.steps + 1):
        si = px.StepInputs(inputs=jnp.asarray(feats[t]), targets=jnp.asarray(targets[t:t + 1]))
        state, _ = run_phase_timed_step(runners, state, si, timer)
        live = int(live_conn_count(state))
        edges_min, edges_max = min(edges_min, live), max(edges_max, live)
    wall = time.perf_counter() - t0

    w1 = np.asarray(state.conns[0][px.WEIGHT.name])
    live_final = int(live_conn_count(state))
    # weight drift over live slots (update path exercised)
    weight_drift = float(np.mean(np.abs(w1[: len(w0)] - w0)))
    fused_ns = measure_fused_step_ns(net, static, state, si0, n=300)

    overflowed = edges_max >= capacity
    print(f"[churn] edges {edges_min}..{edges_max} (final {live_final}), "
          f"weight_drift={weight_drift:.4e}, "
          f"{'OVERFLOW (arena undersized!)' if overflowed else 'no overflow / no retrace'}")

    summary = {
        "workload": "14_churn_scaling",
        "dataset": "synthetic-bernoulli",
        "neurons": n, "inputs": n_in, "depth": depth, "fanin": fanin,
        "grow_k": args.grow_k, "steps": args.steps,
        "wall_seconds": round(wall, 3),
        "n_units": n, "n_edges": n_edges0,
        "edges_min": edges_min, "edges_max": edges_max, "edges_final": live_final,
        "candidate_grid": n * n, "max_candidates": max_candidates, "capacity": capacity,
        "weight_drift": round(weight_drift, 8),
        "lr": args.lr, "decay": args.decay, "seed": args.seed,
        "peak_vram_bytes": int(device_bytes()),
        "fused_step_ns_mean": round(fused_ns, 3),
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    _, summary_path, _ = output_paths(args, "churn_scaling")
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] n={n} fused_step={fused_ns / 1e3:.1f}µs "
          f"(fwd={summary['forward_ns_mean'] / 1e3:.0f} upd={summary['update_ns_mean'] / 1e3:.0f} "
          f"prune={summary['prune_ns_mean'] / 1e3:.0f} grow={summary['grow_ns_mean'] / 1e3:.0f}µs) "
          f"edges {edges_min}..{edges_max} vram={summary['peak_vram_bytes'] / 1e6:.0f}MB")
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
    p.add_argument("--grow-k", type=int, default=4, help="target edges added per source unit")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--decay", type=float, default=0.9, help="eligibility trace decay")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
