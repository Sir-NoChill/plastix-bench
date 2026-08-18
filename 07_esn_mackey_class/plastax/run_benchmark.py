"""Workload 7 — Echo State Network on Mackey-Glass, **plastax** port.

Mirrors 07_esn_mackey_class/plastix (the C++ Plastix impl): a FIXED random
recurrent reservoir driven one step per timestep, with a host-side ridge-
regression readout. Despite the name "..._class" this is REGRESSION —
one-step-ahead prediction of the Mackey-Glass series (metrics MSE/RMSE/R²).

Reservoir dynamics (native oracle):
    x(t) = (1-α)·x(t-1) + α·tanh( W_in·u(t) + W_rec·x(t-1) )   α = leak = 0.3

The reservoir is CYCLIC (res_j → res_i for all i≠j), so it cannot be built
through NetworkBuilder.from_topology (its host-side level pass rejects
cycles). It maps to plastax's **PIPELINE** propagation: one flat synchronous
sweep per timestep — every edge's Map reads the previous step's activation,
every reservoir unit is then re-applied — which is exactly one reservoir
update. State persists in the ACTIVATION column across timesteps. Built via
common/plastax build_pipeline_state (the plastax ReservoirBuilder analogue).

Only ForwardPass is a plastax trait; W_in/W_rec are fixed (never trained) and
the readout W_out is fit once by closed-form ridge over collected reservoir
states — there is no in-graph learning phase (matches the oracle, which only
ever calls DoForwardPass).

Usage:
    uv run python 07_esn_mackey_class/plastax/run_benchmark.py --no-plot
"""

from __future__ import annotations

import argparse
import dataclasses
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
    StructuralLog,
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
from plastax._types import ACTIVATION  # noqa: E402
from plastax.phases import build_phases  # noqa: E402


# --- reservoir forward trait ------------------------------------------------


class LeakyTanhForward(px.ForwardPass):
    """map = w·activation[src] (linear, both W_in and W_rec edges);
    apply = leaky-tanh (oracle Forward::Apply, 07_esn_mackey_class.cpp:104-118):
    x_i = (1-α)·x_i(prev) + α·tanh(Σ). Input unit is protected by the sweep."""

    combine = px.monoid.sum_

    def __init__(self, leak: float) -> None:
        self._leak = jnp.float32(leak)

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[ACTIVATION, src]

    def apply(self, u, i, g, acc):
        del g
        old = u[ACTIVATION, i]
        new = (jnp.float32(1.0) - self._leak) * old + self._leak * jnp.tanh(acc)
        return px.UnitWrite.of((ACTIVATION, new))


def make_net(leak: float) -> type[px.Network[None]]:
    fwd = LeakyTanhForward(leak)

    class EsnNet(px.Network[None]):
        forward_pass = fwd
        propagation = px.Propagation.PIPELINE

    return EsnNet


# --- reservoir construction (fixed random weights) --------------------------


def build_reservoir(net, n: int, sr: float, seed: int):
    """1 input (id 0) + N reservoir units (ids 1..N). Input edges 0->res_i
    (W_in ~ U(-1,1)); recurrent edges res_j->res_i for all i≠j
    (W_rec ~ N(0, sr/√N), zero diagonal). N + N·(N-1) = N² live edges."""
    rng = np.random.default_rng(seed)
    w_in = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    w_rec = rng.normal(0.0, sr / np.sqrt(n), size=(n, n)).astype(np.float32)
    np.fill_diagonal(w_rec, 0.0)

    # recurrent edges: dst i, src j (i≠j), weight W_rec[i, j]
    dst_i, src_j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    off = dst_i != src_j
    rec_src = (src_j[off] + 1).astype(np.int32)
    rec_dst = (dst_i[off] + 1).astype(np.int32)
    rec_w = w_rec[dst_i[off], src_j[off]].astype(np.float32)

    # input edges: 0 -> res_i, weight W_in[i]
    in_src = np.zeros(n, dtype=np.int32)
    in_dst = np.arange(1, n + 1, dtype=np.int32)

    from_ids = np.concatenate([in_src, rec_src])
    to_ids = np.concatenate([in_dst, rec_dst])
    weights = np.concatenate([w_in, rec_w])

    static, state = build_pipeline_state(
        net,
        num_units=n + 1,
        input_ids=(0,),
        output_ids=(),
        from_ids=from_ids,
        to_ids=to_ids,
        weights=weights,
        globals_=None,
    )
    return static, state


# --- Mackey-Glass series (oracle recipe, 07_esn_mackey_class.cpp:57-100) -----


def mackey_glass(n_samples: int, tau: int, seed: int) -> np.ndarray:
    """β=0.2, γ=0.1, h=0.1, τ=17, sub-sample 10, settle τ·20 windows, then
    zero-mean/unit-std normalize (oracle normalizes unconditionally). We seed
    the full delay buffer at 1.2+jitter rather than the oracle's 18 samples —
    a different RNG realization (the jax/pytorch refs already differ), but the
    same chaotic MG(τ=17) attractor for a valid one-step-prediction task."""
    rng = np.random.default_rng(seed)
    beta, gamma, expo, h, sub = 0.2, 0.1, 10, 0.1, 10
    di = tau * sub  # delay in integration steps = 170
    settle = tau * 20 * sub  # discard 3400 integration steps
    n_int = settle + n_samples * sub
    buf = np.empty(di + 1 + n_int, dtype=np.float64)
    buf[: di + 1] = 1.2 + 0.01 * rng.standard_normal(di + 1)
    for t in range(di + 1, di + 1 + n_int):
        delayed = buf[t - 1 - di]
        cur = buf[t - 1]
        buf[t] = cur + h * (beta * delayed / (1.0 + delayed**expo) - gamma * cur)
    series = buf[di + 1 + settle :][::sub][:n_samples].astype(np.float32)
    return (series - series.mean()) / (series.std() + 1e-8)


# --- run --------------------------------------------------------------------


def run(args) -> dict:
    probe = MemoryProbe()
    probe.start()

    series_len = args.series_len if not args.quick else min(args.series_len, 800)
    series = mackey_glass(series_len, args.tau, args.seed)
    n_train_all = int(args.train_frac * series_len)
    warmup = args.warmup
    train_rows = n_train_all - warmup - 1
    test_rows = series_len - n_train_all - 1
    if train_rows < 1 or test_rows < 1:
        raise ValueError(f"series too short: train_rows={train_rows} test_rows={test_rows}")
    probe.end_dataset()

    net = make_net(args.leak)
    static, state = build_reservoir(net, args.units, args.sr, args.seed)
    n_units = int(state.units[ACTIVATION.name].shape[0])
    n_edges = int((~state.conns[0][px.DEAD.name]).sum())
    probe.end_weights()

    # Drive the reservoir continuously over the whole series (warmup→train→
    # test, never reset — oracle behavior), collecting each step's reservoir
    # state in one scan (one device→host transfer, no per-step sync).
    forward_phase = build_phases(net, static)[0]
    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)
    res_ids = jnp.arange(1, args.units + 1, dtype=jnp.int32)
    dummy = px.StepInputs(inputs=jnp.zeros((1,), jnp.float32), targets=None)

    @jax.jit
    def collect(state, u_seq):
        def step(st, u):
            act = st.units[ACTIVATION.name].at[input_ids].set(u)
            st = dataclasses.replace(st, units={**st.units, ACTIVATION.name: act})
            st, _ = forward_phase(st, dummy)
            return st, st.units[ACTIVATION.name][res_ids]

        return jax.lax.scan(step, state, u_seq)

    # inputs series[0..series_len-2] → states aligned so state[k] follows input k.
    u_seq = jnp.asarray(series[: series_len - 1].reshape(-1, 1))
    t0 = time.perf_counter()
    _, states = collect(state, u_seq)
    states = np.asarray(jax.block_until_ready(states))  # (series_len-1, N)
    collect_wall = time.perf_counter() - t0

    # Ridge over the train window: rows warmup..n_train_all-2, targets shifted +1.
    s_train = states[warmup : warmup + train_rows].astype(np.float64)
    y_train = series[warmup + 1 : warmup + 1 + train_rows].astype(np.float64)
    a = s_train.T @ s_train + args.ridge * np.eye(args.units)
    w_out = np.linalg.solve(a, s_train.T @ y_train)  # (N,)

    s_test = states[n_train_all : n_train_all + test_rows].astype(np.float64)
    y_test = series[n_train_all + 1 : n_train_all + 1 + test_rows].astype(np.float64)
    pred = s_test @ w_out
    err = pred - y_test
    test_mse = float(np.mean(err**2))
    test_rmse = float(np.sqrt(test_mse))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    test_r2 = float(1.0 - np.sum(err**2) / (ss_tot + 1e-12))

    print(f"[info] plastax devices={jax.devices()} reservoir={args.units} "
          f"units={n_units} edges={n_edges} series_len={series_len} "
          f"train_rows={train_rows} test_rows={test_rows}")

    # --- per-step forward timing (matches the oracle's DoForwardPass-only step)
    runners = build_phase_runners(net, static, donate=True)
    n_time = min(train_rows, 1500)
    si0 = px.StepInputs(inputs=jnp.asarray(series[warmup:warmup + 1]), targets=None)
    ws = state
    ws, _ = run_phase_timed_step(runners, ws, si0, PhaseTimer())  # warmup/compile
    jax.block_until_ready(ws)

    timer = PhaseTimer()
    t0 = time.perf_counter()
    for k in range(n_time):
        si = px.StepInputs(inputs=jnp.asarray(series[k % (series_len - 1):k % (series_len - 1) + 1]), targets=None)
        ws, _ = run_phase_timed_step(runners, ws, si, timer)
    wall = time.perf_counter() - t0

    # `ws` is the final threaded (undeleted) state; the donating loop above
    # consumed the original `state`'s buffers, so measure off `ws`.
    fused_ns = measure_fused_step_ns(net, static, ws, si0, n=500)

    hist_path, summary_path, _ = output_paths(args, "esn_mackey_class")
    log = StructuralLog(hist_path)
    log.log(0, n_units, n_edges, edges={(0, 0, 0)}, val_loss=test_mse,
            train_loss=None, test_mse=test_mse, test_mae=float(np.mean(np.abs(err))))
    log.flush()

    summary = {
        "workload": "07_esn_mackey_class",
        "dataset": "mackey-glass",
        "reservoir": args.units, "spectral_radius": args.sr, "leak": args.leak,
        "ridge": args.ridge, "tau": args.tau, "series_len": series_len,
        "epochs": 0, "batch": 1, "lr": 0.0,
        "wall_seconds": round(wall, 3),
        "test_mse": round(test_mse, 8), "test_rmse": round(test_rmse, 8),
        "test_r2": round(test_r2, 6),
        "test_mae": round(float(np.mean(np.abs(err))), 8),
        "val_mse_final": round(test_mse, 8),
        "n_units": n_units, "n_edges": n_edges,
        "jaccard_min": 1.0, "jaccard_max": 1.0,
        "seed": args.seed,
        "peak_vram_bytes": int(device_bytes()),
        "fused_step_ns_mean": round(fused_ns, 3),
        "collect_wall_seconds": round(collect_wall, 3),
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] test_mse={test_mse:.3e} rmse={test_rmse:.3e} r2={test_r2:.5f} "
          f"phase_sep_fwd_ns={summary['forward_ns_mean']:.0f} "
          f"fused_step_ns={fused_ns:.0f} vram={summary['peak_vram_bytes'] / 1e6:.0f}MB")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--units", type=int, default=100, help="reservoir size N")
    p.add_argument("--sr", type=float, default=1.25, help="nominal spectral radius")
    p.add_argument("--leak", type=float, default=0.3, help="leak rate α")
    p.add_argument("--ridge", type=float, default=1e-5, help="ridge λ")
    p.add_argument("--tau", type=int, default=17, help="Mackey-Glass delay")
    p.add_argument("--series-len", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--train-frac", type=float, default=0.8)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
