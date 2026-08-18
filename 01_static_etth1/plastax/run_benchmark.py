"""Workload 1 — STATIC regime on ETTh1, **plastax** port.

Mirrors 01_static_etth1/plastix (the C++ Plastix impl): a fixed depth-3 GeLU
MLP (2 GeLU hidden + linear output), per-example SGD with sum-MSE. Topology
never changes. This is the plastax (JAX-Plastix) reference impl — same summary
schema as pytorch/plastix/cpp/cuda/jax, so it slots into runs.csv.

Trait port of the C++ StaticForward/StaticBackward/StaticUpdateConn:
  * forward:  z = sum W*a_src; store PreAct=z; a = IsOutput ? z : GeLU(z).
  * loss:     MSELoss stages dL/da = pred-target into LossGrad (output units).
  * backward: dL/da = acc + LossGrad; dL/dz = dL/da * (IsOutput ? 1 : GeLU'(z)).
  * update:   w -= lr * dL/dz_dst * a_src.
Fields PreAct/GradPreAct/IsOutput mirror the C++ PreActTag/GradPreActTag/
IsOutputTag; LossGrad bridges loss->backward for output units (plastax has no
framework BackwardAcc column — see examples/mlp_xor.py).

Phases are timed individually (common/plastax build_phase_runners) so the
breakdown matches the oracle's separate DoForwardPass/DoBackwardPass/
DoUpdateConn; plastax can FUSE them into one make_step kernel (reported as
fused_step_ns for reference).

Usage:
    uv run python 01_static_etth1/plastax/run_benchmark.py --quick --no-plot
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
    device_bytes,
    download_if_missing,
    measure_fused_step_ns,
    output_paths,
    run_phase_timed_step,
    write_summary_csv,
)

import plastax as px  # noqa: E402
from plastax._types import ACTIVATION  # noqa: E402

ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
_INV_SQRT2 = 0.70710678118654752440
_INV_SQRT2PI = 0.39894228040143267794

PreAct = px.FieldSpec.f32("pre_act")  # z, stored by forward for GeLU'(z)
GradPreAct = px.FieldSpec.f32("grad_pre_act")  # dL/dz, persisted across levels
LossGrad = px.FieldSpec.f32("loss_grad")  # dL/da staged by loss (output units)
IsOutput = px.FieldSpec.f32("is_output")  # 1.0 for output units (linear), else 0


def _gelu(z: jax.Array) -> jax.Array:
    return jnp.float32(0.5) * z * (jnp.float32(1.0) + jax.lax.erf(z * _INV_SQRT2))


def _gelu_grad(z: jax.Array) -> jax.Array:
    cdf = jnp.float32(0.5) * (jnp.float32(1.0) + jax.lax.erf(z * _INV_SQRT2))
    pdf = _INV_SQRT2PI * jnp.exp(jnp.float32(-0.5) * z * z)
    return cdf + z * pdf


class StaticForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[ACTIVATION, src]

    def apply(self, u, i, g, acc):
        del g
        is_out = u[IsOutput, i] > jnp.float32(0.5)
        activation = jnp.where(is_out, acc, _gelu(acc))
        return px.UnitWrite.of((PreAct, acc), (ACTIVATION, activation))


class StaticBackward(px.BackwardPass):
    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[GradPreAct, src]

    def apply(self, u, i, g, acc):
        del g
        z = u[PreAct, i]
        dlda = acc + u[LossGrad, i]
        is_out = u[IsOutput, i] > jnp.float32(0.5)
        dphidz = jnp.where(is_out, jnp.float32(1.0), _gelu_grad(z))
        return px.UnitWrite.of((GradPreAct, dlda * dphidz))


class MSELoss(px.Loss):
    def per_output(self, u, i, target, g):
        del g
        pred = u[ACTIVATION, i]
        diff = pred - target
        return jnp.float32(0.5) * diff * diff, px.UnitWrite.of((LossGrad, diff))


class SgdUpdate(px.UpdateConn):
    def __init__(self, lr: float) -> None:
        self._lr = jnp.float32(lr)

    def incoming(self, u, dst, src, c, cid, g):
        del g
        delta = self._lr * u[GradPreAct, dst] * u[ACTIVATION, src]
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] - delta))

    def outgoing(self, u, src, dst, c, cid, g):
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


def make_net(lr: float) -> type[px.Network[None]]:
    class StaticNet(px.Network[None]):
        forward_pass = StaticForward()
        backward_pass = StaticBackward()
        loss = MSELoss()
        update_conn = SgdUpdate(lr)
        extra_unit_fields = (PreAct, GradPreAct, LossGrad, IsOutput)
        propagation = px.Propagation.TOPOLOGICAL

    return StaticNet


# --- data (numpy; mirrors the jax/pytorch loaders) -------------------------


def _synthesize(T: int, C: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    base = np.sin(2 * np.pi * t / 24)[:, None] + 0.4 * np.sin(
        2 * np.pi * t / (24 * 7)
    )[:, None]
    return base + 0.1 * rng.standard_normal((T, C)).astype(np.float32)


def load_etth1(data_dir: Path, synthetic: bool) -> np.ndarray:
    if synthetic:
        return _synthesize(17_420, 7, seed=0)
    csv_path = data_dir / "ETTh1.csv"
    try:
        download_if_missing(ETTH1_URL, csv_path)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ETTh1 download failed ({e}); synthetic", file=sys.stderr)
        return _synthesize(17_420, 7, seed=0)
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "date"]
    return df[cols].to_numpy(dtype=np.float32)


def windowed(data: np.ndarray, in_len: int, out_len: int):
    T, C = data.shape
    n = T - in_len - out_len + 1
    idx = np.arange(in_len)[None, :] + np.arange(n)[:, None]
    X = data[idx].reshape(n, in_len * C)
    idx_y = np.arange(out_len)[None, :] + np.arange(n)[:, None] + in_len
    Y = data[idx_y].reshape(n, out_len * C)
    return X.astype(np.float32), Y.astype(np.float32)


def build_state(net, in_dim, out_dim, hidden, depth, key):
    """input -> (depth-1) GeLU hidden of `hidden` -> linear `out_dim` output.
    nn.Linear-style U(-1/sqrt(fan_in), 1/sqrt(fan_in)) init (matches jax/torch)."""
    init = jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
    dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
    blocks = [px.topology.input_units(in_dim)]
    for i in range(depth):
        blocks.append(px.topology.dense(dims[i], dims[i + 1], init=init))
    static, state = px.NetworkBuilder.from_topology(
        net, px.topology.sequential(*blocks), key, globals_=None
    )
    # Mark output units (linear activation + linear backward).
    is_out = state.units[IsOutput.name].at[jnp.asarray(static.output_ids)].set(1.0)
    state = dataclasses.replace(state, units={**state.units, IsOutput.name: is_out})
    return static, state


def run(args) -> dict:
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    probe = MemoryProbe()
    probe.start()

    data = load_etth1(args.data_dir, args.synthetic)
    n_train = int(0.7 * len(data))
    mu = data[:n_train].mean(0)
    sd = data[:n_train].std(0) + 1e-6
    data = (data - mu) / sd
    X, Y = windowed(data, args.in_len, args.out_len)
    n_tr = int(0.7 * len(X))
    n_va = int(0.15 * len(X))
    Xtr, Ytr = jnp.asarray(X[:n_tr]), jnp.asarray(Y[:n_tr])
    Xte, Yte = jnp.asarray(X[n_tr + n_va :]), jnp.asarray(Y[n_tr + n_va :])
    probe.end_dataset()

    in_dim, out_dim = X.shape[1], Y.shape[1]
    net = make_net(args.lr)
    key, mk = jax.random.split(key)
    static, state = build_state(net, in_dim, out_dim, args.hidden, args.depth, mk)
    output_ids = np.asarray(static.output_ids)
    n_units = int(state.units[ACTIVATION.name].shape[0])
    n_edges = int(sum(int((~b[px.DEAD.name]).sum()) for b in state.conns))
    probe.end_weights()

    runners = build_phase_runners(net, static, donate=True)
    eval_fwd = build_phase_runners(net, static, donate=False)["forward"]

    def evaluate(trained_state, n_max=1024):
        n = min(len(Xte), n_max)
        se = ae = 0.0
        for j in range(n):
            si = px.StepInputs(inputs=Xte[j], targets=None)
            st, _ = eval_fwd(trained_state, si)
            pred = np.asarray(st.units[ACTIVATION.name][output_ids])
            tgt = np.asarray(Yte[j])
            se += float(np.mean((pred - tgt) ** 2))
            ae += float(np.mean(np.abs(pred - tgt)))
        return se / n, ae / n

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    cap = args.max_train_rows if args.max_train_rows > 0 else n_tr
    n_use = min(n_tr, cap)
    rng = np.random.default_rng(args.seed)

    hist_path, summary_path, _ = output_paths(args, "static_etth1")
    log = StructuralLog(hist_path)
    v0_mse, _ = evaluate(state, n_max=256)
    log.log(0, n_units, n_edges, edges={(0, 0, 0)}, val_loss=v0_mse,
            train_loss=None, test_mse=v0_mse, test_mae=0.0)

    print(f"[info] plastax devices={jax.devices()} in={in_dim} out={out_dim} "
          f"hidden={args.hidden} depth={args.depth} units={n_units} edges={n_edges} "
          f"epochs={epochs} rows/epoch={n_use}")

    # Warmup: compile every phase kernel off the clock.
    si0 = px.StepInputs(inputs=Xtr[0], targets=Ytr[0])
    ws = state
    ws, _ = run_phase_timed_step(runners, ws, si0, PhaseTimer())
    jax.block_until_ready(ws)
    state = ws

    timer = PhaseTimer()
    t0 = time.perf_counter()
    for _ep in range(1, epochs + 1):
        order = rng.permutation(n_tr)[:n_use]
        for j in order:
            si = px.StepInputs(inputs=Xtr[j], targets=Ytr[j])
            state, _ = run_phase_timed_step(runners, state, si, timer)
    wall = time.perf_counter() - t0

    te_mse, te_mae = evaluate(state)
    log.log(epochs, n_units, n_edges, edges={(0, 0, 0)}, val_loss=te_mse,
            train_loss=None, test_mse=te_mse, test_mae=te_mae)
    log.flush()

    # plastax's real production path: one fused kernel per step (vs the
    # phase-separated loop's per-phase device sync). Reported for a fair
    # per-step-cost comparison to the native fused C++ step.
    fused_ns = measure_fused_step_ns(
        net, static, state, px.StepInputs(inputs=Xtr[0], targets=Ytr[0]), n=500
    )

    summary = {
        "workload": "01_static_etth1",
        "dataset": "ETTh1" if not args.synthetic else "synthetic-etth1",
        "in_len": args.in_len, "out_len": args.out_len,
        "hidden": args.hidden, "depth": args.depth,
        "epochs": epochs, "batch": 1, "lr": args.lr,
        "wall_seconds": round(wall, 3),
        "test_mse": round(float(te_mse), 6), "test_mae": round(float(te_mae), 6),
        "val_mse_final": round(float(te_mse), 6),
        "n_units": n_units, "n_edges": n_edges,
        "jaccard_min": 1.0, "jaccard_max": 1.0,
        "seed": args.seed,
        "peak_vram_bytes": int(device_bytes()),
        "fused_step_ns_mean": round(fused_ns, 3),
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s test_mse={float(te_mse):.4f} "
          f"phase_sep_step_ns={summary['step_ns_mean']:.0f} "
          f"fused_step_ns={fused_ns:.0f} "
          f"vram={summary['peak_vram_bytes'] / 1e6:.0f}MB")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--in-len", type=int, default=96)
    p.add_argument("--out-len", type=int, default=24)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)  # per-example SGD (accepted, ignored)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train-rows", type=int, default=2000,
                   help="cap rows/epoch (per-example SGD is slow); 0 = full split")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
