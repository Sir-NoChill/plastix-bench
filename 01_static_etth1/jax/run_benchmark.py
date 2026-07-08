"""Workload 1 / 5 — STATIC regime on ETTh1, JAX port.

Mirrors 01_static_etth1/pytorch: a fixed 3-layer GeLU MLP trained with
sum-reduction MSE + plain SGD. Topology never changes (Jaccard flat 1.0). This
is the JAX reference impl — same summary schema (phase + memory columns) as the
pytorch/plastix/cpp/cuda impls, so it slots into runs.csv and the tables.

JAX timing notes:
  * A warmup step runs each jitted fn once BEFORE the timed loop so XLA compile
    time isn't charged to the first step.
  * The phase split marks forward (an explicit `apply`), loss, backward
    (`jax.grad`, which recomputes the forward — JAX stores no activations, unlike
    autograd), then update. Each mark blocks on its result so async dispatch is
    flushed and the timing is real (see common/jax PhaseTimer).

Usage:
    uv run python 01_static_etth1/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
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
    download_if_missing,
    output_paths,
    write_summary_csv,
)

ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)


# --- data (numpy; mirrors the pytorch loader) ------------------------------

def _synthesize(T: int, C: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    base = (np.sin(2 * np.pi * t / 24)[:, None]
            + 0.4 * np.sin(2 * np.pi * t / (24 * 7))[:, None])
    return base + 0.1 * rng.standard_normal((T, C)).astype(np.float32)


def load_etth1(data_dir: Path, synthetic: bool) -> np.ndarray:
    if synthetic:
        return _synthesize(17_420, 7, seed=0)
    csv_path = data_dir / "ETTh1.csv"
    try:
        download_if_missing(ETTH1_URL, csv_path)
    except Exception as e:
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


# --- model (params = list of (W, b)) ---------------------------------------

def init_params(key, in_dim, out_dim, hidden, depth):
    dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
    params = []
    for i in range(depth):
        key, wk = jax.random.split(key)
        # Kaiming-ish uniform, matching nn.Linear's default init scale.
        lim = 1.0 / (dims[i] ** 0.5)
        W = jax.random.uniform(wk, (dims[i], dims[i + 1]), minval=-lim, maxval=lim)
        b = jnp.zeros((dims[i + 1],))
        params.append((W, b))
    return params


def apply_mlp(params, x):
    last = len(params) - 1
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i != last:
            x = jax.nn.gelu(x)
    return x


def _edge_count(params) -> int:
    return int(sum(W.size for W, _ in params))


def _unit_count(params) -> int:
    return int(sum(W.shape[1] for W, _ in params))


def run(args) -> dict:
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    probe = MemoryProbe()
    probe.start()

    data = load_etth1(args.data_dir, args.synthetic)
    n_train = int(0.7 * len(data))
    mu = data[:n_train].mean(0); sd = data[:n_train].std(0) + 1e-6
    data = (data - mu) / sd
    X, Y = windowed(data, args.in_len, args.out_len)
    n_tr = int(0.7 * len(X)); n_va = int(0.15 * len(X))
    Xtr, Ytr = jnp.asarray(X[:n_tr]), jnp.asarray(Y[:n_tr])
    Xva, Yva = jnp.asarray(X[n_tr:n_tr + n_va]), jnp.asarray(Y[n_tr:n_tr + n_va])
    Xte, Yte = jnp.asarray(X[n_tr + n_va:]), jnp.asarray(Y[n_tr + n_va:])
    probe.end_dataset()

    key, mk = jax.random.split(key)
    params = init_params(mk, X.shape[1], Y.shape[1], args.hidden, args.depth)
    probe.end_weights()

    lr = args.lr

    @jax.jit
    def forward(params, xb):
        return apply_mlp(params, xb)

    @jax.jit
    def loss_from_pred(pred, yb):
        return jnp.sum((pred - yb) ** 2)            # sum reduction (matches torch)

    def _loss(params, xb, yb):
        return jnp.sum((apply_mlp(params, xb) - yb) ** 2)

    grad_fn = jax.jit(jax.grad(_loss))

    @jax.jit
    def sgd(params, grads):
        return [(W - lr * gW, b - lr * gb)
                for (W, b), (gW, gb) in zip(params, grads)]

    @jax.jit
    def eval_mse_mae(params, x, y):
        p = apply_mlp(params, x)
        return jnp.mean((p - y) ** 2), jnp.mean(jnp.abs(p - y))

    def batches(rng):
        idx = np.array(rng.permutation(n_tr))
        for i in range(0, n_tr - args.batch + 1, args.batch):
            sl = idx[i:i + args.batch]
            yield Xtr[sl], Ytr[sl]

    hist_path, summary_path, _ = output_paths(args, "static_etth1")
    log = StructuralLog(hist_path)
    n_units, n_edges = _unit_count(params), _edge_count(params)
    v0, _ = eval_mse_mae(params, Xva, Yva)
    log.log(0, n_units, n_edges, edges={(0, 0, 0)}, val_loss=float(v0),
            train_loss=None, test_mse=float(v0), test_mae=0.0)

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    rng = np.random.default_rng(args.seed)
    print(f"[info] jax devices={jax.devices()} in_dim={X.shape[1]} "
          f"out_dim={Y.shape[1]} hidden={args.hidden} depth={args.depth} "
          f"epochs={epochs} train={n_tr}")

    # Warmup: force XLA compilation of every jitted fn off the clock.
    xb0, yb0 = Xtr[:args.batch], Ytr[:args.batch]
    p0 = forward(params, xb0); l0 = loss_from_pred(p0, yb0)
    g0 = grad_fn(params, xb0, yb0); _ = sgd(params, g0)
    jax.block_until_ready((p0, l0, g0))

    timer = PhaseTimer()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for xb, yb in batches(rng):
            timer.tick()
            pred = forward(params, xb)
            timer.mark_forward(pred)
            loss = loss_from_pred(pred, yb)
            timer.mark_loss(loss)
            grads = grad_fn(params, xb, yb)
            timer.mark_backward(grads)
            params = sgd(params, grads)
            timer.mark_update(params)
            timer.step_done()
        va_mse, va_mae = eval_mse_mae(params, Xva, Yva)
        te_mse, te_mae = eval_mse_mae(params, Xte, Yte)
        log.log(ep, n_units, n_edges, edges={(0, 0, 0)},
                val_loss=float(va_mse), train_loss=None, val_mae=float(va_mae),
                test_mse=float(te_mse), test_mae=float(te_mae))
        print(f"[ep {ep:>3d}] val_mse={float(va_mse):.4f} test_mse={float(te_mse):.4f}")
    wall = time.perf_counter() - t0

    te_mse, te_mae = eval_mse_mae(params, Xte, Yte)
    log.flush()

    summary = {
        "workload": "01_static_etth1",
        "dataset": "ETTh1" if not args.synthetic else "synthetic-etth1",
        "in_len": args.in_len, "out_len": args.out_len,
        "hidden": args.hidden, "depth": args.depth,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "wall_seconds": round(wall, 3),
        "val_mse_final": round(log.records[-1]["val_loss"], 6),
        "test_mse": round(float(te_mse), 6),
        "test_mae": round(float(te_mae), 6),
        "n_units": n_units, "n_edges": n_edges,
        "jaccard_min": 1.0, "jaccard_max": 1.0,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s val_mse={summary['val_mse_final']:.4f} "
          f"test_mse={float(te_mse):.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick, etc.
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--in-len", type=int, default=96)
    p.add_argument("--out-len", type=int, default=24)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
