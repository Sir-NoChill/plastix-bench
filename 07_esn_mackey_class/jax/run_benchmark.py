"""Workload 07 — Echo State Network on Mackey-Glass, JAX port.

Mirrors 07_esn_mackey_class/pytorch (which drives reservoirpy) and the raw-C++
impl at 07_esn_mackey_class/cpp. Here the ESN math is reimplemented directly in
JAX rather than via reservoirpy: a FIXED random reservoir (W_in, W_rec) is
driven over the series and only a linear readout W_out is fit by CLOSED-FORM
ridge regression.

    x(t) = (1-a) x(t-1) + a tanh(W_in u(t) + W_rec x(t-1) + b)
    y(t) = W_out x(t)
    W_out = Y Xᵀ (X Xᵀ + λI)⁻¹        (closed-form ridge, no SGD)

W_in (N x in) and W_rec (N x N) are fixed random; W_rec is scaled to a target
spectral radius (N(0, sr/sqrt(N)), zero diagonal — same recipe as the C++ impl).

JAX timing notes:
  * There is no iterative SGD, so the phase split is unusual. The reservoir
    state-collection scan (the O(N^2)/step forward dynamics) is charged to
    `forward`; the closed-form ridge solve (the analogue of loss+backward+update
    in one shot) is charged to `backward`. loss/update/prune/grow/reset are
    no-ops. To keep ns/step comparable with the iterative benches we amortise
    over the number of training timesteps (step_done() once per training row),
    exactly as the pytorch 07 impl does.
  * A warmup runs each jitted fn once BEFORE the clock so XLA compile time isn't
    charged to the timed region.

Usage:
    uv run python 07_esn_mackey_class/jax/run_benchmark.py --quick --no-plot
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
    output_paths,
    plot_run,
    plot_test_curve,
    test_plot_path,
    write_summary_csv,
)


# --- data (numpy; mirrors the cpp Mackey-Glass Euler recipe) ---------------

def mackey_glass(n: int, tau: int, seed: int) -> np.ndarray:
    """Mackey-Glass series via the same Euler recipe as the C++ impl
    (07_esn_mackey_class/cpp): beta=0.2, gamma=0.1, h=0.1, sub-sample x10,
    settle for tau*20 windows, small gaussian jitter on the seed history."""
    rng = np.random.default_rng(seed)
    beta, gamma, h = 0.2, 0.1, 0.1
    sub = 10
    settle = tau * 20
    buf = [1.2 + rng.normal(0.0, 0.01) for _ in range(tau + 1)]
    cur = buf[-1]
    di = tau * sub

    def step():
        nonlocal cur
        delayed = buf[-di] if len(buf) >= di else buf[0]
        dx = beta * delayed / (1.0 + delayed ** 10.0) - gamma * cur
        cur += h * dx
        buf.append(cur)
        if len(buf) > tau * sub * 2:
            del buf[: len(buf) - tau * sub * 2]

    for _ in range(settle * sub):
        step()
    out = []
    i = 0
    while len(out) < n:
        step()
        if (i % sub) == 0:
            out.append(cur)
        i += 1
    return np.asarray(out[:n], dtype=np.float32)


# --- reservoir (fixed random W_in, W_rec) ----------------------------------

def init_reservoir(key, units: int, sr: float, in_dim: int):
    """W_in ~ U(-1, 1) (N x in); W_rec ~ N(0, sr/sqrt(N)) with zero diagonal
    (N x N), giving an approximate spectral radius of `sr` without a
    power-iteration pass. b = 0. All fixed (never trained)."""
    kin, krec = jax.random.split(key)
    w_in = jax.random.uniform(kin, (units, in_dim), minval=-1.0, maxval=1.0)
    rec_std = sr / (units ** 0.5)
    w_rec = rec_std * jax.random.normal(krec, (units, units))
    w_rec = w_rec - jnp.diag(jnp.diag(w_rec))   # zero self-loops
    return w_in, w_rec


@jax.jit
def collect_states(w_in, w_rec, us, x0, leak):
    """Drive the reservoir over the 1-D input sequence `us` (T,), starting from
    state x0 (N,). Returns states X (T x N) via lax.scan.

        x(t) = (1-a) x(t-1) + a tanh(W_in u(t) + W_rec x(t-1))
    """
    def step(x, u):
        pre = w_in @ jnp.atleast_1d(u) + w_rec @ x
        x_new = (1.0 - leak) * x + leak * jnp.tanh(pre)
        return x_new, x_new

    _, xs = jax.lax.scan(step, x0, us)
    return xs


@jax.jit
def ridge_solve(X, Y, ridge):
    """Closed-form ridge readout. X: (T x N) states, Y: (T x out) targets.
        W_out = Y Xᵀ (X Xᵀ + λI)⁻¹        (out x N)
    Solved via jnp.linalg.solve of the SPD system (X Xᵀ + λI) Aᵀ = X Yᵀ. Done
    in float64 for numerical parity with the C++ double-precision solve."""
    Xd = X.astype(jnp.float64)
    Yd = Y.astype(jnp.float64)
    n = Xd.shape[1]
    XtX = Xd.T @ Xd + ridge * jnp.eye(n, dtype=jnp.float64)   # (N x N)
    XtY = Xd.T @ Yd                                           # (N x out)
    w_out_t = jnp.linalg.solve(XtX, XtY)                      # (N x out)
    return w_out_t.T                                          # (out x N)


def run(args) -> dict:
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    probe = MemoryProbe()
    probe.start()

    # ---- data ----------------------------------------------------------
    n_timesteps = args.series_len // (4 if args.quick else 1)
    n_timesteps = max(500, n_timesteps)
    series = mackey_glass(n_timesteps, tau=17, seed=args.seed)
    if args.normalize:
        series = (series - series.mean()) / (series.std() + 1e-8)

    n_train = int(len(series) * args.train_frac)
    # One-step-ahead forecast: input u(t) -> target series[t + horizon].
    us = series[:-args.horizon]
    ys = series[args.horizon:]
    warmup = min(args.warmup, n_train - 2)

    us_j = jnp.asarray(us)
    ys_j = jnp.asarray(ys)
    probe.end_dataset()

    # ---- fixed reservoir ----------------------------------------------
    key, rk = jax.random.split(key)
    w_in, w_rec = init_reservoir(rk, args.units, args.sr, in_dim=1)
    x0 = jnp.zeros((args.units,), dtype=w_rec.dtype)
    probe.end_weights()

    print(f"[info] jax devices={jax.devices()} series_len={n_timesteps} "
          f"horizon={args.horizon} train={n_train} units={args.units} "
          f"sr={args.sr} leak={args.lr} ridge={args.ridge} warmup={warmup} "
          f"normalize={args.normalize}")

    leak = float(args.lr)
    # Training rows = post-warmup states in the training window.
    train_rows = n_train - warmup
    if train_rows <= 1:
        sys.exit("[fatal] warmup consumed the entire training window")

    # ---- warmup: compile jitted fns off the clock ----------------------
    w0 = collect_states(w_in, w_rec, us_j[: warmup + 4], x0, leak)
    r0 = ridge_solve(w0[-4:], ys_j[warmup: warmup + 4][:, None], args.ridge)
    jax.block_until_ready((w0, r0))

    timer = PhaseTimer()
    t0 = time.perf_counter()

    # ---- forward: collect reservoir states over the whole series -------
    all_states = collect_states(w_in, w_rec, us_j, x0, leak)   # (T x N)
    timer.mark_forward(all_states)
    timer.mark_loss()  # no-op (ridge writes W_out directly, no loss policy)

    # Training states/targets (drop the warmup transient).
    Xtr = all_states[warmup:n_train]           # (train_rows x N)
    Ytr = ys_j[warmup:n_train][:, None]        # (train_rows x 1)

    # ---- backward: closed-form ridge solve (the "learning") ------------
    w_out = ridge_solve(Xtr, Ytr, args.ridge)  # (1 x N)
    timer.mark_backward(w_out)
    timer.mark_update()  # no-op (solve already produced the readout)

    # Amortise the single closed-form fit over the training rows so ns/step
    # is comparable with the iterative benches (mirrors pytorch 07).
    for _ in range(train_rows):
        timer.step_done()

    wall = time.perf_counter() - t0

    # ---- test inference ------------------------------------------------
    Xte = all_states[n_train:]                 # (test_rows x N)
    Yte = ys_j[n_train:]                        # (test_rows,)
    preds = (Xte @ w_out.T).squeeze(-1)         # (test_rows,)
    preds = jax.block_until_ready(preds)

    if not bool(jnp.all(jnp.isfinite(preds))):
        rmse_v, mse_v, r2_v = float("inf"), float("inf"), float("-inf")
    else:
        err = preds - Yte
        mse_v = float(jnp.mean(err ** 2))
        rmse_v = float(jnp.sqrt(jnp.mean(err ** 2)))
        ss_tot = float(jnp.sum((Yte - jnp.mean(Yte)) ** 2)) + 1e-12
        r2_v = float(1.0 - jnp.sum(err ** 2) / ss_tot)

    # ---- logging / summary ---------------------------------------------
    hist_path, summary_path, plot_path = output_paths(args, "esn_mg_baseline")
    log = StructuralLog(hist_path)
    n_units = args.units + 1                    # reservoir + readout output unit
    n_edges = args.units * args.units           # dense frozen recurrent matrix
    log.log(0, n_units=n_units, n_edges=n_edges, edges=None,
            val_loss=rmse_v, test_rmse=rmse_v, test_mse=mse_v, test_r2=r2_v,
            sr=args.sr, lr=args.lr, ridge=args.ridge,
            units=args.units, seed=args.seed)
    log.flush()

    summary = {
        "workload": "esn_mg_baseline",
        "dataset": "mackey-glass",
        "units": args.units,
        "sr": args.sr,
        "lr": args.lr,
        "ridge": args.ridge,
        "warmup": warmup,
        "horizon": args.horizon,
        "series_len": n_timesteps,
        "train_frac": args.train_frac,
        "normalize": int(args.normalize),
        "wall_seconds": round(wall, 3),
        "test_rmse": round(rmse_v, 8),
        "test_mse": round(mse_v, 8),
        "test_r2": round(r2_v, 8),
        "n_units": n_units,
        "n_edges": n_edges,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.3f}s test_rmse={rmse_v:.6f} "
          f"test_mse={mse_v:.6f} test_r2={r2_v:.6f}")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"ESN-MG baseline (jax) -- units={args.units} "
                       f"sr={args.sr} lr={args.lr}")
        tpath = test_plot_path(args, "esn_mg_baseline")
        plot_test_curve(log.records, tpath,
                        title=f"ESN-MG baseline test RMSE (jax) -- sr={args.sr}",
                        metric_key="test_rmse", ylabel="test RMSE",
                        higher_is_better=False)
        print(f"[done] wrote {plot_path}, {tpath}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick
    p.add_argument("--series-len", type=int, default=2000,
                   help="number of Mackey-Glass timesteps to generate")
    p.add_argument("--horizon", type=int, default=1,
                   help="forecast lead time in steps")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--units", type=int, default=100)
    p.add_argument("--sr", type=float, default=1.25,
                   help="spectral radius of the recurrent weight matrix")
    p.add_argument("--lr", type=float, default=0.3,
                   help="leak rate (temporal smoothing of reservoir state)")
    p.add_argument("--ridge", type=float, default=1e-5,
                   help="L2 regularization on the readout")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--normalize", action="store_true", default=False)
    args = p.parse_args()

    summary = run(args)
    if not args.quick and (summary["test_rmse"] > 1e-1
                           or summary["test_r2"] < 0.95):
        print(f"[warn] baseline test_rmse={summary['test_rmse']:.4f} "
              f"R^2={summary['test_r2']:.4f} -- expected RMSE ~1e-3 and "
              f"R^2 ~0.999. Check normalization, warmup, and horizon.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
