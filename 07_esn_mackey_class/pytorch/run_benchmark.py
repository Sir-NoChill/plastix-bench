"""Echo State Network on Mackey-Glass -- baseline smoke test.

Single-shot ESN fit using a fixed (non-trainable) random reservoir and a
ridge-regression readout.  Demonstrates the reservoir-computing thesis:
near-perfect one-step forecasting comes from training ONLY the linear
readout on top of the reservoir's spontaneous dynamics.

Outputs (under results/ and plots/, in the same
naming scheme as the other workloads):
    results/esn_mg_baseline[_tag].history.jsonl
    results/esn_mg_baseline[_tag].summary.csv
    results/esn_mg_baseline[_tag].plot.png
    plots/esn_mg_baseline[_tag].test.png

Usage:
    uv run python esn-mg/esn_baseline.py
    uv run python esn-mg/esn_baseline.py --quick
    uv run python esn-mg/esn_baseline.py --units 300 --sr 1.1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# common.py lives one dir up.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common/pytorch"))

from common import (  # noqa: E402
    StructuralLog,
    add_common_args,
    output_paths,
    plot_run,
    plot_test_curve,
    test_plot_path,
    write_summary_csv,
)


def build_esn(units: int, sr: float, lr: float, ridge: float, seed: int):
    """Build a `Reservoir >> Ridge` ESN.  Import is local so the rest of
    the suite doesn't require reservoirpy when this script isn't run."""
    from reservoirpy.nodes import Reservoir, Ridge
    reservoir = Reservoir(units=units, sr=sr, lr=lr, seed=seed)
    readout = Ridge(ridge=ridge)
    return reservoir >> readout


def fit_and_score(units: int, sr: float, lr: float, ridge: float,
                  warmup: int, x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, y_test: np.ndarray,
                  seed: int) -> tuple[float, float, float, dict]:
    """Returns (rmse, r2, wall_seconds, phase_ns).  RMSE may be inf if the
    reservoir state went non-finite (sr > 1 is allowed to misbehave).

    `phase_ns` carries `fit_ns` (the closed-form ridge fit; reservoirpy
    bundles reservoir-state collection + ridge solve internally so we report
    them as one number) and `inference_ns` (a held-out test forward pass)."""
    from reservoirpy.observables import rmse as _rmse, rsquare as _r2

    esn = build_esn(units=units, sr=sr, lr=lr, ridge=ridge, seed=seed)
    t0 = time.perf_counter()
    tf0 = time.perf_counter_ns()
    esn.fit(x_train, y_train, warmup=warmup)
    tf1 = time.perf_counter_ns()
    pred = esn.run(x_test)
    ti1 = time.perf_counter_ns()
    wall = time.perf_counter() - t0
    phase_ns = {"fit_ns": tf1 - tf0, "inference_ns": ti1 - tf1,
                "n_train_steps": len(x_train)}
    if not np.all(np.isfinite(pred)):
        return float("inf"), float("-inf"), wall, phase_ns
    return float(_rmse(y_test, pred)), float(_r2(y_test, pred)), wall, phase_ns


def run(args) -> dict:
    np.random.seed(args.seed)
    # Lazy import so we can fail-fast with a clean message.
    try:
        from reservoirpy.datasets import mackey_glass, to_forecasting
    except ImportError as e:
        sys.exit(f"[fatal] reservoirpy not installed ({e}); "
                 f"run `uv pip install reservoirpy`")

    n_timesteps = args.series_len // (4 if args.quick else 1)
    n_timesteps = max(500, n_timesteps)
    X = mackey_glass(n_timesteps=n_timesteps)

    # Optional rescale to roughly [-1, 1].  Mackey-Glass sits in ~[0.4, 1.4]
    # natively; zero-meaning makes `sr` and `ridge` interact more predictably
    # and matches the spec's recommended normalization.
    if args.normalize:
        X = (X - X.mean()) / (X.std() + 1e-8)

    x_train, x_test, y_train, y_test = to_forecasting(
        X, forecast=args.horizon, test_size=args.test_size,
    )
    print(f"[info] series_len={n_timesteps}  horizon={args.horizon}  "
          f"train={len(x_train)}  test={len(x_test)}  "
          f"normalize={args.normalize}")
    print(f"[info] units={args.units}  sr={args.sr}  lr={args.lr}  "
          f"ridge={args.ridge}  warmup={args.warmup}  seed={args.seed}")

    rmse_v, r2_v, wall, phase_ns = fit_and_score(
        units=args.units, sr=args.sr, lr=args.lr, ridge=args.ridge,
        warmup=args.warmup,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        seed=args.seed,
    )

    hist_path, summary_path, plot_path = output_paths(args, "esn_mg_baseline")
    log = StructuralLog(hist_path)
    n_units = args.units + 1            # reservoir + readout output unit
    n_edges = args.units * args.units   # the (dense, frozen) recurrent matrix
    log.log(0, n_units=n_units, n_edges=n_edges, edges=None,
            val_loss=rmse_v, test_rmse=rmse_v, test_r2=r2_v,
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
        "warmup": args.warmup,
        "horizon": args.horizon,
        "series_len": n_timesteps,
        "test_size": args.test_size,
        "normalize": int(args.normalize),
        "wall_seconds": round(wall, 3),
        "test_rmse": round(rmse_v, 8),
        "test_r2": round(r2_v, 8),
        "seed": args.seed,
        # Phase columns: "step" = one reservoir timestep over the training
        # window. The ridge fit is a single closed-form call; we amortise it
        # over n_train_steps to keep ns/step comparable with the iterative
        # benches.
        "step_count": int(phase_ns["n_train_steps"]),
        "step_ns_mean": round(wall * 1e9 / max(phase_ns["n_train_steps"], 1), 3),
        "forward_ns_mean": 0.0,
        "loss_ns_mean": 0.0,
        "backward_ns_mean": round(phase_ns["fit_ns"]
                                  / max(phase_ns["n_train_steps"], 1), 3),
        "update_ns_mean": 0.0,
        "structural_ns_mean": 0.0,
        "reset_ns_mean": 0.0,
        "other_ns_mean": round((wall * 1e9 - phase_ns["fit_ns"])
                               / max(phase_ns["n_train_steps"], 1), 3),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.2f}s  test_rmse={rmse_v:.6f}  "
          f"test_r2={r2_v:.6f}")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"ESN-MG baseline -- units={args.units} sr={args.sr} "
                       f"lr={args.lr}")
        print(f"[done] wrote {plot_path}")
        # Single-point test plot for parity with the rest of the suite.
        tpath = test_plot_path(args, "esn_mg_baseline")
        plot_test_curve(log.records, tpath,
                        title=f"ESN-MG baseline test RMSE -- sr={args.sr}",
                        metric_key="test_rmse", ylabel="test RMSE",
                        higher_is_better=False)
        print(f"[done] wrote {tpath}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--series-len", type=int, default=2000,
                   help="number of Mackey-Glass timesteps to generate")
    p.add_argument("--horizon", type=int, default=1,
                   help="forecast lead time in steps")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--units", type=int, default=100)
    p.add_argument("--sr", type=float, default=1.25,
                   help="spectral radius of the recurrent weight matrix")
    p.add_argument("--lr", type=float, default=0.3,
                   help="leak rate (temporal smoothing of reservoir state)")
    p.add_argument("--ridge", type=float, default=1e-5,
                   help="L2 regularization on the readout")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--normalize", action="store_true", default=False,
                   help="zero-mean / unit-var the series before splitting "
                        "(spec calls this optional; off by default so the "
                        "canonical baseline hits the ~1e-3 RMSE target)")
    args = p.parse_args()

    summary = run(args)
    # Sanity check from the spec: with these defaults, RMSE should be
    # order 1e-3 and R^2 near 0.999.  Warn if a non-quick run misses this
    # by more than a couple orders of magnitude.
    if not args.quick and (summary["test_rmse"] > 1e-1
                           or summary["test_r2"] < 0.95):
        print(f"[warn] baseline test_rmse={summary['test_rmse']:.4f} "
              f"R^2={summary['test_r2']:.4f} -- expected RMSE ~1e-3 and "
              f"R^2 ~0.999.  Check normalization, warmup, and horizon.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
