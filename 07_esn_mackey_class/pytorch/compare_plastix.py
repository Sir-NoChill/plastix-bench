"""Compare the C++ plastix ESN against the reservoirpy ridge ESN.

Both models are run on the *same* Mackey-Glass series (generated once by
reservoirpy and written to a temp CSV that the plastix binary loads).
Produces two figures:

  plots/esn_mg_compare.predictions.png
      Truth + both predictions overlaid for the test window. Visualizes
      where the LMS readout (plastix) lags the closed-form ridge
      (reservoirpy) — typically the startup transient and high-curvature
      regions.

  plots/esn_mg_compare.metrics.png
      Bar chart of test RMSE and R^2 per framework.

Also writes a single-row summary CSV at
    results/esn_mg_compare.summary.csv

Usage:
    uv run python esn-mg/compare_plastix.py
    uv run python esn-mg/compare_plastix.py --epochs 200
    uv run python esn-mg/compare_plastix.py --plastix-bin path/to/esn_mg
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = REPO_ROOT / "build" / "examples" / "esn-mg" / "esn_mg"


def find_plastix_bin(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            sys.exit(f"[fatal] --plastix-bin not found: {p}")
        return p
    if DEFAULT_BIN.exists():
        return DEFAULT_BIN
    sys.exit(
        f"[fatal] plastix esn_mg binary not found at {DEFAULT_BIN}.\n"
        f"        build it first:\n"
        f"          cmake -S {REPO_ROOT} -B {REPO_ROOT}/build\n"
        f"          cmake --build {REPO_ROOT}/build --target esn_mg"
    )


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def run_reservoirpy(series: np.ndarray, units: int, sr: float, lr: float,
                    ridge: float, warmup: int, seed: int,
                    train_frac: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Train the reservoirpy ridge ESN on the same series (already
    normalized) and return (preds, truth, rmse, r2) on the test window.

    `train_frac` is held parallel with the plastix run so both models see
    identical train/test splits.
    """
    from reservoirpy.nodes import Reservoir, Ridge
    from reservoirpy.observables import rmse as _rmse, rsquare as _r2

    n = len(series)
    n_train = int(n * train_frac)
    x = series.reshape(-1, 1).astype(np.float64)
    x_train = x[:n_train]
    y_train = x[1:n_train + 1]
    x_test = x[n_train:-1]
    y_test = x[n_train + 1:]
    # `y_test` aligns to `x_test`: x_test[i] is u(t), y_test[i] is u(t+1).
    reservoir = Reservoir(units=units, sr=sr, lr=lr, seed=seed)
    readout = Ridge(ridge=ridge)
    esn = reservoir >> readout
    esn.fit(x_train, y_train, warmup=warmup)
    pred = esn.run(x_test)
    return (
        pred.squeeze(axis=-1),
        y_test.squeeze(axis=-1),
        float(_rmse(y_test, pred)),
        float(_r2(y_test, pred)),
    )


def run_plastix(plastix_bin: Path, series_csv: Path, out_dir: Path,
                epochs: int) -> dict:
    """Invoke the plastix binary on the given series CSV and load the
    learning / predictions / summary CSVs it writes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(plastix_bin), str(out_dir), str(epochs), str(series_csv)]
    print(f"[plastix] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(f"[fatal] plastix binary exited {r.returncode}")
    pred_rows = list(csv.DictReader((out_dir / "predictions.csv").open()))
    sum_row = next(csv.DictReader((out_dir / "summary.csv").open()))
    return {
        "preds": np.array([float(r["pred"]) for r in pred_rows]),
        "truth": np.array([float(r["truth"]) for r in pred_rows]),
        "rmse": float(sum_row["test_rmse"]),
        "r2": float(sum_row["test_r2"]),
    }


def plot_predictions(out_path: Path, plastix: dict,
                     rpy_pred: np.ndarray, rpy_truth: np.ndarray,
                     plastix_metrics: tuple[float, float],
                     rpy_metrics: tuple[float, float],
                     n_show: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n_show, len(plastix["truth"]), len(rpy_truth))
    truth = plastix["truth"][:n]
    rpy_pred = rpy_pred[:n]
    plastix_pred = plastix["preds"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    ax.plot(truth, color="black", linewidth=1.2, label="truth")
    ax.plot(rpy_pred, color="tab:blue", linewidth=1.0,
            label=f"reservoirpy ridge  RMSE={rpy_metrics[0]:.4f}  "
                  f"R²={rpy_metrics[1]:.4f}")
    ax.plot(plastix_pred, color="tab:orange", linewidth=1.0,
            label=f"plastix LMS      RMSE={plastix_metrics[0]:.4f}  "
                  f"R²={plastix_metrics[1]:.4f}")
    ax.set_ylabel("normalized series")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("ESN forecasts on Mackey-Glass test window")

    ax2 = axes[1]
    ax2.plot(rpy_pred - truth, color="tab:blue", linewidth=0.9,
             label="reservoirpy error")
    ax2.plot(plastix_pred - truth, color="tab:orange", linewidth=0.9,
             label="plastix error")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("test step")
    ax2.set_ylabel("pred − truth")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_metrics(out_path: Path, plastix_metrics: tuple[float, float],
                 rpy_metrics: tuple[float, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["reservoirpy\n(ridge)", "plastix\n(LMS)"]
    colors = ["tab:blue", "tab:orange"]
    rmses = [rpy_metrics[0], plastix_metrics[0]]
    r2s = [rpy_metrics[1], plastix_metrics[1]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    bars = ax.bar(labels, rmses, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("test RMSE (log)")
    ax.set_title("Test RMSE")
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(labels, r2s, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test R²")
    ax.set_title("Test R²")
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("plastix LMS readout vs reservoirpy ridge readout")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plastix-bin", type=str, default=None)
    p.add_argument("--series-len", type=int, default=2000)
    p.add_argument("--units", type=int, default=100,
                   help="reservoir size (must match the C++ default)")
    p.add_argument("--sr", type=float, default=1.25)
    p.add_argument("--lr", type=float, default=0.3,
                   help="leak rate")
    p.add_argument("--ridge", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=100,
                   help="plastix LMS epochs over the training window")
    p.add_argument("--n-show", type=int, default=400)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results"))
    p.add_argument("--plots-dir", type=Path,
                   default=Path("plots"))
    p.add_argument("--keep-tmp", action="store_true",
                   help="leave the temp series + CSVs on disk for inspection")
    args = p.parse_args()

    plastix_bin = find_plastix_bin(args.plastix_bin)

    # 1) Generate one Mackey-Glass series via reservoirpy so the comparison
    #    isn't biased by integrator differences.  Normalize (zero-mean,
    #    unit-var) here so what we hand to plastix matches what plastix's
    #    NormalizeSeries branch would do internally.
    try:
        from reservoirpy.datasets import mackey_glass
    except ImportError as e:
        sys.exit(f"[fatal] reservoirpy not installed ({e}); "
                 f"run `uv pip install reservoirpy`")
    raw = mackey_glass(n_timesteps=args.series_len).squeeze(axis=-1).astype(np.float64)
    series = standardize(raw)

    # 2) Dump series to a temp CSV plastix can read.
    tmp = Path(tempfile.mkdtemp(prefix="esn_mg_compare_"))
    series_csv = tmp / "series.csv"
    with series_csv.open("w") as f:
        for v in series:
            f.write(f"{v}\n")

    # 3) Run plastix on it.
    plastix_out = tmp / "plastix_out"
    plastix = run_plastix(plastix_bin, series_csv, plastix_out, args.epochs)

    # 4) Run reservoirpy on the SAME (already-normalized) series.
    rpy_pred, rpy_truth, rpy_rmse, rpy_r2 = run_reservoirpy(
        series=series.astype(np.float32),
        units=args.units, sr=args.sr, lr=args.lr, ridge=args.ridge,
        warmup=args.warmup, seed=args.seed, train_frac=args.train_frac,
    )

    # 5) Sanity check: both runs should produce the same test length.
    if len(rpy_pred) != len(plastix["preds"]):
        print(f"[warn] length mismatch: reservoirpy={len(rpy_pred)} "
              f"plastix={len(plastix['preds'])}.  Plotting the shorter "
              f"prefix.", file=sys.stderr)
    if not np.allclose(rpy_truth[: len(plastix["truth"])],
                        plastix["truth"][: len(rpy_truth)], atol=1e-5):
        print("[warn] truth values differ across runs -- check "
              "normalization order", file=sys.stderr)

    # 6) Persist a single-row comparison summary.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "esn_mg_compare.summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "framework", "units", "sr", "lr", "ridge_or_lmslr",
            "warmup", "epochs", "test_rmse", "test_r2",
        ])
        w.writerow([
            "reservoirpy_ridge", args.units, args.sr, args.lr, args.ridge,
            args.warmup, 1, f"{rpy_rmse:.6g}", f"{rpy_r2:.6g}",
        ])
        w.writerow([
            "plastix_lms", args.units, args.sr, args.lr, "(from binary)",
            args.warmup, args.epochs,
            f"{plastix['rmse']:.6g}", f"{plastix['r2']:.6g}",
        ])
    print(f"[done] wrote {summary_path}")

    # 7) Plots.
    pred_png = args.plots_dir / "esn_mg_compare.predictions.png"
    plot_predictions(pred_png, plastix, rpy_pred, rpy_truth,
                     (plastix["rmse"], plastix["r2"]),
                     (rpy_rmse, rpy_r2),
                     n_show=args.n_show)
    print(f"[done] wrote {pred_png}")

    metrics_png = args.plots_dir / "esn_mg_compare.metrics.png"
    plot_metrics(metrics_png, (plastix["rmse"], plastix["r2"]),
                 (rpy_rmse, rpy_r2))
    print(f"[done] wrote {metrics_png}")

    if args.keep_tmp:
        print(f"[done] tmp dir retained at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
