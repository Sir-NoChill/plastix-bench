"""Plot the plastix ESN's learning curve over epochs.

By default this invokes the C++ binary with --epochs (so the curve is
reproducible from a clean checkout) and reads the `learning.csv` it
writes.  Pass `--from-csv path/to/learning.csv` to skip the run and
plot an existing CSV instead.

Outputs:
    plots/esn_mg_learning_curve.png  (two stacked panels:
                                                 test RMSE log y,
                                                 test R^2 with a 0–1 axis)

Usage:
    uv run python esn-mg/plot_learning_curve.py
    uv run python esn-mg/plot_learning_curve.py --epochs 200
    uv run python esn-mg/plot_learning_curve.py \\
        --from-csv /tmp/esn_mg_compare_*/plastix_out/learning.csv
"""
from __future__ import annotations

import argparse
import csv
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


def run_plastix(plastix_bin: Path, out_dir: Path, epochs: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(plastix_bin), str(out_dir), str(epochs)]
    print(f"[plastix] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(f"[fatal] plastix binary exited {r.returncode}")
    return out_dir / "learning.csv"


def load_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not csv_path.exists():
        sys.exit(f"[fatal] learning csv not found: {csv_path}")
    epochs, rmses, r2s = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            rmses.append(float(row["test_rmse"]))
            r2s.append(float(row["test_r2"]))
    return np.array(epochs), np.array(rmses), np.array(r2s)


def plot_curve(epochs: np.ndarray, rmses: np.ndarray, r2s: np.ndarray,
               out_path: Path, ridge_rmse: float | None,
               ridge_r2: float | None, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter epoch 0 from the log-y plot (RMSE there is ~1 from the zero
    # readout — gives the curve a useful starting point but a 1-step drop
    # of two orders of magnitude crushes the rest of the curve visually).
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax = axes[0]
    ax.plot(epochs, rmses, marker="o", markersize=3, linewidth=1.2,
            color="tab:orange", label="plastix LMS")
    if ridge_rmse is not None and np.isfinite(ridge_rmse):
        ax.axhline(ridge_rmse, color="tab:blue", linestyle="--",
                   linewidth=1.0,
                   label=f"reservoirpy ridge = {ridge_rmse:.4g}")
    # Skip nonpositive entries (R²=0 baseline can produce equally-degenerate
    # RMSE in edge cases) so log scale doesn't blow up.
    if np.any(rmses > 0):
        ax.set_yscale("log")
    ax.set_ylabel("test RMSE (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)

    ax = axes[1]
    ax.plot(epochs, r2s, marker="o", markersize=3, linewidth=1.2,
            color="tab:orange", label="plastix LMS")
    if ridge_r2 is not None and np.isfinite(ridge_r2):
        ax.axhline(ridge_r2, color="tab:blue", linestyle="--",
                   linewidth=1.0,
                   label=f"reservoirpy ridge = {ridge_r2:.4f}")
    ax.set_ylim(min(-0.05, float(r2s.min()) - 0.05), 1.02)
    ax.set_xlabel("epoch (full pass over the training window)")
    ax.set_ylabel("test R²")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_ridge_baseline(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["framework"] == "reservoirpy_ridge":
                return float(row["test_rmse"]), float(row["test_r2"])
    return None, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plastix-bin", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--from-csv", type=Path, default=None,
                   help="skip the C++ run and read this learning.csv")
    p.add_argument("--out", type=Path,
                   default=Path("plots/esn_mg_learning_curve.png"))
    p.add_argument("--compare-csv", type=Path,
                   default=Path("results/esn_mg_compare.summary.csv"),
                   help="if present, overlays the reservoirpy ridge "
                        "baseline as a horizontal reference line")
    args = p.parse_args()

    if args.from_csv:
        csv_path = args.from_csv
    else:
        plastix_bin = find_plastix_bin(args.plastix_bin)
        tmp = Path(tempfile.mkdtemp(prefix="esn_mg_learning_"))
        csv_path = run_plastix(plastix_bin, tmp, args.epochs)

    epochs, rmses, r2s = load_curve(csv_path)
    ridge_rmse, ridge_r2 = load_ridge_baseline(args.compare_csv)

    title = (
        f"plastix ESN learning curve  (units=100, sr=1.25, leak=0.3, "
        f"epochs={int(epochs.max())})"
    )
    plot_curve(epochs, rmses, r2s, args.out, ridge_rmse, ridge_r2, title)
    print(f"[done] wrote {args.out}")

    # Summarize key milestones.
    print(f"[curve] epoch 0    : RMSE={rmses[0]:.5g}  R²={r2s[0]:+.5f}")
    if len(epochs) > 1:
        i1 = 1
        print(f"[curve] epoch {epochs[i1]:>4}: RMSE={rmses[i1]:.5g}  "
              f"R²={r2s[i1]:+.5f}")
        i_mid = len(epochs) // 2
        print(f"[curve] epoch {epochs[i_mid]:>4}: RMSE={rmses[i_mid]:.5g}  "
              f"R²={r2s[i_mid]:+.5f}")
        print(f"[curve] epoch {epochs[-1]:>4}: RMSE={rmses[-1]:.5g}  "
              f"R²={r2s[-1]:+.5f}")
    if ridge_rmse is not None:
        print(f"[ref  ] ridge      : RMSE={ridge_rmse:.5g}  R²={ridge_r2:+.5f}")


if __name__ == "__main__":
    main()
