"""Echo State Network on Mackey-Glass -- spectral-radius sweep.

Scans `sr` over a list of values, averages test RMSE over `--seeds`
random reservoir initializations per `sr`, and plots the mean +/- std
on a log y-axis.  The headline result is a U-shaped curve with its
minimum near `sr ~ 1` (the edge of chaos): too small a spectral radius
loses memory, too large violates the echo-state property.

Outputs (under the standard suite layout):
    results/esn_mg[_tag].history.jsonl      # one record per (sr, seed)
    results/esn_mg[_tag].summary.csv        # single headline row
    results/esn_mg[_tag].sweep.csv          # per-trial rmse/r2
    results/esn_mg[_tag].plot.png           # canonical 3-panel
    plots/esn_mg[_tag].test.png             # rmse-vs-trial-index
    plots/esn_mg[_tag].rmse_vs_sr.png       # the U-curve

Usage:
    uv run python esn-mg/sweep_sr.py
    uv run python esn-mg/sweep_sr.py --quick
    uv run python esn-mg/sweep_sr.py --seeds 5 --units 100
    uv run python esn-mg/sweep_sr.py --sr-values 0.5,1.0,1.5
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common/pytorch"))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    StructuralLog,
    add_common_args,
    output_paths,
    plot_run,
    plot_test_curve,
    test_plot_path,
    write_summary_csv,
)
from esn_baseline import fit_and_score  # noqa: E402


# Spec §6 suggests [0.5..1.5] (denser near 1.0); we extend to 2.5 so the
# right arm of the U-curve (where sr > 1 breaks the echo-state property
# and some seeds diverge) actually shows up at one-step horizon.
DEFAULT_SR_VALUES = (0.5, 0.7, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 1.8, 2.5)
QUICK_SR_VALUES = (0.7, 1.0, 1.5)


def parse_sr_values(s: str | None) -> tuple[float, ...] | None:
    if not s:
        return None
    return tuple(float(x) for x in s.split(",") if x.strip())


def plot_rmse_vs_sr(per_sr: dict[float, dict], out_path: Path,
                    title: str) -> None:
    """U-curve of mean test RMSE vs spectral radius, std as error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    srs = sorted(per_sr.keys())
    means = np.array([per_sr[s]["mean"] for s in srs])
    stds = np.array([per_sr[s]["std"] for s in srs])

    # Guard against NaN/Inf so the log scale doesn't blow up.
    safe_means = np.where(np.isfinite(means), means, np.nan)
    safe_stds = np.where(np.isfinite(stds), stds, 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(srs, safe_means, yerr=safe_stds,
                marker="o", linewidth=1.4, capsize=4, color="tab:purple",
                label="mean +/- std")
    if np.any(np.isfinite(safe_means)):
        best_idx = int(np.nanargmin(safe_means))
        ax.axvline(srs[best_idx], color="tab:gray", linestyle=":",
                   linewidth=0.8,
                   label=f"best sr={srs[best_idx]:.2f} "
                         f"(rmse={safe_means[best_idx]:.4g})")
    ax.set_xlabel("spectral radius (sr)")
    ax.set_ylabel("test RMSE (log)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_per_trial_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sr", "seed", "rmse", "r2", "wall_seconds"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})


def run(args) -> dict:
    np.random.seed(args.seed)
    try:
        from reservoirpy.datasets import mackey_glass, to_forecasting
    except ImportError as e:
        sys.exit(f"[fatal] reservoirpy not installed ({e}); "
                 f"run `uv pip install reservoirpy`")

    sr_values = parse_sr_values(args.sr_values)
    if sr_values is None:
        sr_values = QUICK_SR_VALUES if args.quick else DEFAULT_SR_VALUES
    seeds_per_sr = max(1, args.seeds // (2 if args.quick else 1))
    n_timesteps = args.series_len // (4 if args.quick else 1)
    n_timesteps = max(500, n_timesteps)

    X = mackey_glass(n_timesteps=n_timesteps)
    if args.normalize:
        X = (X - X.mean()) / (X.std() + 1e-8)
    x_train, x_test, y_train, y_test = to_forecasting(
        X, forecast=args.horizon, test_size=args.test_size,
    )

    print(f"[info] series_len={n_timesteps}  train={len(x_train)}  "
          f"test={len(x_test)}  normalize={args.normalize}")
    print(f"[info] sweeping sr={list(sr_values)}  seeds_per_sr={seeds_per_sr}  "
          f"units={args.units}  lr={args.lr}  ridge={args.ridge}")

    hist_path, summary_path, plot_path = output_paths(args, "esn_mg")
    log = StructuralLog(hist_path)
    per_trial: list[dict] = []

    n_units_log = args.units + 1
    n_edges_log = args.units * args.units

    t0 = time.perf_counter()
    trial = 0
    for sr in sr_values:
        for k in range(seeds_per_sr):
            seed = args.seed + 1000 * k + int(round(sr * 100))
            rmse_v, r2_v, wall = fit_and_score(
                units=args.units, sr=sr, lr=args.lr, ridge=args.ridge,
                warmup=args.warmup,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                seed=seed,
            )
            trial += 1
            per_trial.append({
                "sr": sr, "seed": seed,
                "rmse": rmse_v, "r2": r2_v,
                "wall_seconds": round(wall, 3),
            })
            # NaN/Inf-safe value for logging.
            logged_rmse = rmse_v if np.isfinite(rmse_v) else None
            log.log(trial, n_units=n_units_log, n_edges=n_edges_log,
                    edges=None, val_loss=logged_rmse,
                    test_rmse=logged_rmse, test_r2=r2_v if np.isfinite(r2_v) else None,
                    sr=sr, seed=seed, trial_wall=round(wall, 3))
            print(f"[trial {trial:>3d}/{len(sr_values) * seeds_per_sr}] "
                  f"sr={sr:.3f} seed={seed:<6d}  rmse={rmse_v:.4g}  "
                  f"r2={r2_v:.4g}  ({wall:.2f}s)")
    wall_total = time.perf_counter() - t0
    log.flush()

    # Aggregate per sr (treating NaN/Inf rmse as missing).
    per_sr: dict[float, dict] = {}
    for sr in sr_values:
        rmses = np.array([t["rmse"] for t in per_trial if t["sr"] == sr],
                         dtype=float)
        finite = rmses[np.isfinite(rmses)]
        n_finite = int(finite.size)
        per_sr[sr] = {
            "mean": float(finite.mean()) if n_finite else float("nan"),
            "std": float(finite.std()) if n_finite > 1 else 0.0,
            "n_finite": n_finite,
            "n_total": int(rmses.size),
        }

    # Identify best sr by mean RMSE (ignoring NaNs).
    finite_means = {s: per_sr[s]["mean"] for s in sr_values
                    if np.isfinite(per_sr[s]["mean"])}
    if finite_means:
        best_sr = min(finite_means, key=finite_means.get)
        best_rmse = finite_means[best_sr]
        best_std = per_sr[best_sr]["std"]
    else:
        best_sr = float("nan")
        best_rmse = float("nan")
        best_std = float("nan")

    sweep_csv_path = Path(str(summary_path).replace(".summary.csv",
                                                    ".sweep.csv"))
    write_per_trial_csv(per_trial, sweep_csv_path)

    summary = {
        "workload": "esn_mg",
        "dataset": "mackey-glass",
        "units": args.units,
        "lr": args.lr,
        "ridge": args.ridge,
        "warmup": args.warmup,
        "horizon": args.horizon,
        "series_len": n_timesteps,
        "normalize": int(args.normalize),
        "sr_values": ";".join(f"{s:g}" for s in sr_values),
        "seeds_per_sr": seeds_per_sr,
        "n_trials": len(per_trial),
        "wall_seconds": round(wall_total, 3),
        "best_sr": best_sr,
        "best_sr_rmse_mean": round(best_rmse, 8) if np.isfinite(best_rmse) else "",
        "best_sr_rmse_std": round(best_std, 8) if np.isfinite(best_std) else "",
        "trials_nonfinite": sum(1 for t in per_trial
                                 if not np.isfinite(t["rmse"])),
        "seed": args.seed,
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall_total:.1f}s  best_sr={best_sr}  "
          f"best_rmse_mean={best_rmse:.6g} +/- {best_std:.6g}  "
          f"nonfinite={summary['trials_nonfinite']}/{len(per_trial)}")
    print(f"[done] wrote {hist_path}, {summary_path}, {sweep_csv_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"ESN-MG sr-sweep -- units={args.units} "
                       f"seeds_per_sr={seeds_per_sr}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "esn_mg")
        plot_test_curve(log.records, tpath,
                        title=f"ESN-MG sweep RMSE per trial -- "
                              f"best sr={best_sr}",
                        metric_key="test_rmse", ylabel="test RMSE",
                        higher_is_better=False)
        print(f"[done] wrote {tpath}")
        ucurve_path = (Path(args.out_dir).parent / "plots"
                       / (Path(plot_path).stem.replace(".plot", "")
                          + ".rmse_vs_sr.png"))
        plot_rmse_vs_sr(per_sr, ucurve_path,
                        title=f"Edge of chaos -- ESN on Mackey-Glass "
                              f"(units={args.units}, seeds={seeds_per_sr})")
        print(f"[done] wrote {ucurve_path}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--series-len", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--units", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--ridge", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--normalize", action="store_true", default=False,
                   help="zero-mean / unit-var the series (off by default; "
                        "the U-shape is visible either way)")
    p.add_argument("--seeds", type=int, default=5,
                   help="random-reservoir seeds per sr value")
    p.add_argument("--sr-values", type=str, default=None,
                   help="comma-separated sr values; defaults to a denser grid "
                        "near 1.0 (or a tiny grid in --quick)")
    args = p.parse_args()

    summary = run(args)
    # Sanity-check the U-shape: best sr should sit roughly in [0.8, 1.3].
    # In --quick we sweep too few points to enforce this, so only warn on
    # full runs.
    if (not args.quick and isinstance(summary["best_sr"], float)
            and np.isfinite(summary["best_sr"])
            and not (0.8 <= summary["best_sr"] <= 1.5)):
        print(f"[warn] best_sr={summary['best_sr']} sits outside the typical "
              f"[0.8, 1.5] edge-of-chaos band -- inspect rmse_vs_sr.png and "
              f"consider more seeds, longer series, or a denser sr grid. "
              f"(One-step Mackey-Glass keeps the basin shallow on the right.)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
