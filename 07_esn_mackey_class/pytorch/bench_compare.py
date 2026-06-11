"""Wall-clock benchmark: plastix LMS ESN vs reservoirpy ridge ESN.

Both implementations are run on the same Mackey-Glass series (generated
once by reservoirpy, written to a temp CSV the C++ binary reads) with
identical hyperparameters where they overlap (units, sr, leak, warmup,
train/test split).

What's measured:

  - reservoirpy (ridge): time to collect reservoir states over the
    training window, fit ridge regression, run reservoir+readout on the
    test window.

  - plastix (LMS): training-phase totals reported by the C++ binary in
    timing.csv / summary.csv (forward, loss, update, per-epoch eval) and
    inference window time, swept across multiple epoch counts so we can
    walk the wall-clock-vs-accuracy frontier.

What's plotted:

  - Pareto frontier: total wall time vs final test RMSE (log-log). A
    single point for ridge; one point per plastix epoch count.

  - Phase breakdown stacks: where wall time actually goes inside each
    implementation.

  - Per-step microbenchmark: ns per forward step in each framework, the
    most apples-to-apples cost comparison once you ignore algorithm
    choice (online SGD vs batch closed-form).

Outputs (under the standard suite layout):

    results/esn_mg_bench.summary.csv     # one row per config
    results/esn_mg_bench.plastix.csv     # per-epoch detail
    plots/esn_mg_bench.pareto.png        # time vs RMSE
    plots/esn_mg_bench.phases.png        # stacked phase bars
    plots/esn_mg_bench.perstep.png       # ns/step bars

Usage:
    uv run python esn-mg/bench_compare.py
    uv run python esn-mg/bench_compare.py --epochs 1,5,25,100
    uv run python esn-mg/bench_compare.py --repeats 3
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = REPO_ROOT / "build" / "examples" / "esn-mg" / "esn_mg"
DEFAULT_RAW_BIN = REPO_ROOT / "build" / "traditional-raw" / "06_esn_mackey_glass"
DEFAULT_EPOCHS = (1, 5, 10, 25, 50, 100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        f"        cmake -S {REPO_ROOT} -B {REPO_ROOT}/build && \\\n"
        f"        cmake --build {REPO_ROOT}/build --target esn_mg"
    )


def find_raw_bin(explicit: str | None) -> Path | None:
    """The raw-C++/OpenBLAS baseline is optional — skip silently if it's
    not built so this script still works on machines without OpenBLAS."""
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            sys.exit(f"[fatal] --raw-bin not found: {p}")
        return p
    return DEFAULT_RAW_BIN if DEFAULT_RAW_BIN.exists() else None


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def write_series_csv(series: np.ndarray, path: Path) -> None:
    with path.open("w") as f:
        for v in series:
            f.write(f"{v}\n")


# ---------------------------------------------------------------------------
# reservoirpy bench
# ---------------------------------------------------------------------------

def bench_reservoirpy(series: np.ndarray, units: int, sr: float, lr: float,
                      ridge: float, warmup: int, seed: int,
                      train_frac: float, repeats: int) -> dict:
    """Decomposes the reservoirpy pipeline into:
      - reservoir_train_s : run reservoir over training inputs to collect
                            hidden states. Dominated by the dense
                            ``W_rec @ x`` matvec, hit T times. The single
                            biggest cost in the python implementation —
                            confirmed by the measured timings.
      - ridge_fit_s       : closed-form ridge solve on
                            (states[warmup:], y_train[warmup:]).  O(N^3)
                            in reservoir size, but at N=100 it's pure
                            BLAS and negligible next to the reservoir
                            forward pass.
      - reservoir_test_s  : run reservoir over the test window.
      - readout_run_s     : matrix-multiply the readout over test states.

    Times are averaged across ``repeats`` trials.  Each trial rebuilds the
    reservoir+readout from scratch.

    ==== BOTTLENECK (reservoirpy side) ===================================
    Measured: reservoir_train_s ≈ 19 ms over 1500 training steps =
    ~12.8 µs/step.  Of that, the inner loop is a single ``x @ W.T + u @
    W_in.T`` numpy matmul per step (reservoirpy/nodes/reservoirs/base.py),
    which lands as one BLAS gemv against a contiguous 100×100 matrix.
    Almost everything else (tanh + leak combine, state caching, the
    python-level for-loop overhead in reservoirpy.Node._call) is dwarfed
    by that one BLAS call.

    The ridge_fit_s phase is ~2 ms — a single ``scipy.linalg.solve`` on a
    100×100 normal-equation system.  Increasing units to 500 would push
    ridge_fit_s up roughly 125× (cubic) while reservoir_train_s would
    only grow ~25× (quadratic in the matvec), so for larger reservoirs
    the cost picture flips and the ridge solve becomes meaningful.
    ======================================================================
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

    results = {
        "reservoir_train_s": [],
        "ridge_fit_s": [],
        "reservoir_test_s": [],
        "readout_run_s": [],
        "rmse": [],
        "r2": [],
    }
    for _ in range(repeats):
        reservoir = Reservoir(units=units, sr=sr, lr=lr, seed=seed)
        readout = Ridge(ridge=ridge)

        # Fresh nodes — no reset needed; `.run` initializes lazily.
        t0 = time.perf_counter()
        states_train = reservoir.run(x_train)
        t1 = time.perf_counter()
        readout.fit(states_train[warmup:], y_train[warmup:])
        t2 = time.perf_counter()
        states_test = reservoir.run(x_test)
        t3 = time.perf_counter()
        pred = readout.run(states_test)
        t4 = time.perf_counter()

        results["reservoir_train_s"].append(t1 - t0)
        results["ridge_fit_s"].append(t2 - t1)
        results["reservoir_test_s"].append(t3 - t2)
        results["readout_run_s"].append(t4 - t3)
        results["rmse"].append(float(_rmse(y_test, pred)))
        results["r2"].append(float(_r2(y_test, pred)))

    agg = {k: float(np.mean(v)) for k, v in results.items()}
    agg["rmse_std"] = float(np.std(results["rmse"]))
    agg["fit_s"] = agg["reservoir_train_s"] + agg["ridge_fit_s"]
    agg["inference_s"] = agg["reservoir_test_s"] + agg["readout_run_s"]
    agg["total_s"] = agg["fit_s"] + agg["inference_s"]
    agg["n_train"] = n_train - warmup
    agg["n_test"] = len(x_test)
    return agg


# ---------------------------------------------------------------------------
# plastix bench
# ---------------------------------------------------------------------------

def run_plastix_once(plastix_bin: Path, series_csv: Path, out_dir: Path,
                     epochs: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(plastix_bin), str(out_dir), str(epochs), str(series_csv)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # The example binary returns exit-code 1 when R² < 0.95 (its FAIL
    # threshold), which trips at low epoch counts.  For benchmarking,
    # tolerate that as long as the summary.csv landed; only a hard crash
    # (missing CSV) is fatal.
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(f"[fatal] plastix binary exited {r.returncode} and wrote "
                 f"no summary at {summary_path}")
    summary = next(csv.DictReader(summary_path.open()))
    return {
        "epochs": int(summary["epochs"]),
        "n_train_steps": int(summary["n_train_steps"]),
        "rmse": float(summary["test_rmse"]),
        "r2": float(summary["test_r2"]),
        "train_wall_s": int(summary["train_wall_ns"]) / 1e9,
        "forward_s": int(summary["forward_ns"]) / 1e9,
        "loss_s": int(summary["loss_ns"]) / 1e9,
        "update_s": int(summary["update_ns"]) / 1e9,
        "eval_s": int(summary["eval_ns"]) / 1e9,
        "inference_s": int(summary["inference_ns"]) / 1e9,
    }


def run_raw_once(raw_bin: Path, series_csv: Path, out_dir: Path) -> dict:
    """Run the raw-C++/OpenBLAS ESN once.  argv[2] is ignored by that
    binary (one closed-form ridge solve regardless), but we pass "1" for
    CLI parity with the plastix binary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(raw_bin), str(out_dir), "1", str(series_csv)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(f"[fatal] raw-C++ binary exited {r.returncode} and wrote "
                 f"no summary at {summary_path}")
    summary = next(csv.DictReader(summary_path.open()))
    return {
        "epochs": 1,
        "n_train_steps": int(summary["n_train_steps"]),
        "rmse": float(summary["test_rmse"]),
        "r2": float(summary["test_r2"]),
        "train_wall_s": int(summary["train_wall_ns"]) / 1e9,
        "forward_s": int(summary["forward_ns"]) / 1e9,
        "loss_s": int(summary["loss_ns"]) / 1e9,
        # raw-C++ writes the ridge fit time into update_ns (closed-form
        # analogue of LMS update), matching the plastix schema.
        "update_s": int(summary["update_ns"]) / 1e9,
        "eval_s": int(summary["eval_ns"]) / 1e9,
        "inference_s": int(summary["inference_ns"]) / 1e9,
    }


def bench_raw(raw_bin: Path, series_csv: Path, repeats: int) -> dict | None:
    if raw_bin is None:
        return None
    trials = []
    for _ in range(repeats):
        tmp = Path(tempfile.mkdtemp(prefix="esn_bench_raw_"))
        trials.append(run_raw_once(raw_bin, series_csv, tmp))
        shutil.rmtree(tmp, ignore_errors=True)
    agg = {k: float(np.mean([t[k] for t in trials]))
           for k in trials[0]
           if isinstance(trials[0][k], (int, float))}
    agg["epochs"] = 1
    agg["rmse_std"] = float(np.std([t["rmse"] for t in trials]))
    print(f"[raw-cpp ] wall={agg['train_wall_s']:.4f}s "
          f"fwd={agg['forward_s']:.4f}s ridge={agg['update_s']:.4f}s "
          f"inf={agg['inference_s']:.4f}s "
          f"RMSE={agg['rmse']:.4f} R²={agg['r2']:.4f}")
    return agg


    summary = next(csv.DictReader(summary_path.open()))
    # ns → s for arithmetic; keep ns column too for later sanity.
    return {
        "epochs": int(summary["epochs"]),
        "n_train_steps": int(summary["n_train_steps"]),
        "rmse": float(summary["test_rmse"]),
        "r2": float(summary["test_r2"]),
        "train_wall_s": int(summary["train_wall_ns"]) / 1e9,
        "forward_s": int(summary["forward_ns"]) / 1e9,
        "loss_s": int(summary["loss_ns"]) / 1e9,
        "update_s": int(summary["update_ns"]) / 1e9,
        "eval_s": int(summary["eval_ns"]) / 1e9,
        "inference_s": int(summary["inference_ns"]) / 1e9,
    }


def bench_plastix(plastix_bin: Path, series_csv: Path, epochs_list: list[int],
                  repeats: int) -> list[dict]:
    """Sweeps plastix at a list of epoch counts; averages each over `repeats`.
    Returns one dict per epoch count."""
    out_rows = []
    for ep in epochs_list:
        trials = []
        for _ in range(repeats):
            tmp = Path(tempfile.mkdtemp(prefix=f"esn_bench_p{ep}_"))
            trials.append(run_plastix_once(plastix_bin, series_csv, tmp, ep))
            shutil.rmtree(tmp, ignore_errors=True)
        agg = {k: float(np.mean([t[k] for t in trials]))
               for k in trials[0]
               if isinstance(trials[0][k], (int, float))}
        agg["epochs"] = ep
        agg["rmse_std"] = float(np.std([t["rmse"] for t in trials]))
        out_rows.append(agg)
        print(f"[plastix ep={ep:>3}] wall={agg['train_wall_s']:.3f}s "
              f"fwd={agg['forward_s']:.3f}s upd={agg['update_s']:.3f}s "
              f"eval={agg['eval_s']:.3f}s inf={agg['inference_s']:.3f}s "
              f"RMSE={agg['rmse']:.4f} R²={agg['r2']:.4f}")
    return out_rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pareto(out_path: Path, rpy: dict, plastix_rows: list[dict],
                raw: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    # plastix: connected line over epoch counts.
    p_times = [r["train_wall_s"] + r["inference_s"] for r in plastix_rows]
    p_rmse = [r["rmse"] for r in plastix_rows]
    ax.plot(p_times, p_rmse, marker="o", color="tab:orange", linewidth=1.4,
            label="plastix LMS")
    for r in plastix_rows:
        ax.annotate(f"{r['epochs']}ep",
                    xy=(r["train_wall_s"] + r["inference_s"], r["rmse"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8,
                    color="tab:orange")

    # reservoirpy: single point.
    rpy_time = rpy["total_s"]
    ax.scatter([rpy_time], [rpy["rmse"]], color="tab:blue", s=80,
               marker="D", label=f"reservoirpy ridge")
    ax.annotate("1 closed-form solve",
                xy=(rpy_time, rpy["rmse"]),
                xytext=(8, -12), textcoords="offset points", fontsize=8,
                color="tab:blue")

    # raw-C++/OpenBLAS ceiling reference point.
    if raw is not None:
        raw_time = raw["train_wall_s"] + raw["inference_s"]
        ax.scatter([raw_time], [raw["rmse"]], color="tab:green", s=80,
                   marker="s", label=f"raw-C++/OpenBLAS ridge")
        ax.annotate("BLAS ceiling",
                    xy=(raw_time, raw["rmse"]),
                    xytext=(8, -12), textcoords="offset points", fontsize=8,
                    color="tab:green")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total wall-clock time (s, log)")
    ax.set_ylabel("test RMSE (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("ESN Mackey-Glass — wall-clock vs accuracy Pareto frontier")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_phases(out_path: Path, rpy: dict,
                plastix_short: dict, plastix_long: dict,
                raw: dict | None) -> None:
    """Stacked-bar breakdown of where wall time goes.

    Bars side by side (rightmost two are plastix's per-effort points;
    leftmost two are the python and raw-C++ closed-form references):
      - reservoirpy ridge:  reservoir_train + ridge_fit + reservoir_test
                            + readout_run
      - raw-C++/OpenBLAS:   forward + ridge_fit + inference
      - plastix small epoch: forward + loss + update + eval + inference
      - plastix long  epoch: same decomposition

    The contrast between raw-C++ and reservoirpy isolates the per-step
    Python-interpreter / dispatch overhead.  The contrast between either
    of those and plastix isolates the SOA-per-connection penalty.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    stacks = []

    # reservoirpy stacks
    labels.append("reservoirpy\nridge")
    stacks.append([
        ("reservoir (train)", rpy["reservoir_train_s"], "#1f77b4"),
        ("ridge solve",       rpy["ridge_fit_s"],       "#aec7e8"),
        ("reservoir (test)",  rpy["reservoir_test_s"],  "#76b7e8"),
        ("readout (test)",    rpy["readout_run_s"],     "#cce5ff"),
    ])

    # raw-C++ stack (forward, ridge, inference — no eval, no loss).
    if raw is not None:
        labels.append("raw-C++\n(OpenBLAS)")
        stacks.append([
            ("forward (train)", raw["forward_s"],   "#2ca02c"),
            ("ridge solve",     raw["update_s"],    "#98df8a"),
            ("inference",       raw["inference_s"], "#c7e9c0"),
        ])

    # plastix stacks (forward+loss+update sit per-step inside each epoch;
    # eval is the per-epoch test snapshot; inference is the post-training
    # test pass).
    def plastix_stack(p):
        return [
            ("forward",         p["forward_s"],   "#d62728"),
            ("update (LMS)",    p["update_s"],    "#ff9896"),
            ("loss",            p["loss_s"],      "#ffbb78"),
            ("per-epoch eval",  p["eval_s"],      "#ff7f0e"),
            ("inference",       p["inference_s"], "#fdd0a2"),
        ]

    labels.append(f"plastix\n{plastix_short['epochs']}ep")
    stacks.append(plastix_stack(plastix_short))
    labels.append(f"plastix\n{plastix_long['epochs']}ep")
    stacks.append(plastix_stack(plastix_long))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.55
    seen = set()
    for bar_idx, stack in enumerate(stacks):
        b = 0.0
        for name, val, color in stack:
            label = name if name not in seen else None
            seen.add(name)
            ax.bar(labels[bar_idx], val, width=width, bottom=b, color=color,
                   label=label)
            b += val
        ax.text(bar_idx, b * 1.02, f"{b:.3g}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock time (s, log)")
    ax.set_title("Wall-clock decomposition by phase")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_perstep(out_path: Path, rpy: dict, plastix_long: dict,
                 raw: dict | None) -> None:
    """Normalize the forward-pass cost to nanoseconds per timestep so the
    implementations are directly comparable, ignoring algorithm choice.
    raw-C++ is the BLAS ceiling; reservoirpy is BLAS + Python interpreter;
    plastix is the per-connection SOA loop.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_p = max(1, plastix_long["n_train_steps"])
    plastix_fwd_ns = plastix_long["forward_s"] / n_p * 1e9
    plastix_upd_ns = plastix_long["update_s"] / n_p * 1e9
    plastix_inf_ns = (plastix_long["inference_s"] /
                      max(1, rpy["n_test"])) * 1e9
    rpy_fwd_ns = rpy["reservoir_train_s"] / max(1, rpy["n_train"]) * 1e9
    rpy_inf_ns = rpy["reservoir_test_s"] / max(1, rpy["n_test"]) * 1e9
    if raw is not None:
        raw_fwd_ns = raw["forward_s"] / max(1, raw["n_train_steps"]) * 1e9
        raw_inf_ns = (raw["inference_s"] /
                       max(1, rpy["n_test"])) * 1e9

    labels = ["forward\n(train)", "update\n(train)", "forward\n(inference)"]
    n_groups = 3 if raw is not None else 2
    x = np.arange(len(labels))
    w = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(9, 4.8))

    def maybe_label(i, val, xpos):
        if val > 0:
            ax.text(xpos, val, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8)

    offset = -w * (n_groups - 1) / 2
    if raw is not None:
        raw_vals = [raw_fwd_ns, 0.0, raw_inf_ns]
        xs = x + offset
        ax.bar(xs, raw_vals, width=w, color="tab:green",
               label="raw-C++ (OpenBLAS)")
        for i, v in enumerate(raw_vals):
            maybe_label(i, v, xs[i])
        offset += w

    rpy_vals = [rpy_fwd_ns, 0.0, rpy_inf_ns]
    xs = x + offset
    ax.bar(xs, rpy_vals, width=w, color="tab:blue",
           label="reservoirpy (numpy/BLAS)")
    for i, v in enumerate(rpy_vals):
        maybe_label(i, v, xs[i])
    offset += w

    plastix_vals = [plastix_fwd_ns, plastix_upd_ns, plastix_inf_ns]
    xs = x + offset
    ax.bar(xs, plastix_vals, width=w, color="tab:orange",
           label="plastix (SOA loop)")
    for i, v in enumerate(plastix_vals):
        maybe_label(i, v, xs[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("nanoseconds per timestep")
    ax.set_yscale("log")
    ax.set_title("Per-step cost (lower = faster)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plastix-bin", type=str, default=None)
    p.add_argument("--raw-bin", type=str, default=None,
                   help="raw-C++/OpenBLAS ESN binary (default: probe "
                        "build/traditional-raw/06_esn_mackey_glass)")
    p.add_argument("--no-raw", action="store_true",
                   help="skip the raw-C++ comparison even if the binary "
                        "is available")
    p.add_argument("--series-len", type=int, default=2000)
    p.add_argument("--units", type=int, default=100)
    p.add_argument("--sr", type=float, default=1.25)
    p.add_argument("--lr", type=float, default=0.3, help="leak rate")
    p.add_argument("--ridge", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--epochs", type=str,
                   default=",".join(str(e) for e in DEFAULT_EPOCHS),
                   help="comma-separated plastix epoch counts to sweep")
    p.add_argument("--repeats", type=int, default=2,
                   help="trials per config; results are averaged")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results"))
    p.add_argument("--plots-dir", type=Path,
                   default=Path("plots"))
    args = p.parse_args()

    epoch_counts = sorted({int(s) for s in args.epochs.split(",") if s.strip()})
    plastix_bin = find_plastix_bin(args.plastix_bin)
    raw_bin = None if args.no_raw else find_raw_bin(args.raw_bin)
    if raw_bin is None and not args.no_raw:
        print(f"[setup] raw-C++ binary not at {DEFAULT_RAW_BIN}; build with "
              f"`cmake --build {REPO_ROOT}/build --target "
              f"traditional_raw_06_esn_mackey_glass` to include it",
              file=sys.stderr)

    # 1) Build one normalized Mackey-Glass series both implementations share.
    try:
        from reservoirpy.datasets import mackey_glass
    except ImportError as e:
        sys.exit(f"[fatal] reservoirpy not installed ({e}); "
                 f"run `uv pip install reservoirpy`")
    raw = mackey_glass(n_timesteps=args.series_len).squeeze(axis=-1).astype(np.float64)
    series = standardize(raw)

    tmp = Path(tempfile.mkdtemp(prefix="esn_bench_"))
    series_csv = tmp / "series.csv"
    write_series_csv(series, series_csv)

    print(f"[setup] series_len={len(series)}  units={args.units}  "
          f"sr={args.sr}  leak={args.lr}  warmup={args.warmup}  "
          f"epoch_sweep={epoch_counts}  repeats={args.repeats}\n")

    # 2) reservoirpy bench.
    rpy = bench_reservoirpy(
        series=series, units=args.units, sr=args.sr, lr=args.lr,
        ridge=args.ridge, warmup=args.warmup, seed=args.seed,
        train_frac=args.train_frac, repeats=args.repeats,
    )
    print(f"[rpy] fit={rpy['fit_s']:.3f}s  "
          f"(reservoir={rpy['reservoir_train_s']:.3f}s + "
          f"ridge={rpy['ridge_fit_s']:.3f}s)  "
          f"inf={rpy['inference_s']:.3f}s  "
          f"RMSE={rpy['rmse']:.4f} R²={rpy['r2']:.4f}\n")

    # 2b) raw-C++/OpenBLAS bench (optional — the BLAS-ceiling reference).
    raw = bench_raw(raw_bin, series_csv, args.repeats) if raw_bin else None

    # 3) plastix sweep.
    plastix_rows = bench_plastix(plastix_bin, series_csv, epoch_counts,
                                  args.repeats)

    shutil.rmtree(tmp, ignore_errors=True)

    # 4) Persist summary CSV (one row per config, comparable columns).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "esn_mg_bench.summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "framework", "epochs", "n_train_steps",
            "train_wall_s", "forward_s", "update_or_solve_s",
            "loss_s", "eval_s", "inference_s",
            "total_s", "test_rmse", "test_r2",
        ])
        w.writerow([
            "reservoirpy_ridge", 1, rpy["n_train"],
            rpy["fit_s"], rpy["reservoir_train_s"], rpy["ridge_fit_s"],
            0.0, 0.0, rpy["inference_s"],
            rpy["total_s"], rpy["rmse"], rpy["r2"],
        ])
        if raw is not None:
            total = raw["train_wall_s"] + raw["inference_s"]
            w.writerow([
                "raw_cpp_blas", 1, raw["n_train_steps"],
                raw["train_wall_s"], raw["forward_s"], raw["update_s"],
                raw["loss_s"], raw["eval_s"], raw["inference_s"],
                total, raw["rmse"], raw["r2"],
            ])
        for r in plastix_rows:
            total = r["train_wall_s"] + r["inference_s"]
            w.writerow([
                "plastix_lms", r["epochs"], r["n_train_steps"],
                r["train_wall_s"], r["forward_s"], r["update_s"],
                r["loss_s"], r["eval_s"], r["inference_s"],
                total, r["rmse"], r["r2"],
            ])
    print(f"\n[done] wrote {summary_path}")

    plastix_csv = args.out_dir / "esn_mg_bench.plastix.csv"
    with plastix_csv.open("w", newline="") as f:
        cols = ["epochs", "n_train_steps", "train_wall_s", "forward_s",
                "loss_s", "update_s", "eval_s", "inference_s",
                "rmse", "rmse_std", "r2"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in plastix_rows:
            w.writerow({k: r[k] for k in cols if k in r})

    # 5) Plots.
    pareto_path = args.plots_dir / "esn_mg_bench.pareto.png"
    plot_pareto(pareto_path, rpy, plastix_rows, raw)
    print(f"[done] wrote {pareto_path}")

    phases_path = args.plots_dir / "esn_mg_bench.phases.png"
    plot_phases(phases_path, rpy, plastix_rows[0], plastix_rows[-1], raw)
    print(f"[done] wrote {phases_path}")

    perstep_path = args.plots_dir / "esn_mg_bench.perstep.png"
    plot_perstep(perstep_path, rpy, plastix_rows[-1], raw)
    print(f"[done] wrote {perstep_path}")

    # 6) Headline takeaways.
    p_long = plastix_rows[-1]
    p_short = plastix_rows[0]
    print(f"\n[summary] reservoirpy ridge: total {rpy['total_s']:.4f}s, "
          f"RMSE {rpy['rmse']:.4g}")
    if raw is not None:
        raw_total = raw["train_wall_s"] + raw["inference_s"]
        print(f"[summary] raw-C++ BLAS  : total {raw_total:.4f}s, "
              f"RMSE {raw['rmse']:.4g}  "
              f"({rpy['total_s'] / max(1e-6, raw_total):.2f}× faster than rpy)")
    print(f"[summary] plastix {p_short['epochs']} ep: total "
          f"{p_short['train_wall_s'] + p_short['inference_s']:.3f}s, "
          f"RMSE {p_short['rmse']:.4g}")
    print(f"[summary] plastix {p_long['epochs']} ep: total "
          f"{p_long['train_wall_s'] + p_long['inference_s']:.3f}s, "
          f"RMSE {p_long['rmse']:.4g}")
    if raw is not None:
        raw_total = raw["train_wall_s"] + raw["inference_s"]
        ratio_p_over_raw = (p_long["train_wall_s"] + p_long["inference_s"]) / max(
            1e-6, raw_total)
        print(f"[summary] plastix {p_long['epochs']}ep is "
              f"{ratio_p_over_raw:.1f}× slower than raw-C++/OpenBLAS at "
              f"{p_long['rmse'] / raw['rmse']:.2f}× the RMSE")
    ratio = (p_long["train_wall_s"] + p_long["inference_s"]) / max(
        1e-6, rpy["total_s"])
    print(f"[summary] plastix {p_long['epochs']}ep is "
          f"{ratio:.1f}× slower than reservoirpy at "
          f"{p_long['rmse'] / rpy['rmse']:.2f}× the RMSE")


if __name__ == "__main__":
    main()
