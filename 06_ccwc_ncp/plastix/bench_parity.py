"""Parity benchmark: Plastix `07_ccwc_ncp` vs Python `ccwc/`
model A on the same sine task and the same network shape.

Parity surface (kept identical across both implementations):

    * task         : noisy-sine regression (2-D in / 2-D out)
    * units        : neuron count of the NCP layer (sweepable)
    * seq_len      : timesteps per sequence (sweepable)
    * train_seqs   : per-epoch training sequence count
    * val_seqs / test_seqs : eval sizes
    * epochs       : pass count

Differences accepted (documented in the figure caption):

    * neuron model   : ncps LTC (Python) vs leaky-tanh-RNN (Plastix)
    * learning rule  : Adam-BPTT (Python) vs SGD-style e-prop (Plastix)
    * lr default     : 1e-3 (Adam) vs 5e-3 (e-prop)
    * batch          : 64 (Python) vs 1 (Plastix walks one sequence at a time)

For each config we time both implementations end-to-end (train + eval)
and record final test MSE. Both write `wall_seconds` to their existing
summary CSVs; this harness just parses those and aggregates.

Usage:
    # default sweep over units = {16, 32, 48, 64}, epochs = 8
    uv run python traditional-plastix/07-ccwc-ncp/bench_parity.py run

    # custom sweep
    uv run python traditional-plastix/07-ccwc-ncp/bench_parity.py run \\
        --units-grid 16,32,64 --epochs 12 --seq-len 48 --train-seqs 256

    # plot whatever's already in the CSV
    uv run python traditional-plastix/07-ccwc-ncp/bench_parity.py plot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]

# Locations of the three implementations.
PLASTIX_BIN = REPO / "build-host" / "traditional-plastix" / "07_ccwc_ncp"
RAW_BIN     = REPO / "build-host" / "traditional-raw"     / "07_ccwc_ncp"
PY_TRAIN    = REPO / "traditional" / "ccwc" / "train.py"

# Where the harness writes its long-format CSV + comparison plot.
BENCH_CSV = REPO / "traditional-plastix" / "07-ccwc-ncp" / "bench_parity.csv"
BENCH_PNG = REPO / "traditional-plastix" / "plots" / "ccwc_bench_parity.png"


@dataclass
class RunSpec:
    """One row of the sweep grid. `tag` is used as the unique on-disk
    suffix so concurrent or repeated runs don't collide."""
    units: int
    seq_len: int
    train_seqs: int
    val_seqs: int
    test_seqs: int
    epochs: int
    seed: int
    tag: str


@dataclass
class Result:
    impl: str        # "plastix" or "python"
    spec: RunSpec
    wall_s: float
    test_mse: float
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Implementation drivers
# ---------------------------------------------------------------------------

def _parse_summary_csv(path: Path) -> dict[str, str]:
    """One-row CSV reader — returns header→value map."""
    with path.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise RuntimeError(f"unexpected summary CSV shape: {path}")
    return dict(zip(rows[0], rows[1]))


def run_plastix(spec: RunSpec, *, lr: float, quiet: bool) -> Result:
    """Invoke the compiled Plastix binary and parse its summary CSV.

    Returns the implementation's own self-reported wall_seconds — that
    measures training + per-epoch eval inside the binary, which is the
    apples-to-apples number to compare against the Python self-reported
    wall_seconds. The harness also measures wall-clock around the
    subprocess as a sanity check; it always ≥ the self-report.
    """
    if not PLASTIX_BIN.exists():
        raise FileNotFoundError(
            f"Plastix binary missing: {PLASTIX_BIN}\n"
            f"  build with: cmake --build build-host --target 07_ccwc_ncp -j")
    cmd = [
        str(PLASTIX_BIN),
        "--tag", spec.tag,
        "--units", str(spec.units),
        "--seq-len", str(spec.seq_len),
        "--train-seqs", str(spec.train_seqs),
        "--val-seqs", str(spec.val_seqs),
        "--test-seqs", str(spec.test_seqs),
        "--epochs", str(spec.epochs),
        "--lr", str(lr),
        "--seed", str(spec.seed),
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    sub_wall = time.perf_counter() - t0
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        raise RuntimeError(
            f"plastix run failed (rc={cp.returncode}): {' '.join(shlex.quote(c) for c in cmd)}")
    if not quiet:
        print(f"  [plastix subproc wall {sub_wall:.2f}s]")
    summary_path = REPO / "traditional-plastix" / "results" / f"ccwc_ncp_{spec.tag}.summary.csv"
    row = _parse_summary_csv(summary_path)
    return Result(
        impl="plastix",
        spec=spec,
        wall_s=float(row["wall_seconds"]),
        test_mse=float(row["test_mse"]),
        extras={"subproc_wall_s": sub_wall,
                "n_edges": float(row["n_edges"])},
    )


def run_raw(spec: RunSpec, *, lr: float, quiet: bool) -> Result:
    """Invoke the OpenBLAS-backed raw C++ binary. Same CLI surface as the
    Plastix port — both consume `bench::CliArgs` — so this is a one-line
    rename of the path. Summary CSV columns are the same shape too."""
    if not RAW_BIN.exists():
        raise FileNotFoundError(
            f"raw binary missing: {RAW_BIN}\n"
            f"  build with: cmake --build build-host "
            f"--target traditional_raw_07_ccwc_ncp -j")
    cmd = [
        str(RAW_BIN),
        "--tag", spec.tag,
        "--units", str(spec.units),
        "--seq-len", str(spec.seq_len),
        "--train-seqs", str(spec.train_seqs),
        "--val-seqs", str(spec.val_seqs),
        "--test-seqs", str(spec.test_seqs),
        "--epochs", str(spec.epochs),
        "--lr", str(lr),
        "--seed", str(spec.seed),
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
                             "OMP_NUM_THREADS": "1"})
    sub_wall = time.perf_counter() - t0
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        raise RuntimeError(
            f"raw run failed (rc={cp.returncode}): {' '.join(shlex.quote(c) for c in cmd)}")
    if not quiet:
        print(f"  [raw     subproc wall {sub_wall:.2f}s]")
    summary_path = REPO / "traditional-raw" / "results" / f"ccwc_ncp_{spec.tag}.summary.csv"
    row = _parse_summary_csv(summary_path)
    return Result(
        impl="raw",
        spec=spec,
        wall_s=float(row["wall_seconds"]),
        test_mse=float(row["test_mse"]),
        extras={"subproc_wall_s": sub_wall,
                "n_edges": float(row["n_edges"])},
    )


def run_python(spec: RunSpec, *, lr: float, quiet: bool) -> Result:
    """Invoke the Python ccwc trainer on model A only (the wired NCP)
    using uv to ensure the same env the user runs interactively."""
    cmd = [
        "uv", "run", "python", str(PY_TRAIN),
        "--task", "sine", "--model", "A",
        "--tag", spec.tag,
        "--units", str(spec.units),
        "--sine-seq-len", str(spec.seq_len),
        "--sine-train", str(spec.train_seqs),
        "--sine-val", str(spec.val_seqs),
        "--sine-test", str(spec.test_seqs),
        "--epochs", str(spec.epochs),
        "--lr", str(lr),
        "--seed", str(spec.seed),
        "--device", "cpu",     # CPU on both sides so the comparison is fair
        "--no-plot",
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                        env={**os.environ, "OMP_NUM_THREADS": "1",
                             "MKL_NUM_THREADS": "1"})
    sub_wall = time.perf_counter() - t0
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr[-2000:])
        raise RuntimeError(
            f"python run failed (rc={cp.returncode}): {' '.join(shlex.quote(c) for c in cmd)}")
    if not quiet:
        print(f"  [python  subproc wall {sub_wall:.2f}s]")
    summary_path = REPO / "traditional" / "results" / f"ccwc_A_{spec.tag}.summary.csv"
    row = _parse_summary_csv(summary_path)
    return Result(
        impl="python",
        spec=spec,
        wall_s=float(row["wall_seconds"]),
        test_mse=float(row["test_metric"]),
        extras={"subproc_wall_s": sub_wall,
                "n_params": float(row["n_params"])},
    )


# ---------------------------------------------------------------------------
# Sweep + CSV write
# ---------------------------------------------------------------------------

def run_sweep(args) -> None:
    units_grid = [int(u) for u in args.units_grid.split(",") if u.strip()]
    rows: list[Result] = []

    print(f"[bench] sweeping units ∈ {units_grid}  "
          f"seq_len={args.seq_len}  train={args.train_seqs}  "
          f"epochs={args.epochs}  seed={args.seed}")
    print(f"[bench] lr: plastix={args.plastix_lr}  raw={args.raw_lr}  "
          f"python={args.python_lr}")

    for u in units_grid:
        spec = RunSpec(
            units=u,
            seq_len=args.seq_len,
            train_seqs=args.train_seqs,
            val_seqs=args.val_seqs,
            test_seqs=args.test_seqs,
            epochs=args.epochs,
            seed=args.seed,
            tag=f"bench_u{u}",
        )
        print(f"[bench] units={u}")
        rp = run_plastix(spec, lr=args.plastix_lr, quiet=args.quiet)
        rows.append(rp)
        print(f"  plastix  wall={rp.wall_s:.3f}s  test_mse={rp.test_mse:.4f}")
        rr = run_raw(spec, lr=args.raw_lr, quiet=args.quiet)
        rows.append(rr)
        print(f"  raw      wall={rr.wall_s:.3f}s  test_mse={rr.test_mse:.4f}")
        ry = run_python(spec, lr=args.python_lr, quiet=args.quiet)
        rows.append(ry)
        print(f"  python   wall={ry.wall_s:.3f}s  test_mse={ry.test_mse:.4f}")
        print(f"  ratios   raw/plastix={rr.wall_s/rp.wall_s:.2f}x  "
              f"plastix/python={rp.wall_s/ry.wall_s:.2f}x")

    # Long-format CSV: one row per (impl, units).
    BENCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BENCH_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "impl", "units", "seq_len", "train_seqs", "val_seqs", "test_seqs",
            "epochs", "seed", "lr", "wall_s", "test_mse",
            "subproc_wall_s", "n_params_or_edges",
        ])
        for r in rows:
            lr = {"plastix": args.plastix_lr,
                  "raw":     args.raw_lr,
                  "python":  args.python_lr}[r.impl]
            n2 = r.extras.get("n_params", r.extras.get("n_edges", 0))
            w.writerow([
                r.impl, r.spec.units, r.spec.seq_len, r.spec.train_seqs,
                r.spec.val_seqs, r.spec.test_seqs, r.spec.epochs,
                r.spec.seed, lr, r.wall_s, r.test_mse,
                r.extras.get("subproc_wall_s", 0.0), int(n2),
            ])
    print(f"\n[bench] wrote {BENCH_CSV}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

IMPL_COLOUR = {
    "plastix": "#1f9d55",
    "raw":     "#3a7bd5",
    "python":  "#d35a5a",
}
IMPL_LABEL = {
    "plastix": "Plastix (C++, framework + e-prop)",
    "raw":     "Raw C++ + OpenBLAS (dense BLAS + e-prop)",
    "python":  "Python (ncps LTC + Adam-BPTT)",
}
IMPL_MARKER = {"plastix": "o", "raw": "D", "python": "s"}


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run with the 'run' subcommand first")
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_results(args) -> None:
    rows = _load_rows(BENCH_CSV)
    # Pivot to per-impl arrays keyed by units.
    by_impl: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_impl.setdefault(r["impl"], {})[int(r["units"])] = r

    impls = [i for i in ("plastix", "raw", "python") if i in by_impl]
    if not impls:
        raise SystemExit(f"no implementations found in {BENCH_CSV}")
    # Only keep `units` that every implementation produced — keeps the
    # multi-line plots meaningful even when one leg was skipped.
    common_units = set.intersection(*[set(by_impl[i]) for i in impls])
    units = sorted(common_units)
    if not units:
        raise SystemExit(f"no overlapping units across {impls} in {BENCH_CSV}")

    wall  = {i: np.array([float(by_impl[i][u]["wall_s"])   for u in units])
             for i in impls}
    mse   = {i: np.array([float(by_impl[i][u]["test_mse"]) for u in units])
             for i in impls}

    n_units = len(units)
    ref_row = by_impl[impls[0]][units[0]]
    epochs  = int(ref_row["epochs"])
    train_n = int(ref_row["train_seqs"])
    seq_len = int(ref_row["seq_len"])
    steps_per_run = epochs * train_n * seq_len
    step_us = {i: wall[i] / steps_per_run * 1e6 for i in impls}

    fig = plt.figure(figsize=(13.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.9],
                          hspace=0.35, wspace=0.22)

    # --- A: wall time vs units (matched config) ---------------------------
    ax = fig.add_subplot(gs[0, 0])
    for i in impls:
        ax.plot(units, wall[i], marker=IMPL_MARKER[i],
                color=IMPL_COLOUR[i], linewidth=1.8, label=IMPL_LABEL[i])
    ax.set_xlabel("units (NCP neuron count)")
    ax.set_ylabel("wall time (s)")
    ax.set_yscale("log")
    ax.set_title("A — End-to-end wall time vs. network size", loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=8)

    # --- B: per-step time (μs) — grouped bars ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    n_impl = len(impls)
    bar_w = 0.8 / n_impl
    xs = np.arange(n_units)
    for k, i in enumerate(impls):
        offs = (k - (n_impl - 1) / 2) * bar_w
        ax.bar(xs + offs, step_us[i], bar_w, color=IMPL_COLOUR[i],
               label=IMPL_LABEL[i].split(" (")[0],
               edgecolor="white", linewidth=0.5)
    # Annotate each cluster with the python / raw ratio (the headline) when
    # both legs are present.
    if "raw" in impls and "python" in impls:
        for k, _ in enumerate(units):
            r = step_us["python"][k] / step_us["raw"][k]
            ax.text(xs[k], max(step_us[i][k] for i in impls),
                    f"py/raw\n{r:.0f}×", ha="center", va="bottom",
                    fontsize=7, color="#444444")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(u) for u in units])
    ax.set_xlabel("units")
    ax.set_ylabel("μs per DoStep (= per timestep)")
    ax.set_title("B — Per-step cost (log scale)", loc="left")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    # --- C: speedup ribbons (all impls vs python baseline) ----------------
    ax = fig.add_subplot(gs[1, 0])
    if "python" in impls:
        baseline = wall["python"]
        for i in impls:
            if i == "python":
                continue
            sp = baseline / wall[i]
            ax.plot(units, sp, marker=IMPL_MARKER[i],
                    color=IMPL_COLOUR[i], linewidth=2.0, label=f"{i}")
            for u, s in zip(units, sp):
                ax.text(u, s, f"{s:.0f}×", ha="center", va="bottom",
                        fontsize=8, color=IMPL_COLOUR[i])
        ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8,
                   label="parity with Python")
    ax.set_xlabel("units")
    ax.set_ylabel("python wall / impl wall")
    ax.set_title("C — Speedup over Python baseline", loc="left")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=8)

    # --- D: test MSE ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for i in impls:
        ax.plot(units, mse[i], marker=IMPL_MARKER[i],
                color=IMPL_COLOUR[i], linewidth=1.8, label=IMPL_LABEL[i])
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.8, linestyle=":",
               label="zero-predictor (≈0.5)")
    ax.set_xlabel("units")
    ax.set_ylabel("final test MSE")
    ax.set_yscale("log")
    ax.set_title("D — Final test MSE   "
                 "(different learning rules, same task)", loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=7)

    # --- E: caption / config summary (text panel, full-width row) --------
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    plastix_lr = by_impl.get("plastix", {}).get(units[0], {}).get("lr", "—")
    raw_lr     = by_impl.get("raw",     {}).get(units[0], {}).get("lr", "—")
    python_lr  = by_impl.get("python",  {}).get(units[0], {}).get("lr", "—")
    mean_wall = {i: float(np.mean(wall[i])) for i in impls}
    mean_step = {i: float(np.mean(step_us[i])) for i in impls}
    mean_mse  = {i: float(np.mean(mse[i])) for i in impls}

    parts = [
        f"Parity surface: sine regression, seq_len={seq_len}, "
        f"train_seqs={train_n}, epochs={epochs}, identical n_in / n_out=2, "
        f"single thread on the same CPU.",
        "All three implementations use the same AutoNCP-style sparse "
        "wiring, the same leaky-tanh-RNN dynamics, and the same one-step "
        "e-prop learning rule — except Python, which keeps the original "
        "ncps LTC + Adam-BPTT it ships with.",
        f"Mean per-DoStep cost: " + ", ".join(
            f"{i}={mean_step[i]:.2f} μs" for i in impls
        ) + ".",
        f"Mean end-to-end wall: " + ", ".join(
            f"{i}={mean_wall[i]:.3f}s" for i in impls
        ) + ".",
    ]
    if "python" in impls and "raw" in impls:
        parts.append(
            "Raw C++ + OpenBLAS dispatches one cblas_sgemv per layer and "
            "two cblas_sger per timestep, so the inner loop is "
            "hand-vectorised — that's where its lead over the per-edge "
            "Plastix iteration comes from."
        )
    parts.append(
        f"Final test MSE means: " + ", ".join(
            f"{i}={mean_mse[i]:.3f}" for i in impls
        ) + "."
    )
    ax.text(0.0, 1.0, "\n".join(parts), va="top", ha="left",
            fontsize=9, wrap=True, family="serif", color="#333333")

    fig.suptitle(
        "ccwc parity benchmark   ·   "
        "Plastix vs. raw C++ (OpenBLAS) vs. Python (ncps)",
        fontsize=11.5, y=1.01,
    )
    BENCH_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BENCH_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {BENCH_PNG}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Sweep flags shared by the `run` and `rp` subcommands."""
    parser.add_argument("--units-grid", default="16,32,48,64")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--train-seqs", type=int, default=256)
    parser.add_argument("--val-seqs", type=int, default=64)
    parser.add_argument("--test-seqs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    # lr defaults match each implementation's own default so we measure the
    # tools as they ship, not under an artificial common lr that would
    # underfit at least one of them. Plastix and raw share an lr because
    # they share the same e-prop learning rule and same neuron model.
    parser.add_argument("--plastix-lr", type=float, default=5e-3)
    parser.add_argument("--raw-lr",     type=float, default=5e-3)
    parser.add_argument("--python-lr",  type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true")


def run_then_plot(args) -> None:
    run_sweep(args)
    plot_results(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="sweep units and time both impls")
    _add_run_flags(pr)
    pr.set_defaults(func=run_sweep)

    pp = sub.add_parser("plot", help="render the comparison figure")
    pp.set_defaults(func=plot_results)

    prp = sub.add_parser("rp", help="run the sweep, then render the figure")
    _add_run_flags(prp)
    prp.set_defaults(func=run_then_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
