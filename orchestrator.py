#!/bin/env/python3
"""Discover-and-drive harness for the plastix-bench benchmark suite.

Layout convention:

    <bench>/<impl>/run_benchmark.py
        - Python impls: this *is* the trainer.
        - C++ impls:    a thin sentinel that execs the compiled binary
                        at <build-dir>/<bench>/<impl>/run_benchmark.

The orchestrator scans for `*/run_benchmark.py`, invokes each under
matched flags, polls /proc/<descendant>/status to track RSS (peak and
average), parses the standard summary CSV the run emits, and aggregates
results into a single `runs.csv`.

Each per-bench `summary.csv` carries:
  * headline numbers: workload, wall_seconds, test_*, n_units, n_edges
  * per-phase ns/step (forward/loss/backward/update/prune/grow/reset),
    each with mean **and standard deviation** across all optimisation
    steps. See common/{cpp,plastix,pytorch}/common.{hpp,py} for the
    shared `PhaseTimer` that produces them.

`runs.csv` carries one row per (bench, impl, tag) run and hoists the
phase columns + peak/mean RSS up to the aggregate level so the plot
subcommand can read everything from one file.

Subcommands:
    list       enumerate discovered (bench, impl) pairs
    run        execute one or more benchmarks; write runs.csv
    plot       render comparison plots from runs.csv

Standard plots (--kinds flags):
    accuracy        per-bench learning trajectory across impls (one fig/bench)
    overlay         cross-bench normalised curves on a single axis grid
    walltime        wall-clock table + bar chart, grouped by bench and impl
    memory          peak + average RSS panels with tables (VmRSS polling)
    phases          stacked-bar breakdown of mean ns/step per phase
    phase_stats     grouped bars with σ error bars per phase per impl

Usage:
    # Build the C++ binaries (one-time / after editing them). Point Plastix
    # discovery at an install prefix, or build it from a local checkout:
    cmake -S . -B build-host -DCMAKE_PREFIX_PATH=/path/to/plastix-install
    #   ...or:  -DPLASTIX_SOURCE_DIR=/path/to/plastix
    cmake --build build-host -j

    # Run from the repo root:
    uv run python orchestrator.py list
    uv run python orchestrator.py run --bench 01_static_etth1
    uv run python orchestrator.py run --quick     # all, fast
    uv run python orchestrator.py run             # all, full
    uv run python orchestrator.py plot            # render all
    uv run python orchestrator.py plot --kinds phase_stats,memory

    # Pass-through args go after `--`, e.g. shrink a streaming bench:
    uv run python orchestrator.py run \\
        --bench 09_imprintin_learner -- --max-steps 4000

Outputs land under:
    _results/runs.csv                         (aggregate)
    _results/<bench>/<impl>/<file>.summary.csv (per-run)
    _plots/<kind>.png                          (rendered plots)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS_CSV = HERE / "_results" / "runs.csv"
PLOTS_DIR = HERE / "_plots"


# Impls in canonical order — the orchestrator presents them this way in
# tables and plot legends.
IMPL_ORDER = ("pytorch", "plastix", "cpp", "cuda", "jax", "snn", "norse")
IMPL_COLOUR = {
    "pytorch": "#d35a5a",
    "plastix": "#1f9d55",
    "cpp":     "#3a7bd5",
    "cuda":    "#9b1fd6",
    "jax":     "#e08b1f",
    "snn":     "#7b4fd6",
    "norse":   "#d64f9b",
}
IMPL_LABEL = {
    "pytorch": "PyTorch",
    "plastix": "Plastix",
    "cpp":     "C++ (OpenBLAS)",
    "cuda":    "CUDA (cuBLAS)",
    "jax":     "JAX",
    "snn":     "snnTorch (SNN)",
    "norse":   "Norse (SNN)",
}
IMPL_MARKER = {"pytorch": "s", "plastix": "o", "cpp": "D", "cuda": "*",
               "jax": "^", "snn": "v", "norse": "P"}


@dataclass
class Sentinel:
    bench: str
    impl: str
    path: Path                # the run_benchmark.py file


# `baselines` holds the alt-framework comparison harnesses (run via
# run_baselines.py), and `11_scaling_imprint` is driven by the scaling.py sweep —
# neither belongs in the main per-bench suite / paper tables.
_NON_BENCH_DIRS = {"common", "cmake", "baselines", "11_scaling_imprint"}

# Benches discovered normally (so `list` and `run --bench <name>` still reach
# them) but skipped by a no-filter `run`. 01_static_etth1 is a dense-MLP control
# whose comparison is muddied by the plastix impl running batch-1 online SGD
# (~64x the weight updates of every other framework), so it's excluded from the
# default sweep. Pass `--bench 01_static_etth1` to run it explicitly.
_DEFAULT_RUN_EXCLUDE = {"01_static_etth1"}


def discover(root: Path = HERE) -> list[Sentinel]:
    """Walk the suite root and find every <bench>/<impl>/run_benchmark.py.
    Skips dirs whose name starts with `_` plus the shared `common`/`cmake`
    trees and the non-suite harness dirs (metadata / cache / shared utilities)."""
    out: list[Sentinel] = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if bench_dir.name.startswith("_") or bench_dir.name in _NON_BENCH_DIRS:
            continue
        for impl_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            if impl_dir.name.startswith("_") or impl_dir.name == "__pycache__":
                continue
            sentinel = impl_dir / "run_benchmark.py"
            if sentinel.exists():
                out.append(Sentinel(bench_dir.name, impl_dir.name, sentinel))
    return out


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

# Each run writes a summary CSV that the orchestrator parses. Different
# implementations have different column conventions; we read whichever of
# these keys is present and normalise into the orchestrator's runs.csv.
WALL_KEYS   = ("wall_seconds",)
# Order matters: a benchmark might have several of these — we want the
# "final test number" (closest to a holdout score) where possible, then
# fall back to whatever post-training summary it actually emits.
TEST_KEYS   = ("test_mse", "test_rmse", "test_metric", "test_acc",
               "val_acc_final", "val_mse_final", "val_loss_final",
               "test_acc_final", "test_mse_final")
METRIC_KIND_KEYS = ("metric_kind",)
PARAM_KEYS  = ("n_params", "n_edges")
# Per-phase ns/step from benches that report them (currently 09_imprintin_*).
# Missing keys default to NaN so older bench CSVs stay backward-compatible.
PHASE_KEYS = (
    "step_ns_mean", "step_count",
    "forward_ns_mean", "forward_ns_std",
    "loss_ns_mean", "loss_ns_std",
    "backward_ns_mean", "backward_ns_std",
    "update_ns_mean", "update_ns_std",
    "prune_ns_mean", "prune_ns_std",
    "grow_ns_mean", "grow_ns_std",
    "reset_ns_mean", "reset_ns_std",
    "other_ns_mean",
)

# Per-bench RSS milestone breakdown (MemoryProbe). Hoisted alongside PHASE_KEYS
# so memory_table.py can read the categories from runs.csv. Keep in lockstep
# with MEMORY_COLUMNS in common/pytorch/common.py.
MEM_KEYS = (
    "mem_overhead_kb", "mem_dataset_kb", "mem_weights_kb",
)


def _read_one_row_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    return dict(zip(rows[0], rows[1]))


def _first_present(d: dict[str, str], keys) -> str | None:
    for k in keys:
        if k in d:
            return d[k]
    return None


# ---------------------------------------------------------------------------
# Run an individual sentinel
# ---------------------------------------------------------------------------

def _read_vm_field(pid: int, field: str) -> int:
    """Read VmRSS/VmHWM/etc (kB) from /proc/<pid>/status. 0 if PID is gone."""
    needle = f"{field}:"
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(needle):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _read_vmhwm(pid: int) -> int:
    """Per-process peak RSS (kB). Monotonic high-water-mark."""
    return _read_vm_field(pid, "VmHWM")


def _read_vmrss(pid: int) -> int:
    """Current resident set size (kB)."""
    return _read_vm_field(pid, "VmRSS")


def _walk_descendants(root_pid: int) -> set[int]:
    """All currently-live descendants of root_pid (inclusive). Walks
    /proc/*/status for PPid; missing entries are skipped silently."""
    children: dict[int, list[int]] = {}
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return {root_pid}
    for entry in proc_entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        children.setdefault(ppid, []).append(int(entry))
                        break
        except (OSError, ValueError):
            continue
    out = {root_pid}
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _gpu_vram_kb(pids: set[int]) -> int:
    """Sum of GPU VRAM (kB) used by the given PIDs, via nvidia-smi's per-process
    compute-apps query. 0 if no GPU / nvidia-smi absent / no matching process."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0
    total_mib = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in pids:
            try:
                total_mib += int(parts[1])
            except ValueError:
                pass
    return total_mib * 1024


def _poll_memory(pid: int, poll_interval_s: float = 0.05):
    """Background-thread helper. At each tick, sums VmRSS over the live
    descendant set and records both the maximum observation (peak) and the
    running mean (average). VmRSS-based polling is what we want for "average
    memory" — VmHWM is a monotonic high-water-mark per-process and would
    bias the mean upward, especially for short-lived children. GPU VRAM is
    polled on a coarser cadence (nvidia-smi is slow) and tracked as a peak.

    Returns (state, stop_event, thread). state has keys:
        peak_kb:      max(sum over descendants of VmRSS), in kB
        mean_kb:      arithmetic mean across polled timestamps, in kB
        samples:      number of poll ticks
        peak_vram_kb: max(sum over descendants of GPU VRAM), in kB (0 on CPU)
    Notes:
      - Polling stops when stop.set() is called.
      - kB matches /proc convention; divide by 1024 for MiB.
    """
    import threading
    state = {"peak_kb": 0, "mean_kb": 0.0, "samples": 0, "peak_vram_kb": 0}
    stop = threading.Event()

    def loop():
        sum_kb = 0
        n = 0
        vram_every = max(1, int(0.5 / poll_interval_s))  # poll VRAM ~every 0.5s
        while not stop.is_set():
            try:
                pids = _walk_descendants(pid)
            except Exception:
                pids = {pid}
            cur = 0
            for p in pids:
                cur += _read_vmrss(p)
            if cur > state["peak_kb"]:
                state["peak_kb"] = cur
            sum_kb += cur
            n += 1
            # Update mean live so the caller can read partials if needed.
            state["mean_kb"] = sum_kb / max(n, 1)
            state["samples"] = n
            if n % vram_every == 1:
                vram = _gpu_vram_kb(pids)
                if vram > state["peak_vram_kb"]:
                    state["peak_vram_kb"] = vram
            stop.wait(poll_interval_s)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return state, stop, t


# Back-compat alias: existing callers grab the polled tree-wide peak via the
# .v key on the dict, which we now expose as peak_kb. Anything that still
# imports the old name needs a one-line migration below.
def _poll_peak_rss(pid: int, poll_interval_s: float = 0.05):
    """Deprecated: prefer _poll_memory. Kept for any external caller."""
    state, stop, thread = _poll_memory(pid, poll_interval_s)
    # Wrap state so old `.peak["v"]` callers keep working.
    class _Shim:
        def __getitem__(self, _k): return state["peak_kb"]
    return _Shim(), stop, thread


def run_one(s: Sentinel, *, tag: str, build_dir: Path,
            extras: list[str], out_dir: Path,
            quiet: bool = False, device: str = "cpu",
            timeout_s: float = 600.0) -> dict:
    """Invoke a sentinel and parse its summary CSV.

    The sentinel itself decides where to write — we pass `--out-dir` and
    `--tag` so the output path is deterministic and the orchestrator can
    find the resulting summary CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        "uv", "run", "python", str(s.path),
        "--tag", tag,
        "--out-dir", str(out_dir),
        "--seed", "0",
    ]
    if s.impl not in ("pytorch", "jax", "snn", "norse"):
        base_cmd += ["--build-dir", str(build_dir)]
        # The C++ binaries default DataDir to a relative "data" (see
        # common/{plastix,cpp}/common.hpp). Run from the repo root that
        # resolves to ./data, which does not exist — the datasets live under
        # common/pytorch/data. Without this the benches with a synthetic
        # fallback (01-06) silently train on stand-in data and the ones
        # without (08) hard-fail. Point every C++ impl at the real data root.
        # (The PyTorch sentinels resolve this path themselves via common.py.)
        base_cmd += ["--data-dir", str(HERE / "common" / "pytorch" / "data")]
    # `--device cpu` is only understood by the Python impls (PyTorch/JAX/SNN).
    # Passing it to C++ binaries via the sentinel would surface in their
    # CliArgs as an unknown key + non-zero exit; gate on the impl.
    if s.impl in ("pytorch", "jax", "snn", "norse"):
        base_cmd += ["--device", device, "--no-plot"]
    base_cmd += extras

    # JAX picks its backend from JAX_PLATFORMS, not our --device flag, so pin it
    # explicitly: a cpu-tagged pass must run jax on CPU, a gpu pass on the GPU.
    extra_env = {}
    if s.impl == "jax":
        on_gpu = device in ("cuda", "gpu")
        extra_env["JAX_PLATFORMS"] = "cuda" if on_gpu else "cpu"
        if on_gpu:
            # By default XLA preallocates ~75% of VRAM into a pool, which would
            # make the polled per-process VRAM meaningless (~all of it). Disable
            # so peak_vram_kb reflects actual usage.
            extra_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if not quiet:
        print(f"  [run] {s.bench}/{s.impl}   {' '.join(shlex.quote(c) for c in base_cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        base_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=HERE,
        env={**os.environ,
             "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1",
             **extra_env})
    mem_state, stop, poller = _poll_memory(proc.pid)
    # Hard per-benchmark wall-clock cap: kill any run that exceeds it so a
    # single slow (bench, impl) can't stall the whole sweep, and the rest still
    # produce graphable results. Killed runs are reported and skipped.
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stop.set(); poller.join(timeout=0.2)
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        raise RuntimeError(
            f"{s.bench}/{s.impl} exceeded {timeout_s:.0f}s timeout — killed")
    stop.set(); poller.join(timeout=0.2)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(stderr[-2000:])
        raise RuntimeError(
            f"{s.bench}/{s.impl} failed (rc={proc.returncode})")
    peak_rss_kb = mem_state["peak_kb"]
    mean_rss_kb = mem_state["mean_kb"]
    mem_samples = mem_state["samples"]
    peak_vram_kb = mem_state["peak_vram_kb"]

    # The summary CSV lives at `<out-dir>/<workload-name>_<tag>.summary.csv`
    # where <workload-name> varies by impl (`ccwc`, `ccwc_ncp`, `snn_shd`...).
    # Glob for whatever lands at the deterministic tag suffix.
    candidates = sorted(out_dir.glob(f"*_{tag}.summary.csv"))
    if not candidates:
        sys.stderr.write(
            f"[warn] {s.bench}/{s.impl}: no summary CSV at {out_dir} "
            f"matching *_{tag}.summary.csv\n")
        return {
            "bench": s.bench, "impl": s.impl, "tag": tag,
            "wall_seconds": float("nan"), "metric": float("nan"),
            "metric_kind": "?", "peak_rss_kb": peak_rss_kb,
            "mean_rss_kb": round(mean_rss_kb, 1),
            "mem_samples": mem_samples, "peak_vram_kb": peak_vram_kb,
            "subproc_wall_s": wall, "params_or_edges": 0,
            "summary_csv": "",
        }
    summary = _read_one_row_csv(candidates[0]) or {}
    wall_s = float(_first_present(summary, WALL_KEYS) or wall)
    test_v = _first_present(summary, TEST_KEYS)
    metric = float(test_v) if test_v else float("nan")
    metric_kind = _first_present(summary, METRIC_KIND_KEYS)
    if not metric_kind:
        # Heuristic: an accuracy-style column (test_acc, val_acc, _acc_final,
        # etc.) implies a classification metric. We can't just check `"acc"
        # in k` because keys like `jaccard_min` also contain "acc" but mean
        # something else — narrow to keys that begin with `test_acc`,
        # `val_acc`, or contain `_acc_`/`_acc`-suffix.
        def _is_acc_key(k: str) -> bool:
            return (k.startswith(("test_acc", "val_acc"))
                    or k.endswith("_acc")
                    or "_acc_" in k)
        metric_kind = ("acc" if any(_is_acc_key(k) for k in summary)
                       else "mse")
    params = _first_present(summary, PARAM_KEYS) or "0"
    out = {
        "bench": s.bench, "impl": s.impl, "tag": tag,
        "wall_seconds": wall_s,
        "metric": metric,
        "metric_kind": metric_kind,
        "peak_rss_kb": peak_rss_kb,
        "mean_rss_kb": round(mean_rss_kb, 1),
        "mem_samples": mem_samples,
        "peak_vram_kb": peak_vram_kb,
        "subproc_wall_s": wall,
        "params_or_edges": int(float(params)) if params else 0,
        "summary_csv": str(candidates[0]),
    }
    # Hoist any per-phase ns/step columns that the bench emitted so they land
    # in runs.csv next to wall/metric. NaN-fill the rest so a single
    # csv.DictWriter writes a stable schema across benches with and without
    # phase profiling.
    for k in PHASE_KEYS + MEM_KEYS:
        v = summary.get(k)
        try:
            out[k] = float(v) if v is not None and v != "" else float("nan")
        except ValueError:
            out[k] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    sentinels = discover()
    if not sentinels:
        print("(no sentinels found)")
        return
    print(f"{'BENCH':32s}  {'IMPL':10s}  PATH")
    print("-" * 80)
    for s in sentinels:
        print(f"{s.bench:32s}  {s.impl:10s}  {s.path.relative_to(HERE)}")


def _select(sentinels: list[Sentinel], bench: str | None,
            impl: str | None) -> list[Sentinel]:
    if bench:
        bs = {b.strip() for b in bench.split(",")}
        sentinels = [s for s in sentinels if s.bench in bs]
    if impl:
        ms = {m.strip() for m in impl.split(",")}
        sentinels = [s for s in sentinels if s.impl in ms]
    return sentinels


def cmd_run(args) -> None:
    sentinels = _select(discover(), args.bench, args.impl)
    # A no-filter run skips the default-excluded benches (e.g. the dense-MLP
    # control); an explicit --bench overrides the exclusion.
    if not args.bench:
        skipped = sorted({s.bench for s in sentinels} & _DEFAULT_RUN_EXCLUDE)
        if skipped:
            sentinels = [s for s in sentinels
                         if s.bench not in _DEFAULT_RUN_EXCLUDE]
            print(f"[orch] skipping {', '.join(skipped)} in default run "
                  f"(pass --bench <name> to include)")
    if not sentinels:
        print("[err] no sentinels matched the filter", file=sys.stderr)
        sys.exit(1)
    extras = ["--quick"] if args.quick else []
    extras += list(args.passthrough or [])

    out_root = HERE / "_results"
    out_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    print(f"[orch] running {len(sentinels)} benchmark/impl combos "
          f"with tag={args.tag!r}, build_dir={args.build_dir}")
    for s in sentinels:
        per_dir = out_root / s.bench / s.impl
        try:
            r = run_one(s, tag=args.tag, build_dir=args.build_dir,
                        extras=extras, out_dir=per_dir,
                        quiet=args.quiet, device=args.device,
                        timeout_s=args.timeout)
            runs.append(r)
            print(f"  ok  {s.bench}/{s.impl}  wall={r['wall_seconds']:.3f}s  "
                  f"metric={r['metric']:.4f} ({r['metric_kind']})  "
                  f"rss={r['peak_rss_kb']/1024:.1f} MiB")
        except Exception as e:
            print(f"  FAIL  {s.bench}/{s.impl}  {type(e).__name__}: {e}")

    if not runs:
        print("[orch] no successful runs; nothing to write", file=sys.stderr)
        sys.exit(1)
    # Append vs overwrite: easier mental model is overwrite — the user re-runs
    # to update. If they want a long-running archive they can copy runs.csv.
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = list(runs[0].keys())
    with RUNS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"[orch] wrote {RUNS_CSV}  ({len(runs)} rows)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _load_runs() -> list[dict]:
    if not RUNS_CSV.exists():
        raise SystemExit(f"no {RUNS_CSV}; run with 'run' subcommand first")
    with RUNS_CSV.open() as f:
        return list(csv.DictReader(f))


def _load_history_for(run: dict) -> dict | None:
    """Find the per-epoch history JSONL for a given run row, if it exists.
    Convention: same dir as summary CSV, `*_<tag>.history.jsonl`.

    Different impls use different x-axis keys: `epoch`, `step`, `round`,
    `round_`. We accept the first that's present per row and report which
    key won via the returned `x_axis` field. Likewise, different impls
    log `test_*`, `val_*`, or `*_loss`. We collect all available series
    so the plotter can fall back when one is absent."""
    summary = run.get("summary_csv")
    if not summary:
        return None
    p = Path(summary)
    # Replace `.summary.csv` with `.history.jsonl`
    hist = p.with_name(p.name.replace(".summary.csv", ".history.jsonl"))
    if not hist.exists():
        return None
    epochs, train, val, test = [], [], [], []
    metric_kind = run.get("metric_kind", "mse")
    x_keys_seen: list[str] = []
    with hist.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Accept any of these as the x-axis, in priority order.
            x_val = None
            for k in ("epoch", "step", "round_", "round"):
                if k in r:
                    x_val = r[k]
                    if k not in x_keys_seen:
                        x_keys_seen.append(k)
                    break
            if x_val is None:
                continue
            epochs.append(float(x_val))
            tr_v = r.get("train_mse", r.get("train_metric", r.get("train_loss")))
            va_v = r.get("val_mse",   r.get("val_metric",   r.get("val_acc",
                   r.get("val_loss"))))
            te_v = r.get("test_mse",  r.get("test_metric",  r.get("test_acc",
                   r.get("test_loss"))))
            train.append(float(tr_v) if tr_v is not None else float("nan"))
            val.append(  float(va_v) if va_v is not None else float("nan"))
            test.append( float(te_v) if te_v is not None else float("nan"))
    # Pick a single, human-readable x-axis name for the plot.
    x_axis = x_keys_seen[0] if x_keys_seen else "epoch"
    return {"epoch": epochs, "train": train, "val": val, "test": test,
            "metric_kind": metric_kind, "x_axis": x_axis}


def _is_higher_better(metric_kind: str) -> bool:
    return "acc" in metric_kind.lower()


def plot_accuracy(runs: list[dict]) -> None:
    """One figure per benchmark, showing per-impl training trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_bench: dict[str, list[dict]] = {}
    for r in runs:
        by_bench.setdefault(r["bench"], []).append(r)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for bench, rs in sorted(by_bench.items()):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        any_curve = False
        metric_kind = "mse"
        x_axis = "epoch"
        used_val_only = False
        for impl in IMPL_ORDER:
            for r in rs:
                if r["impl"] != impl:
                    continue
                h = _load_history_for(r)
                if h is None or not h["epoch"]:
                    continue
                metric_kind = h["metric_kind"]
                x_axis = h.get("x_axis", "epoch")
                ep = np.array(h["epoch"])
                test_arr = np.array(h["test"])
                val_arr = np.array(h["val"])
                # Prefer test curve; fall back to val if test is all-NaN.
                if np.all(np.isnan(test_arr)) and not np.all(np.isnan(val_arr)):
                    yvals = val_arr
                    label = f"{IMPL_LABEL[impl]} [val]"
                    used_val_only = True
                else:
                    yvals = test_arr
                    label = IMPL_LABEL[impl]
                if np.all(np.isnan(yvals)):
                    continue
                ax.plot(ep, yvals, color=IMPL_COLOUR[impl],
                        marker=IMPL_MARKER[impl], markersize=4,
                        linewidth=1.6, label=label)
                any_curve = True
                break
        if not any_curve:
            plt.close(fig)
            continue
        if _is_higher_better(metric_kind):
            ylabel = "accuracy"
        else:
            ylabel = "loss / MSE" if used_val_only else "test MSE"
        ax.set_xlabel(x_axis)
        ax.set_ylabel(ylabel)
        if not _is_higher_better(metric_kind):
            ax.set_yscale("log")
        ax.set_title(f"{bench}  —  training trajectory across implementations")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", frameon=False)
        out = PLOTS_DIR / f"{bench}_accuracy.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] {out}")


def plot_overlay(runs: list[dict]) -> None:
    """Single figure: each (bench, impl) gets a normalised curve. Useful for
    comparing convergence shape across benches at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_bench: dict[str, list[dict]] = {}
    for r in runs:
        by_bench.setdefault(r["bench"], []).append(r)

    benches = sorted(by_bench)
    n = len(benches)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows),
                              squeeze=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for idx, bench in enumerate(benches):
        ax = axes[idx // cols][idx % cols]
        any_curve = False
        metric_kind = "mse"
        for impl in IMPL_ORDER:
            for r in by_bench[bench]:
                if r["impl"] != impl:
                    continue
                h = _load_history_for(r)
                if h is None or not h["epoch"]:
                    continue
                metric_kind = h["metric_kind"]
                yvals = np.array(h["test"])
                # Normalise so the worst point is 1.0 and the best is 0 — this
                # makes shape comparable across very different absolute scales.
                if _is_higher_better(metric_kind):
                    yvals = 1.0 - yvals
                lo, hi = float(np.nanmin(yvals)), float(np.nanmax(yvals))
                norm = (yvals - lo) / max(hi - lo, 1e-9)
                ax.plot(h["epoch"], norm, color=IMPL_COLOUR[impl],
                        marker=IMPL_MARKER[impl], markersize=3,
                        linewidth=1.4, label=IMPL_LABEL[impl])
                any_curve = True
                break
        ax.set_title(bench, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
        if not any_curve:
            ax.text(0.5, 0.5, "no history available",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#999999", fontsize=9)
        if idx == 0:
            ax.legend(loc="best", frameon=False, fontsize=8)
    # Hide any leftover axes.
    for j in range(len(benches), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Cross-benchmark overlay (normalised test metric, lower=better)")
    fig.supxlabel("epoch"); fig.supylabel("normalised metric")
    out = PLOTS_DIR / "overlay_all_benches.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def _grouped_bar(ax, by_bench: dict[str, dict[str, float]],
                  benches: list[str], impls: list[str],
                  value_fmt: str = "{:.2f}") -> None:
    """Helper: render a grouped bar chart on `ax`. The per-impl bar order
    matches IMPL_ORDER for visual consistency across plots."""
    import numpy as np
    width = 0.8 / max(len(impls), 1)
    xs = np.arange(len(benches))
    for k, impl in enumerate(impls):
        offs = (k - (len(impls) - 1) / 2) * width
        vals = [by_bench[b].get(impl, float("nan")) for b in benches]
        bars = ax.bar(xs + offs, vals, width, color=IMPL_COLOUR[impl],
                       edgecolor="white", linewidth=0.5,
                       label=IMPL_LABEL[impl])
        for x, v in zip(xs + offs, vals):
            if v == v and v > 0:
                ax.text(x, v, value_fmt.format(v),
                        ha="center", va="bottom", fontsize=7,
                        color=IMPL_COLOUR[impl])
    ax.set_xticks(xs)
    ax.set_xticklabels(benches, rotation=25, ha="right", fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def plot_walltime(runs: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_bench: dict[str, dict[str, float]] = {}
    for r in runs:
        by_bench.setdefault(r["bench"], {})[r["impl"]] = float(r["wall_seconds"])
    benches = sorted(by_bench)
    impls = [i for i in IMPL_ORDER
             if any(i in by_bench[b] for b in benches)]
    if not benches or not impls:
        print("[plot] walltime: nothing to plot")
        return

    fig, (ax_tbl, ax_bar) = plt.subplots(2, 1, figsize=(11, 7),
                                           gridspec_kw={"height_ratios": [1, 2]})
    # Table
    ax_tbl.axis("off")
    header = ["benchmark", *(IMPL_LABEL[i] for i in impls)]
    rows = []
    for b in benches:
        row = [b]
        for i in impls:
            v = by_bench[b].get(i)
            row.append(f"{v:.3f}s" if v is not None and v == v else "—")
        rows.append(row)
    table = ax_tbl.table(cellText=rows, colLabels=header,
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax_tbl.set_title("Wall-clock time per benchmark / implementation",
                     loc="left", fontsize=11, pad=8)

    # Bar chart
    _grouped_bar(ax_bar, by_bench, benches, impls, value_fmt="{:.2f}s")
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("wall time (s, log scale)")

    out = PLOTS_DIR / "walltime.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_memory(runs: list[dict]) -> None:
    """Two panels: peak RSS (max sample of VmRSS summed across descendants)
    and average RSS (arithmetic mean of the same series). Both grouped by
    bench/impl on a log scale, with the per-bench tables underneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak_by_bench: dict[str, dict[str, float]] = {}
    mean_by_bench: dict[str, dict[str, float]] = {}
    for r in runs:
        peak = float(r["peak_rss_kb"]) / 1024.0   # MiB
        mean = float(r.get("mean_rss_kb", 0) or 0) / 1024.0
        peak_by_bench.setdefault(r["bench"], {})[r["impl"]] = peak
        mean_by_bench.setdefault(r["bench"], {})[r["impl"]] = mean
    benches = sorted(peak_by_bench)
    impls = [i for i in IMPL_ORDER
             if any(i in peak_by_bench[b] for b in benches)]
    if not benches or not impls:
        print("[plot] memory: nothing to plot")
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9),
        gridspec_kw={"height_ratios": [1, 2], "width_ratios": [1, 1]},
    )
    # ---- tables ----
    for col, (title, data) in enumerate(
        [("Peak RSS (MiB)", peak_by_bench),
         ("Average RSS (MiB)", mean_by_bench)],
    ):
        ax_tbl = axes[0][col]
        ax_tbl.axis("off")
        header = ["benchmark", *(IMPL_LABEL[i] for i in impls)]
        rows = []
        for b in benches:
            row = [b]
            for i in impls:
                v = data[b].get(i)
                row.append(f"{v:.1f}" if v is not None and v > 0 else "—")
            rows.append(row)
        table = ax_tbl.table(cellText=rows, colLabels=header,
                              loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax_tbl.set_title(title, loc="left", fontsize=11, pad=8)

    # ---- bars ----
    for col, (title, data) in enumerate(
        [("peak RSS (MiB, log)", peak_by_bench),
         ("mean RSS (MiB, log)", mean_by_bench)],
    ):
        ax = axes[1][col]
        _grouped_bar(ax, data, benches, impls, value_fmt="{:.0f}M")
        ax.set_yscale("log")
        ax.set_ylabel(title)

    fig.suptitle("Resident-set-size during training "
                 "(VmRSS polled across all descendants, 50 ms interval)")
    out = PLOTS_DIR / "memory.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_phase_stats(runs: list[dict]) -> None:
    """For each bench, one panel per (bench, impl) showing forward / loss /
    backward / update / structural / reset as grouped bars with std error
    bars. Mirrors plot_phases but lays out one column per phase so the
    eye can read the *consistency* of each phase (not just its mean cost)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PHASES = [
        ("forward_ns",    "forward"),
        ("loss_ns",       "loss"),
        ("backward_ns",   "backward"),
        ("update_ns",     "update"),
        ("prune_ns",      "prune"),
        ("grow_ns",       "grow"),
        ("reset_ns",      "reset"),
    ]

    def _has_phase(r):
        v = r.get("step_ns_mean")
        return v not in (None, "") and v == v  # nan filter

    instr = [r for r in runs if _has_phase(r)]
    if not instr:
        print("[plot] phase_stats: no instrumented runs")
        return

    by_bench: dict[str, dict[str, dict]] = {}
    for r in instr:
        by_bench.setdefault(r["bench"], {})[r["impl"]] = r

    benches = sorted(by_bench)
    n = len(benches)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.6 * n), squeeze=False)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for bi, bench in enumerate(benches):
        ax = axes[bi][0]
        impls = [i for i in IMPL_ORDER if i in by_bench[bench]]
        if not impls:
            continue
        xs = np.arange(len(PHASES))
        width = 0.78 / max(len(impls), 1)
        for ii, impl in enumerate(impls):
            r = by_bench[bench][impl]
            means = np.array([float(r.get(f"{k}_mean", 0.0) or 0.0)
                              for k, _ in PHASES])
            # std column is `<phase>_ns_std`. Older summaries (e.g. 09 before
            # the std rollout) may not have it; default to 0 so the bar shows
            # without a hat.
            stds = np.array([float(r.get(f"{k}_std", 0.0) or 0.0)
                             for k, _ in PHASES])
            # Replace zero with NaN so the log-scale axis doesn't choke; the
            # bar itself is hidden behind the y-axis floor.
            display = np.where(means > 0, means, np.nan)
            offset = (ii - (len(impls) - 1) / 2) * width
            ax.bar(xs + offset, display, width,
                   yerr=stds, capsize=2.5,
                   error_kw={"elinewidth": 0.8, "ecolor": "#444"},
                   color=IMPL_COLOUR[impl], edgecolor="white",
                   linewidth=0.4, label=IMPL_LABEL[impl])
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in PHASES], fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel("ns / step (mean ± σ, log)")
        steps = int(float(next(iter(by_bench[bench].values())).get(
            "step_count", 0) or 0))
        ax.set_title(f"{bench}  ·  per-phase ns (avg over {steps:,} steps)")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if bi == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Per-phase latency distribution "
                 "(error bars = std deviation across optimisation steps)")
    fig.tight_layout()
    out = PLOTS_DIR / "phase_stats.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_phases(runs: list[dict]) -> None:
    """Stacked-bar breakdown of per-step ns by phase, one bar per (bench, impl).
    Benches that don't emit phase columns are silently skipped — `phases` only
    makes sense once a bench has been instrumented."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PHASE_PLOT_KEYS = ("forward_ns_mean", "loss_ns_mean",
                       "backward_ns_mean", "update_ns_mean",
                       "prune_ns_mean", "grow_ns_mean", "reset_ns_mean",
                       "other_ns_mean")
    PHASE_LABELS = {
        "forward_ns_mean":    "forward",
        "loss_ns_mean":       "loss",
        "backward_ns_mean":   "backward",
        "update_ns_mean":     "update (opt step)",
        "prune_ns_mean":      "prune (Prune*)",
        "grow_ns_mean":       "grow (Add*)",
        "reset_ns_mean":      "reset",
        "other_ns_mean":      "harness / Python",
    }
    PHASE_COLOURS = {
        "forward_ns_mean":    "#5fa8d3",
        "loss_ns_mean":       "#8aacd6",
        "backward_ns_mean":   "#f4a261",
        "update_ns_mean":     "#f0c987",
        "prune_ns_mean":      "#e76f51",
        "grow_ns_mean":       "#c1440e",
        "reset_ns_mean":      "#bfbfbf",
        "other_ns_mean":      "#bdb2ff",
    }

    # Restrict to runs that actually carry a phase breakdown.
    instr_runs = [r for r in runs if r.get("step_ns_mean") not in (None, "")
                  and r.get("step_ns_mean") == r.get("step_ns_mean")]  # nan filter
    if not instr_runs:
        print("[plot] phases: no instrumented runs in runs.csv")
        return

    by_bench: dict[str, dict[str, dict[str, float]]] = {}
    for r in instr_runs:
        by_bench.setdefault(r["bench"], {})[r["impl"]] = r
    benches = sorted(by_bench)

    n = len(benches)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.2 * n), squeeze=False)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for bi, bench in enumerate(benches):
        ax = axes[bi][0]
        impls = [i for i in IMPL_ORDER if i in by_bench[bench]]
        if not impls:
            continue
        xs = np.arange(len(impls))
        bottoms = np.zeros(len(impls))
        for key in PHASE_PLOT_KEYS:
            vals = np.array(
                [float(by_bench[bench][i].get(key, 0.0) or 0.0)
                 for i in impls])
            if not np.any(vals > 0):
                continue
            ax.bar(xs, vals, bottom=bottoms,
                   color=PHASE_COLOURS[key], edgecolor="white",
                   linewidth=0.6, label=PHASE_LABELS[key])
            bottoms += vals
        # Total step time annotation above each bar.
        for x, impl in zip(xs, impls):
            total = float(by_bench[bench][impl].get("step_ns_mean", 0.0) or 0.0)
            ax.text(x, total, f"{total/1000:.1f}µs", ha="center", va="bottom",
                    fontsize=8, color="#333")
        ax.set_xticks(xs)
        ax.set_xticklabels([IMPL_LABEL[i] for i in impls])
        ax.set_yscale("log")
        ax.set_ylabel("ns / step (log)")
        ax.set_title(f"{bench} — per-step phase breakdown "
                     f"(mean over {int(float(by_bench[bench][impls[0]].get('step_count', 0) or 0))} steps)")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if bi == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8, ncols=2)

    fig.suptitle("Per-step latency breakdown by phase "
                 "(lower=less framework overhead)")
    fig.tight_layout()
    out = PLOTS_DIR / "phases.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def cmd_plot(args) -> None:
    runs = _load_runs()
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if "all" in kinds:
        kinds = {"accuracy", "overlay", "walltime", "memory",
                 "phases", "phase_stats"}
    dispatch = {
        "accuracy":    plot_accuracy,
        "overlay":     plot_overlay,
        "walltime":    plot_walltime,
        "memory":      plot_memory,
        "phases":      plot_phases,
        "phase_stats": plot_phase_stats,
    }
    for k in ("accuracy", "overlay", "walltime", "memory",
              "phases", "phase_stats"):
        if k in kinds:
            dispatch[k](runs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="enumerate discovered sentinels")
    sp_list.set_defaults(func=cmd_list)

    sp_run = sub.add_parser("run", help="execute one or more benchmarks")
    sp_run.add_argument("--bench", default=None,
                        help="comma-separated bench filter (default: all)")
    sp_run.add_argument("--impl", default=None,
                        help="comma-separated impl filter (default: all)")
    sp_run.add_argument("--tag", default="orch")
    sp_run.add_argument("--build-dir", type=Path,
                        default=Path("build-host"),
                        help="where C++ binaries live "
                             "(layout matches the source tree)")
    sp_run.add_argument("--quick", action="store_true",
                        help="pass --quick to each sentinel")
    sp_run.add_argument("--device", default="cpu",
                        help="device for the pytorch impls (cpu|cuda); C++ and "
                             "plastix impls are CPU/GPU by their build dir")
    sp_run.add_argument("--timeout", type=float, default=600.0,
                        help="per-benchmark wall-clock cap in seconds "
                             "(default 600 = 10 min); exceeding it kills the "
                             "run and skips it")
    sp_run.add_argument("--quiet", action="store_true")
    sp_run.add_argument("passthrough", nargs="*",
                        help="extra args forwarded to every sentinel")
    sp_run.set_defaults(func=cmd_run)

    sp_plot = sub.add_parser("plot", help="render the standard plots")
    sp_plot.add_argument("--kinds", default="all",
                         help="comma-separated subset of {accuracy, overlay, "
                              "walltime, memory, phases, phase_stats, all}")
    sp_plot.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
