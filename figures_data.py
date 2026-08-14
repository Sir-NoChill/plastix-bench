#!/usr/bin/env python3
"""ETL: archived benchmark results -> pgfplots figure data CSVs (P0-A).

Projects the measured aggregates under `_results/runs_{cpu,gpu}.csv` into the
exact CSV schema each pgfplots figure consumes, so the figures render from
*real* measurements instead of the hand-authored placeholder rows they shipped
with. Output goes under `_results/figures/` by default (mirroring the paper's
`figures/<fig>/data/` layout) so the whole `_results/` tree is a single,
self-contained data root that `just results-tarball` can bundle. Point it
straight at a paper checkout with `--figures-dir /path/to/plastix-paper/figures`.

Two figures have a measured data source today (paths relative to --figures-dir):

  memory_occupancy   <- memory milestones (MiB)        -> memory_occupancy/data/memory.csv
  segmented_bar_perf <- per-work-unit wall time (s)    -> segmented_bar_perf/data/{phase_table,benchmarks}.csv

Both figures are laid out for a FIXED set of eight benchmark columns (the x
ticks, benchmark braces and radars are sized for eight groups), so this script
emits exactly `CANON_BENCHES` in that order. A benchmark/framework with no
archived row is written as empty cells; the figures already render an italic
"data not gathered" stand-in for those, so a partial rerun degrades cleanly.

Design notes that differ from `phase_table.py` / `memory_table.py` (which emit
the paper *summary tables*, a different schema):

  * segmented_bar_perf's base chart plots the *sum* of the phase cells as a
    wall-clock time. We emit seconds PER UNIT OF WORK per phase =
    phase_fraction * wall_seconds / work_units, NOT the 0-100 fractions the
    summary table carries. work_units is the benchmark's SHARED natural work
    axis (epochs / rounds / stream steps; see bench_meta.WORK_AXIS), NOT each
    impl's SGD-granularity-dependent step_count -- so the framework ratios are
    comparable. The figure re-derives the percentage chart itself.
  * We emit a sixth `uncat` phase column (reset + harness/other slack) so the
    decomposition is loss-less and the base-chart sum equals wall / work_units.
    The figure draws it as its own segment; nothing is hidden.
  * Output goes to both figures/segmented_bar_perf/data/phase_table.csv (read by
    the `combined` figure) and .../benchmarks.csv (read by the subplot_* views),
    with identical content.
  * Only the four "core" frameworks the figures draw (plastix/pytorch/cpp/cuda)
    are emitted; jax/snn/norse live only in the summary tables.

Run via `just figures-data` (or `python figures_data.py`). Pure stdlib.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import glob

from bench_meta import (BENCH_CHARACTERISTICS, CHARACTERISTIC_COLS, RESULTS,
                        acronym, collect_archives, f as _f,
                        work_units as _wu, work_unit_label)

# ---------------------------------------------------------------------------
# The eight benchmarks the two figures are laid out for, in column order.
# EDIT HERE to change which benchmarks appear (and re-check the figures' xtick /
# brace counts if you change the LENGTH -- the .tex is sized for eight groups).
# The canonical nine (01-09) minus LTC-sine (06), which bench_meta flags "do not
# headline as a Plastix win" (the cpp/plastix impls are sine-only smoke tasks).
# XL-NN (10) and Scale (11) are excluded here because they are scaling studies
# that belong in a dedicated scaling figure, not this cross-workload comparison.
# ---------------------------------------------------------------------------
CANON_BENCHES = [
    "01_static_etth1",                  # Dense
    "02_idempotent_imp",                # Sparse
    "03_bursty_elec2",                  # Bursty
    "04_continuous_small_appliances",   # Cont-S
    "05_continuous_large_mackey_glass", # Cont-L
    "07_esn_mackey_class",              # ESN
    "08_snn_shd",                       # SNN
    "09_imprintin_learner",             # Imprint
]

# The four frameworks the figures draw, and where each is sourced from. cpp is
# CPU-only; the dedicated cuda impl is GPU-only -- so two archives, as in the
# summary-table scripts.
FIG_FRAMEWORKS = [
    ("plastix", "runs_cpu.csv", "plastix"),
    ("pytorch", "runs_cpu.csv", "pytorch"),
    ("cpp",     "runs_cpu.csv", "cpp"),
    ("cuda",    "runs_gpu.csv", "cuda"),
]

PHASE_BUCKETS = ("fwd", "bwd", "upd", "prune", "grow", "uncat")
MEM_BUCKETS = ("dataset", "weights", "overhead", "scratch", "max")


def _default_figures_dir() -> Path:
    """`_results/figures`, inside the gitignored results tree.

    Keeping the generated figure CSVs under `_results/` makes that tree the
    single, self-contained data root that `just results-tarball` bundles: raw
    runs, aggregate tables, and figure CSVs ship as one tarball. The layout
    under here mirrors the paper's (`<fig>/data/*.csv`), so refreshing the paper
    is a plain recursive copy into `plastix-paper/figures/`. Override the target
    with `--figures-dir` (e.g. to write straight into the paper checkout).
    """
    return RESULTS / "figures"


# ---------------------------------------------------------------------------
# segmented_bar_perf: wall-seconds per unit of work, split by phase
# ---------------------------------------------------------------------------
def _units_for(bench: str, impl: str, fn: str, row: dict) -> float:
    """Shared natural-work-axis count for one (bench, impl) run.

    Prefer the `work_units` column hoisted into runs.csv by the orchestrator
    (present after a rerun). Older archives lack it, so fall back to reading the
    bench's per-run summary.csv, whose axis column (epochs / rounds / max_steps)
    bench_meta knows. Returns 0.0 if neither is available (caller keeps raw wall).
    """
    wu = _f(row, "work_units")
    if wu > 0.0:
        return wu
    tag = "cpu" if "cpu" in fn else "gpu"
    hits = (glob.glob(str(RESULTS / bench / impl / f"*{tag}*.summary.csv"))
            or glob.glob(str(RESULTS / bench / impl / "*.summary.csv")))
    if hits:
        with open(hits[0]) as fh:
            summ = next(csv.DictReader(fh), {})
        return _wu(bench, summ)
    return 0.0


def _phase_seconds(row: dict, units: float) -> dict[str, float]:
    """Per-phase wall time PER UNIT OF WORK (seconds) for one archived run.

    Split the measured wall (`wall_seconds`) across the six phase buckets in
    proportion to their per-step ns means, then divide by the shared work-axis
    count so the figure's base chart reads seconds-per-work-unit (comparable
    across frameworks that use different SGD granularity). Loss folds into
    backward and a legacy `structural` mean folds into grow, per phase_table.py.
    `units <= 0` means the axis is unknown; fall back to raw wall (units=1).
    """
    fwd = _f(row, "forward_ns_mean")
    bwd = _f(row, "backward_ns_mean") + _f(row, "loss_ns_mean")
    upd = _f(row, "update_ns_mean")
    prune = _f(row, "prune_ns_mean")
    grow = _f(row, "grow_ns_mean")
    legacy = _f(row, "structural_ns_mean")
    if prune == 0.0 and grow == 0.0 and legacy:
        grow = legacy
    uncat = _f(row, "reset_ns_mean") + _f(row, "other_ns_mean")

    raw = {"fwd": fwd, "bwd": bwd, "upd": upd,
           "prune": prune, "grow": grow, "uncat": uncat}
    total_ns = sum(raw.values())
    wall = _f(row, "wall_seconds")
    denom = units if units > 0.0 else 1.0
    if total_ns <= 0.0 or wall <= 0.0:
        return {k: 0.0 for k in PHASE_BUCKETS}
    return {k: round(wall * v / total_ns / denom, 6) for k, v in raw.items()}


def build_phase_rows(archives: dict) -> list[list]:
    rows: list[list] = []
    for bench in CANON_BENCHES:
        row: list = [acronym(bench)]
        for _, fn, impl in FIG_FRAMEWORKS:
            src = archives[fn].get((bench, impl))
            if src is None:
                row += [""] * len(PHASE_BUCKETS)
                continue
            units = _units_for(bench, impl, fn, src)
            sec = _phase_seconds(src, units)
            row += [sec[b] for b in PHASE_BUCKETS]
        row += list(BENCH_CHARACTERISTICS.get(bench, [""] * len(CHARACTERISTIC_COLS)))
        rows.append(row)
    return rows


def phase_header() -> list[str]:
    header = ["benchmark"]
    for fw, _, _ in FIG_FRAMEWORKS:
        header += [f"{fw}_{b}" for b in PHASE_BUCKETS]
    return header + list(CHARACTERISTIC_COLS)


# ---------------------------------------------------------------------------
# memory_occupancy: MiB per RSS bucket (mirrors memory_table.py semantics)
# ---------------------------------------------------------------------------
def _mem_mib(row: dict) -> dict[str, int]:
    overhead = _f(row, "mem_overhead_kb")
    dataset = _f(row, "mem_dataset_kb")
    weights = _f(row, "mem_weights_kb")
    peak = _f(row, "peak_rss_kb")
    scratch = max(0.0, peak - (overhead + dataset + weights))
    mib = lambda kb: int(round(kb / 1024.0))
    return {"dataset": mib(dataset), "weights": mib(weights),
            "overhead": mib(overhead), "scratch": mib(scratch), "max": mib(peak)}


def build_memory_rows(archives: dict) -> list[list]:
    rows: list[list] = []
    for bench in CANON_BENCHES:
        row: list = [acronym(bench)]
        for _, fn, impl in FIG_FRAMEWORKS:
            src = archives[fn].get((bench, impl))
            if src is None or _f(src, "peak_rss_kb") <= 0.0:
                row += [""] * len(MEM_BUCKETS)
                continue
            mem = _mem_mib(src)
            row += [mem[b] for b in MEM_BUCKETS]
        row += list(BENCH_CHARACTERISTICS.get(bench, [""] * len(CHARACTERISTIC_COLS)))
        rows.append(row)
    return rows


def memory_header() -> list[str]:
    header = ["benchmark"]
    for fw, _, _ in FIG_FRAMEWORKS:
        header += [f"{fw}_{b}" for b in MEM_BUCKETS]
    return header + list(CHARACTERISTIC_COLS)


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    n_full = sum(1 for r in rows if r[1] != "")
    print(f"[figures-data] wrote {path}  ({len(rows)} rows, {n_full} with data)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-csv", type=Path, default=RESULTS / "runs_cpu.csv")
    ap.add_argument("--gpu-csv", type=Path, default=RESULTS / "runs_gpu.csv")
    ap.add_argument("--figures-dir", type=Path, default=_default_figures_dir(),
                    help="paper figures/ root (default: %(default)s)")
    args = ap.parse_args()

    archives = collect_archives(args.cpu_csv, args.gpu_csv)
    if not any(archives.values()):
        print("[figures-data] no archives found -- run the suite first "
              "(just run-cpu / run-gpu)", file=sys.stderr)
        sys.exit(1)

    # Create the figures dir on demand, but refuse to scatter a whole missing
    # tree (e.g. a mistyped --figures-dir): the PARENT must already exist. For
    # the default (_results/figures) the parent is _results, created by any run.
    if not args.figures_dir.parent.exists():
        print(f"[figures-data] parent of {args.figures_dir} not found -- "
              f"skipping (pass --figures-dir to a real location)",
              file=sys.stderr)
        return

    # The `combined` segmented figure reads phase_table.csv; the subplot_*
    # variants read benchmarks.csv. Emit identical content to both so every
    # variant renders the same real, work-unit-normalized data. (The subplots
    # use the 5-phase schema and ignore the extra `uncat` column.)
    phase_rows = build_phase_rows(archives)
    seg = args.figures_dir / "segmented_bar_perf" / "data"
    _write(seg / "phase_table.csv", phase_header(), phase_rows)
    _write(seg / "benchmarks.csv", phase_header(), phase_rows)
    _write(args.figures_dir / "memory_occupancy" / "data" / "memory.csv",
           memory_header(), build_memory_rows(archives))


if __name__ == "__main__":
    main()
