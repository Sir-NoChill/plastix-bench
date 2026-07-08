#!/usr/bin/env python3
"""Aggregate per-phase benchmark timings into the paper's phase-fraction table.

Reads the orchestrator's archived aggregates (`_results/runs_*.csv`) and emits
one row per benchmark with, for each framework, the FRACTION of total per-step
runtime spent in each phase bucket:

    fwd    forward
    bwd    backward + loss              (loss is folded into backward)
    upd    update (optimiser step)
    prune  DoPrune* / feature removal   (the shrink half of structural)
    grow   DoAdd*   / feature generation (the spawn half of structural)
    uncat  reset + harness/other slack  (everything not attributed above)

The six buckets are percentages that sum to 100 per framework, so the table
drops straight into a stacked-bar chart. The denominator is the full per-step
wall time (`step_ns_mean`), so `uncat` honestly carries the data-movement /
Python-loop / `.item()`-sync overhead that the phase marks don't cover.

Source mapping — no single archive holds all four impls (cpp is CPU-only, the
dedicated cuda impl is GPU-only), so the four framework columns are drawn from
two archives:

    plastix, pytorch, cpp   <-  _results/runs_cpu.csv   (tag=cpu)
    cuda                    <-  _results/runs_gpu.csv   (tag=gpu, dedicated CUDA impl)

The trailing columns (gen_neuron ... apriori) are intrinsic properties of each
benchmark's algorithm rather than measurements; see BENCH_CHARACTERISTICS below
and adjust to taste. Run via `just phase-table`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from bench_meta import (CHARACTERISTIC_COLS, BENCH_CHARACTERISTICS, FRAMEWORKS,
                        GPU_FRAMEWORKS, RESULTS, acronym, collect_archives,
                        f as _f)

# The five phase buckets emitted per framework, plus the uncategorised catch-all.
PHASE_BUCKETS = ("fwd", "bwd", "upd", "prune", "grow", "uncat")


def phase_fractions(row: dict) -> dict[str, float]:
    """Collapse a runs.csv row's per-phase ns/step means into the six paper
    buckets, returned as percentages that sum to 100."""
    fwd = _f(row, "forward_ns_mean")
    bwd = _f(row, "backward_ns_mean") + _f(row, "loss_ns_mean")   # loss folded in
    upd = _f(row, "update_ns_mean")
    prune = _f(row, "prune_ns_mean")
    grow = _f(row, "grow_ns_mean")
    # Legacy archives (pre-split) only carry `structural`; fold it into grow so
    # nothing is silently dropped. Current impls leave structural absent.
    legacy = _f(row, "structural_ns_mean")
    if prune == 0.0 and grow == 0.0 and legacy:
        grow = legacy
    uncat = _f(row, "reset_ns_mean") + _f(row, "other_ns_mean")

    raw = {"fwd": fwd, "bwd": bwd, "upd": upd,
           "prune": prune, "grow": grow, "uncat": uncat}
    total = sum(raw.values())
    if total <= 0.0:
        return {k: 0.0 for k in PHASE_BUCKETS}
    return {k: round(100.0 * v / total, 2) for k, v in raw.items()}


def build_table(cpu_csv: Path, gpu_csv: Path,
                frameworks=FRAMEWORKS) -> tuple[list[str], list[list]]:
    archives = collect_archives(cpu_csv, gpu_csv)

    # Benchmarks present anywhere, in sorted (numeric-prefix) order.
    benches = sorted({bench for arch in archives.values() for (bench, _) in arch})

    header = ["benchmark"]
    for fw, _, _ in frameworks:
        header += [f"{fw}_{b}" for b in PHASE_BUCKETS]
    header += list(CHARACTERISTIC_COLS)

    rows: list[list] = []
    for bench in benches:
        row: list = [acronym(bench)]
        for fw, fn, impl in frameworks:
            src = archives[fn].get((bench, impl))
            if src is None:
                row += [""] * len(PHASE_BUCKETS)        # impl absent for this bench
                continue
            frac = phase_fractions(src)
            row += [frac[b] for b in PHASE_BUCKETS]
        chars = BENCH_CHARACTERISTICS.get(bench)
        if chars is None:
            print(f"[phase-table] WARNING: no characteristics for {bench}; "
                  f"leaving blank", file=sys.stderr)
            row += [""] * len(CHARACTERISTIC_COLS)
        else:
            row += list(chars)
        rows.append(row)
    return header, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-csv", type=Path, default=RESULTS / "runs_cpu.csv",
                    help="archive for plastix/pytorch/cpp columns (default: %(default)s)")
    ap.add_argument("--gpu-csv", type=Path, default=RESULTS / "runs_gpu.csv",
                    help="archive for the cuda column (default: %(default)s)")
    ap.add_argument("--gpu", action="store_true",
                    help="source EVERY framework from the GPU pass (runs_gpu.csv); "
                         "writes phase_table_gpu.csv")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV path (default: phase_table[_gpu].csv)")
    args = ap.parse_args()

    frameworks = GPU_FRAMEWORKS if args.gpu else FRAMEWORKS
    if args.out is None:
        args.out = RESULTS / ("phase_table_gpu.csv" if args.gpu else "phase_table.csv")
    header, rows = build_table(args.cpu_csv, args.gpu_csv, frameworks)
    if not rows:
        print("[phase-table] no benchmark rows found — run the suite first "
              "(just run-cpu)", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[phase-table] wrote {args.out}  ({len(rows)} benchmarks)")


if __name__ == "__main__":
    main()
