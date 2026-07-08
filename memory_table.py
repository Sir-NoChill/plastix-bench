#!/usr/bin/env python3
"""Aggregate the RSS-milestone memory breakdown into the paper's memory table.

Reads the orchestrator's archived aggregates (`_results/runs_*.csv`) and emits
one row per benchmark with, for each framework, resident memory (in MiB) split
into:

    dataset   ΔRSS across the dataset-load block          (MemoryProbe)
    weights   ΔRSS across the model/optimiser-build block  (MemoryProbe)
    overhead  RSS right after startup/imports              (MemoryProbe)
    scratch   max − (dataset+weights+overhead)             (training transients)
    max       peak process RSS                             (orchestrator poller)

Source mapping — same as the phase table (no single archive holds all four
impls): plastix/pytorch/cpp ← runs_cpu.csv, cuda ← runs_gpu.csv. The cuda
column reflects HOST RSS of the dedicated CUDA impl (which keeps canonical
weights on the host); GPU VRAM is not separately broken out.

The trailing columns (gen_neuron ... apriori) are the same intrinsic
characteristics as the phase table (shared from bench_meta). Run via
`just memory-table` (also regenerated automatically by every benchmark pass).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from bench_meta import (CHARACTERISTIC_COLS, BENCH_CHARACTERISTICS, FRAMEWORKS,
                        GPU_FRAMEWORKS, RESULTS, acronym, collect_archives,
                        f as _f)

# Memory buckets per framework. dataset/weights/overhead/scratch/max are the
# host-RSS breakdown (sum to max); vram is peak GPU device memory (0 for CPU
# runs), polled per-process by the orchestrator — orthogonal to the RSS split.
MEM_BUCKETS = ("dataset", "weights", "overhead", "scratch", "max", "vram")


def _mib(kb: float) -> int:
    """kB -> MiB, rounded to the nearest integer (matches the target format)."""
    return int(round(kb / 1024.0))


def memory_breakdown(row: dict) -> dict[str, int]:
    """Collapse a runs.csv row's RSS milestones into the five MiB buckets."""
    overhead = _f(row, "mem_overhead_kb")
    dataset = _f(row, "mem_dataset_kb")
    weights = _f(row, "mem_weights_kb")
    peak = _f(row, "peak_rss_kb")
    # scratch is whatever the peak exceeds the post-model resident set by.
    scratch = max(0.0, peak - (overhead + dataset + weights))
    return {
        "dataset":  _mib(dataset),
        "weights":  _mib(weights),
        "overhead": _mib(overhead),
        "scratch":  _mib(scratch),
        "max":      _mib(peak),
        "vram":     _mib(_f(row, "peak_vram_kb")),   # peak GPU VRAM (0 on CPU)
    }


def build_table(cpu_csv: Path, gpu_csv: Path,
                frameworks=FRAMEWORKS) -> tuple[list[str], list[list]]:
    archives = collect_archives(cpu_csv, gpu_csv)
    benches = sorted({bench for arch in archives.values() for (bench, _) in arch})

    header = ["benchmark"]
    for fw, _, _ in frameworks:
        header += [f"{fw}_{b}" for b in MEM_BUCKETS]
    header += list(CHARACTERISTIC_COLS)

    rows: list[list] = []
    for bench in benches:
        row: list = [acronym(bench)]
        for fw, fn, impl in frameworks:
            src = archives[fn].get((bench, impl))
            if src is None or _f(src, "peak_rss_kb") <= 0.0:
                row += [""] * len(MEM_BUCKETS)     # impl absent / no memory sample
                continue
            mem = memory_breakdown(src)
            row += [mem[b] for b in MEM_BUCKETS]
        chars = BENCH_CHARACTERISTICS.get(bench)
        if chars is None:
            print(f"[memory-table] WARNING: no characteristics for {bench}; "
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
                    help="source EVERY framework from the GPU pass (runs_gpu.csv) "
                         "so vram + GPU timings show for all; writes memory_table_gpu.csv")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV path (default: memory_table[_gpu].csv)")
    args = ap.parse_args()

    frameworks = GPU_FRAMEWORKS if args.gpu else FRAMEWORKS
    if args.out is None:
        args.out = RESULTS / ("memory_table_gpu.csv" if args.gpu else "memory_table.csv")
    header, rows = build_table(args.cpu_csv, args.gpu_csv, frameworks)
    if not rows:
        print("[memory-table] no benchmark rows found — run the suite first "
              "(just run-cpu)", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[memory-table] wrote {args.out}  ({len(rows)} benchmarks)")


if __name__ == "__main__":
    main()
