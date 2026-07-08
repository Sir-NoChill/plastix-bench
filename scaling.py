#!/usr/bin/env python3
"""Scaling sweep driver — Plastix vs PyTorch, 10k → 10M neurons.

Drives the 11_scaling_imprint bench across a log-spaced sweep of network sizes,
one subprocess per (size, framework) so each gets a clean isolated peak-RSS
reading (polled from /proc, reusing orchestrator._poll_memory). Emits a tidy
long-format CSV that pivots directly into two line charts — compute (wall per
step vs neurons) and memory (peak RSS vs neurons) — for each framework.

If a framework exhausts memory / times out at some size, that size is recorded
with a non-`ok` status and the framework is NOT scaled any further (its ceiling
is the last `ok` row). This is opt-in (`just scaling`); it is heavy and is NOT
part of run-cpu / run-gpu.

Output columns (long format):
    neurons, framework, steps, n_units, n_edges,
    wall_per_step_ns, throughput_steps_per_s, peak_rss_mb, status
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import orchestrator as orch  # reuse _poll_memory / _read_one_row_csv
from bench_meta import HERE, RESULTS

DEFAULT_NEURONS = [10_000, 30_000, 100_000, 300_000,
                   1_000_000, 3_000_000, 10_000_000]
FRAMEWORKS = ("plastix", "pytorch")
BENCH = "11_scaling_imprint"


def _summary_for(out_dir: Path, tag: str) -> dict | None:
    hits = sorted(out_dir.glob(f"*{tag}*.summary.csv"))
    if not hits:
        return None
    return orch._read_one_row_csv(hits[-1])


def run_point(framework: str, neurons: int, steps: int, build_dir: str,
              timeout: float) -> dict:
    out_dir = RESULTS / BENCH / framework
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{neurons}"
    script = HERE / BENCH / framework / "run_benchmark.py"
    cmd = ["uv", "run", "python", str(script),
           "--neurons", str(neurons), "--steps", str(steps),
           "--out-dir", str(out_dir), "--tag", tag, "--seed", "0"]
    if framework == "plastix":
        cmd += ["--build-dir", build_dir,
                "--data-dir", str(HERE / "common/pytorch/data")]
    else:
        cmd += ["--device", "cpu", "--no-plot"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    state, stop, thread = orch._poll_memory(proc.pid)
    status = "ok"
    try:
        out, _ = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            status = "fail"  # most likely OOM at the large sizes
            tail = "\n".join((out or "").strip().splitlines()[-3:])
            print(f"    [{framework} n={neurons}] exit={proc.returncode}: {tail}",
                  file=sys.stderr)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        status = "timeout"
    finally:
        stop.set()
        thread.join(timeout=0.2)

    peak_mb = round(state["peak_kb"] / 1024.0, 1)
    row = {"neurons": neurons, "framework": framework, "steps": steps,
           "n_units": "", "n_edges": "", "wall_per_step_ns": "",
           "throughput_steps_per_s": "", "peak_rss_mb": peak_mb,
           "status": status}
    if status == "ok":
        s = _summary_for(out_dir, tag)
        if s:
            try:
                step_ns = float(s.get("step_ns_mean") or 0.0)
            except ValueError:
                step_ns = 0.0
            row["n_units"] = s.get("n_units", "")
            row["n_edges"] = s.get("n_edges", "")
            row["wall_per_step_ns"] = round(step_ns, 1)
            row["throughput_steps_per_s"] = (round(1e9 / step_ns, 1)
                                             if step_ns > 0 else "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neurons", type=int, nargs="*", default=DEFAULT_NEURONS,
                    help="size sweep (default: 10k..10M log-spaced)")
    ap.add_argument("--max-neurons", type=int, default=None,
                    help="drop sweep points above this (handy for a smoke run)")
    ap.add_argument("--steps", type=int, default=200,
                    help="steps per point — enough for a stable per-step mean")
    ap.add_argument("--build-dir", default="build", help="plastix binary build dir")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-point wall-clock cap (seconds)")
    ap.add_argument("-o", "--out", type=Path, default=RESULTS / "scaling_table.csv")
    args = ap.parse_args()

    sizes = sorted(set(args.neurons))
    if args.max_neurons:
        sizes = [n for n in sizes if n <= args.max_neurons]

    rows: list[dict] = []
    for fw in FRAMEWORKS:
        for n in sizes:
            print(f"==> [{fw}] neurons={n:,} steps={args.steps}")
            r = run_point(fw, n, args.steps, args.build_dir, args.timeout)
            rows.append(r)
            print(f"    -> status={r['status']} "
                  f"wall/step={r['wall_per_step_ns']}ns "
                  f"peak={r['peak_rss_mb']}MiB")
            if r["status"] != "ok":
                print(f"    [{fw}] ceiling reached at {n:,} neurons "
                      f"({r['status']}); not scaling {fw} further.")
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["neurons", "framework", "steps", "n_units", "n_edges",
            "wall_per_step_ns", "throughput_steps_per_s", "peak_rss_mb", "status"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[scaling] wrote {args.out}  ({len(rows)} points)")


if __name__ == "__main__":
    main()
