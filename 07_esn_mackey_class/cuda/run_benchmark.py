"""Sentinel for the raw C++ ESN-on-Mackey-Glass benchmark.

This binary predates the `bench::CliArgs` convention used by the other
C++ benchmarks; its argv layout is positional (`argv[1]` = output dir).
The orchestrator passes flag-shaped args, so we translate here and emit
a `<workload>_<tag>.summary.csv` shape the orchestrator can parse —
mirroring the schema the other impls use.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--build-dir", type=Path, default=Path("build-host"))
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", type=Path, default=Path("_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args, _ignored = p.parse_known_args()

    # Locate the binary using the same layout convention as cpp_wrapper:
    # <build_dir>/<bench>/<impl>/run_benchmark.
    rel = Path(*HERE.parts[-2:])     # <bench>/<impl>
    binary = args.build_dir.resolve() / rel / "run_benchmark"
    if not binary.exists():
        raise SystemExit(f"esn binary missing: {binary}")

    # Stage the output under a tag-suffixed subdir so each run is isolated.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage = args.out_dir / f"esn_stage_{args.tag or 'run'}"
    stage.mkdir(parents=True, exist_ok=True)

    # ESN-MG binary CLI: argv[1]=out_dir, argv[2]=epochs (unused for ridge),
    # argv[3]=series_csv (optional, "" => generate Mackey-Glass internally).
    # `--quick` shrinks nothing here because the ridge solve is one-shot;
    # the binary already runs in well under a second.
    cmd = [str(binary), str(stage), "1", ""]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr[-2000:])
        sys.exit(cp.returncode)

    # Map the binary's summary into the orchestrator's canonical shape,
    # including the per-phase ns/step columns from its companion timing.csv.
    src = stage / "summary.csv"
    dst = args.out_dir / f"esn_mg_{args.tag or 'run'}.summary.csv"
    tcsv = stage / "timing.csv"
    timing = {}
    if tcsv.exists():
        with tcsv.open() as f:
            timing = next(csv.DictReader(f), {})
    def _f(k, default=0.0):
        try:
            return float(timing.get(k, default))
        except (TypeError, ValueError):
            return default
    # "step" here = one reservoir timestep over the training window. The
    # ridge solve is a single closed-form call; we charge it to
    # backward_ns_mean (amortised over n_steps) since that's the algorithm's
    # weight-update analogue.
    n_steps = max(int(_f("n_steps", 1)), 1)
    fwd_total = _f("forward_ns", 0.0)
    upd_total = _f("update_ns", 0.0)
    wall_total_ns = _f("wall_ns", wall * 1e9)
    step_ns_mean = wall_total_ns / n_steps
    fwd_mean = fwd_total / n_steps
    bwd_mean = upd_total / n_steps
    other_mean = max(0.0, step_ns_mean - fwd_mean - bwd_mean)
    if src.exists():
        with src.open() as f:
            row = next(csv.DictReader(f), {})
        wall_s = row.get("wall_seconds", f"{wall:.6f}")
        test_v = (row.get("test_rmse")
                  or row.get("test_nrmse")
                  or row.get("test_mse")
                  or row.get("nrmse")
                  or "")
        cols = ["workload", "wall_seconds", "test_mse", "n_units", "seed",
                "step_count", "step_ns_mean", "forward_ns_mean",
                "loss_ns_mean", "backward_ns_mean", "update_ns_mean",
                "structural_ns_mean", "reset_ns_mean", "other_ns_mean"]
        with dst.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow(["esn_mackey_glass", wall_s, test_v,
                        row.get("reservoir", "0"), args.seed,
                        n_steps, round(step_ns_mean, 3),
                        round(fwd_mean, 3), 0.0,
                        round(bwd_mean, 3), 0.0, 0.0, 0.0,
                        round(other_mean, 3)])
        print(f"[esn-sentinel] wrote {dst}")
    else:
        with dst.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["workload", "wall_seconds", "test_mse", "seed"])
            w.writerow(["esn_mackey_glass", f"{wall:.6f}", "", args.seed])
        print(f"[esn-sentinel] wrote {dst} (no upstream summary found)")


if __name__ == "__main__":
    main()
