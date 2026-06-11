#!/usr/bin/env python3
"""Run the il_audio_pred harness and plot it, with hyperparameter sweeps.

Wraps the two manual steps (run the C++ harness, then plot_il.py) into one
command, and turns any hyperparameter into a comma-separated sweep: the cartesian
product of all multi-valued knobs is run, each producing its own predictions CSV
and overlay PNG named after the config.

Each knob maps to an IL_* env var the harness already reads, so nothing is
recompiled between runs. Paths default relative to the repo, so this works from
anywhere.

Examples:
  # Single default run, last 180 s (the common case):
  python examples/sweep_il.py

  # Sweep eta over three values (3 runs), last 180 s:
  python examples/sweep_il.py --eta 0.05,0.1,0.2

  # Sweep memory delay vs window, full-hour overview, 4 runs in parallel:
  python examples/sweep_il.py --delay-max 20,64 --window-max 3,6 \
      --time-range 0 3600 --no-spectrogram --jobs 4

  # Re-plot a different window without re-running the harness:
  python examples/sweep_il.py --eta 0.05,0.1,0.2 --last 60 --reuse-csv
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Friendly knob name -> (IL_* env var, short label token). Order defines the
# label order. Every knob accepts a comma-separated list for sweeping.
KNOBS = {
    "gamma": ("IL_GAMMA", "g"),
    "lambda": ("IL_LAMBDA", "lam"),
    "alpha": ("IL_ALPHA", "a"),
    "eta": ("IL_ETA", "eta"),
    "epsilon_z": ("IL_EPSILON_Z", "ez"),
    "k_pattern": ("IL_K_PATTERN", "kp"),
    "k_memory": ("IL_K_MEMORY", "km"),
    "delay_min": ("IL_MEMORY_DELAY_MIN", "dmin"),
    "delay_max": ("IL_MEMORY_DELAY_MAX", "dmax"),
    "window_min": ("IL_MEMORY_WINDOW_MIN", "wmin"),
    "window_max": ("IL_MEMORY_WINDOW_MAX", "wmax"),
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def total_duration(data_dir):
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)
    return meta["n_steps"] * meta["step_size"] / meta["sample_rate"]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in KNOBS:
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None,
                        metavar="V[,V...]",
                        help="value or comma-separated sweep for this knob")
    ap.add_argument("--dataset", default=os.path.join(REPO, "examples/output/dataset.bin"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "examples/output"))
    ap.add_argument("--binary", default=os.path.join(REPO, "build/default/il_audio_pred"))
    ap.add_argument("--plot-script", default=os.path.join(REPO, "examples/plot_il.py"))
    ap.add_argument("--python", default=os.path.join(REPO, "examples/venv/bin/python"),
                    help="interpreter for plot_il.py (needs matplotlib/numpy)")
    ap.add_argument("--out-dir", default=None, help="PNG dir (default <data-dir>/plots)")
    ap.add_argument("--csv-dir", default=None, help="CSV dir (default <data-dir>/runs)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole dataset")
    last = ap.add_mutually_exclusive_group()
    last.add_argument("--last", type=float, default=180.0,
                      help="plot the last N seconds (default 180)")
    last.add_argument("--time-range", type=float, nargs=2, metavar=("START", "END"))
    ap.add_argument("--no-spectrogram", action="store_true")
    ap.add_argument("--tag", default=None, help="label prefix for outputs")
    ap.add_argument("--jobs", type=int, default=1, help="configs to run in parallel")
    ap.add_argument("--reuse-csv", action="store_true",
                    help="skip the harness run if the CSV already exists")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    return ap.parse_args()


def build_grid(args):
    """Cartesian product of all knobs the user set, as a list of {knob: value}."""
    set_knobs = {n: args.__dict__[n].split(",") for n in KNOBS if args.__dict__[n]}
    if not set_knobs:
        return [{}]
    names = list(set_knobs)
    return [dict(zip(names, combo)) for combo in itertools.product(*set_knobs.values())]


def label_for(config, tag):
    parts = [tag] if tag else []
    parts += [f"{KNOBS[n][1]}{v}" for n, v in config.items()]
    return "_".join(parts) if parts else "default"


def run_one(config, args, t0, t1, range_tag, out_dir, csv_dir):
    label = label_for(config, args.tag)
    csv_path = os.path.join(csv_dir, f"il_pred_{label}.csv")
    png_path = os.path.join(out_dir, f"il_{range_tag}_{label}.png")

    env = dict(os.environ)
    for name, value in config.items():
        env[KNOBS[name][0]] = value

    run_cmd = [args.binary, args.dataset, str(args.max_steps), csv_path]
    plot_cmd = [args.python, args.plot_script, "--predictions", csv_path,
                "--data-dir", args.data_dir, "--time-range", str(t0), str(t1),
                "--output", png_path]
    if args.no_spectrogram:
        plot_cmd.append("--no-spectrogram")

    if args.dry_run:
        envstr = " ".join(f"{KNOBS[n][0]}={v}" for n, v in config.items())
        print(f"[{label}] {envstr} {' '.join(run_cmd)}")
        print(f"[{label}] {' '.join(plot_cmd)}")
        return label, png_path

    if not (args.reuse_csv and os.path.exists(csv_path)):
        subprocess.run(run_cmd, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(plot_cmd, check=True)
    print(f"[{label}] -> {png_path}")
    return label, png_path


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.data_dir, "plots")
    csv_dir = args.csv_dir or os.path.join(args.data_dir, "runs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if args.time_range:
        t0, t1 = args.time_range
        range_tag = f"{int(t0)}-{int(t1)}s"
    else:
        total = total_duration(args.data_dir)
        t0, t1 = max(0.0, total - args.last), total
        range_tag = f"last{int(args.last)}s"

    grid = build_grid(args)
    print(f"{len(grid)} config(s), time range [{t0:.0f}, {t1:.0f}] s, jobs={args.jobs}")

    results = []
    if args.jobs > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, c, args, t0, t1, range_tag, out_dir, csv_dir)
                       for c in grid]
            results = [f.result() for f in futures]
    else:
        results = [run_one(c, args, t0, t1, range_tag, out_dir, csv_dir) for c in grid]

    if not args.dry_run:
        print("\nsummary:")
        for label, png in results:
            print(f"  {label:24s} {png}")


if __name__ == "__main__":
    sys.exit(main())
