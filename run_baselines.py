#!/usr/bin/env python3
"""Baseline comparison driver — run the bio-inspired / alt-framework baselines.

Runs each `baselines/<fw>/run_baseline.py` as its own subprocess (so peak RSS is
a clean per-framework reading, polled via orchestrator._poll_memory) and collects
their one-row outputs into `_results/baselines_table.csv`.

These frameworks (TensorNEAT, an SNN stack, Nengo, evosax) are paradigm-different
from the per-step Plastix interface, so they live in this SEPARATE comparison
table rather than the phase/memory tables. Frameworks whose optional dependency
is not installed report `status=skipped` — the harness still runs end-to-end.

    just baselines                              # run all discovered baselines
    uv run python run_baselines.py --only snn   # just one
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import orchestrator as orch  # reuse _poll_memory / _read_one_row_csv
from baselines.common import ROW_COLUMNS
from bench_meta import HERE, RESULTS

BASELINES_DIR = HERE / "baselines"


def discover() -> list[str]:
    return sorted(p.parent.name for p in BASELINES_DIR.glob("*/run_baseline.py"))


def run_one(fw: str, steps_flags: list[str], timeout: float) -> dict:
    script = BASELINES_DIR / fw / "run_baseline.py"
    with tempfile.TemporaryDirectory() as td:
        rowcsv = Path(td) / "row.csv"
        cmd = ["uv", "run", "python", str(script), "--seed", "0",
               "--out", str(rowcsv), *steps_flags]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        state, stop, thread = orch._poll_memory(proc.pid)
        try:
            out, _ = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate(); rc = -1; out = "(timeout)"
        finally:
            stop.set(); thread.join(timeout=0.2)

        row = orch._read_one_row_csv(rowcsv) if rowcsv.exists() else None
        if row is None:
            tail = "\n".join((out or "").strip().splitlines()[-3:])
            print(f"    [{fw}] no row emitted (exit {rc}): {tail}", file=sys.stderr)
            row = {"framework": fw, "status": "error", "notes": f"exit {rc}"}
    # Attach the externally-polled peak RSS (the stubs can't self-report it).
    if not row.get("peak_rss_mb"):
        row["peak_rss_mb"] = round(state["peak_kb"] / 1024.0, 1)
    return {c: row.get(c, "") for c in ROW_COLUMNS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="subset of frameworks to run")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("-o", "--out", type=Path, default=RESULTS / "baselines_table.csv")
    args = ap.parse_args()

    fws = discover()
    if args.only:
        fws = [f for f in fws if f in set(args.only)]
    if not fws:
        sys.exit("[baselines] nothing to run (no baselines/*/run_baseline.py)")

    flags = ["--quick"] if args.quick else []
    rows = []
    for fw in fws:
        print(f"==> baseline: {fw}")
        rows.append(run_one(fw, flags, args.timeout))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    by_status = Counter(r.get("status") or "?" for r in rows)
    print(f"[baselines] wrote {args.out}  ({len(rows)} frameworks; "
          + ", ".join(f"{n} {s}" for s, n in sorted(by_status.items())) + ")")


if __name__ == "__main__":
    main()
