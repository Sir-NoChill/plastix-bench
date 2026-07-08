"""Shared helpers for the bio-inspired / alt-framework baseline harnesses.

These baselines are paradigm-different from the per-step Plastix interface
(evolutionary population search, neuromorphic NEF, spiking e-prop), so they do
NOT slot into the phase/memory tables. Instead each `baselines/<fw>/run_baseline.py`
emits ONE row in a flat comparison schema, collected by `baselines.py` into
`_results/baselines_table.csv`.

A stub that can't import its framework should emit `status="skipped"` with an
install hint in `notes` — the harness still runs end-to-end, and the row shows
what's missing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "_results"

# Flat comparison-table schema (one row per framework run).
ROW_COLUMNS = ("framework", "task", "paradigm", "metric", "metric_kind",
               "wall_seconds", "peak_rss_mb", "n_params", "status", "notes")


def add_baseline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=None,
                   help="write this single row as a 1-row CSV (used by baselines.py)")


def try_import(module: str):
    """Import a framework module, or return None (stub emits status=skipped)."""
    try:
        return __import__(module)
    except Exception:  # ImportError, or heavier init failures
        return None


def emit(row: dict, out: Path | None) -> None:
    """Print the row and, if `out` is given, write it as a 1-row CSV."""
    full = {c: row.get(c, "") for c in ROW_COLUMNS}
    print("[baseline] " + "  ".join(f"{k}={full[k]}" for k in ROW_COLUMNS))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
            w.writeheader()
            w.writerow(full)
