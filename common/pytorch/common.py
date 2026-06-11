"""
Shared infrastructure for the dynamism-regime benchmark suite.

Provides:
  - StructuralLog : append-only JSONL history of (step, n_units, n_edges,
    topology_hash, val_loss, extras) tuples.
  - edge_jaccard  : Jaccard similarity over consecutive live-edge sets.
  - topology_hash : stable hash of a sorted adjacency list.
  - plot_run      : matplotlib figure with (loss, size, jaccard) panels.
  - add_common_args : CLI args shared across all workloads.
  - download_if_missing : tiny urllib-based fetcher with cached file paths.

All five workload scripts (01_..05_) import this module.  No script should
need a network connection if its dataset is already cached under --data-dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# matplotlib is imported lazily inside plot_run() so that --no-plot runs
# don't pay the import cost (and don't fail if Agg is misconfigured).


# ---------------------------------------------------------------------------
# Topology / structure utilities
# ---------------------------------------------------------------------------

def topology_hash(edges: Iterable[tuple[int, ...]]) -> str:
    """Stable 16-hex-digit hash of a live-edge set.  Order-independent."""
    h = hashlib.blake2b(digest_size=8)
    for e in sorted(edges):
        h.update(repr(e).encode("ascii"))
    return h.hexdigest()


def edge_jaccard(prev: set, curr: set) -> float:
    """|A intersect B| / |A union B|.  Returns 1.0 if both are empty."""
    if not prev and not curr:
        return 1.0
    inter = len(prev & curr)
    union = len(prev | curr)
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------
# Structural history log (JSONL)
# ---------------------------------------------------------------------------

@dataclass
class StructuralLog:
    """Append-only JSONL log of one record per training step.

    Records: {step, n_units, n_edges, topology_hash, val_loss, **extras}.
    The file is opened lazily so a workload that crashes before its first
    record doesn't leave an empty file behind.
    """
    path: Path
    records: list[dict] = field(default_factory=list)
    _prev_edges: set | None = None

    def log(
        self,
        step: int,
        n_units: int,
        n_edges: int,
        edges: set | None = None,
        val_loss: float | None = None,
        **extras: Any,
    ) -> None:
        if edges is None:
            edges = set()
            jacc = 1.0
            thash = ""
        else:
            thash = topology_hash(edges)
            jacc = (
                edge_jaccard(self._prev_edges, edges)
                if self._prev_edges is not None else 1.0
            )
            self._prev_edges = edges
        rec = {
            "step": int(step),
            "n_units": int(n_units),
            "n_edges": int(n_edges),
            "topology_hash": thash,
            "jaccard": jacc,
            "val_loss": None if val_loss is None else float(val_loss),
        }
        rec.update({k: _jsonable(v) for k, v in extras.items()})
        self.records.append(rec)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")


def _jsonable(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_run(records: list[dict], out_path: Path, title: str) -> None:
    """Three-panel figure: validation loss, network size, Jaccard stability.

    Skips panels whose source column is empty (e.g. val_loss==None or
    Jaccard always 1.0 for the Static workload)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        print("[warn] no records to plot", file=sys.stderr)
        return
    steps = [r["step"] for r in records]
    losses = [r.get("val_loss") for r in records]
    n_units = [r["n_units"] for r in records]
    n_edges = [r["n_edges"] for r in records]
    jaccs = [r.get("jaccard", 1.0) for r in records]
    have_loss = any(v is not None for v in losses)

    n_panels = 1 + int(have_loss) + 1  # size always plotted; jaccard always
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    i = 0
    if have_loss:
        ax = axes[i]; i += 1
        ax.plot(steps, [np.nan if v is None else v for v in losses], color="tab:blue")
        ax.set_ylabel("val loss")
        ax.grid(True, alpha=0.3)

    ax = axes[i]; i += 1
    ax.plot(steps, n_units, color="tab:orange", label="units")
    ax2 = ax.twinx()
    ax2.plot(steps, n_edges, color="tab:green", label="edges", linestyle="--")
    ax.set_ylabel("# units", color="tab:orange")
    ax2.set_ylabel("# edges", color="tab:green")
    ax.grid(True, alpha=0.3)

    ax = axes[i]; i += 1
    ax.plot(steps, jaccs, color="tab:red")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("edge Jaccard\n(step vs prev)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    rows: list[dict],
    path: Path,
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# PhaseTimer — per-step phase nanosecond accumulator. Mirrors the C++ helper
# under common/{cpp,plastix}/common.hpp so the summary CSV (and
# downstream runs.csv) carries the same canonical phase columns regardless of
# which framework the bench is written against.
# ---------------------------------------------------------------------------

import math as _math


class _Welford:
    """Online mean + variance accumulator. Per-sample overhead is ~150ns in
    pure-Python (mostly attribute access); negligible against perf_counter_ns
    and tensor-op dispatch."""
    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        return _math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


class PhaseTimer:
    """Track per-step ns spent in named phases of the training inner loop,
    including the standard deviation across steps (Welford's algorithm).

    Usage in a typical PyTorch training step:

        timer = PhaseTimer()
        for batch in loader:
            timer.tick()
            pred = model(x)
            timer.mark_forward()
            loss = F.mse_loss(pred, y)
            timer.mark_loss()
            loss.backward()
            timer.mark_backward()
            opt.step(); opt.zero_grad()
            timer.mark_update()
            timer.step_done()

    `summary_fields(wall_seconds)` returns a dict of {column: value} suitable
    for splatting into the bench's summary dict before write_summary_csv.
    `step_ns_mean` is computed from wall_seconds, so "other" absorbs anything
    that happens between tick()/step_done() but outside the mark_* calls
    (data loading, batching, .item() syncs, host<->device transfers).
    """

    def __init__(self) -> None:
        self._forward = _Welford()
        self._loss = _Welford()
        self._backward = _Welford()
        self._update = _Welford()
        self._structural = _Welford()
        self._reset = _Welford()
        self._steps = 0
        self._last: int | None = None

    def tick(self) -> None:
        self._last = time.perf_counter_ns()

    def _delta(self) -> int:
        now = time.perf_counter_ns()
        assert self._last is not None, "PhaseTimer.tick() not called"
        dt = now - self._last
        self._last = now
        return dt

    def mark_forward(self)    -> None: self._forward.add(self._delta())
    def mark_loss(self)       -> None: self._loss.add(self._delta())
    def mark_backward(self)   -> None: self._backward.add(self._delta())
    def mark_update(self)     -> None: self._update.add(self._delta())
    def mark_structural(self) -> None: self._structural.add(self._delta())
    def mark_reset(self)      -> None: self._reset.add(self._delta())
    def step_done(self)       -> None: self._steps += 1

    @property
    def step_count(self) -> int:
        return self._steps

    def summary_fields(self, wall_seconds: float) -> dict:
        steps = max(self._steps, 1)
        step_ns_mean = (wall_seconds * 1e9) / steps
        accounted = (self._forward.mean + self._loss.mean +
                     self._backward.mean + self._update.mean +
                     self._structural.mean + self._reset.mean)
        return {
            "step_count":         int(steps),
            "step_ns_mean":       round(step_ns_mean, 3),
            "forward_ns_mean":    round(self._forward.mean, 3),
            "forward_ns_std":     round(self._forward.std(), 3),
            "loss_ns_mean":       round(self._loss.mean, 3),
            "loss_ns_std":        round(self._loss.std(), 3),
            "backward_ns_mean":   round(self._backward.mean, 3),
            "backward_ns_std":    round(self._backward.std(), 3),
            "update_ns_mean":     round(self._update.mean, 3),
            "update_ns_std":      round(self._update.std(), 3),
            "structural_ns_mean": round(self._structural.mean, 3),
            "structural_ns_std":  round(self._structural.std(), 3),
            "reset_ns_mean":      round(self._reset.mean, 3),
            "reset_ns_std":       round(self._reset.std(), 3),
            "other_ns_mean":      round(max(0.0, step_ns_mean - accounted), 3),
        }


# Canonical phase column list for benches that report PhaseTimer summaries.
# Update orchestrator's PHASE_KEYS in lockstep if you extend this.
PHASE_COLUMNS = (
    "step_count", "step_ns_mean",
    "forward_ns_mean", "forward_ns_std",
    "loss_ns_mean", "loss_ns_std",
    "backward_ns_mean", "backward_ns_std",
    "update_ns_mean", "update_ns_std",
    "structural_ns_mean", "structural_ns_std",
    "reset_ns_mean", "reset_ns_std",
    "other_ns_mean",
)


# ---------------------------------------------------------------------------
# Dataset download helper
# ---------------------------------------------------------------------------

def download_if_missing(url: str, dest: Path, max_seconds: float = 30.0) -> Path:
    """Best-effort download.  Skips if dest exists.  Raises on failure so the
    caller can fall back to --synthetic mode with a clear message."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url} -> {dest}", file=sys.stderr)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "plastix-bench/0.1"})
    with urllib.request.urlopen(req, timeout=max_seconds) as r, dest.open("wb") as f:
        f.write(r.read())
    print(f"[data] fetched {dest.stat().st_size} bytes in {time.time()-t0:.1f}s",
          file=sys.stderr)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("PLASTIX_DATA",
                                               "common/pytorch/data")))
    p.add_argument("--out-dir", type=Path, default=Path("_results"))
    p.add_argument("--no-plot", action="store_true",
                   help="skip matplotlib output")
    p.add_argument("--quick", action="store_true",
                   help="reduce epoch / step counts for fast smoke tests")
    p.add_argument("--tag", type=str, default="",
                   help="suffix appended to output filenames; useful for sweeps")


def resolve_device(arg: str) -> str:
    if arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return arg


def output_paths(args, name: str) -> tuple[Path, Path, Path]:
    """Return (history_jsonl, summary_csv, plot_png) for a workload."""
    suffix = f"_{args.tag}" if args.tag else ""
    base = args.out_dir / f"{name}{suffix}"
    return (
        Path(str(base) + ".history.jsonl"),
        Path(str(base) + ".summary.csv"),
        Path(str(base) + ".plot.png"),
    )


def test_plot_path(args, name: str) -> Path:
    """Path for the per-benchmark test-set accuracy/loss curve."""
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_dir).parent / "plots"
    return out_dir / f"{name}{suffix}.test.png"


def plot_test_curve(records: list[dict], out_path: Path, title: str,
                    metric_key: str, ylabel: str,
                    higher_is_better: bool = False) -> None:
    """Single-panel plot of one test-set metric vs step. Records is the same
    StructuralLog records list used everywhere else; each record must carry
    `metric_key` as an extra alongside its `step` value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(r["step"], r[metric_key]) for r in records if r.get(metric_key) is not None]
    if not pts:
        print(f"[warn] no {metric_key} in records; skipping plot", file=sys.stderr)
        return
    steps, vals = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.2,
            color="tab:purple")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    best = max(vals) if higher_is_better else min(vals)
    ax.axhline(best, color="tab:gray", linestyle=":", linewidth=0.8,
               label=f"best={best:.4f}")
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
