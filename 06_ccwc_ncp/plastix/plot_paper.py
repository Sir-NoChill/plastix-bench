"""Paper-style figure for the 07-ccwc-ncp benchmark.

Builds one multi-panel figure per run-tag that visually conveys:

  A. The training curve         (train / val / test MSE across epochs)
  B. The NCP wiring schematic   (input → sensory → inter → command → motor,
                                 with sparse fan-in lines, command-recurrent
                                 loops, and motor→command feedback)
  C. Prediction vs target       (the learnt model's sin/cos output overlaid
                                 on the held-out sequence + the noisy input
                                 the model actually saw)

Reads three artefacts emitted by 07_ccwc_ncp.cpp for each `--tag`:

  traditional-plastix/results/ccwc_ncp[_tag].history.jsonl
  traditional-plastix/csv/ccwc_ncp[_tag].pred.csv
  traditional-plastix/csv/ccwc_ncp[_tag].topology.csv

Writes:

  traditional-plastix/plots/ccwc_ncp[_tag].paper.png

Usage:
    uv run python traditional-plastix/plot_ccwc_ncp.py --tag paper
    uv run python traditional-plastix/plot_ccwc_ncp.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np


# AutoNCP-style colour palette. Picked for print clarity (each kind sits in
# a different hue band) and to echo the colour conventions in the original
# Lechner 2020 NCP paper figures.
KIND_COLOUR = {
    255: "#444444",   # input (sentinel)
    0:   "#3a7bd5",   # sensory
    1:   "#7c5cd6",   # inter
    2:   "#d35a5a",   # command
    3:   "#1f9d55",   # motor
}
KIND_NAME = {255: "input", 0: "sensory", 1: "inter", 2: "command", 3: "motor"}


@dataclass
class Topology:
    n_input: int
    n_sensory: int
    n_inter: int
    n_command: int
    n_motor: int
    edges: list[tuple[int, int, int, int, float]]  # (from, to, fkind, tkind, w)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_history(path: Path) -> dict[str, list[float]]:
    """Read the per-epoch JSONL log and extract the four columns we care
    about for panel A. Skips records that are missing one of them."""
    epochs: list[float] = []
    train: list[float] = []
    val:   list[float] = []
    test:  list[float] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if "epoch" not in r:
                continue
            epochs.append(float(r["epoch"]))
            train.append(float(r.get("train_mse", float("nan"))))
            val.append(float(r.get("val_mse", float("nan"))))
            test.append(float(r.get("test_mse", float("nan"))))
    return {"epoch": epochs, "train": train, "val": val, "test": test}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {k: arr[k] for k in arr.dtype.names}


def load_topology(path: Path) -> Topology:
    text = path.read_text().splitlines()
    header_re = re.compile(
        r"#\s*partition\s+n_input=(\d+)\s+n_sensory=(\d+)\s+n_inter=(\d+)"
        r"\s+n_command=(\d+)\s+n_motor=(\d+)"
    )
    n_in = n_s = n_i = n_c = n_m = 0
    edges: list[tuple[int, int, int, int, float]] = []
    for line in text:
        if line.startswith("#"):
            m = header_re.search(line)
            if m:
                n_in, n_s, n_i, n_c, n_m = (int(x) for x in m.groups())
            continue
        if line.startswith("from_id"):
            continue
        if not line.strip():
            continue
        parts = line.split(",")
        edges.append((int(parts[0]), int(parts[1]), int(parts[2]),
                      int(parts[3]), float(parts[4])))
    return Topology(n_in, n_s, n_i, n_c, n_m, edges)


# ---------------------------------------------------------------------------
# Panel A — training curves
# ---------------------------------------------------------------------------

def plot_training(ax, hist: dict[str, list[float]]) -> None:
    ep = np.array(hist["epoch"])
    ax.plot(ep, hist["train"], color="#888888", marker="o", markersize=3,
            linewidth=1.4, label="train")
    ax.plot(ep, hist["val"],   color="#3a7bd5", marker="s", markersize=3,
            linewidth=1.4, label="val")
    ax.plot(ep, hist["test"],  color="#d35a5a", marker="^", markersize=3,
            linewidth=1.6, label="test")
    # Constant-zero predictor baseline. For sin/cos targets in [-1, 1] drawn
    # from random phases, E[target²] ≈ 0.5 — that is the floor a network
    # which collapses to "predict zero" achieves. Sitting well below it is
    # the "learning happened" sanity check.
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.8, linestyle=":",
               label="zero-predictor (≈0.5)")
    final = hist["test"][-1] if hist["test"] else float("nan")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(f"A — Training curves   (final test MSE = {final:.3f})",
                 loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# Panel B — NCP wiring schematic
# ---------------------------------------------------------------------------

def _layer_positions(n: int, x: float, y_lo: float, y_hi: float):
    """N points stacked vertically at column x, spread between y_lo..y_hi.
    For n==1 we centre the single point."""
    if n <= 0:
        return []
    if n == 1:
        return [(x, 0.5 * (y_lo + y_hi))]
    ys = np.linspace(y_hi, y_lo, n)
    return [(x, float(y)) for y in ys]


def plot_topology(ax, topo: Topology) -> None:
    # Layout: five vertical columns (input, sensory, inter, command, motor)
    # spaced uniformly along x ∈ [0, 1], with node y reserved to [0.05, 0.82]
    # so column-name labels at y ≈ 0.94 sit above the topmost node without
    # overlap. Edges are drawn as straight lines except for the recurrent
    # and feedback edges which use curved arrows so they don't visually
    # merge with the feed-forward stack.
    y_top, y_bot = 0.82, 0.05
    columns: dict[int, list[tuple[float, float]]] = {}
    columns[255] = _layer_positions(topo.n_input,   0.05, y_bot + 0.10,
                                    y_top - 0.10)
    columns[0]   = _layer_positions(topo.n_sensory, 0.25, y_bot, y_top)
    columns[1]   = _layer_positions(topo.n_inter,   0.45, y_bot, y_top)
    columns[2]   = _layer_positions(topo.n_command, 0.65, y_bot, y_top)
    columns[3]   = _layer_positions(topo.n_motor,   0.85, y_bot + 0.25,
                                    y_top - 0.25)

    # Build a global id -> (x, y) map. Plastix unit ids are sequential
    # starting with the inputs, then sensory, inter, command, motor — the
    # order in which NCPWiringBuilder allocates them.
    id_to_xy: dict[int, tuple[float, float]] = {}
    cursor = 0
    for kind in (255, 0, 1, 2, 3):
        for pos in columns[kind]:
            id_to_xy[cursor] = pos
            cursor += 1

    # Edge weights → opacity scaling. Higher |w| draws as more opaque; this
    # is the visual analogue of the "strength" lines in NCP figures.
    if topo.edges:
        wmax = max(abs(w) for *_, w in topo.edges)
    else:
        wmax = 1.0
    wmax = max(wmax, 1e-6)

    # Categorise edges so curved / straight rendering can branch on kind.
    feedfwd, recurrent, feedback = [], [], []
    for e in topo.edges:
        f, t, fk, tk, w = e
        if fk == 2 and tk == 2:
            recurrent.append(e)
        elif fk == 3 and tk == 2:
            feedback.append(e)
        else:
            feedfwd.append(e)

    # 1. Straight feed-forward edges (input→sensory, sensory→inter, etc.)
    for f, t, fk, tk, w in feedfwd:
        x0, y0 = id_to_xy[f]
        x1, y1 = id_to_xy[t]
        a = 0.15 + 0.65 * (abs(w) / wmax)
        c = "#1f9d55" if w >= 0 else "#d35a5a"   # excitatory vs inhibitory
        ax.plot([x0, x1], [y0, y1], color=c, alpha=a, linewidth=0.6,
                zorder=1)

    # 2. Command-recurrent edges: drawn as small arcs to the right of the
    # command column so they don't collide with the feedforward bundle.
    for f, t, fk, tk, w in recurrent:
        x0, y0 = id_to_xy[f]
        x1, y1 = id_to_xy[t]
        a = 0.2 + 0.6 * (abs(w) / wmax)
        c = "#1f9d55" if w >= 0 else "#d35a5a"
        arc = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle="arc3,rad=0.45",
            arrowstyle="-",
            color=c, alpha=a, linewidth=0.6, zorder=1,
        )
        ax.add_patch(arc)

    # 3. Motor → command feedback: curved going back to the left.
    for f, t, fk, tk, w in feedback:
        x0, y0 = id_to_xy[f]
        x1, y1 = id_to_xy[t]
        a = 0.25 + 0.55 * (abs(w) / wmax)
        arc = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle="arc3,rad=-0.5",
            arrowstyle="-|>", mutation_scale=8,
            color="#7c5cd6", alpha=a, linewidth=0.9, zorder=1,
        )
        ax.add_patch(arc)

    # Nodes on top of the edges.
    for kind in (255, 0, 1, 2, 3):
        for (x, y) in columns[kind]:
            ax.plot(x, y, "o", markersize=8,
                    markerfacecolor=KIND_COLOUR[kind],
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)

    # Column labels above the top of each stack.
    for kind, label_x in ((255, 0.05), (0, 0.25), (1, 0.45),
                          (2, 0.65), (3, 0.85)):
        n = len(columns[kind])
        ax.text(label_x, 0.94, f"{KIND_NAME[kind]}\n(n={n})",
                ha="center", va="bottom", fontsize=8,
                color=KIND_COLOUR[kind], fontweight="bold")

    n_ff = len(feedfwd); n_rec = len(recurrent); n_fb = len(feedback)
    ax.set_title(
        f"B — NCP wiring   ({n_ff} feed-fwd, {n_rec} recurrent, "
        f"{n_fb} feedback, total {len(topo.edges)})",
        loc="left",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.06, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend below the diagram (outside the node footprint).
    legend_patches = [
        mpatches.Patch(color="#1f9d55", label="excitatory (w>0)"),
        mpatches.Patch(color="#d35a5a", label="inhibitory (w<0)"),
        mpatches.Patch(color="#7c5cd6", label="motor→cmd feedback"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=3,
              frameon=False, fontsize=7)


# ---------------------------------------------------------------------------
# Panel C — prediction vs target
# ---------------------------------------------------------------------------

def plot_predictions(axes, pred: dict[str, np.ndarray]) -> None:
    t = pred["t"]
    for ax, key_pred, key_tgt, key_noise, lab in (
        (axes[0], "pred_sin", "target_sin", "noisy_sin", "sin"),
        (axes[1], "pred_cos", "target_cos", "noisy_cos", "cos"),
    ):
        ax.scatter(t, pred[key_noise], s=8, color="#cccccc",
                   label="input (noisy)", zorder=1)
        ax.plot(t, pred[key_tgt],  color="#222222", linewidth=1.4,
                label="target (clean)", zorder=2)
        ax.plot(t, pred[key_pred], color="#d35a5a", linewidth=1.6,
                label="prediction", zorder=3)
        ax.set_ylabel(lab)
        ax.set_ylim(-1.4, 1.4)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("C — Prediction on held-out test sequence",
                      loc="left")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    axes[1].set_xlabel("timestep t")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def make_figure(history: dict[str, list[float]], pred: dict[str, np.ndarray],
                topo: Topology, tag: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12.0, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 0.9, 0.9],
                          hspace=0.30, wspace=0.18)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_topo  = fig.add_subplot(gs[0, 1])
    ax_sin   = fig.add_subplot(gs[1, :])
    ax_cos   = fig.add_subplot(gs[2, :], sharex=ax_sin)

    plot_training(ax_train, history)
    plot_topology(ax_topo, topo)
    plot_predictions([ax_sin, ax_cos], pred)

    n_units = topo.n_sensory + topo.n_inter + topo.n_command + topo.n_motor
    fig.suptitle(
        f"07-ccwc-ncp  ·  tag={tag!r}  ·  {n_units} NCP neurons, "
        f"{len(topo.edges)} edges  ·  e-prop one-step training",
        fontsize=11, y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def discover_tags(results_dir: Path) -> list[str]:
    """Find every tag with a complete artefact triple. A run with one
    missing file is skipped (we'd just produce an empty panel)."""
    tags = set()
    for h in results_dir.glob("ccwc_ncp_*.history.jsonl"):
        stem = h.stem            # ccwc_ncp_<tag>.history
        tag = stem[len("ccwc_ncp_"):-len(".history")]
        tags.add(tag)
    # The no-tag run uses 'ccwc_ncp.history.jsonl' (no underscore).
    if (results_dir / "ccwc_ncp.history.jsonl").exists():
        tags.add("")
    return sorted(tags)


def paths_for(tag: str, root: Path) -> tuple[Path, Path, Path, Path]:
    suf = f"_{tag}" if tag else ""
    res = root / "results"
    csv_d = root / "csv"
    plot_d = root / "plots"
    return (
        res / f"ccwc_ncp{suf}.history.jsonl",
        csv_d / f"ccwc_ncp{suf}.pred.csv",
        csv_d / f"ccwc_ncp{suf}.topology.csv",
        plot_d / f"ccwc_ncp{suf}.paper.png",
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None,
                   help="run tag to plot (default: every tag with a full "
                        "artefact triple)")
    p.add_argument("--all", action="store_true",
                   help="plot every tag found under results/")
    p.add_argument("--root", type=Path, default=here,
                   help="traditional-plastix/ root (default: this file's dir)")
    args = p.parse_args()

    if args.all or args.tag is None:
        tags = discover_tags(args.root / "results")
        if not tags:
            print("[err] no ccwc_ncp_*.history.jsonl files found under "
                  f"{args.root/'results'}", file=sys.stderr)
            sys.exit(1)
    else:
        tags = [args.tag]

    for tag in tags:
        hist_p, pred_p, topo_p, out_p = paths_for(tag, args.root)
        missing = [str(p) for p in (hist_p, pred_p, topo_p) if not p.exists()]
        if missing:
            print(f"[skip] tag={tag!r}: missing {missing}", file=sys.stderr)
            continue
        hist = load_history(hist_p)
        pred = load_pred(pred_p)
        topo = load_topology(topo_p)
        make_figure(hist, pred, topo, tag, out_p)


if __name__ == "__main__":
    main()
