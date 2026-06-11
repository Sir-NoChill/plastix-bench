"""Paper-style figure for the Python ccwc benchmark (`ccwc/`).

Builds one multi-panel figure per run-tag that visually conveys the
headline NCP claim — a compact wired NCP matches a much larger dense
baseline on a clean task and degrades more gracefully under noise:

  A. Training curves       test MSE across epochs for A (wired NCP),
                           B (dense LTC), C (LSTM baseline).
  B. Parameter counts      bar chart of each model's `n_params`, with
                           the ratio to A annotated — the "wired NCP
                           does it with ~10× fewer params" story.
  C. Robustness sweep      test MSE vs test-time input-noise σ for each
                           model — Lechner 2020's headline figure
                           (auditable autonomy / robustness).
  D. Sample prediction     A's pred sin/cos overlaid on the held-out
                           target + noisy input — the "look it works"
                           plot, matching the Plastix port's figure.

Reads three artefact families emitted by the ccwc port for each `--tag`:

  results/ccwc_{A,B,C}[_tag].history.jsonl
  results/ccwc_all[_tag].summary.csv
  results/ccwc[_tag].robustness.csv
  results/ccwc[_tag].pred.csv         (from predictions.py)

Writes:

  plots/ccwc[_tag].paper.png

Usage:
    uv run python plot_ccwc.py --tag paper
    uv run python plot_ccwc.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("A", "B", "C")
MODEL_COLOUR = {"A": "#3a7bd5", "B": "#7c5cd6", "C": "#d35a5a"}
MODEL_LABEL = {
    "A": "A — wired NCP (LTC, AutoNCP)",
    "B": "B — dense LTC",
    "C": "C — LSTM baseline",
}
MODEL_MARKER = {"A": "o", "B": "s", "C": "^"}


@dataclass
class RunBundle:
    tag: str
    history: dict[str, dict[str, list[float]]]   # model -> {epoch, train, val, test}
    params: dict[str, int]                         # model -> n_params
    robustness: dict[str, list[tuple[float, float]]] | None  # model -> [(sigma, metric)]
    metric_kind: str | None                        # "mse" / "acc"
    pred: dict[str, np.ndarray] | None             # column name -> array


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_history(path: Path) -> dict[str, list[float]]:
    epochs, train, val, test = [], [], [], []
    metric_key_train = "train_metric"
    metric_key_val = "val_metric"
    metric_key_test = "test_metric"
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if "epoch" not in r:
                continue
            epochs.append(float(r["epoch"]))
            train.append(float(r.get(metric_key_train, float("nan"))))
            val.append(float(r.get(metric_key_val, float("nan"))))
            test.append(float(r.get(metric_key_test, float("nan"))))
    return {"epoch": epochs, "train": train, "val": val, "test": test}


def _load_params(out_dir: Path, tag: str) -> dict[str, int]:
    """Read n_params from the joint summary if available, else from each
    per-model summary CSV."""
    suf = f"_{tag}" if tag else ""
    joint = out_dir / f"ccwc_all{suf}.summary.csv"
    if joint.exists():
        out: dict[str, int] = {}
        with joint.open() as f:
            for row in csv.DictReader(f):
                out[row["model"]] = int(row["n_params"])
        return out
    out = {}
    for m in MODELS:
        path = out_dir / f"ccwc_{m}{suf}.summary.csv"
        if not path.exists():
            continue
        with path.open() as f:
            row = next(csv.DictReader(f))
            out[m] = int(row["n_params"])
    return out


def _load_robustness(path: Path):
    if not path.exists():
        return None, None
    table: dict[str, list[tuple[float, float]]] = {m: [] for m in MODELS}
    metric_kind = None
    with path.open() as f:
        for row in csv.DictReader(f):
            m = row["model"]
            metric_kind = row["metric_kind"]
            table.setdefault(m, []).append(
                (float(row["sigma"]), float(row["metric"]))
            )
    # Sort each by sigma so the plot lines are monotonic.
    for m in list(table):
        table[m].sort(key=lambda r: r[0])
        if not table[m]:
            del table[m]
    return table, metric_kind


def _load_predictions(path: Path):
    if not path.exists():
        return None
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {k: arr[k] for k in arr.dtype.names}


def load_bundle(out_dir: Path, tag: str) -> RunBundle:
    suf = f"_{tag}" if tag else ""
    history = {}
    for m in MODELS:
        p = out_dir / f"ccwc_{m}{suf}.history.jsonl"
        if p.exists():
            history[m] = _load_history(p)
    params = _load_params(out_dir, tag)
    robustness, metric_kind = _load_robustness(
        out_dir / f"ccwc{suf}.robustness.csv")
    pred = _load_predictions(out_dir / f"ccwc{suf}.pred.csv")
    return RunBundle(tag, history, params, robustness, metric_kind, pred)


# ---------------------------------------------------------------------------
# Panel A — training curves
# ---------------------------------------------------------------------------

def plot_training(ax, bundle: RunBundle) -> None:
    for m in MODELS:
        if m not in bundle.history:
            continue
        h = bundle.history[m]
        ep = np.array(h["epoch"])
        ax.plot(ep, h["test"],
                color=MODEL_COLOUR[m], marker=MODEL_MARKER[m],
                markersize=3.5, linewidth=1.5,
                label=MODEL_LABEL[m])
    # Reference floor: a constant-zero predictor on sin/cos targets in
    # [−1, 1] drawn from random phases sits at MSE ≈ 0.5.
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.8, linestyle=":",
               label="zero-predictor (≈0.5)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test MSE")
    ax.set_yscale("log")
    ax.set_title("A — Training curves   (sine, all three models)",
                 loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# Panel B — parameter count comparison
# ---------------------------------------------------------------------------

def plot_params(ax, bundle: RunBundle) -> None:
    if not bundle.params:
        ax.text(0.5, 0.5, "no params data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("B — Parameter counts", loc="left")
        return
    names = [m for m in MODELS if m in bundle.params]
    counts = [bundle.params[m] for m in names]
    colours = [MODEL_COLOUR[m] for m in names]
    base = bundle.params.get("A", min(counts) if counts else 1)
    base = max(base, 1)

    xs = np.arange(len(names))
    bars = ax.bar(xs, counts, color=colours,
                  edgecolor="white", linewidth=0.8)
    # Per-bar annotation: absolute count + ratio to A.
    for x, n, c in zip(xs, names, counts):
        ratio = c / base
        ax.text(x, c, f"{c:,}\n({ratio:.1f}× A)",
                ha="center", va="bottom", fontsize=8,
                color=MODEL_COLOUR[n])
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL[n].split(" — ")[0] for n in names])
    ax.set_ylabel("parameter count")
    ax.set_title("B — Parameter counts   (lower is leaner)", loc="left")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(counts) * 1.30 if counts else 1)
    # Hide top/right spines for a paper aesthetic.
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# Panel C — robustness sweep
# ---------------------------------------------------------------------------

def plot_robustness(ax, bundle: RunBundle) -> None:
    if not bundle.robustness:
        ax.text(0.5, 0.5,
                "no robustness sweep available — run robustness.py",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("C — Robustness sweep", loc="left")
        return
    for m in MODELS:
        if m not in bundle.robustness:
            continue
        rows = bundle.robustness[m]
        sigmas = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        ax.plot(sigmas, vals,
                color=MODEL_COLOUR[m], marker=MODEL_MARKER[m],
                markersize=5, linewidth=1.7,
                label=MODEL_LABEL[m])
    ax.set_xlabel("test-time input-noise σ")
    if bundle.metric_kind == "acc":
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel("test MSE")
    ax.set_title(
        "C — Robustness sweep   (graceful degradation under noise)",
        loc="left",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# Panel D — prediction overlay (A's pred, with target + noisy input)
# ---------------------------------------------------------------------------

def plot_predictions(axes, bundle: RunBundle) -> None:
    pred = bundle.pred
    if pred is None:
        for ax in axes:
            ax.text(0.5, 0.5, "no predictions dump — run predictions.py",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].set_title("D — Prediction on held-out test sequence",
                          loc="left")
        return

    t = pred["t"]
    for ax, key_tgt, key_noise, lab in (
        (axes[0], "target_sin", "noisy_sin", "sin"),
        (axes[1], "target_cos", "noisy_cos", "cos"),
    ):
        ax.scatter(t, pred[key_noise], s=6, color="#cccccc",
                   label="input (noisy)", zorder=1)
        ax.plot(t, pred[key_tgt], color="#222222", linewidth=1.4,
                label="target (clean)", zorder=2)
        for m in MODELS:
            key = f"pred_{m}_{lab}"
            if key in pred:
                ax.plot(t, pred[key],
                        color=MODEL_COLOUR[m], linewidth=1.4,
                        label=f"pred {m}", zorder=3)
        ax.set_ylabel(lab)
        ax.set_ylim(-1.6, 1.6)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(
        "D — Prediction on held-out test sequence   "
        "(all three models overlaid on target + noisy input)",
        loc="left",
    )
    axes[0].legend(loc="upper right", frameon=False, fontsize=7, ncol=5)
    axes[1].set_xlabel("timestep t")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def make_figure(bundle: RunBundle, out_path: Path) -> None:
    fig = plt.figure(figsize=(13.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.1, 1.1, 0.9, 0.9],
        hspace=0.35, wspace=0.20,
    )
    ax_train = fig.add_subplot(gs[0, 0])
    ax_params = fig.add_subplot(gs[0, 1])
    ax_robust = fig.add_subplot(gs[1, :])
    ax_sin = fig.add_subplot(gs[2, :])
    ax_cos = fig.add_subplot(gs[3, :], sharex=ax_sin)

    plot_training(ax_train, bundle)
    plot_params(ax_params, bundle)
    plot_robustness(ax_robust, bundle)
    plot_predictions([ax_sin, ax_cos], bundle)

    # Headline summary in the suptitle.
    final_test = {
        m: bundle.history[m]["test"][-1]
        for m in MODELS
        if m in bundle.history and bundle.history[m]["test"]
    }
    headline = "  ·  ".join(
        f"{m}: clean MSE={final_test[m]:.3f}" for m in MODELS
        if m in final_test
    )
    title_tag = bundle.tag if bundle.tag else "(no tag)"
    fig.suptitle(
        f"ccwc — Compact C. elegans-Wired Controller   "
        f"·  tag={title_tag!r}   ·   {headline}",
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
    """Find every tag with at least one of A/B/C's history.jsonl."""
    tags = set()
    for h in results_dir.glob("ccwc_A*.history.jsonl"):
        stem = h.stem  # ccwc_A[_<tag>].history
        body = stem[len("ccwc_A"):-len(".history")]
        if body.startswith("_"):
            tags.add(body[1:])
        else:
            tags.add("")
    return sorted(tags)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None)
    p.add_argument("--all", action="store_true",
                   help="plot every tag found under results/")
    p.add_argument("--results-dir", type=Path,
                   default=here / "results")
    p.add_argument("--plots-dir", type=Path,
                   default=here / "plots")
    args = p.parse_args()

    if args.all or args.tag is None:
        tags = discover_tags(args.results_dir)
        if not tags:
            print(f"[err] no ccwc_A*.history.jsonl files under "
                  f"{args.results_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        tags = [args.tag]

    for tag in tags:
        bundle = load_bundle(args.results_dir, tag)
        if not bundle.history:
            print(f"[skip] tag={tag!r}: no per-model history files",
                  file=sys.stderr)
            continue
        suf = f"_{tag}" if tag else ""
        out_path = args.plots_dir / f"ccwc{suf}.paper.png"
        make_figure(bundle, out_path)


if __name__ == "__main__":
    main()
