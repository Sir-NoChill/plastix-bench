"""Evaluate a trained SNN-SHD checkpoint and write an ablation bar chart.

train.py already runs the time-shuffle ablation once at the end of
training; this script is for re-running it (and additional cuts) on a
saved checkpoint without retraining. Produces:

  - results/snn_shd_<tag>.ablation.json
  - plots/snn_shd_<tag>.ablation.png   (bar chart of test_acc per ablation)

Usage:
    uv run python snn-shd/evaluate.py \\
        --ckpt results/snn_shd_plot_smoke.ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "_shared_python"))
sys.path.insert(0, str(HERE))

from common import resolve_device  # noqa: E402
from data import load_shd, make_loaders, time_shuffle  # noqa: E402
from model import build_model  # noqa: E402


def evaluate_acc(model, loader, device, shuffle_seed: int | None = None):
    model.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if shuffle_seed is not None:
                xb = time_shuffle(xb, seed=shuffle_seed + seen)
            xb_t = xb.permute(1, 0, 2)
            logits, _ = model(xb_t)
            correct += (logits.argmax(-1) == yb).sum().item()
            seen += yb.size(0)
    return correct / max(seen, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="path to .ckpt.pt produced by train.py")
    p.add_argument("--data-dir", type=Path,
                   default=Path("data"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-shuffle-runs", type=int, default=3,
                   help="averaging across this many time-shuffle seeds")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results"))
    p.add_argument("--plots-dir", type=Path,
                   default=Path("plots"))
    args = p.parse_args()

    device = resolve_device(args.device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = state["args"]
    summary = state["summary"]

    _, _, test_ds, n_in, n_classes = load_shd(
        Path(train_args["data_dir"]),
        n_bins=summary["n_bins"], val_frac=train_args["val_frac"],
        seed=train_args["seed"],
    )
    _, _, test_loader = make_loaders(test_ds, test_ds, test_ds,
                                     batch=train_args["batch"])

    if train_args["model"] == "gru":
        # The GRU's hidden size was solved from the SNN's param count;
        # the saved checkpoint already encodes the right shape, so build
        # without `match_to` and use the stored n_hid.
        model = build_model("gru", n_in, summary["n_hid"], n_classes,
                            beta=train_args["beta"],
                            surrogate_slope=train_args["surrogate_slope"])
    else:
        model = build_model(train_args["model"], n_in, summary["n_hid"],
                            n_classes, beta=train_args["beta"],
                            surrogate_slope=train_args["surrogate_slope"])
    model.load_state_dict(state["state_dict"])
    model.to(device)

    natural_acc = evaluate_acc(model, test_loader, device)
    shuffled_accs = []
    for k in range(args.n_shuffle_runs):
        shuffled_accs.append(
            evaluate_acc(model, test_loader, device,
                         shuffle_seed=train_args["seed"] + 17 * (k + 1))
        )
    shuffled_mean = float(np.mean(shuffled_accs))
    shuffled_std = float(np.std(shuffled_accs))

    print(f"[eval] natural test_acc = {natural_acc:.4f}")
    print(f"[eval] time-shuffled    = {shuffled_mean:.4f} +/- {shuffled_std:.4f}"
          f"  (n={args.n_shuffle_runs} runs)")
    print(f"[eval] ablation drop    = {natural_acc - shuffled_mean:+.4f}")

    payload = {
        "ckpt": str(args.ckpt),
        "model": train_args["model"],
        "natural_acc": natural_acc,
        "shuffled_accs": shuffled_accs,
        "shuffled_mean": shuffled_mean,
        "shuffled_std": shuffled_std,
        "ablation_drop": natural_acc - shuffled_mean,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.ckpt.stem.replace(".ckpt", "")
    out_json = args.out_dir / f"{tag}.ablation.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[eval] wrote {out_json}")

    # Bar chart.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["natural", "time-shuffled"],
                  [natural_acc, shuffled_mean],
                  color=["tab:purple", "tab:gray"])
    if shuffled_std > 0:
        ax.errorbar(["time-shuffled"], [shuffled_mean], yerr=[shuffled_std],
                    fmt="none", color="black", capsize=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"SHD time-shuffle ablation ({train_args['model']})\n"
                 f"drop = {natural_acc - shuffled_mean:+.3f}")
    for b, v in zip(bars, [natural_acc, shuffled_mean]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom")
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.plots_dir / f"{tag}.ablation.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {out_png}")


if __name__ == "__main__":
    main()
