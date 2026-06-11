"""Test-time noise robustness sweep for the CCWC benchmark (spec section 6).

Loads the A / B / C checkpoints produced by `train.py`, evaluates each on
the test set under increasing Gaussian input noise, and emits:

  - results/ccwc[_tag].robustness.json  : raw (model, sigma, metric) grid
  - results/ccwc[_tag].robustness.csv   : same as a flat CSV (model, sigma, metric)
  - plots/ccwc[_tag].robustness_vs_sigma.png : one line per model

No retraining, just inference with noise injection. The script is meant to
be run *after* `train.py --model all`.

Usage:
    uv run python ccwc/robustness.py --task psmnist --tag run0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "_shared_python"))
sys.path.insert(0, str(HERE))

from common import resolve_device  # noqa: E402
from data import (  # noqa: E402
    load_psmnist,
    make_loaders,
    make_sine_dataset,
)
from models import build_model  # noqa: E402
from train import evaluate, WORKLOAD  # noqa: E402


def _ckpt_path(out_dir: Path, model: str, tag: str) -> Path:
    sub_tag = f"{model}" + (f"_{tag}" if tag else "")
    return out_dir / f"{WORKLOAD}_{sub_tag}.ckpt.pt"


def _load_one(out_dir: Path, model_name: str, tag: str, device: str):
    path = _ckpt_path(out_dir, model_name, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint for model {model_name}: {path}\n"
            f"  run train.py --model all (and matching --tag) first")
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        model_name, input_size=state["input_size"], units=state["units"],
        n_out=state["n_out"], task=state["task"],
        variant=state.get("variant", "ltc"),
        lstm_hidden=state.get("lstm_hidden", 128),
        mixed_memory=state.get("mixed_memory", False),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    return model, state


def _build_test_loader(args, ref_state: dict):
    """Rebuild the test loader using the same hyperparameters the checkpoint
    was trained with — the user only specifies the task on the CLI."""
    task = ref_state["task"]
    train_args = ref_state["args"]
    batch = train_args.get("batch", 64)
    if task == "sine":
        _, _, test_ds = make_sine_dataset(
            n_train=train_args.get("sine_train", 512),
            n_val=train_args.get("sine_val", 128),
            n_test=train_args.get("sine_test", 128),
            seq_len=train_args.get("sine_seq_len", 64),
            noise_std=train_args.get("sine_noise", 0.1),
            seed=train_args.get("seed", 0),
        )
        train_ds = val_ds = test_ds
        loaders = make_loaders(train_ds, val_ds, test_ds, batch=batch)
        return loaders[2], "mse"
    if task == "psmnist":
        _, _, test_ds, *_ = load_psmnist(
            Path(train_args.get("data_dir", "data")),
            permute=not train_args.get("no_permute", False),
            perm_seed=train_args.get("perm_seed", 12345),
            val_frac=train_args.get("val_frac", 0.10),
            seed=train_args.get("seed", 0),
        )
        train_ds = val_ds = test_ds
        loaders = make_loaders(train_ds, val_ds, test_ds, batch=batch)
        return loaders[2], "acc"
    raise ValueError(f"unknown task in checkpoint: {task!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="where train.py wrote the checkpoints")
    p.add_argument("--plots-dir", type=Path, default=Path("plots"))
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--sigmas", type=str, default="0.0,0.05,0.1,0.2,0.4,0.8",
                   help="comma-separated test-time noise std values")
    p.add_argument("--noise-seed", type=int, default=2026,
                   help="seed for the per-sigma deterministic noise draw")
    p.add_argument("--models", type=str, default="A,B,C",
                   help="subset of {A,B,C} to sweep")
    args = p.parse_args()

    device = resolve_device(args.device)
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    # Load every requested checkpoint up front; bail with a clear message
    # if one is missing so the user knows which to re-train.
    models = {}
    states = {}
    for name in model_names:
        models[name], states[name] = _load_one(args.out_dir, name,
                                                args.tag, device)

    # All checkpoints in a single sweep must agree on the task; otherwise
    # the X-axis isn't comparable.
    tasks = {st["task"] for st in states.values()}
    if len(tasks) != 1:
        raise SystemExit(f"checkpoints disagree on task: {tasks}")
    task = next(iter(tasks))

    test_loader, metric_kind = _build_test_loader(args, next(iter(states.values())))

    # Sweep.
    table = {name: [] for name in model_names}
    print(f"[sweep] task={task}  metric={metric_kind}  sigmas={sigmas}")
    for sigma in sigmas:
        for name in model_names:
            _, m_val = evaluate(models[name], test_loader, device,
                                 metric_kind,
                                 noise_sigma=sigma,
                                 noise_seed=args.noise_seed)
            table[name].append({"sigma": sigma, "metric": m_val})
            print(f"  sigma={sigma:>4.2f}  model={name}  "
                  f"{metric_kind}={m_val:.4f}  "
                  f"params={states[name]['summary']['n_params']}")

    # JSON + CSV summaries.
    suffix = f"_{args.tag}" if args.tag else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{WORKLOAD}{suffix}.robustness.json"
    payload = {
        "task": task, "metric_kind": metric_kind, "sigmas": sigmas,
        "models": {name: states[name]["summary"] for name in model_names},
        "table": table,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[sweep] wrote {json_path}")

    csv_path = args.out_dir / f"{WORKLOAD}{suffix}.robustness.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "n_params", "sigma", "metric_kind", "metric"])
        for name in model_names:
            np_n = states[name]["summary"]["n_params"]
            for row in table[name]:
                w.writerow([task, name, np_n, row["sigma"], metric_kind,
                            row["metric"]])
    print(f"[sweep] wrote {csv_path}")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"A": "tab:blue", "B": "tab:green", "C": "tab:red"}
    for name in model_names:
        xs = [r["sigma"] for r in table[name]]
        ys = [r["metric"] for r in table[name]]
        n_params = states[name]["summary"]["n_params"]
        ax.plot(xs, ys, marker="o", label=f"{name} (params={n_params})",
                color=colours.get(name))
    ax.set_xlabel("test-time noise std (sigma)")
    if metric_kind == "acc":
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel("test MSE")
    ax.set_title(f"CCWC robustness sweep -- task={task}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    png = args.plots_dir / f"{WORKLOAD}{suffix}.robustness_vs_sigma.png"
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"[sweep] wrote {png}")


if __name__ == "__main__":
    main()
