"""Dump prediction traces for the three CCWC models on one held-out sine
sequence.

Loads the A / B / C checkpoints produced by `train.py` (sine task) and
emits `results/ccwc[_tag].pred.csv` with columns

    t, target_sin, target_cos, noisy_sin, noisy_cos,
       pred_A_sin, pred_A_cos, pred_B_sin, pred_B_cos,
       pred_C_sin, pred_C_cos

Used downstream by `plot_ccwc.py` for the prediction-overlay
panel of the paper-style figure. Mirrors the schema of the Plastix port's
`traditional-plastix/csv/ccwc_ncp[_tag].pred.csv` so the two flows render
the same kind of plot.

Usage:
    uv run python ccwc/predictions.py --tag paper
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "_shared_python"))
sys.path.insert(0, str(HERE))

from common import resolve_device  # noqa: E402
from data import make_sine_dataset  # noqa: E402
from models import build_model  # noqa: E402

WORKLOAD = "ccwc"


def _load_one(out_dir: Path, model_name: str, tag: str, device: str):
    suf = f"{model_name}" + (f"_{tag}" if tag else "")
    path = out_dir / f"{WORKLOAD}_{suf}.ckpt.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint for model {model_name}: {path}\n"
            f"  run train.py --model all (and matching --tag) first")
    state = torch.load(path, map_location=device, weights_only=False)
    if state["task"] != "sine":
        raise SystemExit(
            f"predictions.py only handles sine-task checkpoints; "
            f"{path} has task={state['task']!r}")
    model = build_model(
        model_name, input_size=state["input_size"], units=state["units"],
        n_out=state["n_out"], task=state["task"],
        variant=state.get("variant", "ltc"),
        lstm_hidden=state.get("lstm_hidden", 128),
        mixed_memory=state.get("mixed_memory", False),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="where train.py wrote the checkpoints")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seq-idx", type=int, default=0,
                   help="index into the test set to dump (default 0)")
    p.add_argument("--models", type=str, default="A,B,C")
    args = p.parse_args()

    device = resolve_device(args.device)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    # Load every checkpoint up front (sine task is enforced by _load_one).
    models, states = {}, {}
    for name in model_names:
        models[name], states[name] = _load_one(
            args.out_dir, name, args.tag, device)

    # Rebuild the same test set the checkpoints were trained against, so
    # the held-out sequence we dump is one the trained models actually
    # never saw during training (matches robustness.py's reconstruction).
    train_args = next(iter(states.values()))["args"]
    _, _, test_ds = make_sine_dataset(
        n_train=train_args.get("sine_train", 512),
        n_val=train_args.get("sine_val", 128),
        n_test=train_args.get("sine_test", 128),
        seq_len=train_args.get("sine_seq_len", 64),
        noise_std=train_args.get("sine_noise", 0.1),
        seed=train_args.get("seed", 0),
    )
    x, y = test_ds[args.seq_idx]            # (T, 2) input, (T, 2) target
    x_batch = x.unsqueeze(0).to(device)      # (1, T, 2)

    preds: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in model_names:
            preds[name] = models[name](x_batch).squeeze(0).cpu()  # (T, 2)

    suf = f"_{args.tag}" if args.tag else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{WORKLOAD}{suf}.pred.csv"
    cols = ["t", "target_sin", "target_cos", "noisy_sin", "noisy_cos"]
    for n in model_names:
        cols += [f"pred_{n}_sin", f"pred_{n}_cos"]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        T = x.shape[0]
        for t in range(T):
            row = [t,
                   float(y[t, 0]), float(y[t, 1]),
                   float(x[t, 0]), float(x[t, 1])]
            for n in model_names:
                row += [float(preds[n][t, 0]), float(preds[n][t, 1])]
            w.writerow(row)
    print(f"[pred] wrote {out}  (T={T}, models={','.join(model_names)})")


if __name__ == "__main__":
    main()
