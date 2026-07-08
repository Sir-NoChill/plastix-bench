"""Workload 1 / 5 — STATIC ETTh1 forecasting, Norse (SNN) port.

Expressibility: EXPRESSIBLE. A fixed-topology feedforward net maps cleanly onto a
spiking MLP — LIF hidden layers (surrogate-gradient) with the continuous forecast
read out from a non-spiking leaky *integrator* output layer, averaged over the
T-step spike window. The regression target is matched by the accumulated membrane
potential rather than a spike rate. Trained with BPTT (surrogate gradient) + plain
SGD — same schema/algorithm as the snnTorch port.

Norse (norse.torch, eager torch) equivalents of the snnTorch layers:
  - hidden LIF neuron  -> nt.LIFCell()   spk, state = cell(current, state); state=None init
  - non-spiking readout-> nt.LICell()    _, state = cell(current, state); read state.v
The readout integrator has no spike/reset (LICell is a pure leaky integrator),
mirroring snnTorch's reset_mechanism="none" membrane readout; its voltage is
averaged over the T-step window.

The only substantive change vs the ANN benches is the extra TIME dimension: each
`forward` runs the network for T spiking timesteps with the (static) input
injected as a constant current. No structural change (Jaccard flat 1.0).

Usage:
    uv run python 01_static_etth1/norse/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import norse.torch as nt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    download_if_missing,
    output_paths,
    resolve_device,
    write_summary_csv,
)

ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)


def _synthesize(T, C, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    base = (np.sin(2 * np.pi * t / 24)[:, None]
            + 0.4 * np.sin(2 * np.pi * t / (24 * 7))[:, None])
    return base + 0.1 * rng.standard_normal((T, C)).astype(np.float32)


def load_etth1(data_dir, synthetic):
    if synthetic:
        return _synthesize(17_420, 7, seed=0)
    csv_path = data_dir / "ETTh1.csv"
    try:
        download_if_missing(ETTH1_URL, csv_path)
    except Exception as e:
        print(f"[warn] ETTh1 download failed ({e}); synthetic", file=sys.stderr)
        return _synthesize(17_420, 7, seed=0)
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "date"]
    return df[cols].to_numpy(dtype=np.float32)


def windowed(data, in_len, out_len):
    T, C = data.shape
    n = T - in_len - out_len + 1
    idx = np.arange(in_len)[None, :] + np.arange(n)[:, None]
    X = data[idx].reshape(n, in_len * C)
    idx_y = np.arange(out_len)[None, :] + np.arange(n)[:, None] + in_len
    Y = data[idx_y].reshape(n, out_len * C)
    return torch.from_numpy(X), torch.from_numpy(Y)


class SpikingMLP(nn.Module):
    """LIF hidden layers + a non-spiking leaky integrator readout (regression)."""

    def __init__(self, in_dim, out_dim, hidden, depth, T):
        super().__init__()
        self.T = T
        dims = [in_dim] + [hidden] * (depth - 1)
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1])
                                for i in range(depth - 1))
        # Norse LIFCell: spiking hidden neuron (surrogate-gradient by default).
        self.lif = nn.ModuleList(nt.LIFCell() for _ in range(depth - 1))
        self.readout = nn.Linear(dims[-1], out_dim)
        # Norse LICell: pure leaky integrator (no spike/reset) — its membrane
        # voltage is the regression head (mirrors snnTorch reset_mechanism="none").
        self.out_li = nt.LICell()

    def forward(self, x):
        states = [None] * len(self.lif)   # LIFCell states init None on first call
        li_state = None                   # LICell state
        acc = 0.0
        for _ in range(self.T):
            cur = x                                   # constant-current input
            for i, (fc, lif) in enumerate(zip(self.fc, self.lif)):
                spk, states[i] = lif(fc(cur), states[i])
                cur = spk
            _, li_state = self.out_li(self.readout(cur), li_state)
            acc = acc + li_state.v                    # membrane voltage
        return acc / self.T                           # mean-membrane readout


def _counts(model, in_dim):
    n_edges = sum(l.weight.numel() for l in list(model.fc) + [model.readout])
    n_units = sum(l.out_features for l in list(model.fc) + [model.readout])
    return n_units, n_edges


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    probe = MemoryProbe()
    probe.start()

    data = load_etth1(args.data_dir, args.synthetic)
    n_train = int(0.7 * len(data))
    mu = data[:n_train].mean(0); sd = data[:n_train].std(0) + 1e-6
    data = (data - mu) / sd
    X, Y = windowed(data, args.in_len, args.out_len)
    n_tr = int(0.7 * len(X)); n_va = int(0.15 * len(X))
    Xtr, Ytr = X[:n_tr].to(device), Y[:n_tr].to(device)
    Xva, Yva = X[n_tr:n_tr + n_va].to(device), Y[n_tr:n_tr + n_va].to(device)
    Xte, Yte = X[n_tr + n_va:].to(device), Y[n_tr + n_va:].to(device)
    probe.end_dataset()

    T = max(3, args.timesteps // (2 if args.quick else 1))
    model = SpikingMLP(X.shape[1], Y.shape[1], args.hidden, args.depth, T).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    probe.end_weights()

    n_units, n_edges = _counts(model, X.shape[1])
    hist_path, summary_path, _ = output_paths(args, "static_etth1")
    log = StructuralLog(hist_path)

    @torch.no_grad()
    def evaluate(x, y):
        model.eval()
        p = model(x)
        return float(((p - y) ** 2).mean()), float((p - y).abs().mean())

    v0, _ = evaluate(Xva, Yva)
    log.log(0, n_units, n_edges, edges={(0, 0, 0)}, val_loss=v0,
            train_loss=None, test_mse=v0, test_mae=0.0)

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    bs = args.batch
    print(f"[info] device={device} T={T} in_dim={X.shape[1]} hidden={args.hidden} "
          f"depth={args.depth} epochs={epochs} train={n_tr}")

    timer = PhaseTimer()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr - bs + 1, bs):
            sl = perm[i:i + bs]
            xb, yb = Xtr[sl], Ytr[sl]
            timer.tick()
            pred = model(xb)
            timer.mark_forward()
            loss = ((pred - yb) ** 2).sum()
            timer.mark_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            timer.mark_backward()
            opt.step()
            timer.mark_update()
            timer.mark_prune(); timer.mark_grow(); timer.mark_reset()  # static
            timer.step_done()
        va_mse, va_mae = evaluate(Xva, Yva)
        te_mse, te_mae = evaluate(Xte, Yte)
        log.log(ep, n_units, n_edges, edges={(0, 0, 0)}, val_loss=va_mse,
                train_loss=None, val_mae=va_mae, test_mse=te_mse, test_mae=te_mae)
        print(f"[ep {ep:>3d}] val_mse={va_mse:.4f} test_mse={te_mse:.4f}")
    wall = time.perf_counter() - t0

    te_mse, te_mae = evaluate(Xte, Yte)
    log.flush()

    summary = {
        "workload": "01_static_etth1", "framework": "norse",
        "dataset": "ETTh1" if not args.synthetic else "synthetic-etth1",
        "in_len": args.in_len, "out_len": args.out_len, "timesteps": T,
        "hidden": args.hidden, "depth": args.depth,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "wall_seconds": round(wall, 3),
        "val_mse_final": round(log.records[-1]["val_loss"], 6),
        "test_mse": round(te_mse, 6), "test_mae": round(te_mae, 6),
        "n_units": n_units, "n_edges": n_edges,
        "jaccard_min": 1.0, "jaccard_max": 1.0, "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s val_mse={summary['val_mse_final']:.4f} "
          f"test_mse={te_mse:.4f}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--in-len", type=int, default=96)
    p.add_argument("--out-len", type=int, default=24)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=20, help="SNN spike-window T")
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
