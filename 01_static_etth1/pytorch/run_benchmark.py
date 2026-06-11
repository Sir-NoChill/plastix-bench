"""
Workload 1 / 5 -- STATIC regime on ETTh1 long-horizon forecasting.

Architecture: 3-layer feedforward MLP, fixed at init, trained with MSE.
This is the reference baseline against which every other workload (which
involves structural change) is compared.  Topology hash should never change;
edge Jaccard should be a flat 1.0.  If it isn't, instrumentation is broken.

Dataset: ETTh1 (Electricity Transformer Temperature, hourly), Zhou et al.
2021 (AAAI).  17,420 rows x 7 channels, the canonical long-horizon split
(input length 96, forecast horizon 24 by default).

Auto-downloads ETTh1.csv from the upstream GitHub mirror on first run.
Falls back to a deterministic synthetic series with --synthetic.

Usage:
    uv run python 01_static_etth1.py                # full run
    uv run python 01_static_etth1.py --quick        # smoke test
    uv run python 01_static_etth1.py --hidden 512   # sweep knob

Outputs (under --out-dir, default results/):
    static_etth1.history.jsonl     # per-epoch structural log
    static_etth1.summary.csv       # one row of headline numbers
    static_etth1.plot.png          # loss + size + Jaccard
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared_python"))
from common import (  # noqa: E402
    PHASE_COLUMNS,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    download_if_missing,
    output_paths,
    plot_run,
    plot_test_curve,
    resolve_device,
    test_plot_path,
    write_summary_csv,
)


ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_etth1(data_dir: Path, synthetic: bool) -> np.ndarray:
    if synthetic:
        return _synthesize(17_420, 7, seed=0)
    csv_path = data_dir / "ETTh1.csv"
    try:
        download_if_missing(ETTH1_URL, csv_path)
    except Exception as e:
        print(f"[warn] ETTh1 download failed ({e}); falling back to synthetic",
              file=sys.stderr)
        return _synthesize(17_420, 7, seed=0)
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "date"]
    return df[cols].to_numpy(dtype=np.float32)


def _synthesize(T: int, C: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    base = (
        np.sin(2 * np.pi * t / 24)[:, None]
        + 0.4 * np.sin(2 * np.pi * t / (24 * 7))[:, None]
    )
    return base + 0.1 * rng.standard_normal((T, C)).astype(np.float32)


def windowed(data: np.ndarray, in_len: int, out_len: int):
    T, C = data.shape
    n = T - in_len - out_len + 1
    idx = np.arange(in_len)[None, :] + np.arange(n)[:, None]
    X = data[idx].reshape(n, in_len * C)
    idx_y = np.arange(out_len)[None, :] + np.arange(n)[:, None] + in_len
    Y = data[idx_y].reshape(n, out_len * C)
    return torch.from_numpy(X), torch.from_numpy(Y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StaticMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != last:
                x = F.gelu(x)
        return x

    def edge_set(self) -> set[tuple[int, int, int]]:
        """All weight positions count as a live edge in the Static regime
        (no masking, no zeros enforced).  This makes Jaccard trivially 1.0
        for every step, which is exactly the diagnostic we want."""
        edges = set()
        for li, layer in enumerate(self.layers):
            out_d, in_d = layer.weight.shape
            for r in range(out_d):
                for c in range(in_d):
                    edges.add((li, r, c))
        return edges


def edge_count(model: StaticMLP) -> int:
    return sum(l.weight.numel() for l in model.layers)


def unit_count(model: StaticMLP) -> int:
    # Input units are implicit; count hidden + output units.
    return sum(l.out_features for l in model.layers)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, opt, device, timer: PhaseTimer) -> float:
    model.train()
    total = 0.0; n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        timer.tick()
        # reduction='sum' so the per-batch gradient magnitude equals the
        # cumulative magnitude of |batch| per-example gradients — matches
        # what Plastix's per-example SGD loop accumulates over the same
        # batch. The 2x factor between (sum of (p-t)^2) and Plastix's
        # (1/2 sum (p-t)^2) MSELoss is folded into the lr default.
        pred = model(xb)
        timer.mark_forward()
        loss = F.mse_loss(pred, yb, reduction="sum")
        timer.mark_loss()
        loss.backward()
        timer.mark_backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        timer.mark_update()
        timer.step_done()
        total += loss.item(); n += xb.size(0) * yb.size(1)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, x, y, batch: int = 1024) -> tuple[float, float]:
    model.eval()
    mse_sum = 0.0; mae_sum = 0.0; n = 0
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch]; yb = y[i:i + batch]
        p = model(xb)
        mse_sum += F.mse_loss(p, yb, reduction="sum").item()
        mae_sum += (p - yb).abs().sum().item()
        n += yb.numel()
    return mse_sum / n, mae_sum / n


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    data = load_etth1(args.data_dir, args.synthetic)
    n_train = int(0.7 * len(data))
    mu = data[:n_train].mean(0)
    sd = data[:n_train].std(0) + 1e-6
    data = (data - mu) / sd

    X, Y = windowed(data, args.in_len, args.out_len)
    n_tr = int(0.7 * len(X))
    n_va = int(0.15 * len(X))
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva = X[n_tr:n_tr + n_va].to(device)
    Yva = Y[n_tr:n_tr + n_va].to(device)
    Xte = X[n_tr + n_va:].to(device)
    Yte = Y[n_tr + n_va:].to(device)

    loader = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=args.batch, shuffle=True, drop_last=True,
    )

    model = StaticMLP(
        in_dim=X.shape[1], out_dim=Y.shape[1],
        hidden=args.hidden, depth=args.depth,
    ).to(device)
    # SGD (no momentum) — matches Plastix's plain per-connection SGD policy.
    # With reduction='sum' the cumulative per-batch gradient magnitude is
    # already comparable to Plastix's, so the same nominal lr works.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    hist_path, summary_path, plot_path = output_paths(args, "static_etth1")
    log = StructuralLog(hist_path)

    init_edges = model.edge_set()
    init_va_mse = evaluate(model, Xva, Yva)[0]
    init_te_mse, init_te_mae = evaluate(model, Xte, Yte)
    log.log(0, unit_count(model), edge_count(model),
            edges=init_edges, val_loss=init_va_mse,
            train_loss=None, test_mse=init_te_mse, test_mae=init_te_mae)

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    print(f"[info] device={device}  in_dim={X.shape[1]}  out_dim={Y.shape[1]}  "
          f"hidden={args.hidden}  depth={args.depth}  epochs={epochs}  "
          f"train={n_tr}  val={n_va}  test={len(Xte)}")

    timer = PhaseTimer()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, loader, opt, device, timer)
        va_mse, va_mae = evaluate(model, Xva, Yva)
        te_mse, te_mae = evaluate(model, Xte, Yte)
        log.log(ep, unit_count(model), edge_count(model),
                edges=init_edges,  # static: same set every step
                val_loss=va_mse, train_loss=tr, val_mae=va_mae,
                test_mse=te_mse, test_mae=te_mae)
        print(f"[ep {ep:>3d}] train={tr:.4f}  val_mse={va_mse:.4f}  "
              f"val_mae={va_mae:.4f}  test_mse={te_mse:.4f}")
    wall = time.perf_counter() - t0

    te_mse, te_mae = evaluate(model, Xte, Yte)
    log.flush()

    summary = {
        "workload": "01_static_etth1",
        "dataset": "ETTh1" if not args.synthetic else "synthetic-etth1",
        "in_len": args.in_len, "out_len": args.out_len,
        "hidden": args.hidden, "depth": args.depth,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "wall_seconds": round(wall, 3),
        "val_mse_final": round(log.records[-1]["val_loss"], 6),
        "test_mse": round(te_mse, 6),
        "test_mae": round(te_mae, 6),
        "n_units": unit_count(model),
        "n_edges": edge_count(model),
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_max": round(max(r["jaccard"] for r in log.records), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
    }
    write_summary_csv(
        [summary], summary_path,
        columns=list(summary.keys()),
    )
    print(f"[done] wall={wall:.1f}s  val_mse={summary['val_mse_final']:.4f}  "
          f"test_mse={te_mse:.4f}  test_mae={te_mae:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Static ETTh1 -- hidden={args.hidden} depth={args.depth}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "static_etth1")
        plot_test_curve(log.records, tpath,
                        title=f"Static ETTh1 test MSE -- hidden={args.hidden} "
                              f"depth={args.depth}",
                        metric_key="test_mse", ylabel="test MSE",
                        higher_is_better=False)
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--synthetic", action="store_true",
                   help="use a deterministic sinusoidal stand-in for ETTh1")
    p.add_argument("--in-len", type=int, default=96)
    p.add_argument("--out-len", type=int, default=24)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    # PyTorch SGD default tuned for sum-reduction MSE at batch=64. The
    # batched gradient is roughly batch*out_dim*2 times larger than a
    # single Plastix per-example update would produce; we offset that by
    # dropping the nominal lr well below 1e-3.
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()

    summary = run(args)
    if summary["val_mse_final"] > 1.2 and not args.quick:
        print(f"[warn] val_mse={summary['val_mse_final']:.3f} is high; "
              f"persistence baseline is ~1.0 on standardised data",
              file=sys.stderr)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Plastix phase-strategy description
# ---------------------------------------------------------------------------
#
# Static regime <-> Plastix policy slots:
#
#   Forward      : standard layered/topological forward; pass-policy is
#                  Map = w * x, Combine = +, Apply = activation.
#   Backward     : standard backprop with mirror Map/Combine/Apply.
#   UpdateUnit   : NoX (bias-as-unit-state can stay implicit).
#   UpdateConn   : SGD/Adam step on the weight field; one per connection.
#   PruneUnit    : NoX.
#   PruneConn    : NoX.
#   AddUnit      : NoX.
#   AddConn      : NoX.
#   ResetGlobal  : NoX (the loss closure handles aggregation).
#
# Step ordering reduces to: Forward -> Loss -> Backward -> UpdateConn ->
# ResetGlobal.  The other phases are compiled out via `if constexpr` on the
# NoX sentinels.  No level recomputation ever fires.  This is the cheapest
# possible step Plastix can execute, and serves as the denominator for every
# other workload's per-step cost.
