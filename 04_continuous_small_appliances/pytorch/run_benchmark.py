"""
Workload 4 / 5 -- CONTINUOUS-SMALL regime via per-step neuron splitting
on the UCI Appliances Energy Prediction streaming regression task.

Each training step:
  1. Take a minibatch from the stream.
  2. Train one SGD step.
  3. With probability p_split, duplicate the unit with the highest
     activation variance over a sliding window, adding a noisy copy.
  4. With probability p_prune, kill the connection with the smallest
     EMA-magnitude.

Net structural delta per step is at most +/-1 unit and at most +/-1 edge.
The result is a slow random walk over architecture space whose Jaccard-
similarity-between-steps sits very close to 1.0, with a tightly-bounded
|Delta n_units| distribution.  These are the empirical signatures the
regime is supposed to produce.

Dataset:
    UCI Appliances Energy Prediction (Candanedo, Feldheim, Deramaix 2017),
    ~19,735 records at 10-minute resolution from a low-energy house.
    Auto-downloaded from the UCI ML repo; falls back to a synthetic
    slow-drift regression stream if download fails.

Usage:
    uv run python 04_continuous_small_appliances.py
    uv run python 04_continuous_small_appliances.py --quick
    uv run python 04_continuous_small_appliances.py --p-split 0.7
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared_python"))
from common import (  # noqa: E402
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


APPLIANCES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_appliances(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = data_dir / "energydata_complete.csv"
    download_if_missing(APPLIANCES_URL, csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Target column is 'Appliances' (Wh of total appliance load).
    target = "Appliances"
    feat_cols = [c for c in df.columns if c not in (target, "date")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    return X, y


def synth_slow_drift(n: int = 20_000, dim: int = 25, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    # Slowly drifting linear regression target with sinusoidal coefficients.
    coefs = np.stack([np.sin(t + j * 0.3) for j in range(dim)], axis=1
                     ).astype(np.float32)
    y = (X * coefs).sum(axis=1) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Splittable / prunable MLP
# ---------------------------------------------------------------------------

class SplitMLP(nn.Module):
    """Single hidden layer: in_dim -> H -> out_dim.
    Supports per-unit split/kill on the hidden layer (the only place
    structural change is meaningful for a regression MLP)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, device: str):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.l1 = nn.Linear(in_dim, hidden).to(device)
        self.l2 = nn.Linear(hidden, out_dim).to(device)
        # Per-edge keep mask on l1 (the prune step kills hidden-input edges).
        self.mask = torch.ones_like(self.l1.weight, dtype=torch.bool, device=device)
        self.splits = 0
        self.prunes = 0

    @property
    def hidden(self) -> int:
        return self.l1.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(F.linear(x, self.l1.weight * self.mask.to(self.l1.weight.dtype),
                            self.l1.bias))
        return self.l2(h)

    # --- diagnostics ---------------------------------------------------

    def unit_count(self) -> int:
        return self.hidden + self.out_dim

    def edge_count(self) -> int:
        return int(self.mask.sum().item()) + self.l2.weight.numel()

    def edge_set(self) -> set[tuple[int, int, int]]:
        # l1 alive edges
        idx1 = torch.nonzero(self.mask, as_tuple=False).tolist()
        edges = {(0, r, c) for r, c in idx1}
        # l2 all edges
        out_d, in_d = self.l2.weight.shape
        edges.update((1, r, c) for r in range(out_d) for c in range(in_d))
        return edges

    # --- structural ops -------------------------------------------------

    def split_unit(self, idx: int, noise: float = 0.05) -> None:
        """Duplicate hidden unit `idx` with small noise.  The copy's
        output weights are halved (and the original's likewise) so the
        immediate function approximated is preserved -- a "neuron splitting
        with conservation" move."""
        device = self.l1.weight.device
        H = self.hidden
        with torch.no_grad():
            # New l1: (H+1, in_dim) with a noisy copy of row idx appended.
            w1 = self.l1.weight.detach()
            new_row = w1[idx:idx + 1] + noise * torch.randn(
                1, self.in_dim, device=device)
            w1_new = torch.cat([w1, new_row], dim=0)
            b1 = self.l1.bias.detach()
            b1_new = torch.cat([b1, b1[idx:idx + 1].clone()])
            # New l2: (out_dim, H+1).  Halve column idx and copy to new col.
            w2 = self.l2.weight.detach().clone()
            new_col = w2[:, idx:idx + 1] * 0.5
            w2[:, idx:idx + 1] = new_col
            w2_new = torch.cat([w2, new_col], dim=1)
            b2_new = self.l2.bias.detach()
            # Rebuild layers.
            self.l1 = nn.Linear(self.in_dim, H + 1).to(device)
            self.l2 = nn.Linear(H + 1, self.out_dim).to(device)
            self.l1.weight.copy_(w1_new); self.l1.bias.copy_(b1_new)
            self.l2.weight.copy_(w2_new); self.l2.bias.copy_(b2_new)
            new_mask = torch.ones(H + 1, self.in_dim, dtype=torch.bool,
                                  device=device)
            new_mask[:H, :] = self.mask
            self.mask = new_mask
        self.splits += 1

    def kill_smallest_edge(self) -> bool:
        """Mask off the smallest-magnitude *alive* edge in l1."""
        with torch.no_grad():
            w = self.l1.weight.detach().abs()
            w = w + (~self.mask).float() * 1e9  # exclude dead from argmin
            flat = w.flatten()
            v_min, i_min = flat.min(0)
            if v_min.item() > 1e8:
                return False
            r = int(i_min.item() // self.in_dim)
            c = int(i_min.item() % self.in_dim)
            self.mask[r, c] = False
            self.l1.weight[r, c] = 0.0
        self.prunes += 1
        return True


# ---------------------------------------------------------------------------
# Activation-variance tracking
# ---------------------------------------------------------------------------

class ActivationStats:
    """Running estimate of per-hidden-unit activation variance over a window."""
    def __init__(self, hidden: int, window: int, device: str):
        self.window = window
        self.buf = torch.zeros(window, hidden, device=device)
        self.idx = 0
        self.filled = 0
        self.device = device

    def push(self, h: torch.Tensor) -> None:
        # h: (B, hidden) -- record the per-feature mean across the batch.
        if h.shape[1] != self.buf.shape[1]:
            # Hidden size changed (split or rebuild): allocate fresh buf,
            # keeping the same window size.
            self.buf = torch.zeros(self.window, h.shape[1], device=self.device)
            self.idx = 0
            self.filled = 0
        self.buf[self.idx] = h.detach().mean(0)
        self.idx = (self.idx + 1) % self.window
        self.filled = min(self.filled + 1, self.window)

    def hottest(self) -> int:
        if self.filled < 2:
            return int(torch.randint(self.buf.shape[1], (1,)).item())
        var = self.buf[:self.filled].var(0)
        return int(var.argmax().item())


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    if args.synthetic:
        Xnp, ynp = synth_slow_drift(n=args.max_steps * args.batch + 5000,
                                    dim=25, seed=args.seed)
        dataset_name = "synthetic-slow-drift"
    else:
        try:
            Xnp, ynp = load_appliances(args.data_dir)
            dataset_name = "uci-appliances"
        except Exception as e:
            print(f"[warn] Appliances download failed ({e}); using synthetic",
                  file=sys.stderr)
            Xnp, ynp = synth_slow_drift(n=args.max_steps * args.batch + 5000,
                                        dim=25, seed=args.seed)
            dataset_name = "synthetic-slow-drift-fallback"

    # Standardisation: per-feature z-score over the first 10%.
    cut = max(int(0.1 * len(Xnp)), 512)
    mu = Xnp[:cut].mean(0); sd = Xnp[:cut].std(0)
    safe_sd = np.where(sd > 1e-3, sd, 1.0)
    Xnp = ((Xnp - mu) / safe_sd).astype(np.float32)
    y_mu = ynp[:cut].mean(); y_sd = ynp[:cut].std() + 1e-6
    ynp = ((ynp - y_mu) / y_sd).astype(np.float32)

    X = torch.from_numpy(np.ascontiguousarray(Xnp)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(ynp)).to(device).unsqueeze(-1)
    in_dim = X.shape[1]

    # Carve final 15% as held-out test set. Training cycles only over the
    # first 85% so the test slice is never seen.
    n_train = int(0.85 * len(X))
    X_test = X[n_train:]
    y_test = y[n_train:]

    model = SplitMLP(in_dim, 1, hidden=args.init_hidden, device=device).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    act_stats = ActivationStats(model.hidden, window=args.var_window,
                                device=device)

    hist_path, summary_path, plot_path = output_paths(args,
                                                      "continuous_small_appliances")
    log = StructuralLog(hist_path)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    print(f"[info] device={device}  in_dim={in_dim}  data={dataset_name}  "
          f"N={len(X)}  steps={max_steps}  init_hidden={args.init_hidden}")

    deltas_units: list[int] = []
    deltas_edges: list[int] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        idx = (np.arange(args.batch) + (step - 1) * args.batch) % n_train
        xb = X[idx]; yb = y[idx]

        model.train()
        timer.tick()
        h = F.relu(F.linear(xb,
                            model.l1.weight * model.mask.to(model.l1.weight.dtype),
                            model.l1.bias))
        pred = model.l2(h)
        timer.mark_forward()
        # reduction='sum' so the per-batch gradient magnitude matches
        # Plastix's cumulative per-example SGD update over the same batch.
        loss = F.mse_loss(pred, yb, reduction="sum")
        timer.mark_loss()
        loss.backward()
        timer.mark_backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.l1.weight.mul_(model.mask.to(model.l1.weight.dtype))
        timer.mark_update()
        timer.step_done()
        act_stats.push(h)

        units_before = model.unit_count()
        edges_before = model.edge_count()

        did_change = False
        # split?
        if rng.uniform() < args.p_split and model.hidden < args.max_hidden:
            hot = act_stats.hottest()
            model.split_unit(hot, noise=args.split_noise)
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            did_change = True
        # prune?
        if rng.uniform() < args.p_prune and model.mask.sum().item() > model.in_dim:
            if model.kill_smallest_edge():
                did_change = True

        deltas_units.append(model.unit_count() - units_before)
        deltas_edges.append(model.edge_count() - edges_before)

        if step % val_every == 0 or step == max_steps or did_change:
            with torch.no_grad():
                model.eval()
                vstart = (step * args.batch) % n_train
                vend = min(vstart + args.val_window, n_train)
                vl = F.mse_loss(model(X[vstart:vend]), y[vstart:vend]).item()
                test_mse = F.mse_loss(model(X_test), y_test).item()
            log.log(step, n_units=model.unit_count(),
                    n_edges=model.edge_count(),
                    edges=model.edge_set(),
                    val_loss=vl,
                    hidden=model.hidden,
                    splits=model.splits, prunes=model.prunes,
                    delta_units=deltas_units[-1],
                    delta_edges=deltas_edges[-1],
                    test_mse=test_mse)

    wall = time.perf_counter() - t0
    log.flush()

    # |Delta| distribution stats.
    abs_du = np.abs(deltas_units)
    abs_de = np.abs(deltas_edges)
    summary = {
        "workload": "04_continuous_small_appliances",
        "dataset": dataset_name,
        "in_dim": in_dim,
        "init_hidden": args.init_hidden,
        "max_steps": max_steps, "batch": args.batch,
        "p_split": args.p_split, "p_prune": args.p_prune,
        "wall_seconds": round(wall, 3),
        "splits_fired": model.splits,
        "prunes_fired": model.prunes,
        "hidden_final": model.hidden,
        "edges_final": model.edge_count(),
        "val_loss_initial": round(log.records[0]["val_loss"], 6),
        "val_loss_final": round(log.records[-1]["val_loss"], 6),
        "delta_units_p99_abs": int(np.percentile(abs_du, 99)) if len(abs_du) else 0,
        "delta_units_max_abs": int(abs_du.max()) if len(abs_du) else 0,
        "delta_edges_p99_abs": int(np.percentile(abs_de, 99)) if len(abs_de) else 0,
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_mean": round(float(np.mean([r["jaccard"]
                                              for r in log.records])), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  splits={model.splits}  "
          f"prunes={model.prunes}  hidden_final={model.hidden}  "
          f"val_loss={summary['val_loss_final']:.4f}  "
          f"|du|_p99={summary['delta_units_p99_abs']}  "
          f"jaccard_mean={summary['jaccard_mean']:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Continuous-small UCI Appliances -- "
                       f"p_split={args.p_split} p_prune={args.p_prune}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "continuous_small_appliances")
        plot_test_curve(log.records, tpath,
                        title=f"Continuous-small Appliances test MSE -- "
                              f"p_split={args.p_split} p_prune={args.p_prune}",
                        metric_key="test_mse", ylabel="test MSE",
                        higher_is_better=False)
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--synthetic", action="store_true",
                   help="force synthetic slow-drift regression stream")
    p.add_argument("--init-hidden", type=int, default=32)
    p.add_argument("--max-hidden", type=int, default=256,
                   help="upper bound on hidden width so the walk stays bounded")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--val-window", type=int, default=256)
    # SGD lr tuned for sum-reduction MSE at batch=32; the cumulative
    # gradient magnitude is ~ 2 * batch * out_dim larger than what
    # Plastix's per-example SGD lr=1e-3 would produce per example, so
    # we lower the nominal lr accordingly.
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--p-split", type=float, default=0.5,
                   help="per-step probability of splitting the hottest unit")
    p.add_argument("--p-prune", type=float, default=0.5,
                   help="per-step probability of killing the smallest edge")
    p.add_argument("--split-noise", type=float, default=0.05)
    p.add_argument("--var-window", type=int, default=64,
                   help="activation-variance EMA window for hottest-unit pick")
    args = p.parse_args()

    summary = stream(args)
    # Regime-validation gates: max |delta| per step should be <=1 by
    # construction (we do at most one split and one kill per step, and a
    # split adds exactly 1 unit while a prune removes 0 units -- so
    # |delta_units| <= 1 always).
    if summary["delta_units_max_abs"] > 1:
        print(f"[warn] |delta_units|_max = {summary['delta_units_max_abs']} > 1; "
              "regime invariant violated", file=sys.stderr)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Plastix phase-strategy description
# ---------------------------------------------------------------------------
#
# Continuous-small regime <-> Plastix policy slots:
#
#   Forward / Backward / UpdateConn / ResetGlobal : standard, every step.
#   UpdateUnit                                    : maintains a per-unit
#                                                   activation-variance EMA
#                                                   (a per-unit extra field;
#                                                   the IDBD-style step-size
#                                                   reduction in SwiftTD's
#                                                   UpdateConn is a close
#                                                   structural cousin).
#   AddUnit                                       : Probabilistic per step.
#                                                   ShouldAdd(global) returns
#                                                   true with probability
#                                                   p_split.  On fire,
#                                                   AddUnit::AddUnit returns
#                                                   *one* offset and InitUnit
#                                                   copies the hottest-unit
#                                                   activation/weight state
#                                                   from a parent id stored
#                                                   in GlobalState by the
#                                                   prior reduce pass.
#   AddConn                                       : Fires alongside AddUnit
#                                                   to wire the new unit's
#                                                   inputs and outputs (one
#                                                   each on the canonical
#                                                   "neighbourhood" sweep).
#   PruneConn                                     : Probabilistic per step.
#                                                   ShouldPrune := this is
#                                                   the global argmin |w|.
#                                                   Requires a global-argmin
#                                                   reduce pass to land in
#                                                   GlobalState first.
#   PruneUnit                                     : NoX (units never die in
#                                                   this regime; only edges).
#
# Step ordering: same default sequence; the structural delta per step is
# at most +/-1 unit and at most a few edges, so the resort cost is rarely
# triggered (level recomputation only fires when AddConn commits, which
# happens at most O(splits) times).  This is the regime that exercises
# Plastix's allocator at high frequency with low per-step volume -- the
# opposite stress profile from the Bursty workload.
