"""
Workload 3 / 5 -- BURSTY (punctuated equilibrium) regime via plateau-
triggered neurogenesis on the Electricity (Elec2) streaming benchmark.

The network sits structurally idle for many steps, then -- when a sliding
window of validation losses goes flat -- fires a burst event that adds
new hidden units (roughly 10-20% of current size) plus their attendant
connections, and immediately schedules a magnitude-prune sweep that
trims a chunk of the smallest weights to keep total size bounded.
This is the "punctuated equilibrium" pattern: long flat stretches in
unit_count interrupted by sharp jumps and matching small dips.

Dataset:
    Elec2 / Electricity NSW (Harries 1999), 45,312 half-hour records of
    NSW electricity prices, binary "price up vs price down" target.
    Auto-downloaded from a public CSV mirror; falls back to a synthetic
    drifting-classification stream if the download fails.

Usage:
    uv run python 03_bursty_elec2.py                 # full
    uv run python 03_bursty_elec2.py --quick         # smoke
    uv run python 03_bursty_elec2.py --burst-frac 0.25
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


# Public CSV mirror of the Elec2 dataset, hosted by the scikit-multiflow
# streaming-datasets repo.  The dataset has been redistributed widely; this
# URL is one of the more stable copies but is not authoritative.
ELEC2_URL = (
    "https://raw.githubusercontent.com/scikit-multiflow/"
    "streaming-datasets/master/elec.csv"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_elec2(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = data_dir / "elec2.csv"
    download_if_missing(ELEC2_URL, csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
    # The mirror uses column name 'class' for the target (up/down).
    # Some variants use 'UP'/'DOWN' strings, others 0/1.  Normalise to {0,1}.
    target_col = "class" if "class" in df.columns else df.columns[-1]
    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col]
    if y_raw.dtype == object:
        uniq = sorted(y_raw.unique())
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    else:
        y = y_raw.to_numpy(dtype=np.int64)
    return X, y


def synth_drift(n: int = 30_000, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic drifting binary classification stream.  The decision
    hyperplane rotates over `n` steps so the network is forced to keep
    learning (and plateauing then growing) throughout."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    t = np.linspace(0, 4 * np.pi, n)
    for i in range(n):
        w = np.array([math.sin(t[i] + j) for j in range(dim)], dtype=np.float32)
        y[i] = int(X[i] @ w > 0)
    return X, y


# ---------------------------------------------------------------------------
# Growable MLP
# ---------------------------------------------------------------------------

class GrowableMLP(nn.Module):
    """MLP whose middle hidden layer can grow.  Internally three Linear
    layers: in_dim -> H -> H -> out_dim.  Growing means swapping out
    layers[1].weight (H_old x H_old) for (H_new x H_old) and similarly
    enlarging layers[0] (H_new x in_dim) and layers[2] (out_dim x H_new).

    For simplicity and reproducibility every grow operation widens BOTH
    hidden layers by the same amount."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, device: str):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.hidden = hidden
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, out_dim),
        ]).to(device)
        # Per-layer weight masks for prune-after-burst bounding.
        self.masks = [torch.ones_like(l.weight, dtype=torch.bool) for l in self.layers]
        self.bursts = 0
        self.prunes = 0

    # --- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.linear(x, layer.weight * self.masks[i].to(layer.weight.dtype),
                         layer.bias)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

    # --- size / topology -----------------------------------------------

    def unit_count(self) -> int:
        return self.layers[0].out_features + self.layers[1].out_features + self.out_dim

    def edge_count(self) -> int:
        return int(sum(m.sum().item() for m in self.masks))

    def edge_set(self) -> set[tuple[int, int, int]]:
        edges = set()
        for li, m in enumerate(self.masks):
            idx = torch.nonzero(m, as_tuple=False).tolist()
            edges.update((li, r, c) for r, c in idx)
        return edges

    # --- growth ---------------------------------------------------------

    def grow(self, n_new: int, noise: float = 0.05) -> int:
        """Append n_new units to BOTH hidden layers, expanding adjacent
        weight matrices accordingly.  Returns the new total hidden size."""
        new_h = self.hidden + n_new
        # Layer 0: (hidden, in_dim) -> (new_h, in_dim)
        l0 = self.layers[0]
        w_new0 = torch.cat([
            l0.weight.detach(),
            noise * torch.randn(n_new, self.in_dim, device=l0.weight.device),
        ], dim=0)
        b_new0 = torch.cat([l0.bias.detach(),
                             torch.zeros(n_new, device=l0.bias.device)])
        # Layer 1: (hidden, hidden) -> (new_h, new_h).
        l1 = self.layers[1]
        # First widen rows ("more outputs"), then widen columns ("more inputs").
        w_new1 = torch.cat([
            l1.weight.detach(),
            noise * torch.randn(n_new, self.hidden, device=l1.weight.device),
        ], dim=0)
        w_new1 = torch.cat([
            w_new1,
            noise * torch.randn(new_h, n_new, device=l1.weight.device),
        ], dim=1)
        b_new1 = torch.cat([l1.bias.detach(),
                             torch.zeros(n_new, device=l1.bias.device)])
        # Layer 2: (out_dim, hidden) -> (out_dim, new_h)
        l2 = self.layers[2]
        w_new2 = torch.cat([
            l2.weight.detach(),
            noise * torch.randn(self.out_dim, n_new, device=l2.weight.device),
        ], dim=1)
        b_new2 = l2.bias.detach()

        # rebuild
        device = l0.weight.device
        self.layers = nn.ModuleList([
            nn.Linear(self.in_dim, new_h),
            nn.Linear(new_h, new_h),
            nn.Linear(new_h, self.out_dim),
        ]).to(device)
        with torch.no_grad():
            self.layers[0].weight.copy_(w_new0); self.layers[0].bias.copy_(b_new0)
            self.layers[1].weight.copy_(w_new1); self.layers[1].bias.copy_(b_new1)
            self.layers[2].weight.copy_(w_new2); self.layers[2].bias.copy_(b_new2)
        # Expand masks: new edges all start alive.
        new_masks = [torch.ones_like(l.weight, dtype=torch.bool) for l in self.layers]
        # Preserve old dead-edge structure on the original block.
        old_m0 = self.masks[0]
        new_masks[0][:old_m0.shape[0], :old_m0.shape[1]] = old_m0
        old_m1 = self.masks[1]
        new_masks[1][:old_m1.shape[0], :old_m1.shape[1]] = old_m1
        old_m2 = self.masks[2]
        new_masks[2][:, :old_m2.shape[1]] = old_m2
        self.masks = new_masks
        self.hidden = new_h
        self.bursts += 1
        return new_h

    def magnitude_prune(self, prune_frac: float) -> int:
        """Kill `prune_frac` of currently-alive weights by smallest magnitude
        (global).  Returns count killed.  Bounds growth after a burst."""
        flat = torch.cat([
            l.weight[self.masks[i]].abs().detach().flatten()
            for i, l in enumerate(self.layers)
        ])
        k = int(prune_frac * flat.numel())
        if k <= 0:
            return 0
        thresh = torch.kthvalue(flat, k).values.item()
        killed = 0
        for i, l in enumerate(self.layers):
            kill = (l.weight.detach().abs() <= thresh) & self.masks[i]
            killed += int(kill.sum().item())
            self.masks[i] = self.masks[i] & ~kill
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                l.weight.mul_(self.masks[i].to(l.weight.dtype))
        self.prunes += 1
        return killed


# ---------------------------------------------------------------------------
# Plateau detector
# ---------------------------------------------------------------------------

class PlateauDetector:
    def __init__(self, window: int, rel_tol: float, cooldown: int):
        self.win = deque(maxlen=window)
        self.window = window
        self.rel_tol = rel_tol
        self.cooldown = cooldown
        self.steps_since_burst = cooldown  # allow firing immediately if data warrants

    def update(self, loss: float) -> bool:
        self.win.append(float(loss))
        self.steps_since_burst += 1
        if len(self.win) < self.window:
            return False
        if self.steps_since_burst < self.cooldown:
            return False
        mu = sum(self.win) / len(self.win)
        if mu <= 0:
            return False
        std = (sum((x - mu) ** 2 for x in self.win) / len(self.win)) ** 0.5
        if std / mu < self.rel_tol:
            self.steps_since_burst = 0
            return True
        return False


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    if args.synthetic:
        Xnp, ynp = synth_drift(n=args.max_steps * args.batch + 5000, dim=8,
                               seed=args.seed)
        dataset_name = "synthetic-drift"
    else:
        try:
            Xnp, ynp = load_elec2(args.data_dir)
            dataset_name = "elec2"
        except Exception as e:
            print(f"[warn] Elec2 download failed ({e}); using synthetic drift",
                  file=sys.stderr)
            Xnp, ynp = synth_drift(n=args.max_steps * args.batch + 5000, dim=8,
                                   seed=args.seed)
            dataset_name = "synthetic-drift-fallback"

    # Per-feature standardisation over the first 10% of the stream (avoids
    # leakage from later concept-drift phases).  Features with near-zero
    # variance in the early stretch (e.g. vicprice in Elec2 is constant
    # for the first few hundred rows) get left at zero rather than
    # amplified into the thousands.
    cut = max(int(0.1 * len(Xnp)), 256)
    mu = Xnp[:cut].mean(0); sd = Xnp[:cut].std(0)
    safe_sd = np.where(sd > 1e-3, sd, 1.0)
    Xnp = ((Xnp - mu) / safe_sd).astype(np.float32)
    in_dim = Xnp.shape[1]
    n_classes = int(ynp.max()) + 1

    X = torch.from_numpy(np.ascontiguousarray(Xnp)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(ynp)).to(device)

    # Carve final 15% as held-out test set. Training cycles only over the
    # first 85% so the test slice is never seen.
    n_train = int(0.85 * len(X))
    X_test = X[n_train:]
    y_test = y[n_train:]

    model = GrowableMLP(in_dim, n_classes, hidden=args.init_hidden,
                        device=device).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    hist_path, summary_path, plot_path = output_paths(args, "bursty_elec2")
    log = StructuralLog(hist_path)

    detector = PlateauDetector(window=args.plateau_window,
                               rel_tol=args.plateau_rel_tol,
                               cooldown=args.plateau_cooldown)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    # Static held-out window used solely for the plateau detector.  Reading
    # the *same* slice each time isolates "model has stopped learning" from
    # "the next slice happens to be hard" -- the latter is concept drift,
    # not a learning plateau, and would fire spurious bursts. Confined to
    # the training portion so it never overlaps the test split.
    plateau_start = min(int(0.1 * len(X)), n_train - args.val_window - 1)
    plateau_end = plateau_start + args.val_window
    X_plateau = X[plateau_start:plateau_end]
    y_plateau = y[plateau_start:plateau_end]
    print(f"[info] device={device}  in_dim={in_dim}  classes={n_classes}  "
          f"data={dataset_name}  N={len(X)}  steps={max_steps}")

    burst_log: list[dict] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    correct = 0; seen = 0
    for step in range(1, max_steps + 1):
        # Sample a minibatch (treat the stream as cyclic so synthetic / short
        # data can still drive `max_steps` training updates). Indices wrap
        # within the training portion only so the held-out test slice is
        # never seen during training.
        idx = (np.arange(args.batch) + (step - 1) * args.batch) % n_train
        xb = X[idx]; yb = y[idx]

        model.train()
        timer.tick()
        logits = model(xb)
        timer.mark_forward()
        # reduction='sum' so the per-batch gradient magnitude matches
        # Plastix's cumulative per-example SGD update over the same batch.
        loss = F.cross_entropy(logits, yb, reduction="sum")
        timer.mark_loss()
        loss.backward()
        timer.mark_backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for i, l in enumerate(model.layers):
                l.weight.mul_(model.masks[i].to(l.weight.dtype))
        timer.mark_update()
        timer.step_done()
        with torch.no_grad():
            correct += (logits.argmax(-1) == yb).sum().item()
            seen += yb.size(0)

        if step % val_every == 0 or step == max_steps:
            # Three reads:
            #   - plateau_loss is the static held-out slice -- noiseless
            #     enough to drive the plateau detector
            #   - stream_loss is the upcoming (training-portion) slice --
            #     captures concept-drift effects so the plot tells the
            #     right story
            #   - test_loss / test_acc is the final 15% slice (never
            #     trained on) -- the per-step generalisation curve
            with torch.no_grad():
                model.eval()
                plateau_loss = F.cross_entropy(model(X_plateau),
                                               y_plateau).item()
                vstart = (step * args.batch) % n_train
                vend = min(vstart + args.val_window, n_train)
                vl = F.cross_entropy(model(X[vstart:vend]),
                                     y[vstart:vend]).item()
                va = (model(X[vstart:vend]).argmax(-1) == y[vstart:vend]
                      ).float().mean().item()
                test_logits = model(X_test)
                test_loss = F.cross_entropy(test_logits, y_test).item()
                test_acc = (test_logits.argmax(-1) == y_test
                            ).float().mean().item()
            should_burst = detector.update(plateau_loss)

            extras = {
                "train_acc_run": correct / max(seen, 1),
                "val_acc": va,
                "plateau_loss": plateau_loss,
                "bursts": model.bursts, "prunes": model.prunes,
                "test_acc": test_acc,
                "test_loss": test_loss,
            }
            log.log(step, n_units=model.unit_count(),
                    n_edges=model.edge_count(),
                    edges=model.edge_set(),
                    val_loss=vl, **extras)

            if should_burst:
                n_new = max(1, int(args.burst_frac * model.hidden))
                old_h = model.hidden
                new_h = model.grow(n_new, noise=args.burst_noise)
                killed = model.magnitude_prune(args.post_burst_prune_frac)
                burst_log.append({"step": step, "old_h": old_h, "new_h": new_h,
                                  "killed": killed, "val_loss": vl})
                print(f"[burst {len(burst_log):>2d}] step={step:>6d}  "
                      f"hidden {old_h}->{new_h}  killed={killed}  vl={vl:.4f}")
                # Rebuild optimizer state for the resized parameters.
                opt = torch.optim.SGD(model.parameters(), lr=args.lr)
                correct = 0; seen = 0  # reset running tally to focus on post-burst

    wall = time.perf_counter() - t0
    log.flush()

    summary = {
        "workload": "03_bursty_elec2",
        "dataset": dataset_name,
        "in_dim": in_dim, "n_classes": n_classes,
        "init_hidden": args.init_hidden,
        "max_steps": max_steps, "batch": args.batch,
        "burst_frac": args.burst_frac,
        "post_burst_prune_frac": args.post_burst_prune_frac,
        "plateau_window": args.plateau_window,
        "plateau_rel_tol": args.plateau_rel_tol,
        "plateau_cooldown": args.plateau_cooldown,
        "wall_seconds": round(wall, 3),
        "bursts_fired": model.bursts,
        "prunes_fired": model.prunes,
        "hidden_final": model.hidden,
        "edges_final": model.edge_count(),
        "val_loss_initial": round(log.records[0]["val_loss"], 6),
        "val_loss_final": round(log.records[-1]["val_loss"], 6),
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_max": round(max(r["jaccard"] for r in log.records), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  bursts={model.bursts}  "
          f"hidden_final={model.hidden}  edges={model.edge_count()}  "
          f"val_loss_final={summary['val_loss_final']:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Bursty Elec2 -- init_h={args.init_hidden} "
                       f"burst={args.burst_frac:.2f}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "bursty_elec2")
        plot_test_curve(log.records, tpath,
                        title=f"Bursty Elec2 test accuracy -- "
                              f"init_h={args.init_hidden} "
                              f"burst={args.burst_frac:.2f}",
                        metric_key="test_acc", ylabel="test accuracy",
                        higher_is_better=True)
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--synthetic", action="store_true",
                   help="force synthetic drifting stream instead of Elec2")
    p.add_argument("--init-hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-window", type=int, default=512)
    # SGD lr tuned for sum-reduction cross-entropy at batch=64. Plastix's
    # per-example SGD lr=1e-3 with sum-reduction CE corresponds to
    # roughly 1e-3/64 here.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--burst-frac", type=float, default=0.15)
    p.add_argument("--burst-noise", type=float, default=0.05)
    p.add_argument("--post-burst-prune-frac", type=float, default=0.10)
    p.add_argument("--plateau-window", type=int, default=8,
                   help="number of validation snapshots in the plateau window")
    p.add_argument("--plateau-rel-tol", type=float, default=0.03,
                   help="std/mean threshold below which we count as flat")
    p.add_argument("--plateau-cooldown", type=int, default=4,
                   help="min validation snapshots between bursts")
    args = p.parse_args()

    summary = stream(args)
    if not args.quick and summary["bursts_fired"] == 0:
        print("[warn] no bursts fired; try smaller --plateau-rel-tol or "
              "longer --max-steps", file=sys.stderr)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Plastix phase-strategy description
# ---------------------------------------------------------------------------
#
# Bursty regime <-> Plastix policy slots:
#
#   Forward / Backward / UpdateConn   : standard, fires every step.
#   ResetGlobal                       : Increments a step counter; refreshes
#                                       the plateau-detector window (kept in
#                                       GlobalState as a small ring buffer
#                                       of recent losses and a "fire burst"
#                                       boolean).
#   AddUnit                           : Guarded by `global.fire_burst`.  On
#                                       fire, AddUnit::AddUnit returns N
#                                       offsets (one per new unit) for a
#                                       single existing parent, and
#                                       InitUnit zero-initialises the
#                                       activation/forward-acc fields.
#                                       The "N at once" pattern is the
#                                       single most natural way to express
#                                       a burst in Plastix's per-step API.
#   AddConn                           : Also guarded by `global.fire_burst`.
#                                       ShouldAddIncomingConnection returns
#                                       true for source units within the
#                                       neighbourhood of newly-added units
#                                       (probabilistic, density ~ 0.3 in
#                                       this reference).
#   PruneConn                         : Fires only on the *post-burst*
#                                       step, when GlobalState's
#                                       `post_burst` flag is set.
#                                       ShouldPrune := |w| <= threshold,
#                                       with threshold computed in a
#                                       prior reduce pass.  Bounded growth
#                                       requires this prune to be paired
#                                       with the AddConn burst.
#
# Step ordering reduces to:
#   Forward -> Loss -> Backward -> UpdateConn ->
#   (PruneConn if post_burst) -> AddUnit -> AddConn ->
#   Resort (because AddConn committed) -> ResetGlobal
#
# Cost profile: most steps are static-equivalent; a few steps every couple
# hundred to a thousand pay the full structural-modification price.  That's
# the latency-spike profile the workload is built to expose -- Plastix's
# p95 / p99 forward-pass time *immediately after a burst* is the metric of
# interest, not steady-state throughput.
