"""Workload 3 / 5 -- BURSTY (punctuated equilibrium) regime, snnTorch (SNN) port.

Expressibility: EXPRESSIBLE (with the same host-side surgery the pytorch/jax
ports use). A streaming binary classifier that sits structurally idle for many
steps, then -- when a sliding window of validation losses goes flat -- fires a
burst that adds hidden units + connections, followed immediately by a
magnitude-prune sweep to bound growth.

The spiking model is a 3-Linear-layer LIF MLP: in -> H -> H -> out, with LIF
hidden layers (surrogate gradient) and a non-spiking leaky *integrator* readout
(reset_mechanism="none") whose accumulated membrane over T timesteps is the
classification logit. Input is injected as a constant current each timestep
(same schema as 01 snn). Trained with BPTT (surrogate gradient) + plain SGD,
sum-reduction cross-entropy.

The crucial expressibility point: snnTorch's `snn.Leaky` layers carry NO
persistent per-neuron state across steps -- their membrane is re-initialised
every forward via `init_leaky()`. So growing a hidden layer is pure host
surgery on the adjacent `nn.Linear` weight matrices (exactly like the pytorch
impl); there is no LIF state tensor to resize. After each burst we rebuild the
growable Linear layers, expand the boolean weight masks, magnitude-prune, and
rebuild the optimizer -- then the very next forward runs the enlarged spiking
net over T timesteps with no state-shape mismatch.

Usage:
    uv run python 03_bursty_elec2/snn/run_benchmark.py --quick --no-plot
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
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
    MemoryProbe,
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


ELEC2_URL = (
    "https://raw.githubusercontent.com/scikit-multiflow/"
    "streaming-datasets/master/elec.csv"
)


# ---------------------------------------------------------------------------
# Data (mirrors the pytorch loader)
# ---------------------------------------------------------------------------

def load_elec2(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = data_dir / "elec2.csv"
    download_if_missing(ELEC2_URL, csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
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
    """Synthetic drifting binary classification stream. The decision
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
# Growable spiking MLP
# ---------------------------------------------------------------------------

class GrowableSpikingMLP(nn.Module):
    """Spiking MLP whose two hidden layers can grow at runtime.

    Structure: in_dim -> H (LIF) -> H (LIF) -> out_dim (leaky integrator).
    The three growable connection matrices are plain `nn.Linear` layers; the
    LIF units (`snn.Leaky`) are stateless between forwards -- their membrane is
    re-initialised each forward via init_leaky(), so nothing per-neuron needs
    resizing when H changes. Growth/prune are host surgery on the Linear
    weights (mirrors the pytorch impl), with per-layer boolean masks tracking
    pruned (dead) edges; the forward multiplies weights by their mask.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int, T: int,
                 device: str, beta: float = 0.9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.T = T
        self.device = device
        self.beta = beta
        self._grad = surrogate.fast_sigmoid()
        self._build_linears(
            [nn.Linear(in_dim, hidden),
             nn.Linear(hidden, hidden),
             nn.Linear(hidden, out_dim)])
        # Two spiking hidden layers + a non-spiking leaky integrator readout.
        # The LIF modules hold no state between forwards, so they are never
        # resized -- only the Linear weight matrices grow/shrink around them.
        self.lif1 = snn.Leaky(beta=beta, spike_grad=self._grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=self._grad)
        self.out_lif = snn.Leaky(beta=beta, spike_grad=self._grad,
                                 reset_mechanism="none")
        self.masks = [torch.ones_like(l.weight, dtype=torch.bool)
                      for l in self.layers]
        self.bursts = 0
        self.prunes = 0

    def _build_linears(self, layers) -> None:
        self.layers = nn.ModuleList(layers).to(self.device)

    # --- forward: T-step spike window ------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w0 = self.layers[0].weight * self.masks[0].to(self.layers[0].weight.dtype)
        w1 = self.layers[1].weight * self.masks[1].to(self.layers[1].weight.dtype)
        w2 = self.layers[2].weight * self.masks[2].to(self.layers[2].weight.dtype)
        b0, b1, b2 = (self.layers[0].bias, self.layers[1].bias,
                      self.layers[2].bias)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_mem = self.out_lif.init_leaky()
        acc = 0.0
        for _ in range(self.T):
            cur = x                              # constant-current input
            spk1, mem1 = self.lif1(F.linear(cur, w0, b0), mem1)
            spk2, mem2 = self.lif2(F.linear(spk1, w1, b1), mem2)
            _, out_mem = self.out_lif(F.linear(spk2, w2, b2), out_mem)
            acc = acc + out_mem
        return acc / self.T                      # mean-membrane readout logits

    # --- size / topology -------------------------------------------------

    def unit_count(self) -> int:
        return (self.layers[0].out_features + self.layers[1].out_features
                + self.out_dim)

    def edge_count(self) -> int:
        return int(sum(m.sum().item() for m in self.masks))

    def edge_set(self) -> set[tuple[int, int, int]]:
        edges = set()
        for li, m in enumerate(self.masks):
            idx = torch.nonzero(m, as_tuple=False).tolist()
            edges.update((li, r, c) for r, c in idx)
        return edges

    # --- growth ----------------------------------------------------------

    def grow(self, n_new: int, noise: float = 0.05) -> int:
        """Append n_new units to BOTH hidden layers, expanding adjacent weight
        matrices. Mirrors the pytorch grow(); the LIF layers are untouched
        because they hold no per-neuron state. Returns new hidden size."""
        new_h = self.hidden + n_new
        dev = self.layers[0].weight.device
        l0, l1, l2 = self.layers[0], self.layers[1], self.layers[2]
        # Layer 0: (hidden, in_dim) -> (new_h, in_dim): more output rows.
        w_new0 = torch.cat([
            l0.weight.detach(),
            noise * torch.randn(n_new, self.in_dim, device=dev),
        ], dim=0)
        b_new0 = torch.cat([l0.bias.detach(),
                            torch.zeros(n_new, device=dev)])
        # Layer 1: (hidden, hidden) -> (new_h, new_h): rows then columns.
        w_new1 = torch.cat([
            l1.weight.detach(),
            noise * torch.randn(n_new, self.hidden, device=dev),
        ], dim=0)
        w_new1 = torch.cat([
            w_new1,
            noise * torch.randn(new_h, n_new, device=dev),
        ], dim=1)
        b_new1 = torch.cat([l1.bias.detach(),
                            torch.zeros(n_new, device=dev)])
        # Layer 2: (out_dim, hidden) -> (out_dim, new_h): more input columns.
        w_new2 = torch.cat([
            l2.weight.detach(),
            noise * torch.randn(self.out_dim, n_new, device=dev),
        ], dim=1)
        b_new2 = l2.bias.detach().clone()

        self._build_linears([
            nn.Linear(self.in_dim, new_h),
            nn.Linear(new_h, new_h),
            nn.Linear(new_h, self.out_dim),
        ])
        with torch.no_grad():
            self.layers[0].weight.copy_(w_new0); self.layers[0].bias.copy_(b_new0)
            self.layers[1].weight.copy_(w_new1); self.layers[1].bias.copy_(b_new1)
            self.layers[2].weight.copy_(w_new2); self.layers[2].bias.copy_(b_new2)
        # Expand masks: new edges start alive; preserve old dead structure.
        new_masks = [torch.ones_like(l.weight, dtype=torch.bool)
                     for l in self.layers]
        m0 = self.masks[0]
        new_masks[0][:m0.shape[0], :m0.shape[1]] = m0
        m1 = self.masks[1]
        new_masks[1][:m1.shape[0], :m1.shape[1]] = m1
        m2 = self.masks[2]
        new_masks[2][:, :m2.shape[1]] = m2
        self.masks = new_masks
        self.hidden = new_h
        self.bursts += 1
        return new_h

    def magnitude_prune(self, prune_frac: float) -> int:
        """Kill `prune_frac` of currently-alive weights by smallest magnitude
        (global). Returns count killed. Bounds growth after a burst."""
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
        self.steps_since_burst = cooldown

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

    probe = MemoryProbe()
    probe.start()

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

    cut = max(int(0.1 * len(Xnp)), 256)
    mu = Xnp[:cut].mean(0); sd = Xnp[:cut].std(0)
    safe_sd = np.where(sd > 1e-3, sd, 1.0)
    Xnp = ((Xnp - mu) / safe_sd).astype(np.float32)
    in_dim = Xnp.shape[1]
    n_classes = int(ynp.max()) + 1

    X = torch.from_numpy(np.ascontiguousarray(Xnp)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(ynp)).to(device)

    n_train = int(0.85 * len(X))
    X_test = X[n_train:]
    y_test = y[n_train:]
    probe.end_dataset()

    T = max(3, args.timesteps // (2 if args.quick else 1))
    model = GrowableSpikingMLP(in_dim, n_classes, hidden=args.init_hidden,
                               T=T, device=device).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    probe.end_weights()

    hist_path, summary_path, plot_path = output_paths(args, "bursty_elec2")
    log = StructuralLog(hist_path)

    detector = PlateauDetector(window=args.plateau_window,
                               rel_tol=args.plateau_rel_tol,
                               cooldown=args.plateau_cooldown)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    plateau_start = min(int(0.1 * len(X)), n_train - args.val_window - 1)
    plateau_end = plateau_start + args.val_window
    X_plateau = X[plateau_start:plateau_end]
    y_plateau = y[plateau_start:plateau_end]
    print(f"[info] device={device}  T={T}  in_dim={in_dim}  "
          f"classes={n_classes}  data={dataset_name}  N={len(X)}  "
          f"steps={max_steps}")

    burst_log: list[dict] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    correct = 0; seen = 0
    for step in range(1, max_steps + 1):
        idx = (np.arange(args.batch) + (step - 1) * args.batch) % n_train
        xb = X[idx]; yb = y[idx]

        model.train()
        timer.tick()
        logits = model(xb)
        timer.mark_forward()
        loss = F.cross_entropy(logits, yb, reduction="sum")
        timer.mark_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        timer.mark_backward()
        opt.step()
        with torch.no_grad():
            for i, l in enumerate(model.layers):
                l.weight.mul_(model.masks[i].to(l.weight.dtype))
        timer.mark_update()
        with torch.no_grad():
            correct += (logits.argmax(-1) == yb).sum().item()
            seen += yb.size(0)

        should_burst = False
        vl = None
        if step % val_every == 0 or step == max_steps:
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

        # --- structural phases: sampled every step (mostly no-ops) --------
        if should_burst:
            n_new = max(1, int(args.burst_frac * model.hidden))
            old_h = model.hidden
            new_h = model.grow(n_new, noise=args.burst_noise)
            timer.mark_grow()
            killed = model.magnitude_prune(args.post_burst_prune_frac)
            timer.mark_prune()
            # Rebuild optimizer state for the resized parameters.
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            burst_log.append({"step": step, "old_h": old_h, "new_h": new_h,
                              "killed": killed, "val_loss": vl})
            print(f"[burst {len(burst_log):>2d}] step={step:>6d}  "
                  f"hidden {old_h}->{new_h}  killed={killed}  vl={vl:.4f}")
            correct = 0; seen = 0
        else:
            timer.mark_grow()
            timer.mark_prune()
        timer.mark_reset()
        timer.step_done()

        if vl is not None:
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

    wall = time.perf_counter() - t0
    log.flush()

    summary = {
        "workload": "03_bursty_elec2",
        "framework": "snntorch",
        "dataset": dataset_name,
        "in_dim": in_dim, "n_classes": n_classes,
        "init_hidden": args.init_hidden,
        "timesteps": T,
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
        "n_units": model.unit_count(),
        "n_edges": model.edge_count(),
        "edges_final": model.edge_count(),
        "val_loss_initial": round(log.records[0]["val_loss"], 6),
        "val_loss_final": round(log.records[-1]["val_loss"], 6),
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_max": round(max(r["jaccard"] for r in log.records), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  bursts={model.bursts}  "
          f"hidden_final={model.hidden}  edges={model.edge_count()}  "
          f"val_loss_final={summary['val_loss_final']:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Bursty Elec2 (SNN) -- init_h={args.init_hidden} "
                       f"burst={args.burst_frac:.2f} T={T}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "bursty_elec2")
        plot_test_curve(log.records, tpath,
                        title=f"Bursty Elec2 (SNN) test accuracy -- "
                              f"init_h={args.init_hidden} "
                              f"burst={args.burst_frac:.2f}",
                        metric_key="test_acc", ylabel="test accuracy",
                        higher_is_better=True)
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot
    p.add_argument("--synthetic", action="store_true",
                   help="force synthetic drifting stream instead of Elec2")
    p.add_argument("--init-hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-window", type=int, default=512)
    p.add_argument("--timesteps", type=int, default=20,
                   help="SNN spike-window T (--quick halves)")
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
