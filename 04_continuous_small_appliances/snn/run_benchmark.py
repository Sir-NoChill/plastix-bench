"""Workload 4 / 5 -- CONTINUOUS-SMALL regime, snnTorch (SNN) port.

Mirrors 04_continuous_small_appliances/pytorch: a single-hidden-layer streaming
regression MLP trained with sum-reduction MSE + plain SGD on the UCI Appliances
task. Each step *maybe* splits the highest-activation-variance hidden unit (a
noisy, conservation-preserving copy wired to all inputs + the output) and *maybe*
prunes the single smallest-magnitude alive l1 edge. Net structural delta per step
is at most +/-1 unit and a small number of edges, so Jaccard-between-steps ~ 1.0.

Spiking realisation
-------------------
  * Hidden layer: nn.Linear(in -> H) -> snn.Leaky (LIF, surrogate gradient).
  * Readout:      nn.Linear(H -> 1)  -> snn.Leaky(reset_mechanism="none") -- a
    non-spiking leaky *integrator* whose membrane, averaged over the T-step spike
    window, is the continuous regression output (same readout as bench 01 snn).
  * Each forward runs T spiking timesteps with the (static, per-step) minibatch
    injected as a constant input current. LIF membranes are re-initialised each
    forward, so there is NO persistent recurrent state to resize -- the ONLY state
    that a split/prune touches is the two nn.Linear weight tensors.

Expressibility / cost
---------------------
Because the LIF membranes carry no cross-step state, per-step structural surgery
is exactly the host-side nn.Linear rebuild the pytorch impl uses -- snnTorch adds
no obstacle to it. The real cost is that this bench changes structure EVERY step,
so every step pays a full T-step BPTT (forward + backward unrolled T times) on top
of the rebuild. There is no torch.compile here (eager), so the "rebuild" is just a
tensor realloc, not a recompile -- cheap; the dominant cost is the T-fold BPTT.

Usage:
    uv run python 04_continuous_small_appliances/snn/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    resolve_device,
    write_summary_csv,
)

APPLIANCES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)


# --- data (mirrors the pytorch/jax loaders) --------------------------------

def load_appliances(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = data_dir / "energydata_complete.csv"
    download_if_missing(APPLIANCES_URL, csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
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
    coefs = np.stack([np.sin(t + j * 0.3) for j in range(dim)], axis=1
                     ).astype(np.float32)
    y = (X * coefs).sum(axis=1) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return X, y


# --- splittable / prunable spiking MLP -------------------------------------

class SpikingSplitMLP(nn.Module):
    """in_dim -> H (LIF) -> out_dim (leaky integrator readout).

    Structural change touches only l1 / l2 nn.Linear weights; the LIF/readout
    modules are stateless between forwards (membranes re-init each forward), so
    a split/prune is a plain host-side weight rebuild -- no persistent spiking
    state to resize."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, T: int,
                 device: str, beta: float = 0.9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T = T
        self.device = device
        grad = surrogate.fast_sigmoid()
        self.l1 = nn.Linear(in_dim, hidden).to(device)
        self.lif = snn.Leaky(beta=beta, spike_grad=grad)
        self.l2 = nn.Linear(hidden, out_dim).to(device)
        # Non-spiking leaky integrator readout: its membrane is the regression head.
        self.out_lif = snn.Leaky(beta=beta, spike_grad=grad,
                                 reset_mechanism="none")
        # Per-edge keep mask on l1 (prune kills hidden<-input edges).
        self.mask = torch.ones_like(self.l1.weight, dtype=torch.bool,
                                    device=device)
        self.splits = 0
        self.prunes = 0

    @property
    def hidden(self) -> int:
        return self.l1.out_features

    def forward(self, x: torch.Tensor, return_rate: bool = False):
        """Run T spiking steps; read out mean membrane of the output integrator.

        If return_rate, also return the per-unit mean hidden spike rate over the
        window (B, H) -- the SNN analogue of the ANN's hidden activation, used to
        pick the highest-variance unit to split."""
        w1 = self.l1.weight * self.mask.to(self.l1.weight.dtype)
        mem1 = self.lif.init_leaky()
        mem_out = self.out_lif.init_leaky()
        acc_out = 0.0
        rate = 0.0
        for _ in range(self.T):
            cur1 = torch.nn.functional.linear(x, w1, self.l1.bias)
            spk1, mem1 = self.lif(cur1, mem1)
            _, mem_out = self.out_lif(self.l2(spk1), mem_out)
            acc_out = acc_out + mem_out
            if return_rate:
                rate = rate + spk1
        out = acc_out / self.T
        if return_rate:
            return out, rate / self.T
        return out

    # --- diagnostics ---------------------------------------------------

    def unit_count(self) -> int:
        return self.hidden + self.out_dim

    def edge_count(self) -> int:
        return int(self.mask.sum().item()) + self.l2.weight.numel()

    def edge_set(self) -> set[tuple[int, int, int]]:
        idx1 = torch.nonzero(self.mask, as_tuple=False).tolist()
        edges = {(0, r, c) for r, c in idx1}
        out_d, in_d = self.l2.weight.shape
        edges.update((1, r, c) for r in range(out_d) for c in range(in_d))
        return edges

    # --- structural ops -------------------------------------------------

    def split_unit(self, idx: int, noise: float = 0.05) -> None:
        """Duplicate hidden unit `idx` (noisy copy), wired to all inputs and the
        output. Output weights halved (original + copy) so the function is
        preserved -- conservation-preserving neuron split."""
        device = self.l1.weight.device
        H = self.hidden
        with torch.no_grad():
            w1 = self.l1.weight.detach()
            new_row = w1[idx:idx + 1] + noise * torch.randn(
                1, self.in_dim, device=device)
            w1_new = torch.cat([w1, new_row], dim=0)
            b1 = self.l1.bias.detach()
            b1_new = torch.cat([b1, b1[idx:idx + 1].clone()])
            w2 = self.l2.weight.detach().clone()
            new_col = w2[:, idx:idx + 1] * 0.5
            w2[:, idx:idx + 1] = new_col
            w2_new = torch.cat([w2, new_col], dim=1)
            b2_new = self.l2.bias.detach()
            # Rebuild the two Linear layers at the new width.
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
        with torch.no_grad():
            w = self.l1.weight.detach().abs()
            w = w + (~self.mask).float() * 1e9
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


# --- activation (spike-rate) variance tracking -----------------------------

class ActivationStats:
    """Running per-hidden-unit spike-rate variance over a sliding window."""
    def __init__(self, hidden: int, window: int, device: str):
        self.window = window
        self.buf = torch.zeros(window, hidden, device=device)
        self.idx = 0
        self.filled = 0
        self.device = device

    def push(self, h: torch.Tensor) -> None:
        if h.shape[1] != self.buf.shape[1]:
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


# --- streaming loop --------------------------------------------------------

def stream(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    probe = MemoryProbe()
    probe.start()

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

    cut = max(int(0.1 * len(Xnp)), 512)
    mu = Xnp[:cut].mean(0); sd = Xnp[:cut].std(0)
    safe_sd = np.where(sd > 1e-3, sd, 1.0)
    Xnp = ((Xnp - mu) / safe_sd).astype(np.float32)
    y_mu = ynp[:cut].mean(); y_sd = ynp[:cut].std() + 1e-6
    ynp = ((ynp - y_mu) / y_sd).astype(np.float32)

    X = torch.from_numpy(np.ascontiguousarray(Xnp)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(ynp)).to(device).unsqueeze(-1)
    in_dim = X.shape[1]

    n_train = int(0.85 * len(X))
    X_test = X[n_train:]
    y_test = y[n_train:]
    probe.end_dataset()

    T = max(3, args.timesteps // (2 if args.quick else 1))
    model = SpikingSplitMLP(in_dim, 1, hidden=args.init_hidden, T=T,
                            device=device).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    probe.end_weights()
    act_stats = ActivationStats(model.hidden, window=args.var_window,
                                device=device)

    hist_path, summary_path, plot_path = output_paths(
        args, "continuous_small_appliances")
    log = StructuralLog(hist_path)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    print(f"[info] device={device} T={T} in_dim={in_dim} data={dataset_name} "
          f"N={len(X)} steps={max_steps} init_hidden={args.init_hidden}")

    @torch.no_grad()
    def eval_mse(x, yv):
        model.eval()
        p = model(x)
        model.train()
        return float(((p - yv) ** 2).mean())

    deltas_units: list[int] = []
    deltas_edges: list[int] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        idx = (np.arange(args.batch) + (step - 1) * args.batch) % n_train
        xb = X[idx]; yb = y[idx]

        model.train()
        timer.tick()
        pred, rate = model(xb, return_rate=True)
        timer.mark_forward()
        loss = ((pred - yb) ** 2).sum()          # sum reduction (matches torch/jax)
        timer.mark_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        timer.mark_backward()
        opt.step()
        with torch.no_grad():
            model.l1.weight.mul_(model.mask.to(model.l1.weight.dtype))
        timer.mark_update()
        act_stats.push(rate)

        units_before = model.unit_count()
        edges_before = model.edge_count()

        did_change = False
        # split?
        timer.tick()
        if rng.uniform() < args.p_split and model.hidden < args.max_hidden:
            hot = act_stats.hottest()
            model.split_unit(hot, noise=args.split_noise)
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            did_change = True
        timer.mark_grow()
        # prune?
        timer.tick()
        if rng.uniform() < args.p_prune and model.mask.sum().item() > model.in_dim:
            if model.kill_smallest_edge():
                did_change = True
        timer.mark_prune()
        timer.mark_reset()
        timer.step_done()

        deltas_units.append(model.unit_count() - units_before)
        deltas_edges.append(model.edge_count() - edges_before)

        if step % val_every == 0 or step == max_steps or did_change:
            vstart = (step * args.batch) % n_train
            vend = min(vstart + args.val_window, n_train)
            vl = eval_mse(X[vstart:vend], y[vstart:vend])
            test_mse = eval_mse(X_test, y_test)
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

    abs_du = np.abs(deltas_units)
    abs_de = np.abs(deltas_edges)
    test_mse_final = eval_mse(X_test, y_test)
    summary = {
        "workload": "04_continuous_small_appliances",
        "framework": "snntorch",
        "dataset": dataset_name,
        "in_dim": in_dim,
        "init_hidden": args.init_hidden,
        "timesteps": T,
        "max_steps": max_steps, "batch": args.batch,
        "p_split": args.p_split, "p_prune": args.p_prune,
        "wall_seconds": round(wall, 3),
        "splits_fired": model.splits,
        "prunes_fired": model.prunes,
        "hidden_final": model.hidden,
        "n_units": model.unit_count(),
        "n_edges": model.edge_count(),
        "edges_final": model.edge_count(),
        "test_mse": round(test_mse_final, 6),
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
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s splits={model.splits} "
          f"prunes={model.prunes} hidden_final={model.hidden} "
          f"test_mse={summary['test_mse']:.4f} "
          f"|du|_p99={summary['delta_units_p99_abs']} "
          f"jaccard_mean={summary['jaccard_mean']:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    if not args.no_plot:
        try:
            from common import plot_run, plot_test_curve, test_plot_path
            plot_run(log.records, plot_path,
                     title=f"Continuous-small Appliances (SNN) -- "
                           f"p_split={args.p_split} p_prune={args.p_prune}")
            tpath = test_plot_path(args, "continuous_small_appliances")
            plot_test_curve(log.records, tpath,
                            title="Continuous-small Appliances (SNN) test MSE",
                            metric_key="test_mse", ylabel="test MSE",
                            higher_is_better=False)
            print(f"[done] wrote {plot_path}, {tpath}")
        except Exception as e:
            print(f"[warn] plotting skipped ({e})", file=sys.stderr)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot/--tag
    p.add_argument("--synthetic", action="store_true",
                   help="force synthetic slow-drift regression stream")
    p.add_argument("--init-hidden", type=int, default=32)
    p.add_argument("--max-hidden", type=int, default=256,
                   help="upper bound on hidden width so the walk stays bounded")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--val-window", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--p-split", type=float, default=0.5,
                   help="per-step probability of splitting the hottest unit")
    p.add_argument("--p-prune", type=float, default=0.5,
                   help="per-step probability of killing the smallest edge")
    p.add_argument("--split-noise", type=float, default=0.05)
    p.add_argument("--var-window", type=int, default=64,
                   help="spike-rate-variance window for hottest-unit pick")
    p.add_argument("--timesteps", type=int, default=16,
                   help="SNN spike-window T (--quick halves it)")
    args = p.parse_args()

    summary = stream(args)
    if summary["delta_units_max_abs"] > 1:
        print(f"[warn] |delta_units|_max = {summary['delta_units_max_abs']} > 1; "
              "regime invariant violated", file=sys.stderr)


if __name__ == "__main__":
    main()
