"""Workload 5 / 5 — CONTINUOUS-LARGE Mackey-Glass regression, snnTorch (SNN) port.

Mirrors 05_continuous_large_mackey_glass/pytorch: a sparse "reservoir-like"
network is trained one SGD step per minibatch, and the topology is mutated
*every step* — heavy-tailed Pareto grow/shrink, Watts-Strogatz recurrent
rewires every `rewire_every` steps, and growth-momentum burst-grows.

Spiking version (schema of 01_static_etth1/snn):
  * `l_in` (dense) drives a LIF hidden layer; the masked recurrent block
    `l_rec` is fed back as an additional input current INSIDE the T-step
    spike loop (one recurrent spike hop per timestep, akin to an ESN tick);
    `l_out` (dense) drives a NON-spiking leaky integrator readout
    (reset_mechanism="none") whose membrane, averaged over T timesteps, is
    the continuous regression head. The MSE target is matched by accumulated
    membrane potential, not a spike rate.
  * Trained with BPTT (surrogate gradient) + Adam, one step per minibatch.

Structural surgery is EAGER host-side torch: because snnTorch is plain torch
and the LIF membranes are re-initialised at the start of every forward pass,
growing/shrinking/rewiring is exactly the pytorch impl's `_resize_to` weight
rebuild — the spiking dynamics add no obstacle to the churn. Same cadence.

Regime signatures: consecutive-live-edge Jaccard sits well below 1.0, and
|Delta n_units| is heavy-tailed. Same headline summary schema as the pytorch
and jax impls, plus framework/timesteps.

Mackey-Glass is generated internally (no download); --synthetic is a no-op.

Usage:
    uv run python 05_continuous_large_mackey_glass/snn/run_benchmark.py --quick --no-plot
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
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    plot_run,
    plot_test_curve,
    resolve_device,
    test_plot_path,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Mackey-Glass integration (identical to the pytorch / jax impls)
# ---------------------------------------------------------------------------

def mackey_glass(n: int, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                 step: float = 1.0, x0: float = 1.2, seed: int = 0) -> np.ndarray:
    """Euler integration of dx/dt = beta*x(t-tau)/(1+x(t-tau)^10) - gamma*x."""
    rng = np.random.default_rng(seed)
    history_len = tau + 1
    x = np.full(history_len, x0, dtype=np.float64) + 0.01 * rng.standard_normal(history_len)
    out = np.zeros(n, dtype=np.float32)
    cur = x[-1]
    buf = list(x)
    h = 0.1
    sub = int(step / h)
    if sub < 1:
        sub = 1
    out_i = 0
    burn_steps = (tau * 20) * sub
    for _ in range(burn_steps):
        delayed = buf[-tau * sub] if len(buf) >= tau * sub else buf[0]
        dx = beta * delayed / (1.0 + delayed ** 10) - gamma * cur
        cur = cur + h * dx
        buf.append(cur)
        if len(buf) > tau * sub * 2:
            buf = buf[-tau * sub * 2:]
    for i in range(n * sub):
        delayed = buf[-tau * sub] if len(buf) >= tau * sub else buf[0]
        dx = beta * delayed / (1.0 + delayed ** 10) - gamma * cur
        cur = cur + h * dx
        buf.append(cur)
        if len(buf) > tau * sub * 2:
            buf = buf[-tau * sub * 2:]
        if i % sub == 0:
            out[out_i] = cur
            out_i += 1
            if out_i >= n:
                break
    return out


def windowed(series: np.ndarray, in_len: int, horizon: int
             ) -> tuple[np.ndarray, np.ndarray]:
    n = len(series) - in_len - horizon + 1
    idx = np.arange(in_len)[None, :] + np.arange(n)[:, None]
    X = series[idx].astype(np.float32)
    Y = series[np.arange(n) + in_len + horizon - 1].astype(np.float32)
    return X, Y.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Spiking sparse-recurrent net with mutable connectivity
# ---------------------------------------------------------------------------

class SpikingReservoir(nn.Module):
    """Spiking reservoir: dense input->hidden drives a LIF layer whose spikes
    are fed back through a masked recurrent block each timestep; a non-spiking
    leaky integrator reads out the regression value (mean membrane over T).

    The recurrent mask is what the continuous-large regime rewires; grow/shrink
    rebuild the `Linear` submodules exactly as the pytorch reservoir does."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int,
                 recur_density: float, T: int, device: str,
                 beta: float = 0.9, seed: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T = T
        self.beta = beta
        self.device = device
        grad = surrogate.fast_sigmoid()
        self.l_in = nn.Linear(in_dim, hidden).to(device)
        self.l_rec = nn.Linear(hidden, hidden).to(device)
        self.l_out = nn.Linear(hidden, out_dim).to(device)
        self.lif = snn.Leaky(beta=beta, spike_grad=grad)
        # Output integrator: leaky, no reset — its membrane is the regression head.
        self.out_lif = snn.Leaky(beta=beta, spike_grad=grad,
                                 reset_mechanism="none")
        g = torch.Generator(device="cpu").manual_seed(seed)
        rec_mask = (torch.rand(hidden, hidden, generator=g) < recur_density)
        rec_mask.fill_diagonal_(False)
        self.rec_mask = rec_mask.to(device)
        with torch.no_grad():
            self.l_rec.weight.mul_(0.3)
            self.l_rec.weight.mul_(self.rec_mask.to(self.l_rec.weight.dtype))
        self.rewires = 0
        self.grows = 0
        self.shrinks = 0
        self.bursts = 0

    @property
    def hidden(self) -> int:
        return self.l_in.out_features

    # --- forward (BPTT over T spiking timesteps) ------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LIF membranes re-init every forward — so a between-step resize of the
        # weights leaves no stale recurrent state to reconcile.
        mem = self.lif.init_leaky()
        out_mem = self.out_lif.init_leaky()
        rec_mask = self.rec_mask.to(self.l_rec.weight.dtype)
        spk = torch.zeros(x.shape[0], self.hidden, device=x.device)
        in_cur = self.l_in(x)                          # constant-current input
        acc = 0.0
        for _ in range(self.T):
            rec = F.linear(spk, self.l_rec.weight * rec_mask, self.l_rec.bias)
            spk, mem = self.lif(in_cur + rec, mem)
            _, out_mem = self.out_lif(self.l_out(spk), out_mem)
            acc = acc + out_mem
        return acc / self.T                            # mean-membrane readout

    # --- topology stats -------------------------------------------------

    def unit_count(self) -> int:
        return self.hidden + self.out_dim

    def edge_count(self) -> int:
        return (self.l_in.weight.numel()
                + int(self.rec_mask.sum().item())
                + self.l_out.weight.numel())

    def edge_set(self) -> set[tuple[int, int, int]]:
        out0, in0 = self.l_in.weight.shape
        edges = {(0, r, c) for r in range(out0) for c in range(in0)}
        idx = torch.nonzero(self.rec_mask, as_tuple=False).tolist()
        edges.update((1, r, c) for r, c in idx)
        out2, in2 = self.l_out.weight.shape
        edges.update((2, r, c) for r in range(out2) for c in range(in2))
        return edges

    # --- structural ops (host-side torch rebuild) -----------------------

    def _resize_to(self, new_h: int, init_scale: float = 0.05) -> None:
        if new_h == self.hidden:
            return
        device = self.l_in.weight.device
        old_h = self.hidden
        old_w_in = self.l_in.weight.detach()
        old_b_in = self.l_in.bias.detach()
        if new_h > old_h:
            extra_w = init_scale * torch.randn(new_h - old_h, self.in_dim,
                                               device=device)
            w_in_new = torch.cat([old_w_in, extra_w], dim=0)
            b_in_new = torch.cat([old_b_in,
                                  torch.zeros(new_h - old_h, device=device)])
        else:
            w_in_new = old_w_in[:new_h]
            b_in_new = old_b_in[:new_h]
        old_w_rec = self.l_rec.weight.detach()
        old_b_rec = self.l_rec.bias.detach()
        old_mask = self.rec_mask
        if new_h > old_h:
            pad_rows = init_scale * torch.randn(new_h - old_h, old_h, device=device)
            w_rec_grown = torch.cat([old_w_rec, pad_rows], dim=0)
            pad_cols = init_scale * torch.randn(new_h, new_h - old_h, device=device)
            w_rec_new = torch.cat([w_rec_grown, pad_cols], dim=1)
            b_rec_new = torch.cat([old_b_rec,
                                   torch.zeros(new_h - old_h, device=device)])
            mask_grown = torch.cat([old_mask,
                                    torch.zeros(new_h - old_h, old_h,
                                                dtype=torch.bool,
                                                device=device)], dim=0)
            new_pad_cols = (torch.rand(new_h, new_h - old_h, device=device) < 0.05)
            mask_new = torch.cat([mask_grown, new_pad_cols], dim=1)
            mask_new.fill_diagonal_(False)
        else:
            w_rec_new = old_w_rec[:new_h, :new_h]
            b_rec_new = old_b_rec[:new_h]
            mask_new = old_mask[:new_h, :new_h]
        old_w_out = self.l_out.weight.detach()
        old_b_out = self.l_out.bias.detach()
        if new_h > old_h:
            extra = init_scale * torch.randn(self.out_dim, new_h - old_h,
                                             device=device)
            w_out_new = torch.cat([old_w_out, extra], dim=1)
        else:
            w_out_new = old_w_out[:, :new_h]
        b_out_new = old_b_out

        self.l_in = nn.Linear(self.in_dim, new_h).to(device)
        self.l_rec = nn.Linear(new_h, new_h).to(device)
        self.l_out = nn.Linear(new_h, self.out_dim).to(device)
        with torch.no_grad():
            self.l_in.weight.copy_(w_in_new); self.l_in.bias.copy_(b_in_new)
            self.l_rec.weight.copy_(w_rec_new); self.l_rec.bias.copy_(b_rec_new)
            self.l_out.weight.copy_(w_out_new); self.l_out.bias.copy_(b_out_new)
        self.rec_mask = mask_new

    def grow(self, n: int, init_scale: float = 0.05) -> None:
        self._resize_to(self.hidden + n, init_scale=init_scale)
        self.grows += 1

    def shrink(self, n: int) -> None:
        target = max(8, self.hidden - n)
        self._resize_to(target)
        self.shrinks += 1

    def rewire(self, frac: float, rng: np.random.Generator) -> int:
        """Watts-Strogatz-style rewire of the masked recurrent block."""
        with torch.no_grad():
            alive = torch.nonzero(self.rec_mask, as_tuple=False)
            n_alive = alive.shape[0]
            if n_alive == 0:
                return 0
            n_rewire = max(1, int(frac * n_alive))
            sel = rng.choice(n_alive, size=n_rewire, replace=False)
            sel_idx = alive[sel]
            self.rec_mask[sel_idx[:, 0], sel_idx[:, 1]] = False
            old_weights = self.l_rec.weight[sel_idx[:, 0], sel_idx[:, 1]].clone()
            self.l_rec.weight[sel_idx[:, 0], sel_idx[:, 1]] = 0.0
            n_h = self.hidden
            new_src = rng.integers(0, n_h, size=n_rewire)
            new_dst = rng.integers(0, n_h, size=n_rewire)
            committed = 0
            for w_val, s, d in zip(old_weights.tolist(), new_src, new_dst):
                if s == d:
                    continue
                if self.rec_mask[s, d]:
                    continue
                self.rec_mask[s, d] = True
                self.l_rec.weight[s, d] = w_val
                committed += 1
            self.rewires += 1
            return committed


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    probe = MemoryProbe()
    probe.start()

    series = mackey_glass(n=args.series_len, tau=args.tau, seed=args.seed)
    series = (series - series.mean()) / (series.std() + 1e-6)

    X, Y = windowed(series, in_len=args.in_len, horizon=args.horizon)
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    n = len(X)
    n_tr = int(0.7 * n); n_va = int(0.15 * n)
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva, Yva = X[n_tr:n_tr + n_va], Y[n_tr:n_tr + n_va]
    Xte, Yte = X[n_tr + n_va:], Y[n_tr + n_va:]
    probe.end_dataset()

    T = max(3, args.timesteps // (2 if args.quick else 1))
    model = SpikingReservoir(in_dim=args.in_len, out_dim=1,
                             hidden=args.init_hidden,
                             recur_density=args.recur_density,
                             T=T, device=device, seed=args.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    probe.end_weights()

    hist_path, summary_path, plot_path = output_paths(args, "continuous_large_mg")
    log = StructuralLog(hist_path)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    rewire_every = max(1, args.rewire_every)
    print(f"[info] device={device}  T={T}  N={n}  train={n_tr}  val={n_va}  "
          f"test={n - n_tr - n_va}  steps={max_steps}  init_hidden={args.init_hidden}  "
          f"recur_density={args.recur_density}")

    @torch.no_grad()
    def evaluate(x, y):
        model.eval()
        return F.mse_loss(model(x), y).item()

    growth_momentum = 0.0
    deltas_units: list[int] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        idx = rng.integers(0, n_tr, size=args.batch)
        xb = Xtr[idx]; yb = Ytr[idx]

        model.train()
        timer.tick()
        pred = model(xb)
        timer.mark_forward()
        loss = F.mse_loss(pred, yb)
        timer.mark_loss()
        loss.backward()
        timer.mark_backward()
        gnorm = sum(p.grad.norm().item() for p in model.parameters()
                    if p.grad is not None)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.l_rec.weight.mul_(model.rec_mask.to(model.l_rec.weight.dtype))
        timer.mark_update()

        units_before = model.unit_count()

        # --- structural perturbation -------------------------------------
        # Heavy-tailed unit delta: sign +/-, magnitude 1 + ceil(Pareto(alpha)).
        did_grow = False
        did_shrink = False
        sign = 1 if rng.uniform() > 0.5 else -1
        mag = max(1, int(math.ceil(rng.pareto(args.pareto_alpha))))
        mag = min(mag, args.max_delta_per_step)
        if sign > 0 and model.hidden + mag <= args.max_hidden:
            model.grow(mag, init_scale=0.05)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            did_grow = True
        elif sign < 0 and model.hidden - mag >= args.min_hidden:
            model.shrink(mag)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            did_shrink = True

        # Periodic rewiring (Watts-Strogatz sever+reconnect).
        did_rewire = False
        if step % rewire_every == 0 and model.hidden >= 8:
            model.rewire(args.rewire_frac, rng)
            did_rewire = True

        # Growth-momentum burst.
        did_burst = False
        growth_momentum += gnorm
        if growth_momentum > args.momentum_threshold:
            burst = max(2, args.momentum_burst)
            if model.hidden + burst <= args.max_hidden:
                model.grow(burst)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                model.bursts += 1
                did_burst = True
            growth_momentum = 0.0

        # Phase marks. The pytorch PhaseTimer marks are no-arg and fire every
        # step (grow covers grow/rewire-reconnect/burst, prune covers
        # shrink/rewire-sever); the timed work happened above via the eager
        # host-side surgery, so an unconditional mark each step keeps the
        # accumulator fed exactly as 01_snn does.
        _ = (did_grow or did_rewire or did_burst, did_shrink or did_rewire)
        timer.mark_grow()
        timer.mark_prune()
        timer.mark_reset()
        timer.step_done()

        deltas_units.append(model.unit_count() - units_before)

        if step % val_every == 0 or step == max_steps:
            vl = evaluate(Xva, Yva)
            te = evaluate(Xte, Yte)
            log.log(step, n_units=model.unit_count(),
                    n_edges=model.edge_count(),
                    edges=model.edge_set(),
                    val_loss=vl,
                    hidden=model.hidden,
                    grows=model.grows, shrinks=model.shrinks,
                    rewires=model.rewires, bursts=model.bursts,
                    delta_units=deltas_units[-1],
                    test_mse=te)
            if step % (val_every * 5) == 0 or step == max_steps:
                print(f"[step {step:>5d}] hidden={model.hidden:>4d}  "
                      f"edges={model.edge_count():>6d}  vl={vl:.4f}  "
                      f"jaccard={log.records[-1]['jaccard']:.3f}")

    wall = time.perf_counter() - t0
    log.flush()

    test_mse = evaluate(Xte, Yte)

    abs_du = np.abs(deltas_units)
    nz_du = abs_du[abs_du > 0]
    summary = {
        "workload": "05_continuous_large_mg", "framework": "snntorch",
        "dataset": "mackey-glass", "timesteps": T,
        "tau": args.tau, "in_len": args.in_len, "horizon": args.horizon,
        "init_hidden": args.init_hidden, "recur_density": args.recur_density,
        "pareto_alpha": args.pareto_alpha,
        "rewire_every": args.rewire_every, "rewire_frac": args.rewire_frac,
        "max_steps": max_steps, "batch": args.batch,
        "wall_seconds": round(wall, 3),
        "grows": model.grows, "shrinks": model.shrinks,
        "rewires": model.rewires, "bursts": model.bursts,
        "hidden_final": model.hidden,
        "edges_final": model.edge_count(),
        "val_loss_initial": round(log.records[0]["val_loss"], 6),
        "val_loss_final": round(log.records[-1]["val_loss"], 6),
        "test_mse": round(test_mse, 6),
        "delta_units_p50_abs": int(np.percentile(abs_du, 50)) if len(abs_du) else 0,
        "delta_units_p95_abs": int(np.percentile(abs_du, 95)) if len(abs_du) else 0,
        "delta_units_max_abs": int(abs_du.max()) if len(abs_du) else 0,
        "delta_units_mean_nonzero": (float(nz_du.mean()) if len(nz_du)
                                     else 0.0),
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_mean": round(float(np.mean([r["jaccard"]
                                             for r in log.records])), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  grows={model.grows}  shrinks={model.shrinks}  "
          f"rewires={model.rewires}  hidden={model.hidden}  "
          f"test_mse={test_mse:.4f}  jaccard_mean={summary['jaccard_mean']:.3f}  "
          f"|du|_p95={summary['delta_units_p95_abs']}  "
          f"|du|_max={summary['delta_units_max_abs']}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Continuous-large Mackey-Glass (SNN) -- "
                       f"alpha={args.pareto_alpha} "
                       f"rewire_frac={args.rewire_frac}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "continuous_large_mg")
        plot_test_curve(log.records, tpath,
                        title=f"Continuous-large Mackey-Glass (SNN) test MSE -- "
                              f"alpha={args.pareto_alpha} "
                              f"rewire_frac={args.rewire_frac}",
                        metric_key="test_mse", ylabel="test MSE",
                        higher_is_better=False)
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot/--tag
    p.add_argument("--synthetic", action="store_true",
                   help="no-op; Mackey-Glass is always generated internally")
    p.add_argument("--series-len", type=int, default=5000)
    p.add_argument("--tau", type=int, default=17)
    p.add_argument("--in-len", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--init-hidden", type=int, default=128)
    p.add_argument("--min-hidden", type=int, default=16)
    p.add_argument("--max-hidden", type=int, default=512)
    p.add_argument("--recur-density", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--timesteps", type=int, default=15,
                   help="SNN spike-window T (--quick halves)")
    p.add_argument("--pareto-alpha", type=float, default=1.5,
                   help="shape parameter; smaller = heavier tail")
    p.add_argument("--max-delta-per-step", type=int, default=20)
    p.add_argument("--rewire-every", type=int, default=3,
                   help="rewire the recurrent layer every N steps")
    p.add_argument("--rewire-frac", type=float, default=0.25,
                   help="fraction of alive recurrent edges rewired each fire")
    p.add_argument("--momentum-threshold", type=float, default=200.0)
    p.add_argument("--momentum-burst", type=int, default=10)
    args = p.parse_args()

    summary = stream(args)
    if summary["jaccard_mean"] > 0.95 and not args.quick:
        print(f"[warn] jaccard_mean={summary['jaccard_mean']:.3f} is high; "
              "rewire/grow rates may be too low to qualify as continuous-large",
              file=sys.stderr)


if __name__ == "__main__":
    main()
