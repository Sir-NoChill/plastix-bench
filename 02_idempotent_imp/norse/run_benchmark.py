"""Workload 2 / 5 — IDEMPOTENT SHRINKAGE via iterative magnitude pruning (IMP)
of a wide MLP CLASSIFIER on a UCR-style time-series task, Norse (SNN) port.

Expressibility: EXPRESSIBLE. IMP maps cleanly onto a spiking classifier under
Norse just as it does under snnTorch. The ReLU MLP becomes an LIF MLP: every
hidden Linear is followed by a `nt.LIFCell` (a leaky integrate-and-fire neuron
with the built-in surrogate-gradient spike). Classification logits are read out
from a non-spiking leaky integrator `nt.LICell` on the output layer — its
accumulated membrane voltage (`state.v`) summed over the T-step window is used
as the class logits, and cross-entropy is taken on those. This is the SAME fix
the snnTorch port needed: a hard spike-count readout collapses to zero spikes
cold-start and yields no gradient, whereas the accumulated membrane carries the
summed drive and stays differentiable from step 0. The IMP machinery is
identical to every other bench: a persistent 0/1 keep-mask per Linear weight is
multiplied into the forward, each prune round zeros the smallest-magnitude
ALIVE weights globally (monotonic), and gradient hooks keep pruned weights at
zero. The loop converges to a topology re-pruning leaves unchanged (fixed
point). Mask/threshold bookkeeping is host numpy (np.partition), exactly like
the pytorch/jax/snnTorch impls; the only substantive change from the ANN bench
is the extra TIME dimension.

Algorithm (Frankle & Carbin 2018):
    train(model, epochs=E0)
    repeat:
        keep_mask = abs(W) > kth_value(abs(W), p * |alive W|)
        W <- W * keep_mask
        train(model, epochs=Ek)            # finetune the survivors
    until no further edges are removed (fixed point).

Usage:
    uv run python 02_idempotent_imp/norse/run_benchmark.py --quick --synthetic --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import norse.torch as nt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    resolve_device,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Data (mirrors the pytorch/jax/snnTorch loaders)
# ---------------------------------------------------------------------------

def synth_ucr(
    n_per_class: int = 200, n_classes: int = 5, length: int = 128,
    snr: float = 1.5, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-class time-series classification dataset with sinusoidal
    shapelets. Class k is sin(2 pi (k+1) t / L) + 0.4 sin(2 pi (2k+3) t / L)
    plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / length
    Xs = []; ys = []
    for k in range(n_classes):
        base = (
            np.sin(2 * np.pi * (k + 1) * t)
            + 0.4 * np.sin(2 * np.pi * (2 * k + 3) * t)
        ).astype(np.float32)
        sig = base[None, :].repeat(n_per_class, axis=0)
        noise = (rng.standard_normal((n_per_class, length)) / snr).astype(np.float32)
        Xs.append(sig + noise)
        ys.append(np.full(n_per_class, k, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def load_ucr_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """UCR archive format: each row is `label\\tval1\\tval2\\t...`."""
    arr = np.loadtxt(path, dtype=np.float32)
    y = arr[:, 0].astype(np.int64)
    uniq = sorted(set(y.tolist()))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y], dtype=np.int64)
    X = arr[:, 1:]
    return X, y


def load_data(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if args.data is not None:
        train_path = args.data / "TRAIN.tsv"
        test_path = args.data / "TEST.tsv"
        if not train_path.exists() or not test_path.exists():
            raise SystemExit(f"need TRAIN.tsv and TEST.tsv under {args.data}")
        Xtr_np, ytr_np = load_ucr_tsv(train_path)
        Xte_np, yte_np = load_ucr_tsv(test_path)
    else:
        Xtr_np, ytr_np = synth_ucr(
            n_per_class=args.n_per_class, n_classes=args.n_classes,
            length=args.length, snr=args.snr, seed=args.seed,
        )
        Xte_np, yte_np = synth_ucr(
            n_per_class=max(50, args.n_per_class // 4),
            n_classes=args.n_classes, length=args.length, snr=args.snr,
            seed=args.seed + 1_000,
        )
    n_classes = int(max(ytr_np.max(), yte_np.max())) + 1
    # standardise per-instance (matches UCR convention)
    Xtr_np = (Xtr_np - Xtr_np.mean(1, keepdims=True)) / (Xtr_np.std(1, keepdims=True) + 1e-6)
    Xte_np = (Xte_np - Xte_np.mean(1, keepdims=True)) / (Xte_np.std(1, keepdims=True) + 1e-6)
    return (
        torch.from_numpy(Xtr_np), torch.from_numpy(ytr_np),
        torch.from_numpy(Xte_np), torch.from_numpy(yte_np),
        n_classes,
    )


# ---------------------------------------------------------------------------
# Spiking model + IMP machinery
# ---------------------------------------------------------------------------

class SpikingMLP(nn.Module):
    """Wide LIF MLP classifier, Norse edition. Runs T spiking timesteps with
    the (static) input injected as constant current each step; class logits are
    the SUMMED output-layer membrane voltage over the window (leaky-integrator
    readout). Per-Linear 0/1 keep masks are multiplied into every weight in the
    forward — the IMP hook."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int, depth: int,
                 T: int, beta: float = 0.9):
        super().__init__()
        self.T = T
        # We reproduce snnTorch's `Leaky(beta)` recurrence  v[t] = beta*v + I[t]
        # exactly using Norse's *box* cells, whose Euler update is
        #     v[t] = (1 - dt*tau_mem_inv)*v + (dt*tau_mem_inv)*input.
        # Choosing dt*tau_mem_inv = 1 - beta gives the matching decay `beta`,
        # but Norse then attenuates the input by (1-beta); we pre-scale the
        # pre-neuron current by 1/(1-beta) so the drive reaches the membrane
        # un-attenuated — recovering v = beta*v + I identically. (The plain
        # nt.LIFCell / nt.LICell carry an extra synaptic-current stage and, with
        # the default dt scaling, charge the membrane too slowly to spike from
        # random init — the box cells match snnTorch's single-state Leaky.)
        self.beta = beta
        self.cur_scale = 1.0 / (1.0 - beta)
        self.dt = (1.0 - beta) / 100.0            # dt*tau_mem_inv = 1 - beta
        lif_p = nt.LIFBoxParameters(tau_mem_inv=torch.tensor(100.0))
        li_p = nt.LIBoxParameters(tau_mem_inv=torch.tensor(100.0))
        dims = [in_dim] + [hidden] * (depth - 1) + [n_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        )
        # A Norse LIF (box) cell — leaky integrate-and-fire with surrogate-
        # gradient spike and subtract-reset — follows every HIDDEN Linear.
        #     cell(input_current, state) -> (spk, state);  state.v is membrane.
        self.lif = nn.ModuleList(
            nt.LIFBoxCell(p=lif_p, dt=self.dt) for _ in range(depth - 1)
        )
        # Output layer is a non-spiking leaky INTEGRATOR (nt.LIBoxCell): the
        # class logits are its accumulated membrane (state.v) over the T-step
        # window — the always-differentiable readout for classification. (A
        # hard-spiking output collapses to zero spikes cold-start and gives no
        # gradient; the membrane readout carries the summed drive instead.)
        self.out_li = nt.LIBoxCell(p=li_p, dt=self.dt)
        # Persistent boolean keep-masks, one per linear layer.
        self.masks = [torch.ones_like(l.weight, dtype=torch.bool) for l in self.layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [None] * (len(self.layers) - 1)   # LIF cell states, init None
        out_state = None                           # LI cell state, init None
        acc = 0.0
        last = len(self.layers) - 1
        for _ in range(self.T):
            cur = x                                # constant-current input
            for i, layer in enumerate(self.layers):
                w = layer.weight * self.masks[i].to(layer.weight.device)
                # Pre-scale the pre-neuron current so the box cell's implicit
                # (1-beta) input attenuation is cancelled -> v = beta*v + I.
                z = F.linear(cur, w, layer.bias) * self.cur_scale
                if i != last:                      # hidden: spiking LIF cell
                    spk, states[i] = self.lif[i](z, states[i])
                    cur = spk
                else:                              # output: leaky integrator
                    _, out_state = self.out_li(z, out_state)
                    acc = acc + out_state.v        # accumulate membrane voltage
        return acc                                 # summed-membrane logits

    def alive_count(self) -> int:
        return sum(int(m.sum().item()) for m in self.masks)

    def total_count(self) -> int:
        return sum(m.numel() for m in self.masks)

    def edge_set(self) -> set[tuple[int, int, int]]:
        edges = set()
        for li, m in enumerate(self.masks):
            idx = torch.nonzero(m, as_tuple=False).tolist()
            edges.update((li, r, c) for r, c in idx)
        return edges


def global_imp_step(model: SpikingMLP, prune_frac: float) -> int:
    """Prune `prune_frac` of currently-alive weights by smallest magnitude
    over the whole model. Mask/threshold bookkeeping in host numpy (mirrors the
    pytorch/jax/snnTorch impls). Returns how many edges were newly killed."""
    Ws = [layer.weight.detach().abs().cpu().numpy() for layer in model.layers]
    masks_b = [m.cpu().numpy().astype(bool) for m in model.masks]
    alive_w = [Ws[i][masks_b[i]].flatten() for i in range(len(Ws))]
    flat = np.concatenate(alive_w)
    k = int(prune_frac * flat.size)
    if k <= 0:
        return 0
    # kth smallest (1-indexed k), matching torch.kthvalue(flat, k).
    threshold = float(np.partition(flat, k - 1)[k - 1])
    newly_killed = 0
    for i, layer in enumerate(model.layers):
        kill = (Ws[i] <= threshold) & masks_b[i]
        newly_killed += int(kill.sum())
        new_mask = masks_b[i] & ~kill
        model.masks[i] = torch.from_numpy(new_mask).to(layer.weight.device)
    return newly_killed


def apply_masks_inplace(model: SpikingMLP) -> None:
    """Zero out pruned weights (so finetuning's grad-mul keeps them zero and
    accuracy reflects the masked forward exactly)."""
    with torch.no_grad():
        for layer, mask in zip(model.layers, model.masks):
            layer.weight.mul_(mask.to(layer.weight.dtype))


def install_grad_hooks(model: SpikingMLP) -> list:
    handles = []
    for layer, mask in zip(model.layers, model.masks):
        def make_hook(m_ref):
            def hook(grad):
                return grad * m_ref.to(grad.dtype)
            return hook
        handles.append(layer.weight.register_hook(make_hook(mask)))
    return handles


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train(model, loader, opt, epochs, device, timer) -> float:
    model.train()
    total = 0.0; n = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            timer.tick()
            logits = model(xb)                    # T-step spiking forward
            timer.mark_forward()
            # sum-reduction cross-entropy on the membrane logits — matches
            # the ANN benches' per-batch gradient scale.
            loss = F.cross_entropy(logits, yb, reduction="sum")
            timer.mark_loss()
            loss.backward()                       # BPTT (surrogate gradient)
            timer.mark_backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            timer.mark_update()
            timer.step_done()
            total += loss.item(); n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, x, y, batch=512) -> tuple[float, float]:
    model.eval()
    nll = 0.0; correct = 0; n = 0
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch]; yb = y[i:i + batch]
        logits = model(xb)
        nll += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
        n += yb.size(0)
    return nll / n, correct / n


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    probe = MemoryProbe()
    probe.start()

    Xtr, ytr, Xte, yte, n_classes = load_data(args)
    print(f"[info] device={device}  in_dim={Xtr.shape[1]}  classes={n_classes}  "
          f"train={Xtr.shape[0]}  test={Xte.shape[0]}")

    Xte_d = Xte.to(device); yte_d = yte.to(device)
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=args.batch, shuffle=True, drop_last=False,
    )
    probe.end_dataset()

    T = max(3, args.timesteps // (2 if args.quick else 1))
    model = SpikingMLP(Xtr.shape[1], n_classes,
                       hidden=args.hidden, depth=args.depth, T=T).to(device)
    # Move masks alongside parameters.
    model.masks = [m.to(device) for m in model.masks]
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    probe.end_weights()

    hist_path, summary_path, plot_path = output_paths(args, "idempotent_imp")
    log = StructuralLog(hist_path)

    initial_edges = model.alive_count()
    timer = PhaseTimer()
    e0 = max(1, args.init_epochs // (4 if args.quick else 1))
    print(f"[imp] device={device} T={T} initial dense training, {e0} epochs ...")
    tr_loss = train(model, loader, opt, epochs=e0, device=device, timer=timer)
    val_nll, val_acc = evaluate(model, Xte_d, yte_d)
    log.log(0, n_units=sum(l.out_features for l in model.layers),
            n_edges=model.alive_count(), edges=model.edge_set(),
            val_loss=val_nll, val_acc=val_acc, train_loss=tr_loss, round_=0,
            test_acc=val_acc, test_nll=val_nll)
    print(f"[imp] round 0  edges={model.alive_count():>8d}  val_acc={val_acc:.3f}")

    rounds = max(1, args.max_rounds // (3 if args.quick else 1))
    fine_eps = max(1, args.finetune_epochs // (2 if args.quick else 1))
    t0 = time.perf_counter()
    round_logs = []
    for r in range(1, rounds + 1):
        prev_alive = model.alive_count()
        timer.tick()
        killed = global_imp_step(model, args.prune_frac)
        apply_masks_inplace(model)
        timer.mark_prune()
        timer.mark_grow()      # static topology: IMP only prunes
        timer.mark_reset()
        timer.step_done()
        # refresh grad hooks against the new masks
        handles = install_grad_hooks(model)
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr * args.finetune_lr_scale,
        )
        tr_loss = train(model, loader, opt, epochs=fine_eps, device=device,
                        timer=timer)
        for h in handles: h.remove()
        apply_masks_inplace(model)
        val_nll, val_acc = evaluate(model, Xte_d, yte_d)
        alive = model.alive_count()
        sparsity = 1.0 - alive / model.total_count()
        log.log(r, n_units=sum(l.out_features for l in model.layers),
                n_edges=alive, edges=model.edge_set(),
                val_loss=val_nll, val_acc=val_acc, train_loss=tr_loss,
                round_=r, killed_this_round=killed, sparsity=sparsity,
                test_acc=val_acc, test_nll=val_nll)
        round_logs.append({
            "round": r, "alive": alive, "killed": killed,
            "sparsity": sparsity, "val_acc": val_acc, "val_nll": val_nll,
        })
        print(f"[imp] round {r}  killed={killed:>7d}  alive={alive:>8d}  "
              f"sparsity={sparsity:.3f}  val_acc={val_acc:.3f}")
        if killed == 0:
            print(f"[imp] reached fixed point at round {r}")
            break
        if alive == prev_alive:  # belt and braces
            break
    wall = time.perf_counter() - t0

    log.flush()

    final = round_logs[-1] if round_logs else {"round": 0, "alive": initial_edges,
                                               "sparsity": 0.0, "val_acc": val_acc}
    summary = {
        "workload": "02_idempotent_imp",
        "framework": "norse",
        "dataset": str(args.data) if args.data else "synthetic-ucr",
        "n_classes": n_classes,
        "length": Xtr.shape[1],
        "timesteps": T,
        "hidden": args.hidden, "depth": args.depth,
        "init_epochs": e0, "finetune_epochs": fine_eps,
        "prune_frac": args.prune_frac, "max_rounds": rounds,
        "wall_seconds": round(wall, 3),
        "rounds_run": final["round"],
        "n_units": sum(l.out_features for l in model.layers),
        "n_edges": final["alive"],
        "edges_initial": initial_edges,
        "edges_final": final["alive"],
        "sparsity_final": round(final["sparsity"], 6),
        "val_acc_initial": round(log.records[0]["val_acc"], 6),
        "val_acc_final": round(final["val_acc"], 6),
        "accuracy": round(final["val_acc"], 6),
        "fixed_point_reached": int(round_logs[-1]["killed"] == 0) if round_logs else 0,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  rounds={final['round']}  "
          f"sparsity={final['sparsity']:.3f}  val_acc={final['val_acc']:.3f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot
    p.add_argument("--synthetic", action="store_true",
                   help="use the deterministic UCR-style synthesizer (default)")
    p.add_argument("--data", type=Path, default=None,
                   help="optional UCR-format dir containing TRAIN.tsv / TEST.tsv")
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--snr", type=float, default=1.5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--init-epochs", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--prune-frac", type=float, default=0.2,
                   help="fraction of currently-alive weights pruned per round")
    p.add_argument("--timesteps", type=int, default=20, help="SNN spike-window T")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--finetune-lr-scale", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    summary = run(args)
    drop = summary["val_acc_initial"] - summary["val_acc_final"]
    if drop > 0.10 and not args.quick:
        print(f"[warn] accuracy dropped {drop:.3f} during IMP", file=sys.stderr)


if __name__ == "__main__":
    main()
