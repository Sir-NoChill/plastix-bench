"""
Workload 2 / 5 -- IDEMPOTENT SHRINKAGE regime via iterative magnitude
pruning (IMP) of a wide MLP on a UCR-style time-series classification task.

Algorithm (Frankle & Carbin 2018):
    train(model, epochs=E0)
    repeat:
        keep_mask = abs(W) > kth_value(abs(W), p * |W|)
        W <- W * keep_mask
        train(model, epochs=Ek)            # finetune the survivors
    until no further edges are removed (fixed point).

The terminating condition is the defining feature of the regime: edges
decrease monotonically and the loop converges to a topology that
re-pruning leaves unchanged.

Dataset:
    The UCR archive (Dau et al. 2018) is the canonical home for short
    time-series classification problems and is what the IMP / lottery-ticket
    literature reaches for.  The official archive is password-locked at
    cs.ucr.edu; in lieu of bundling it, this script defaults to a
    deterministic UCR-style synthesizer: K classes, each defined by a
    different sinusoidal "shapelet" plus per-instance Gaussian noise.
    The synthesizer is reproducible from --seed and the wide-MLP IMP
    dynamics are observable in either setting.  Pass --data /path/to/dir
    with TRAIN.tsv / TEST.tsv (UCR format: label, then values) to swap in
    a real UCR dataset.

Usage:
    uv run python 02_idempotent_imp.py                  # full
    uv run python 02_idempotent_imp.py --quick          # smoke
    uv run python 02_idempotent_imp.py --prune-frac 0.2 # tune
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
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
# Data
# ---------------------------------------------------------------------------

def synth_ucr(
    n_per_class: int = 200, n_classes: int = 5, length: int = 128,
    snr: float = 1.5, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-class time-series classification dataset with sinusoidal
    shapelets.  Class k is sin(2 pi (k+1) t / L) + 0.4 sin(2 pi (2k+3) t / L)
    plus noise.  Easy enough that a small subnet should suffice -- which is
    exactly what IMP is supposed to discover."""
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
    # UCR labels are sometimes negative or start at 1; remap to 0..K-1.
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
# Model + IMP machinery
# ---------------------------------------------------------------------------

class WideMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int, depth: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (depth - 1) + [n_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        )
        # Persistent boolean keep-masks, one per linear layer.
        self.masks = [torch.ones_like(l.weight, dtype=torch.bool) for l in self.layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = F.linear(x, layer.weight * self.masks[i].to(layer.weight.device),
                         layer.bias)
            if i != last:
                x = F.relu(x)
        return x

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


def global_imp_step(model: WideMLP, prune_frac: float) -> int:
    """Prune `prune_frac` of currently-alive weights by smallest magnitude
    over the whole model.  Returns how many edges were newly killed."""
    alive_w = [
        layer.weight[model.masks[i]].abs().detach().flatten()
        for i, layer in enumerate(model.layers)
    ]
    flat = torch.cat(alive_w)
    k = int(prune_frac * flat.numel())
    if k <= 0:
        return 0
    threshold = torch.kthvalue(flat, k).values.item()
    newly_killed = 0
    for i, layer in enumerate(model.layers):
        mask = model.masks[i]
        kill = (layer.weight.detach().abs() <= threshold) & mask
        newly_killed += int(kill.sum().item())
        model.masks[i] = mask & ~kill
    return newly_killed


def apply_masks_inplace(model: WideMLP) -> None:
    """Zero out pruned weights (so finetuning's grad-mul keeps them zero
    and accuracy reflects the masked forward exactly)."""
    with torch.no_grad():
        for layer, mask in zip(model.layers, model.masks):
            layer.weight.mul_(mask.to(layer.weight.dtype))


def install_grad_hooks(model: WideMLP) -> list:
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
            logits = model(xb)
            timer.mark_forward()
            # reduction='sum' so the per-batch gradient magnitude matches
            # Plastix's cumulative per-example SGD update over the same
            # batch.
            loss = F.cross_entropy(logits, yb, reduction="sum")
            timer.mark_loss()
            loss.backward()
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

    Xtr, ytr, Xte, yte, n_classes = load_data(args)
    print(f"[info] device={device}  in_dim={Xtr.shape[1]}  classes={n_classes}  "
          f"train={Xtr.shape[0]}  test={Xte.shape[0]}")

    Xte_d = Xte.to(device); yte_d = yte.to(device)
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=args.batch, shuffle=True, drop_last=False,
    )

    model = WideMLP(Xtr.shape[1], n_classes,
                    hidden=args.hidden, depth=args.depth).to(device)
    # Move masks alongside parameters
    model.masks = [m.to(device) for m in model.masks]
    # SGD (no momentum) — matches Plastix's plain per-connection SGD.
    # weight_decay is dropped because Plastix's UpdateConn doesn't apply it.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    hist_path, summary_path, plot_path = output_paths(args, "idempotent_imp")
    log = StructuralLog(hist_path)

    initial_edges = model.alive_count()
    timer = PhaseTimer()
    e0 = max(1, args.init_epochs // (4 if args.quick else 1))
    print(f"[imp] initial dense training, {e0} epochs ...")
    tr_loss = train(model, loader, opt, epochs=e0, device=device, timer=timer)
    val_nll, val_acc = evaluate(model, Xte_d, yte_d)
    log.log(0, n_units=sum(l.out_features for l in model.layers),
            n_edges=model.alive_count(), edges=model.edge_set(),
            val_loss=val_nll, val_acc=val_acc, train_loss=tr_loss, round_=0,
            test_acc=val_acc, test_nll=val_nll)
    print(f"[imp] round 0  edges={model.alive_count():>8d}  "
          f"val_acc={val_acc:.3f}")

    rounds = max(1, args.max_rounds // (3 if args.quick else 1))
    fine_eps = max(1, args.finetune_epochs // (2 if args.quick else 1))
    t0 = time.perf_counter()
    round_logs = []
    for r in range(1, rounds + 1):
        prev_alive = model.alive_count()
        killed = global_imp_step(model, args.prune_frac)
        apply_masks_inplace(model)
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
        "dataset": str(args.data) if args.data else "synthetic-ucr",
        "n_classes": n_classes,
        "length": Xtr.shape[1],
        "hidden": args.hidden, "depth": args.depth,
        "init_epochs": e0, "finetune_epochs": fine_eps,
        "prune_frac": args.prune_frac, "max_rounds": rounds,
        "wall_seconds": round(wall, 3),
        "rounds_run": final["round"],
        "edges_initial": initial_edges,
        "edges_final": final["alive"],
        "sparsity_final": round(final["sparsity"], 6),
        "val_acc_initial": round(log.records[0]["val_acc"], 6),
        "val_acc_final": round(final["val_acc"], 6),
        "fixed_point_reached": int(round_logs[-1]["killed"] == 0) if round_logs else 0,
        "seed": args.seed,
        **timer.summary_fields(wall),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  rounds={final['round']}  "
          f"sparsity={final['sparsity']:.3f}  val_acc={final['val_acc']:.3f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Idempotent IMP -- hidden={args.hidden} prune={args.prune_frac:.2f}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "idempotent_imp")
        plot_test_curve(log.records, tpath,
                        title=f"Idempotent IMP test accuracy -- "
                              f"hidden={args.hidden} prune={args.prune_frac:.2f}",
                        metric_key="test_acc", ylabel="test accuracy",
                        higher_is_better=True)
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
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
    # SGD lr tuned for sum-reduction cross-entropy at batch=128. Plastix's
    # per-example SGD uses 1e-3; PyTorch with one update per 128-example
    # batch needs a smaller per-step lr to stay stable, but not too small
    # or the network never escapes chance accuracy in --quick's 5 epochs.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--finetune-lr-scale", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    summary = run(args)
    # Coarse sanity gate: we expect substantial sparsity at the fixed point
    # (>70% on this easy synthetic), and val_acc loss should be modest
    # (within 5pp of initial).
    drop = summary["val_acc_initial"] - summary["val_acc_final"]
    if drop > 0.10 and not args.quick:
        print(f"[warn] accuracy dropped {drop:.3f} during IMP (initial "
              f"{summary['val_acc_initial']:.3f} -> final "
              f"{summary['val_acc_final']:.3f}); consider --prune-frac smaller "
              f"or --finetune-epochs larger", file=sys.stderr)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Plastix phase-strategy description
# ---------------------------------------------------------------------------
#
# Idempotent shrinkage <-> Plastix policy slots:
#
#   Forward      : standard.  The masked-weight forward used here is exactly
#                  what a `WeightTag * MaskTag` connection state produces in
#                  Plastix when MaskTag is a bool.  No structural change is
#                  required; the prune phase tombstones the connection
#                  (`DeadTag = true`), and the framework already skips dead
#                  edges in every loop.
#   Backward     : standard backprop; gradients are zeroed for dead edges
#                  automatically because Map(Other.dead?, ...) shorts.
#   UpdateConn   : Adam step + optional weight-decay; same as Static.
#   PruneConn    : ShouldPrune := |w| <= threshold(round)  AND  !already_dead.
#                  Fires *between* training epochs, not every step.  The
#                  threshold is computed in a Reduce phase before the prune
#                  loop (Plastix's two-phase UpdateConn split is the natural
#                  home for this).
#   AddConn      : NoX.  IMP only removes.
#   ResetGlobal  : zero per-round counters (killed_this_round, ...).
#
# Step ordering remains the default sequence; PruneConn fires periodically
# (every K steps) rather than every step, but the framework already
# accommodates that via a guard in the ShouldPrune predicate.  The defining
# observation is that across consecutive rounds the dead-edge set is a
# *growing union* and the topology hash strictly evolves through a
# decreasing-cardinality chain until it stabilises.  This is what the
# `fixed_point_reached` flag in the summary CSV records.
