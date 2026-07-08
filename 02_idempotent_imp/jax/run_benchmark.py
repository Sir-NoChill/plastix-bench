"""Workload 2 / 5 — IDEMPOTENT SHRINKAGE regime via iterative magnitude
pruning (IMP) of a wide MLP on a UCR-style time-series classification task,
JAX port.

Mirrors 02_idempotent_imp/pytorch: a wide ReLU MLP trained with sum-reduction
cross-entropy + plain SGD, then iteratively magnitude-pruned. A persistent
0/1 keep-mask per weight matrix is multiplied into the forward; each prune
round zeros the smallest-magnitude ALIVE weights globally (monotonic) and the
loop converges to a topology re-pruning leaves unchanged (fixed point). This
is the JAX reference impl — same summary schema (phase + memory columns) as
the pytorch/plastix/cpp/cuda impls.

Algorithm (Frankle & Carbin 2018):
    train(model, epochs=E0)
    repeat:
        keep_mask = abs(W) > kth_value(abs(W), p * |alive W|)
        W <- W * keep_mask
        train(model, epochs=Ek)            # finetune the survivors
    until no further edges are removed (fixed point).

JAX timing notes:
  * A warmup step runs each jitted fn once BEFORE the timed loop so XLA
    compile time isn't charged to the first step.
  * The forward multiplies each W by its mask; gradients are masked in the
    update so pruned weights stay zero (like the pytorch grad hooks). Masks
    are host-numpy; the global-prune threshold is computed in numpy exactly
    as in the pytorch impl.

Usage:
    uv run python 02_idempotent_imp/jax/run_benchmark.py --quick --synthetic --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/jax"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Data (numpy; mirrors the pytorch loader)
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


def load_data(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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
        Xtr_np.astype(np.float32), ytr_np.astype(np.int64),
        Xte_np.astype(np.float32), yte_np.astype(np.int64),
        n_classes,
    )


# ---------------------------------------------------------------------------
# Model (params = list of (W, b)) + IMP machinery
# ---------------------------------------------------------------------------

def init_params(key, in_dim, n_classes, hidden, depth):
    dims = [in_dim] + [hidden] * (depth - 1) + [n_classes]
    params = []
    for i in range(depth):
        key, wk = jax.random.split(key)
        # Kaiming-ish uniform, matching nn.Linear's default init scale.
        lim = 1.0 / (dims[i] ** 0.5)
        W = jax.random.uniform(wk, (dims[i], dims[i + 1]), minval=-lim, maxval=lim)
        b = jnp.zeros((dims[i + 1],))
        params.append((W, b))
    return params


def apply_mlp(params, masks, x):
    last = len(params) - 1
    for i, (W, b) in enumerate(params):
        x = x @ (W * masks[i]) + b
        if i != last:
            x = jax.nn.relu(x)
    return x


def global_imp_step(params, masks, prune_frac: float):
    """Prune `prune_frac` of currently-alive weights by smallest magnitude
    over the whole model, in host numpy (mirrors the pytorch impl). Returns
    (new_masks, newly_killed)."""
    Ws = [np.asarray(W) for W, _ in params]
    alive_w = [np.abs(Ws[i])[masks[i].astype(bool)].flatten() for i in range(len(Ws))]
    flat = np.concatenate(alive_w)
    k = int(prune_frac * flat.size)
    if k <= 0:
        return masks, 0
    # kth smallest (1-indexed k), matching torch.kthvalue(flat, k).
    threshold = float(np.partition(flat, k - 1)[k - 1])
    new_masks = []
    newly_killed = 0
    for i in range(len(Ws)):
        mask_b = masks[i].astype(bool)
        kill = (np.abs(Ws[i]) <= threshold) & mask_b
        newly_killed += int(kill.sum())
        new_masks.append((mask_b & ~kill).astype(np.float32))
    return new_masks, newly_killed


def _alive_count(masks) -> int:
    return int(sum(int(m.sum()) for m in masks))


def _total_count(masks) -> int:
    return int(sum(m.size for m in masks))


def _edge_set(masks) -> set:
    edges = set()
    for li, m in enumerate(masks):
        idx = np.argwhere(m.astype(bool))
        edges.update((li, int(r), int(c)) for r, c in idx)
    return edges


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> dict:
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    probe = MemoryProbe()
    probe.start()

    Xtr_np, ytr_np, Xte_np, yte_np, n_classes = load_data(args)
    Xtr, ytr = jnp.asarray(Xtr_np), jnp.asarray(ytr_np)
    Xte, yte = jnp.asarray(Xte_np), jnp.asarray(yte_np)
    n_tr = Xtr.shape[0]
    print(f"[info] jax devices={jax.devices()} in_dim={Xtr_np.shape[1]} "
          f"classes={n_classes} train={Xtr_np.shape[0]} test={Xte_np.shape[0]}")
    probe.end_dataset()

    key, mk = jax.random.split(key)
    params = init_params(mk, Xtr_np.shape[1], n_classes,
                         hidden=args.hidden, depth=args.depth)
    masks = [np.ones_like(np.asarray(W), dtype=np.float32) for W, _ in params]
    probe.end_weights()

    def _loss(params, masks, xb, yb):
        logits = apply_mlp(params, masks, xb)
        # reduction='sum' cross-entropy (matches torch F.cross_entropy sum).
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(logp[jnp.arange(yb.shape[0]), yb])

    @jax.jit
    def forward(params, masks, xb):
        return apply_mlp(params, masks, xb)

    @jax.jit
    def loss_from_logits(logits, yb):
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(logp[jnp.arange(yb.shape[0]), yb])

    grad_fn = jax.jit(jax.grad(_loss))

    @jax.jit
    def sgd(params, grads, masks, lr):
        # mask gradients so pruned weights stay exactly zero (grad-hook analogue)
        return [(W - lr * (gW * masks[i]), b - lr * gb)
                for i, ((W, b), (gW, gb)) in enumerate(zip(params, grads))]

    @jax.jit
    def eval_nll_acc(params, masks, x, y):
        logits = apply_mlp(params, masks, x)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.sum(logp[jnp.arange(y.shape[0]), y])
        acc = jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32))
        return nll / y.shape[0], acc

    def batches(rng):
        idx = np.array(rng.permutation(n_tr))
        for i in range(0, n_tr, args.batch):
            sl = idx[i:i + args.batch]
            yield Xtr[sl], ytr[sl]

    def evaluate():
        nll = 0.0; correct = 0; n = 0
        for i in range(0, Xte.shape[0], 512):
            xb = Xte[i:i + 512]; yb = yte[i:i + 512]
            logits = forward(params, jmasks, xb)
            logp = jax.nn.log_softmax(logits, axis=-1)
            nll += float(-jnp.sum(logp[jnp.arange(yb.shape[0]), yb]))
            correct += int(jnp.sum(jnp.argmax(logits, axis=-1) == yb))
            n += int(yb.shape[0])
        return nll / n, correct / n

    hist_path, summary_path, plot_path = output_paths(args, "idempotent_imp")
    log = StructuralLog(hist_path)

    jmasks = [jnp.asarray(m) for m in masks]
    initial_edges = _alive_count(masks)
    n_units = sum(W.shape[1] for W, _ in params)
    timer = PhaseTimer()

    def train_loop(rng, lr, epochs):
        nonlocal params
        total = 0.0; n = 0
        lr_arr = jnp.asarray(lr, dtype=jnp.float32)
        for _ in range(epochs):
            for xb, yb in batches(rng):
                timer.tick()
                logits = forward(params, jmasks, xb)
                timer.mark_forward(logits)
                loss = loss_from_logits(logits, yb)
                timer.mark_loss(loss)
                grads = grad_fn(params, jmasks, xb, yb)
                timer.mark_backward(grads)
                params = sgd(params, grads, jmasks, lr_arr)
                timer.mark_update(params)
                timer.step_done()
                total += float(loss); n += int(xb.shape[0])
        return total / max(n, 1)

    e0 = max(1, args.init_epochs // (4 if args.quick else 1))
    rounds = max(1, args.max_rounds // (3 if args.quick else 1))
    fine_eps = max(1, args.finetune_epochs // (2 if args.quick else 1))

    # Warmup: force XLA compilation of every jitted fn off the clock.
    xb0, yb0 = Xtr[:args.batch], ytr[:args.batch]
    lr0 = jnp.asarray(args.lr, dtype=jnp.float32)
    p0 = forward(params, jmasks, xb0)
    l0 = loss_from_logits(p0, yb0)
    g0 = grad_fn(params, jmasks, xb0, yb0)
    _ = sgd(params, g0, jmasks, lr0)
    _ = eval_nll_acc(params, jmasks, xb0, yb0)
    jax.block_until_ready((p0, l0, g0))

    rng = np.random.default_rng(args.seed)
    print(f"[imp] initial dense training, {e0} epochs ...")
    t0 = time.perf_counter()
    tr_loss = train_loop(rng, args.lr, e0)
    val_nll, val_acc = evaluate()
    log.log(0, n_units=n_units, n_edges=_alive_count(masks),
            edges=_edge_set(masks),
            val_loss=val_nll, val_acc=val_acc, train_loss=tr_loss, round_=0,
            test_acc=val_acc, test_nll=val_nll)
    print(f"[imp] round 0  edges={_alive_count(masks):>8d}  val_acc={val_acc:.3f}")

    round_logs = []
    for r in range(1, rounds + 1):
        prev_alive = _alive_count(masks)
        timer.tick()
        masks, killed = global_imp_step(params, masks, args.prune_frac)
        jmasks = [jnp.asarray(m) for m in masks]
        # zero out pruned weights so the masked forward reflects them exactly
        params = [(W * jmasks[i], b) for i, (W, b) in enumerate(params)]
        timer.mark_prune(params)
        timer.mark_grow()
        timer.mark_reset()
        timer.step_done()

        tr_loss = train_loop(rng, args.lr * args.finetune_lr_scale, fine_eps)
        params = [(W * jmasks[i], b) for i, (W, b) in enumerate(params)]
        val_nll, val_acc = evaluate()
        alive = _alive_count(masks)
        sparsity = 1.0 - alive / _total_count(masks)
        log.log(r, n_units=n_units, n_edges=alive, edges=_edge_set(masks),
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
        if alive == prev_alive:
            break
    wall = time.perf_counter() - t0

    log.flush()

    final = round_logs[-1] if round_logs else {"round": 0, "alive": initial_edges,
                                               "sparsity": 0.0, "val_acc": val_acc}
    summary = {
        "workload": "02_idempotent_imp",
        "dataset": str(args.data) if args.data else "synthetic-ucr",
        "n_classes": n_classes,
        "length": Xtr_np.shape[1],
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
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  rounds={final['round']}  "
          f"sparsity={final['sparsity']:.3f}  val_acc={final['val_acc']:.3f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
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
