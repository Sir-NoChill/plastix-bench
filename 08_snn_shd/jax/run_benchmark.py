"""Workload 8 / — sparse spiking SNN on SHD, JAX port.

Mirrors 08_snn_shd/pytorch (snnTorch RecurrentSNN): a recurrent LIF network
classifying the Spiking Heidelberg Digits. Topology is fixed, so everything
jits cleanly.

Architecture (matches the pytorch `rsnn`):
    Linear(700 -> H) -> recurrent LIF(H) -> Linear(H -> 20) -> LIF readout
The readout LIF uses reset_mechanism="none" (a clean membrane accumulator);
its membrane summed over T is the logits, cross-entropy against the label.

JAX specifics for spiking:
  * LIF dynamics roll over the T time bins via `jax.lax.scan` (carry = hidden
    spike + membrane, output membrane). No Python time loop.
  * The Heaviside spike is non-differentiable; we wrap it in a `jax.custom_jvp`
    whose forward pass is the exact step function but whose JVP is a fast-
    sigmoid surrogate (a straight-through/surrogate-gradient estimator). This
    is the crux of a jax SNN.
  * A warmup step compiles every jitted fn off the clock (matches 01 jax).

Usage:
    uv run python 08_snn_shd/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# SHD data loader is framework-agnostic numpy binning; reuse it directly.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common/pytorch"))
sys.path.insert(0, str(HERE.parent / "pytorch"))

sys.path.insert(0, str(HERE.parents[1] / "common/jax"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    write_summary_csv,
)
from data import load_shd  # noqa: E402  (08_snn_shd/pytorch/data.py)


# --- surrogate-gradient spike function --------------------------------------

@partial(jax.custom_jvp, nondiff_argnums=(1,))
def spike(v, slope):
    """Heaviside(v): forward is the exact step; backward is a fast-sigmoid
    surrogate. `slope` steepens the surrogate (snnTorch fast_sigmoid slope)."""
    return (v >= 0.0).astype(v.dtype)


@spike.defjvp
def _spike_jvp(slope, primals, tangents):
    (v,), (dv,) = primals, tangents
    out = spike(v, slope)
    # fast-sigmoid surrogate derivative: 1 / (1 + slope*|v|)^2
    grad = 1.0 / (1.0 + slope * jnp.abs(v)) ** 2
    return out, grad * dv


# --- model (params = dict of arrays) ----------------------------------------

def init_params(key, n_in, n_hid, n_out):
    kW_in, kW_rec, kW_out = jax.random.split(key, 3)
    # Kaiming-uniform fan-in for fc_in (matches the pytorch kaiming init that
    # drives the sparse binary SHD input above the LIF threshold).
    lim_in = (6.0 / n_in) ** 0.5
    W_in = jax.random.uniform(kW_in, (n_in, n_hid), minval=-lim_in, maxval=lim_in)
    lim_rec = 1.0 / (n_hid ** 0.5)
    W_rec = jax.random.uniform(kW_rec, (n_hid, n_hid),
                               minval=-lim_rec, maxval=lim_rec)
    lim_out = 1.0 / (n_hid ** 0.5)
    W_out = jax.random.uniform(kW_out, (n_hid, n_out),
                               minval=-lim_out, maxval=lim_out)
    return {
        "W_in": W_in, "b_in": jnp.zeros((n_hid,)),
        "W_rec": W_rec,
        "W_out": W_out, "b_out": jnp.zeros((n_out,)),
    }


def forward_full(params, x, beta, slope):
    """x: (T, B, n_in). Returns (logits (B,n_out), mean_firing_rate).

    A single scan carries hidden spike + both membranes + the running readout-
    membrane sum (the logits) + the running hidden-spike sum (firing rate)."""
    T, B, _ = x.shape
    n_hid = params["b_in"].shape[0]
    n_out = params["b_out"].shape[0]

    def step(carry, x_t):
        spk_h, mem_h, mem_o, logit_sum, spk_sum = carry
        # Recurrent LIF hidden: input current + recurrent spike feedback.
        cur_h = x_t @ params["W_in"] + params["b_in"] + spk_h @ params["W_rec"]
        mem_h = beta * mem_h + cur_h
        spk_h = spike(mem_h - 1.0, slope)              # threshold = 1.0
        mem_h = mem_h - spk_h                           # subtract-reset
        # Readout LIF: reset_mechanism="none" -> pure leaky accumulator.
        cur_o = spk_h @ params["W_out"] + params["b_out"]
        mem_o = beta * mem_o + cur_o
        return (spk_h, mem_h, mem_o, logit_sum + mem_o,
                spk_sum + spk_h.mean()), None

    init = (jnp.zeros((B, n_hid)), jnp.zeros((B, n_hid)), jnp.zeros((B, n_out)),
            jnp.zeros((B, n_out)), jnp.zeros(()))
    (_, _, _, logits, spk_sum), _ = jax.lax.scan(step, init, x)
    firing_rate = spk_sum / T
    return logits, firing_rate


def _softmax_ce(logits, y):
    logp = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.take_along_axis(logp, y[:, None], axis=-1).mean()


# --- counts -----------------------------------------------------------------

def _edge_count(params) -> int:
    return int(params["W_in"].size + params["W_rec"].size + params["W_out"].size)


def run(args) -> dict:
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    probe = MemoryProbe()
    probe.start()

    n_bins = args.n_bins // (2 if args.quick else 1)
    n_bins = max(20, n_bins)
    train_ds, val_ds, test_ds, n_in, n_classes = load_shd(
        args.data_dir, n_bins=n_bins, val_frac=args.val_frac, seed=args.seed,
    )

    # Datasets hold torch tensors; pull the raw numpy arrays.
    def to_np(ds):
        return np.asarray(ds.X.numpy()), np.asarray(ds.y.numpy())

    Xtr, ytr = to_np(train_ds)
    Xva, yva = to_np(val_ds)
    Xte, yte = to_np(test_ds)

    if args.quick:
        # SHD is heavy: cap sample counts so --quick is a genuine smoke test.
        Xtr, ytr = Xtr[:2 * args.batch], ytr[:2 * args.batch]
        Xva, yva = Xva[:args.batch], yva[:args.batch]
        Xte, yte = Xte[:args.batch], yte[:args.batch]

    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr)
    Xva_j, yva_j = jnp.asarray(Xva), jnp.asarray(yva)
    Xte_j, yte_j = jnp.asarray(Xte), jnp.asarray(yte)
    probe.end_dataset()

    key, mk = jax.random.split(key)
    params = init_params(mk, n_in, args.n_hid, n_classes)
    probe.end_weights()

    beta = args.beta
    slope = args.surrogate_slope
    lr = args.lr

    @jax.jit
    def forward(params, xb):                            # xb: (B, T, C)
        x_t = jnp.transpose(xb, (1, 0, 2))              # (T, B, C)
        return forward_full(params, x_t, beta, slope)

    @jax.jit
    def loss_from_pred(logits, yb):
        return _softmax_ce(logits, yb)

    def _loss(params, xb, yb):
        x_t = jnp.transpose(xb, (1, 0, 2))
        logits, fr = forward_full(params, x_t, beta, slope)
        loss = _softmax_ce(logits, yb)
        if args.rate_reg > 0:
            loss = loss + args.rate_reg * fr ** 2
        return loss

    grad_fn = jax.jit(jax.grad(_loss))

    @jax.jit
    def adam_update(params, grads, m, v, t):
        b1, b2, eps = 0.9, 0.999, 1e-8
        # Global-norm gradient clipping — surrogate-gradient BPTT through the
        # T-step scan can explode and drive weights (then membranes, then the
        # softmax) to nan. Clip to a max global norm of 5.0.
        gnorm = jnp.sqrt(sum(jnp.sum(g * g)
                             for g in jax.tree_util.tree_leaves(grads)))
        scale = jnp.minimum(1.0, 5.0 / (gnorm + 1e-6))
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        t = t + 1
        new_m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g,
                                       m, grads)
        new_v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * g * g,
                                       v, grads)
        mhat = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** t), new_m)
        vhat = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** t), new_v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            params, mhat, vhat)
        return new_params, new_m, new_v, t

    @jax.jit
    def eval_acc(params, x, y):
        logits, _ = forward(params, x)
        return jnp.mean((logits.argmax(-1) == y).astype(jnp.float32))

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t_step = jnp.array(0, dtype=jnp.int32)

    def batches(rng, X, y):
        n = len(y)
        idx = np.array(rng.permutation(n))
        for i in range(0, n - args.batch + 1, args.batch):
            sl = idx[i:i + args.batch]
            yield jnp.asarray(X[sl]), jnp.asarray(y[sl])

    hist_path, summary_path, plot_path = output_paths(args, "snn_shd")
    log = StructuralLog(hist_path)
    n_units = n_in + args.n_hid + n_classes
    n_edges = _edge_count(params)

    va0 = float(eval_acc(params, Xva_j, yva_j))
    te0 = float(eval_acc(params, Xte_j, yte_j))
    log.log(0, n_units=n_units, n_edges=n_edges, edges=None,
            val_acc=va0, test_acc=te0, epoch=0, train_loss=None)

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    print(f"[info] jax devices={jax.devices()} n_in={n_in} n_hid={args.n_hid} "
          f"n_out={n_classes} n_bins={n_bins} epochs={epochs} "
          f"train={len(ytr)} val={len(yva)} test={len(yte)}")

    # Warmup: compile every jitted fn off the clock.
    rng = np.random.default_rng(args.seed)
    xb0 = jnp.asarray(Xtr[:args.batch]); yb0 = jnp.asarray(ytr[:args.batch])
    p0, _ = forward(params, xb0); l0 = loss_from_pred(p0, yb0)
    g0 = grad_fn(params, xb0, yb0)
    _ = adam_update(params, g0, m, v, t_step)
    jax.block_until_ready((p0, l0, g0))

    best_val = -1.0
    best_params = params
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for xb, yb in batches(rng, Xtr, ytr):
            timer.tick()
            pred, _ = forward(params, xb)
            timer.mark_forward(pred)
            loss = loss_from_pred(pred, yb)
            timer.mark_loss(loss)
            grads = grad_fn(params, xb, yb)
            timer.mark_backward(grads)
            params, m, v, t_step = adam_update(params, grads, m, v, t_step)
            timer.mark_update(params)
            timer.mark_prune(); timer.mark_grow(); timer.mark_reset()  # static
            timer.step_done()
        va = float(eval_acc(params, Xva_j, yva_j))
        te = float(eval_acc(params, Xte_j, yte_j))
        improved = va > best_val
        if improved:
            best_val = va
            best_params = jax.tree_util.tree_map(lambda a: a, params)
        log.log(ep, n_units=n_units, n_edges=n_edges, edges=None,
                val_acc=va, test_acc=te, epoch=ep, train_loss=float(loss))
        print(f"[ep {ep:>3d}] loss={float(loss):.4f} val_acc={va:.3f} "
              f"test_acc={te:.3f}{'  *' if improved else ''}")
    wall = time.perf_counter() - t0

    test_acc = float(eval_acc(best_params, Xte_j, yte_j))
    _, fr_final = forward(best_params, Xte_j)
    fr_final = float(fr_final)
    log.flush()

    summary = {
        "workload": "08_snn_shd",
        "dataset": "SHD",
        "model": "rsnn",
        "n_in": n_in, "n_hid": args.n_hid, "n_out": n_classes,
        "n_bins": n_bins,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "beta": args.beta, "surrogate_slope": args.surrogate_slope,
        "rate_reg": args.rate_reg,
        "wall_seconds": round(wall, 3),
        "val_acc_best": round(best_val, 6),
        "test_acc": round(test_acc, 6),   # key matches pytorch/cpp so orchestrator finds it
        "firing_rate_final": round(fr_final, 6),
        "n_units": n_units, "n_edges": n_edges,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s test_accuracy={test_acc:.3f} "
          f"firing_rate={fr_final:.3f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device/--quick/--out-dir/--seed/--no-plot
    p.add_argument("--n-bins", type=int, default=100,
                   help="time bins per sample (also the SNN's T)")
    p.add_argument("--n-hid", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.9,
                   help="LIF membrane decay (closer to 1 = slower leak)")
    p.add_argument("--surrogate-slope", type=float, default=25.0,
                   help="slope of the fast-sigmoid surrogate gradient")
    p.add_argument("--timesteps", type=int, default=0,
                   help="override T (0 = use n_bins)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--rate-reg", type=float, default=0.0,
                   help="L2 penalty on mean hidden firing rate")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
