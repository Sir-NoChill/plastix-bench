"""Workload 6 — CCWC wired-NCP LTC on the sine task, JAX port.

Mirrors 06_ccwc_ncp/pytorch (model A: the sparse, fixed-topology AutoNCP-wired
Liquid Time-Constant network) trained online on the noisy-sine next-step task.
This is the JAX reference impl — same summary schema (phase + memory columns)
as the pytorch/plastix/cpp/cuda impls, so it slots into runs.csv and the tables.

The topology is a *static* AutoNCP wiring: recurrent + sensory connectivity are
frozen 0/1 masks (plus a fixed reversal-potential matrix `erev`) multiplied into
the LTC synapse activations every ODE unfold. Because the mask is a compile-time
constant, the whole scan jits once and never recompiles (Jaccard flat 1.0).

JAX specifics for the recurrence:
  * The LTC is rolled over the time axis with `jax.lax.scan`, carrying the
    hidden state `v` (the membrane potential). The per-step cell body is a direct
    port of ncps.torch.LTCCell._ode_solver (6 fused ODE unfolds, softplus
    positivity constraints — implicit_param_constraints=True, the LTC default).
  * The sparse NCP wiring is applied as `w_activation * sparsity_mask` inside the
    dynamics, exactly as the torch cell does; topology is a static jnp constant.
  * A warmup step runs each jitted fn once BEFORE the timed loop so XLA compile
    time isn't charged. Each PhaseTimer.mark_* is passed its phase result so
    async dispatch is flushed and the timing is real.

Usage:
    uv run python 06_ccwc_ncp/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common/jax"))
# Reuse the pytorch bench's numpy data generation + ncps wiring directly.
sys.path.insert(0, str(HERE.parents[0] / "pytorch"))

from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    write_summary_csv,
)
from data import make_sine_dataset  # noqa: E402
from ncps.wirings import AutoNCP  # noqa: E402

WORKLOAD = "ccwc"


# ---------------------------------------------------------------------------
# Data (numpy; reuses the pytorch TensorDatasets, materialised as jnp arrays)
# ---------------------------------------------------------------------------

def _ds_arrays(ds):
    """Pull the (X, Y) tensors out of a torch TensorDataset as numpy."""
    x, y = ds.tensors
    return np.asarray(x.numpy()), np.asarray(y.numpy())


def load_sine(args):
    train_ds, val_ds, test_ds = make_sine_dataset(
        n_train=args.sine_train, n_val=args.sine_val, n_test=args.sine_test,
        seq_len=args.sine_seq_len, noise_std=args.sine_noise, seed=args.seed,
    )
    Xtr, Ytr = _ds_arrays(train_ds)
    Xva, Yva = _ds_arrays(val_ds)
    Xte, Yte = _ds_arrays(test_ds)
    return (jnp.asarray(Xtr), jnp.asarray(Ytr),
            jnp.asarray(Xva), jnp.asarray(Yva),
            jnp.asarray(Xte), jnp.asarray(Yte))


# ---------------------------------------------------------------------------
# Wiring — static AutoNCP masks + erev matrices (numpy -> jnp constants)
# ---------------------------------------------------------------------------

class Wiring:
    """Frozen NCP connectivity + reversal potentials for the LTC dynamics."""

    def __init__(self, units: int, n_out: int, input_size: int,
                 sparsity: float, seed: int):
        w = AutoNCP(units, n_out, sparsity_level=sparsity, seed=seed)
        w.build(input_size)
        adj = np.asarray(w.adjacency_matrix)              # (U, U) in {-1,0,1}
        sadj = np.asarray(w.sensory_adjacency_matrix)     # (S, U)
        self.units = int(w.units)
        self.input_dim = int(w.input_dim)
        self.output_dim = int(w.output_dim)
        self.sparsity_mask = jnp.asarray(np.abs(adj), dtype=jnp.float32)
        self.sensory_sparsity_mask = jnp.asarray(np.abs(sadj), dtype=jnp.float32)
        self.erev = jnp.asarray(adj, dtype=jnp.float32)
        self.sensory_erev = jnp.asarray(sadj, dtype=jnp.float32)
        # Edge / synapse counts (headline structural numbers).
        self.n_recurrent = int(np.sum(np.abs(adj)))
        self.n_sensory = int(np.sum(np.abs(sadj)))


# ---------------------------------------------------------------------------
# Parameters (JAX pytree — mirrors LTCCell._allocate_parameters init ranges)
# ---------------------------------------------------------------------------

_INIT_RANGES = {
    "gleak": (0.001, 1.0),
    "vleak": (-0.2, 0.2),
    "cm": (0.4, 0.6),
    "w": (0.001, 1.0),
    "sigma": (3.0, 8.0),
    "mu": (0.3, 0.8),
    "sensory_w": (0.001, 1.0),
    "sensory_sigma": (3.0, 8.0),
    "sensory_mu": (0.3, 0.8),
}


def _uniform(key, shape, name):
    lo, hi = _INIT_RANGES[name]
    return jax.random.uniform(key, shape, minval=lo, maxval=hi)


def init_params(key, wiring: Wiring):
    U, S = wiring.units, wiring.input_dim
    M = wiring.output_dim
    keys = jax.random.split(key, 9)
    p = {
        "gleak": _uniform(keys[0], (U,), "gleak"),
        "vleak": _uniform(keys[1], (U,), "vleak"),
        "cm": _uniform(keys[2], (U,), "cm"),
        "sigma": _uniform(keys[3], (U, U), "sigma"),
        "mu": _uniform(keys[4], (U, U), "mu"),
        "w": _uniform(keys[5], (U, U), "w"),
        "sensory_sigma": _uniform(keys[6], (S, U), "sensory_sigma"),
        "sensory_mu": _uniform(keys[7], (S, U), "sensory_mu"),
        "sensory_w": _uniform(keys[8], (S, U), "sensory_w"),
        # affine input / output maps (input_mapping=output_mapping="affine")
        "input_w": jnp.ones((S,)),
        "input_b": jnp.zeros((S,)),
        "output_w": jnp.ones((M,)),
        "output_b": jnp.zeros((M,)),
    }
    return p


def _param_count(params) -> int:
    return int(sum(np.prod(v.shape) for v in params.values()))


# ---------------------------------------------------------------------------
# LTC dynamics — direct port of ncps.torch.LTCCell (implicit constraints)
# ---------------------------------------------------------------------------

_ODE_UNFOLDS = 6
_EPSILON = 1e-8


def _sigmoid(v_pre, mu, sigma):
    # v_pre: (B, U) -> (B, U, 1); mu/sigma: (U, U) or (S, U)
    x = sigma * (v_pre[..., None] - mu)
    return jax.nn.sigmoid(x)


def ltc_step(params, wiring: Wiring, v, inputs):
    """One LTC RNN step: fuse `_ODE_UNFOLDS` ODE solves. Returns (v_next, out).

    `inputs` is (B, S) already affine-mapped. `v` is (B, U)."""
    sp = jax.nn.softplus  # make_positive_fn (implicit_param_constraints=True)

    # Sensory synapses are loop-invariant across the ODE unfolds.
    sw = sp(params["sensory_w"]) * _sigmoid(
        inputs, params["sensory_mu"], params["sensory_sigma"])
    sw = sw * wiring.sensory_sparsity_mask
    srev = sw * wiring.sensory_erev
    w_num_sensory = jnp.sum(srev, axis=1)     # sum over source sensory neurons
    w_den_sensory = jnp.sum(sw, axis=1)

    cm_t = sp(params["cm"]) / (1.0 / _ODE_UNFOLDS)   # elapsed_time == 1.0
    gleak = sp(params["gleak"])
    w_param = sp(params["w"])

    def body(v_pre, _):
        wa = w_param * _sigmoid(v_pre, params["mu"], params["sigma"])
        wa = wa * wiring.sparsity_mask
        rev = wa * wiring.erev
        w_num = jnp.sum(rev, axis=1) + w_num_sensory
        w_den = jnp.sum(wa, axis=1) + w_den_sensory
        numerator = cm_t * v_pre + gleak * params["vleak"] + w_num
        denominator = cm_t + gleak + w_den
        return numerator / (denominator + _EPSILON), None

    v_next, _ = jax.lax.scan(body, v, None, length=_ODE_UNFOLDS)
    # Output map: slice motor neurons, then affine.
    out = v_next[:, : wiring.output_dim]
    out = out * params["output_w"] + params["output_b"]
    return v_next, out


def forward_seq(params, wiring: Wiring, x):
    """Roll the LTC over the time axis. x: (B, T, S) -> (B, T, M)."""
    B = x.shape[0]
    v0 = jnp.zeros((B, wiring.units))
    xw, xb = params["input_w"], params["input_b"]

    def step(v, x_t):
        inp = x_t * xw + xb              # affine input map
        v_next, out = ltc_step(params, wiring, v, inp)
        return v_next, out

    # scan over time (axis 1): move time to leading axis for lax.scan.
    x_tbf = jnp.swapaxes(x, 0, 1)        # (T, B, S)
    _, outs = jax.lax.scan(step, v0, x_tbf)   # outs: (T, B, M)
    return jnp.swapaxes(outs, 0, 1)     # (B, T, M)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args) -> dict:
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    probe = MemoryProbe()
    probe.start()

    Xtr, Ytr, Xva, Yva, Xte, Yte = load_sine(args)
    input_size, n_out = int(Xtr.shape[-1]), int(Ytr.shape[-1])
    n_tr = int(Xtr.shape[0])
    probe.end_dataset()

    wiring = Wiring(args.units, n_out, input_size,
                    sparsity=args.sparsity, seed=args.wiring_seed)
    key, mk = jax.random.split(key)
    params = init_params(mk, wiring)
    n_params = _param_count(params)
    n_units = input_size + wiring.units + n_out
    n_edges = n_params
    probe.end_weights()

    # Adam + global-norm grad clip (matches torch Adam + clip_grad_norm_).
    if args.grad_clip > 0:
        tx = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                         optax.adam(args.lr))
    else:
        tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    @jax.jit
    def forward(params, xb):
        return forward_seq(params, wiring, xb)

    @jax.jit
    def loss_from_pred(pred, yb):
        return jnp.mean((pred - yb) ** 2)          # mse (mean reduction)

    def _loss(params, xb, yb):
        return jnp.mean((forward_seq(params, wiring, xb) - yb) ** 2)

    grad_fn = jax.jit(jax.grad(_loss))

    @jax.jit
    def update(params, grads, opt_state):
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def eval_mse(params, x, y):
        return jnp.mean((forward_seq(params, wiring, x) - y) ** 2)

    def batches(rng):
        idx = np.array(rng.permutation(n_tr))
        for i in range(0, n_tr - args.batch + 1, args.batch):
            sl = idx[i:i + args.batch]
            yield Xtr[sl], Ytr[sl]

    sub_tag = "A" + (f"_{args.tag}" if args.tag else "")
    saved_tag = args.tag
    args.tag = sub_tag
    hist_path, summary_path, _ = output_paths(args, WORKLOAD)
    args.tag = saved_tag

    log = StructuralLog(hist_path)
    v0 = eval_mse(params, Xva, Yva)
    t0m = eval_mse(params, Xte, Yte)
    log.log(0, n_units=n_units, n_edges=n_edges, edges=None,
            val_loss=float(v0), val_metric=float(v0),
            test_loss=float(t0m), test_metric=float(t0m),
            train_loss=None, model="A")

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    print(f"[info] jax devices={jax.devices()} model=A input_size={input_size} "
          f"units={args.units} n_out={n_out} params={n_params} "
          f"n_recurrent={wiring.n_recurrent} n_sensory={wiring.n_sensory} "
          f"epochs={epochs} lr={args.lr} batch={args.batch} train={n_tr}")

    # Warmup: force XLA compilation of every jitted fn off the clock.
    xb0, yb0 = Xtr[:args.batch], Ytr[:args.batch]
    p0 = forward(params, xb0); l0 = loss_from_pred(p0, yb0)
    g0 = grad_fn(params, xb0, yb0)
    _pw, _os = update(params, g0, opt_state)
    jax.block_until_ready((p0, l0, g0, _pw))

    best_metric = float("inf")
    best_params = params

    timer = PhaseTimer()
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for xb, yb in batches(rng):
            timer.tick()
            pred = forward(params, xb)
            timer.mark_forward(pred)
            loss = loss_from_pred(pred, yb)
            timer.mark_loss(loss)
            grads = grad_fn(params, xb, yb)
            timer.mark_backward(grads)
            params, opt_state = update(params, grads, opt_state)
            timer.mark_update((params, opt_state))
            timer.mark_prune()   # static topology (no structural mutation)
            timer.mark_grow()
            timer.mark_reset()
            timer.step_done()
        v_mse = float(eval_mse(params, Xva, Yva))
        t_mse = float(eval_mse(params, Xte, Yte))
        log.log(ep, n_units=n_units, n_edges=n_edges, edges=None,
                val_loss=v_mse, val_metric=v_mse,
                test_loss=t_mse, test_metric=t_mse,
                train_loss=v_mse, train_metric=v_mse, model="A", epoch=ep)
        improved = v_mse < best_metric
        if improved:
            best_metric = v_mse
            best_params = jax.tree_util.tree_map(lambda a: a, params)
        flag = "  *" if improved else ""
        print(f"[A ep {ep:>3d}] val_mse={v_mse:.4f} test_mse={t_mse:.4f}{flag}")
    wall = time.perf_counter() - t0
    log.flush()

    # Headline test number from the best-val checkpoint.
    t_metric = float(eval_mse(best_params, Xte, Yte))

    summary = {
        "workload": WORKLOAD,
        "task": "sine",
        "model": "A",
        "variant": args.variant,
        "input_size": input_size,
        "units": args.units,
        "n_out": n_out,
        "n_params": n_params,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "mixed_memory": 0,
        "wall_seconds": round(wall, 3),
        "metric_kind": "mse",
        "val_metric_best": round(best_metric, 6),
        "test_metric": round(t_metric, 6),
        "n_units": n_units, "n_edges": n_edges,
        "n_recurrent_synapses": wiring.n_recurrent,
        "n_sensory_synapses": wiring.n_sensory,
        "jaccard_min": 1.0, "jaccard_max": 1.0,
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] model=A wall={wall:.1f}s params={n_params} "
          f"best_val={best_metric:.4f} test={t_metric:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick
    p.add_argument("--variant", choices=["ltc"], default="ltc",
                   help="dynamical-neuron variant (JAX port implements LTC)")
    p.add_argument("--units", type=int, default=32,
                   help="neuron count (AutoNCP requires units > n_out)")
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="AutoNCP sparsity level")
    p.add_argument("--wiring-seed", type=int, default=22222,
                   help="seed for the AutoNCP wiring generation")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # sine-specific
    p.add_argument("--sine-train", type=int, default=512)
    p.add_argument("--sine-val", type=int, default=128)
    p.add_argument("--sine-test", type=int, default=128)
    p.add_argument("--sine-seq-len", type=int, default=64)
    p.add_argument("--sine-noise", type=float, default=0.1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
