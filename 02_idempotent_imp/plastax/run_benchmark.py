"""Workload 2 — Iterative Magnitude Pruning MLP, **plastax** port.

Mirrors 02_idempotent_imp/plastix (the C++ Plastix oracle): a bias-free ReLU
MLP (depth-1 ReLU hidden layers + linear output), softmax cross-entropy over
NClasses, per-example SGD, and Iterative Magnitude Pruning (IMP) — prune the
smallest-|w| fraction of live weights between short finetune rounds until a
round removes nothing.

Feedforward DAG → plastax TOPOLOGICAL propagation (same level-walk as bench
01). Pruning uses plastax's **PruneConn** trait: the host computes the per-
round magnitude threshold over the currently-alive weights, arms the prune
predicate via globals, and runs DoPruneConnections ONCE (a pure tombstone-OR
into the DEAD column). This is the "mask prune" the feasibility triage flagged
— connection death only, no unit growth.

Trait port of the C++ Forward/Backward/UpdateConn/PruneConn:
  * forward:  z = Σ w·a_src; store PreAct=z; a = IsOutput ? z : ReLU(z).
  * loss:     softmax-CE; stage dL/dz = softmax(z)_i - target_i into LossGrad.
  * backward: dL/dz = (acc + LossGrad) · (IsOutput ? 1 : 1[z>0]).
  * update:   w -= lr · dL/dz_dst · a_src   (lr from globals; finetune lowers it).
  * prune:    armed & |w| <= threshold  → DEAD (host arms + sets threshold/round).

Globals is a dict {lr, prune_threshold, prune_armed}; changing its VALUES
(never its keys/shapes) reconfigures lr and the prune round without retracing.

Usage:
    uv run python 02_idempotent_imp/plastax/run_benchmark.py --no-plot --quick
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "plastax"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    build_phase_runners,
    device_bytes,
    measure_fused_step_ns,
    output_paths,
    run_phase_timed_step,
    write_summary_csv,
)

import plastax as px  # noqa: E402
from plastax._types import ACTIVATION  # noqa: E402
from plastax.state import live_conn_count  # noqa: E402

PreAct = px.FieldSpec.f32("pre_act")  # z, stored by forward for ReLU'(z)
GradPreAct = px.FieldSpec.f32("grad_pre_act")  # dL/dz, persisted across levels
LossGrad = px.FieldSpec.f32("loss_grad")  # dL/dz staged by loss (output units)
IsOutput = px.FieldSpec.f32("is_output")  # 1.0 for output units (linear), else 0


# --- traits -----------------------------------------------------------------


class ReluForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[ACTIVATION, src]

    def apply(self, u, i, g, acc):
        del g
        is_out = u[IsOutput, i] > jnp.float32(0.5)
        activation = jnp.where(is_out, acc, jnp.maximum(acc, jnp.float32(0.0)))
        return px.UnitWrite.of((PreAct, acc), (ACTIVATION, activation))


class ReluBackward(px.BackwardPass):
    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * u[GradPreAct, src]

    def apply(self, u, i, g, acc):
        del g
        z = u[PreAct, i]
        dlda = acc + u[LossGrad, i]
        is_out = u[IsOutput, i] > jnp.float32(0.5)
        dphidz = jnp.where(is_out, jnp.float32(1.0), (z > jnp.float32(0.0)).astype(jnp.float32))
        return px.UnitWrite.of((GradPreAct, dlda * dphidz))


class SoftmaxCELoss(px.Loss):
    """Softmax cross-entropy over the NClasses output logits vs a one-hot
    target. per_output(i) reads ALL output logits to form the shared softmax
    denominator (logsumexp), returns -target_i·(z_i - lse) so the loss_phase
    sum reproduces -Σ target log softmax, and stages dL/dz_i = p_i - target_i
    into LossGrad (== the oracle's softmax-target backward accumulator)."""

    def __init__(self, out_ids) -> None:
        self._out_ids = tuple(int(i) for i in out_ids)

    def per_output(self, u, i, target, g):
        del g
        logits = jnp.stack([u[ACTIVATION, j] for j in self._out_ids])
        lse = logsumexp(logits)
        z_i = u[ACTIVATION, i]
        loss_i = -target * (z_i - lse)
        p_i = jnp.exp(z_i - lse)
        return loss_i, px.UnitWrite.of((LossGrad, p_i - target))


class SgdUpdate(px.UpdateConn):
    """w -= lr · dL/dz_dst · a_src; lr read from globals so the finetune
    rounds can lower it without retracing."""

    def incoming(self, u, dst, src, c, cid, g):
        delta = g["lr"] * u[GradPreAct, dst] * u[ACTIVATION, src]
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] - delta))

    def outgoing(self, u, src, dst, c, cid, g):
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


class ImpPrune(px.PruneConn):
    """armed & |w| <= threshold  (oracle PruneConn::ShouldPrune). Host sets
    both globals per round; disarmed the rest of the time so DoPruneConnections
    is a no-op tombstone-OR."""

    def predicate(self, u, c, cid, g):
        del u
        return g["prune_armed"] & (jnp.abs(c[px.WEIGHT, cid]) <= g["prune_threshold"])


def make_nets(out_ids):
    fwd, bwd = ReluForward(), ReluBackward()
    loss_t, upd, prune_t = SoftmaxCELoss(out_ids), SgdUpdate(), ImpPrune()
    fields = (PreAct, GradPreAct, LossGrad, IsOutput)

    class ImpNet(px.Network[dict]):
        forward_pass = fwd
        backward_pass = bwd
        loss = loss_t
        update_conn = upd
        prune_conn = prune_t
        extra_unit_fields = fields
        propagation = px.Propagation.TOPOLOGICAL

    class ImpTrainNet(px.Network[dict]):
        # Same field sets as ImpNet (mlp_xor's XorNet/XorNetEval pattern) so
        # one (static, state) drives either; no prune → fused make_step and
        # the per-step timed loop never pay a prune pass they don't need.
        forward_pass = fwd
        backward_pass = bwd
        loss = loss_t
        update_conn = upd
        extra_unit_fields = fields
        propagation = px.Propagation.TOPOLOGICAL

    return ImpNet, ImpTrainNet


# --- data (synthetic UCR shapelets, oracle SynthUcr) ------------------------


def synth_ucr(n_classes, length, n_per_class, snr, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32) / length
    xs, ys = [], []
    for k in range(n_classes):
        base = np.sin(2 * np.pi * (k + 1) * t) + 0.4 * np.sin(2 * np.pi * (2 * k + 3) * t)
        for _ in range(n_per_class):
            xs.append(base + rng.normal(0.0, 1.0 / snr, size=length).astype(np.float32))
            ys.append(k)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)  # per-instance z
    return x, y


def build_state(net, in_dim, n_classes, hidden, depth, key):
    """input -> (depth-1) ReLU hidden of `hidden` -> linear `n_classes` output.
    nn.Linear-style init (matches the jax/pytorch refs; the oracle uses a
    single global sqrt(6/(in+hidden)) limit instead — init affects the learned
    weights, not per-step cost)."""
    init = jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
    dims = [in_dim] + [hidden] * (depth - 1) + [n_classes]
    blocks = [px.topology.input_units(in_dim)]
    for i in range(depth):
        blocks.append(px.topology.dense(dims[i], dims[i + 1], init=init))
    globals0 = {
        "lr": jnp.float32(1e-3),
        "prune_threshold": jnp.float32(0.0),
        "prune_armed": jnp.bool_(False),
    }
    static, state = px.NetworkBuilder.from_topology(
        net, px.topology.sequential(*blocks), key, globals_=globals0
    )
    is_out = state.units[IsOutput.name].at[jnp.asarray(static.output_ids)].set(1.0)
    state = dataclasses.replace(state, units={**state.units, IsOutput.name: is_out})
    return static, state


def set_globals(state, *, lr=None, threshold=None, armed=None):
    g = dict(state.globals_)
    if lr is not None:
        g["lr"] = jnp.float32(lr)
    if threshold is not None:
        g["prune_threshold"] = jnp.float32(threshold)
    if armed is not None:
        g["prune_armed"] = jnp.bool_(armed)
    return dataclasses.replace(state, globals_=g)


def magnitude_threshold(state, prune_frac):
    """k-th smallest |w| among ALIVE edges, k = int(prune_frac·alive), k≥1
    (oracle ComputeThreshold). Host-side numpy over the current arenas."""
    alive_abs = []
    for bucket in state.conns:
        w = np.asarray(bucket[px.WEIGHT.name])
        dead = np.asarray(bucket[px.DEAD.name])
        alive_abs.append(np.abs(w[~dead]))
    aw = np.concatenate(alive_abs)
    alive = int(aw.size)
    if alive == 0:
        return 0.0, 0
    k = min(max(1, int(prune_frac * alive)), alive)
    return float(np.partition(aw, k - 1)[k - 1]), alive


# --- run --------------------------------------------------------------------


def run(args) -> dict:
    key = jax.random.PRNGKey(args.seed)
    probe = MemoryProbe()
    probe.start()

    n_classes, length = args.n_classes, args.length
    Xtr, Ytr = synth_ucr(n_classes, length, args.n_per_class, args.snr, args.seed)
    n_test = max(16, args.n_per_class // 4)
    Xte, Yte = synth_ucr(n_classes, length, n_test, args.snr, args.seed + 1000)
    Xtr_j, Xte_j = jnp.asarray(Xtr), jnp.asarray(Xte)
    onehot = np.eye(n_classes, dtype=np.float32)
    n_train = len(Xtr)
    probe.end_dataset()

    out_ids = tuple(range(length + hidden_units(args) - n_classes, length + hidden_units(args)))
    imp_net, train_net = make_nets(out_ids)
    key, mk = jax.random.split(key)
    static, state = build_state(imp_net, length, n_classes, args.hidden, args.depth, mk)
    # verify analytic out_ids == builder's
    assert tuple(static.output_ids) == out_ids, (static.output_ids, out_ids)
    n_units = int(state.units[ACTIVATION.name].shape[0])
    n_edges0 = int(live_conn_count(state))
    probe.end_weights()

    train_runners = build_phase_runners(train_net, static, donate=True)
    prune_runner = build_phase_runners(imp_net, static, donate=True)["prune"]
    eval_fwd = build_phase_runners(train_net, static, donate=False)["forward"]

    def accuracy(st, n_max=1024):
        n = min(len(Xte), n_max)
        correct = 0
        for j in range(n):
            si = px.StepInputs(inputs=Xte_j[j], targets=None)
            s2, _ = eval_fwd(st, si)
            logits = np.asarray(s2.units[ACTIVATION.name][np.asarray(static.output_ids)])
            correct += int(np.argmax(logits) == int(Yte[j]))
        return correct / n

    init_epochs = max(1, args.init_epochs // (4 if args.quick else 1))
    finetune_epochs = max(1, args.finetune_epochs // (2 if args.quick else 1))
    max_rounds = max(2, args.max_rounds // (3 if args.quick else 1))
    rng = np.random.default_rng(args.seed)

    hist_path, summary_path, _ = output_paths(args, "idempotent_imp")
    log = StructuralLog(hist_path)

    print(f"[info] plastax devices={jax.devices()} in={length} classes={n_classes} "
          f"hidden={args.hidden} depth={args.depth} units={n_units} edges={n_edges0} "
          f"train={n_train} init_ep={init_epochs} ft_ep={finetune_epochs} rounds={max_rounds}")

    # warmup / compile every train kernel off the clock
    si0 = px.StepInputs(inputs=Xtr_j[0], targets=jnp.asarray(onehot[Ytr[0]]))
    state, _ = run_phase_timed_step(train_runners, state, si0, PhaseTimer())
    jax.block_until_ready(state)

    def train_epochs(state, n_epochs, timer):
        for _ep in range(n_epochs):
            order = rng.permutation(n_train)
            for j in order:
                si = px.StepInputs(inputs=Xtr_j[j], targets=jnp.asarray(onehot[Ytr[j]]))
                state, _ = run_phase_timed_step(train_runners, state, si, timer)
        return state

    timer = PhaseTimer()
    t0 = time.perf_counter()
    # Dense training.
    state = set_globals(state, lr=args.lr)
    state = train_epochs(state, init_epochs, timer)
    acc_dense = accuracy(state)
    log.log(0, n_units, n_edges0, edges={(0, 0, 0)}, val_loss=1 - acc_dense,
            train_loss=None, test_mse=1 - acc_dense, test_mae=0.0)

    # IMP rounds: threshold → arm → prune once → disarm → finetune.
    ft_lr = args.lr * args.finetune_lr_scale
    prune_ns = []
    n_edges_final = n_edges0
    for rnd in range(max_rounds):
        thr, alive = magnitude_threshold(state, args.prune_frac)
        state = set_globals(state, lr=ft_lr, threshold=thr, armed=True)
        tp = time.perf_counter_ns()
        state, _ = prune_runner(state, si0)
        jax.block_until_ready(state)
        prune_ns.append(time.perf_counter_ns() - tp)
        state = set_globals(state, armed=False)
        alive_after = int(live_conn_count(state))
        killed = alive - alive_after
        n_edges_final = alive_after
        if killed <= 0:
            print(f"[info] round {rnd}: fixed point (killed 0), stopping IMP")
            break
        state = train_epochs(state, finetune_epochs, timer)
        acc_r = accuracy(state)
        log.log(rnd + 1, n_units, alive_after, edges={(0, 0, 0)}, val_loss=1 - acc_r,
                train_loss=None, test_mse=1 - acc_r, test_mae=0.0)
        print(f"[info] round {rnd}: killed {killed} → {alive_after} live "
              f"({100 * alive_after / n_edges0:.1f}%), acc={acc_r:.3f}")
    wall = time.perf_counter() - t0
    log.flush()

    acc_final = accuracy(state)
    sparsity = 1.0 - n_edges_final / n_edges0
    fused_ns = measure_fused_step_ns(train_net, static, state, si0, n=500)

    summary = {
        "workload": "02_idempotent_imp",
        "dataset": "synthetic-ucr",
        "length": length, "n_classes": n_classes,
        "hidden": args.hidden, "depth": args.depth,
        "epochs": init_epochs, "batch": 1, "lr": args.lr,
        "prune_frac": args.prune_frac,
        "wall_seconds": round(wall, 3),
        "acc_dense": round(acc_dense, 4), "acc_final": round(acc_final, 4),
        "sparsity": round(sparsity, 4),
        "test_mse": round(1 - acc_final, 6), "test_mae": round(1 - acc_final, 6),
        "val_mse_final": round(1 - acc_final, 6),
        "n_units": n_units, "n_edges": n_edges0, "n_edges_final": n_edges_final,
        "jaccard_min": 1.0, "jaccard_max": 1.0,
        "seed": args.seed,
        "peak_vram_bytes": int(device_bytes()),
        "fused_step_ns_mean": round(fused_ns, 3),
        "prune_call_ns_mean": round(float(np.mean(prune_ns)) if prune_ns else 0.0, 3),
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] acc {acc_dense:.3f}→{acc_final:.3f} sparsity={sparsity:.1%} "
          f"edges {n_edges0}→{n_edges_final} phase_sep_step_ns={summary['step_ns_mean']:.0f} "
          f"fused_step_ns={fused_ns:.0f} prune_call_ns={summary['prune_call_ns_mean']:.0f} "
          f"vram={summary['peak_vram_bytes'] / 1e6:.0f}MB")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def hidden_units(args) -> int:
    """Total non-input units = hidden·(depth-1) + n_classes (output)."""
    return args.hidden * (args.depth - 1) + args.n_classes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--snr", type=float, default=1.5)
    p.add_argument("--init-epochs", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--prune-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr-scale", type=float, default=0.3)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
