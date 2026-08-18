"""Workload 6 — NCP "LTC-sine" recurrent net, **plastax** port.

Mirrors 06_ccwc_ncp/plastix (the C++ Plastix ORACLE — note: NOT the conductance
LTC ODE that lives in jax/pytorch; the oracle is a leaky-tanh recurrent RNN,
"LTC-lite", trained by one-step e-prop). A fixed Neural-Circuit-Policy wiring
(sensory→inter→command→command↺→motor, plus motor→command feedback) is driven
one timestep per step under **PIPELINE** propagation; the cyclic command/
feedback synapses make TOPOLOGICAL impossible.

Neuron update (oracle Forward, 07_ccwc_ncp.cpp:117-143):
    x_i[t] = tanh( α_i·x_i[t-1] + Σ_{j→i} w_ij·tanh(x_j[t-1]) )   α_i = exp(-Δt/τ_i)

Learning is e-prop, NOT BPTT — three local per-step rules:
  * loss:     MSE on motor units, stages dL/dx = pred-target into LossGrad.
  * backward: a transposed-weight feedback PROJECTION (PIPELINE backward sweep,
              accumulates onto the edge SOURCE), giving each unit a learning
              signal L; motor units take L = pred-target fresh from the loss,
              all others L = Σ_{i→k} w·(1-x_k²)·L_k (depth-1 truncated — reads
              last step's L).
  * update:   per-edge eligibility trace e = β·e + (1-x_dst²)·tanh(x_src), then
              w -= clip(lr·L_dst·e); weights clipped to ±w_max.

State carried across timesteps in ACTIVATION / LearningSignal (units) and the
Elig conn column; reset host-side between sequences. Built via the custom
cyclic-graph PIPELINE builder (common/plastax build_pipeline_state).

Usage:
    uv run python 06_ccwc_ncp/plastax/run_benchmark.py --no-plot --quick
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "plastax"))
from common import (  # noqa: E402
    MemoryProbe,
    PhaseTimer,
    StructuralLog,
    add_common_args,
    build_phase_runners,
    build_pipeline_state,
    device_bytes,
    measure_fused_step_ns,
    output_paths,
    run_phase_timed_step,
    write_summary_csv,
)

import plastax as px  # noqa: E402
from plastax._types import ACTIVATION  # noqa: E402
from plastax.phases import build_phases  # noqa: E402

Alpha = px.FieldSpec.f32("alpha")  # α_i = exp(-Δt/τ_i), fixed per unit
LearningSignal = px.FieldSpec.f32("learning_signal")  # e-prop L_i
IsMotor = px.FieldSpec.f32("is_motor")  # 1.0 for motor (output) units
LossGrad = px.FieldSpec.f32("loss_grad")  # dL/dx = pred-target on motors
Elig = px.FieldSpec.f32("elig")  # per-edge eligibility trace e_ij


# --- traits -----------------------------------------------------------------


class LtcForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        return c[px.WEIGHT, cid] * jnp.tanh(u[ACTIVATION, src])

    def apply(self, u, i, g, acc):
        del g
        x_prev = u[ACTIVATION, i]
        return px.UnitWrite.of((ACTIVATION, jnp.tanh(u[Alpha, i] * x_prev + acc)))


class LtcMSELoss(px.Loss):
    def per_output(self, u, i, target, g):
        del g
        diff = u[ACTIVATION, i] - target
        return jnp.float32(0.5) * diff * diff, px.UnitWrite.of((LossGrad, diff))


class LtcBackward(px.BackwardPass):
    """Feedback projection (PIPELINE backward: accumulates onto the edge
    SOURCE). In the trait's map the framework binds `dst`=edge-source
    (accumulator target) and `src`=downstream unit, so the projected term
    w·(1-x_downstream²)·L_downstream lands on the source. apply: motor units
    take the fresh loss gradient, all others the accumulated projection."""

    combine = px.monoid.sum_

    def map(self, u, dst, src, c, cid, g):
        del dst, g
        x = u[ACTIVATION, src]
        return c[px.WEIGHT, cid] * (jnp.float32(1.0) - x * x) * u[LearningSignal, src]

    def apply(self, u, i, g, acc):
        del g
        is_motor = u[IsMotor, i] > jnp.float32(0.5)
        return px.UnitWrite.of((LearningSignal, jnp.where(is_motor, u[LossGrad, i], acc)))


class EpropUpdate(px.UpdateConn):
    def __init__(self, lr, beta, clip, wmax) -> None:
        self._lr = jnp.float32(lr)
        self._beta = jnp.float32(beta)
        self._clip = jnp.float32(clip)
        self._wmax = jnp.float32(wmax)

    def incoming(self, u, dst, src, c, cid, g):
        del g
        hpre = jnp.tanh(u[ACTIVATION, src])
        x_dst = u[ACTIVATION, dst]
        sens = (jnp.float32(1.0) - x_dst * x_dst) * hpre
        e_new = self._beta * c[Elig, cid] + sens
        delta = jnp.clip(self._lr * u[LearningSignal, dst] * e_new, -self._clip, self._clip)
        w_new = jnp.clip(c[px.WEIGHT, cid] - delta, -self._wmax, self._wmax)
        return px.ConnWrite.of((Elig, e_new), (px.WEIGHT, w_new))

    def outgoing(self, u, src, dst, c, cid, g):
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


def make_nets(lr, beta, clip, wmax):
    fwd = LtcForward()
    fields = (Alpha, LearningSignal, IsMotor, LossGrad)

    class NcpNet(px.Network[None]):
        forward_pass = fwd
        backward_pass = LtcBackward()
        loss = LtcMSELoss()
        update_conn = EpropUpdate(lr, beta, clip, wmax)
        extra_unit_fields = fields
        extra_conn_fields = (Elig,)
        propagation = px.Propagation.PIPELINE

    class NcpEvalNet(px.Network[None]):
        forward_pass = fwd
        extra_unit_fields = fields
        extra_conn_fields = (Elig,)
        propagation = px.Propagation.PIPELINE

    return NcpNet, NcpEvalNet


# --- NCP wiring (fixed, seeded — oracle NCPWiringBuilder) --------------------


def build_ncp(net, units, out_dim, seed, dt, w_init, k_sparse, k_rec, k_fb):
    rng = np.random.default_rng(seed)
    in_dim = 2
    motors = out_dim
    rest = units - motors
    ns = max(1, rest // 3)
    ni = (rest - ns) // 2
    nc = (rest - ns) - ni

    in_ids = np.arange(0, in_dim)
    sens = np.arange(in_dim, in_dim + ns)
    inter = np.arange(in_dim + ns, in_dim + ns + ni)
    cmd = np.arange(in_dim + ns + ni, in_dim + ns + ni + nc)
    motor = np.arange(in_dim + ns + ni + nc, in_dim + ns + ni + nc + motors)
    num_units = in_dim + units

    froms: list[int] = []
    tos: list[int] = []

    def fanin(dst_pool, src_pool, k):
        for d in dst_pool:
            pool = src_pool[src_pool != d]
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for s in sel:
                froms.append(int(s))
                tos.append(int(d))

    for s in sens:  # 1. input -> sensory (dense)
        for i in in_ids:
            froms.append(int(i))
            tos.append(int(s))
    fanin(inter, sens, k_sparse)  # 2. sensory -> inter
    fanin(cmd, inter, k_sparse)  # 3. inter -> command
    fanin(cmd, cmd, k_rec)  # 4. command -> command (recurrent, self excluded)
    for m in motor:  # 5. command -> motor (dense)
        for c in cmd:
            froms.append(int(c))
            tos.append(int(m))
    for m in motor:  # 6. motor -> command feedback (recurrent)
        for c in rng.choice(cmd, size=min(k_fb, len(cmd)), replace=False):
            froms.append(int(m))
            tos.append(int(c))

    froms_a = np.asarray(froms, dtype=np.int32)
    tos_a = np.asarray(tos, dtype=np.int32)
    weights = rng.uniform(-w_init, w_init, size=froms_a.shape[0]).astype(np.float32)

    tau = rng.uniform(0.5, 1.5, size=num_units).astype(np.float32)
    alpha = np.exp(-dt / tau).astype(np.float32)
    is_motor = np.zeros(num_units, dtype=np.float32)
    is_motor[motor] = 1.0

    static, state = build_pipeline_state(
        net,
        num_units=num_units,
        input_ids=tuple(int(i) for i in in_ids),
        output_ids=tuple(int(i) for i in motor),
        from_ids=froms_a,
        to_ids=tos_a,
        weights=weights,
        extra_unit_cols={Alpha.name: alpha, IsMotor.name: is_motor},
        globals_=None,
    )
    return static, state, tuple(int(i) for i in motor)


def reset_seq(state):
    """Zero the recurrent dynamics between sequences (oracle ResetPerSequence),
    keeping learned weights, α, IsMotor, wiring."""
    units = dict(state.units)
    for f in (ACTIVATION, LearningSignal, LossGrad):
        units[f.name] = jnp.zeros_like(units[f.name])
    conns0 = dict(state.conns[0])
    conns0[Elig.name] = jnp.zeros_like(conns0[Elig.name])
    return dataclasses.replace(state, units=units, conns=(conns0,))


# --- data (noisy sine, next-step; oracle MakeSine) --------------------------


def make_sine(n_seq, seq_len, noise_std, seed):
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.5, 2.0, size=n_seq).astype(np.float32)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_seq).astype(np.float32)
    t = np.arange(seq_len, dtype=np.float32)
    a0 = freqs[:, None] * (2 * np.pi * t[None, :] / seq_len) + phases[:, None]
    a1 = freqs[:, None] * (2 * np.pi * (t[None, :] + 1) / seq_len) + phases[:, None]
    noise = rng.normal(0.0, noise_std, size=(n_seq, seq_len, 2)).astype(np.float32)
    x = np.stack([np.sin(a0), np.cos(a0)], axis=-1) + noise  # noisy current
    y = np.stack([np.sin(a1), np.cos(a1)], axis=-1)  # clean next
    return x.astype(np.float32), y.astype(np.float32)


# --- run --------------------------------------------------------------------


def run(args) -> dict:
    probe = MemoryProbe()
    probe.start()

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    train_seqs = args.train_seqs if not args.quick else min(args.train_seqs, 48)
    test_seqs = args.test_seqs if not args.quick else min(args.test_seqs, 32)
    Xtr, Ytr = make_sine(train_seqs, args.seq_len, args.noise_std, args.seed)
    Xte, Yte = make_sine(test_seqs, args.seq_len, args.noise_std, args.seed + 999)
    probe.end_dataset()

    net, eval_net = make_nets(args.lr, args.beta_trace, args.clip_delta, args.w_max)
    static, state, motor_ids = build_ncp(
        net, args.units, args.out, args.seed, args.dt, args.w_init,
        args.k_sparse, args.k_rec, args.k_fb,
    )
    n_units_total = int(state.units[ACTIVATION.name].shape[0])
    n_units = n_units_total - len(static.input_ids)  # oracle reports ex-inputs
    n_edges = int((~state.conns[0][px.DEAD.name]).sum())
    probe.end_weights()

    train_runners = build_phase_runners(net, static, donate=True)
    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)
    motor_arr = jnp.asarray(motor_ids, dtype=jnp.int32)
    eval_fwd = build_phases(eval_net, static)[0]
    dummy = px.StepInputs(inputs=jnp.zeros((len(static.input_ids),), jnp.float32), targets=None)

    @jax.jit
    def eval_seq(state, x_seq):
        def step(st, x):
            act = st.units[ACTIVATION.name].at[input_ids].set(x)
            st = dataclasses.replace(st, units={**st.units, ACTIVATION.name: act})
            st, _ = eval_fwd(st, dummy)
            return st, st.units[ACTIVATION.name][motor_arr]

        return jax.lax.scan(step, state, x_seq)

    def test_mse(state) -> float:
        se = 0.0
        for s in range(test_seqs):
            _, preds = eval_seq(reset_seq(state), jnp.asarray(Xte[s]))
            se += float(np.mean((np.asarray(preds) - Yte[s]) ** 2))
        return se / test_seqs

    print(f"[info] plastax devices={jax.devices()} units={n_units} edges={n_edges} "
          f"seq_len={args.seq_len} train_seqs={train_seqs} epochs={epochs} lr={args.lr}")

    hist_path, summary_path, _ = output_paths(args, "ccwc_ncp")
    log = StructuralLog(hist_path)
    mse0 = test_mse(state)
    log.log(0, n_units, n_edges, edges={(0, 0, 0)}, val_loss=mse0, train_loss=None,
            test_mse=mse0, test_mae=0.0)

    # warmup/compile the train kernels off the clock
    si0 = px.StepInputs(inputs=jnp.asarray(Xtr[0, 0]), targets=jnp.asarray(Ytr[0, 0]))
    state, _ = run_phase_timed_step(train_runners, reset_seq(state), si0, PhaseTimer())
    jax.block_until_ready(state)

    rng = np.random.default_rng(args.seed)
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for _ep in range(epochs):
        for s in rng.permutation(train_seqs):
            state = reset_seq(state)
            for tt in range(args.seq_len):
                si = px.StepInputs(inputs=jnp.asarray(Xtr[s, tt]), targets=jnp.asarray(Ytr[s, tt]))
                state, _ = run_phase_timed_step(train_runners, state, si, timer)
    wall = time.perf_counter() - t0

    mse_final = test_mse(state)
    log.log(epochs, n_units, n_edges, edges={(0, 0, 0)}, val_loss=mse_final,
            train_loss=None, test_mse=mse_final, test_mae=0.0)
    log.flush()

    fused_ns = measure_fused_step_ns(net, static, state, si0, n=500)

    summary = {
        "workload": "06_ccwc_ncp",
        "dataset": "synthetic-sine",
        "units": args.units, "seq_len": args.seq_len,
        "epochs": epochs, "batch": 1, "lr": args.lr,
        "wall_seconds": round(wall, 3),
        "test_mse": round(mse_final, 6), "test_mse_init": round(mse0, 6),
        "test_mae": round(mse_final, 6), "val_mse_final": round(mse_final, 6),
        "n_units": n_units, "n_edges": n_edges,
        "jaccard_min": 1.0, "jaccard_max": 1.0,
        "seed": args.seed,
        "peak_vram_bytes": int(device_bytes()),
        "fused_step_ns_mean": round(fused_ns, 3),
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] test_mse {mse0:.4f}→{mse_final:.4f} "
          f"phase_sep_step_ns={summary['step_ns_mean']:.0f} "
          f"fused_step_ns={fused_ns:.0f} vram={summary['peak_vram_bytes'] / 1e6:.0f}MB")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--out", type=int, default=2, help="motor / output dim")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--train-seqs", type=int, default=256)
    p.add_argument("--val-seqs", type=int, default=64)
    p.add_argument("--test-seqs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--beta-trace", type=float, default=0.9)
    p.add_argument("--clip-delta", type=float, default=0.1)
    p.add_argument("--w-max", type=float, default=5.0)
    p.add_argument("--w-init", type=float, default=0.5)
    p.add_argument("--k-sparse", type=int, default=4)
    p.add_argument("--k-rec", type=int, default=4)
    p.add_argument("--k-fb", type=int, default=2)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
