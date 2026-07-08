"""Workload 5 / 5 — CONTINUOUS-LARGE regime on Mackey-Glass, JAX port.

Mirrors 05_continuous_large_mackey_glass/pytorch: a sparse "reservoir-like"
MLP (dense input->hidden, masked hidden recurrent, dense hidden->output) is
trained one Adam SGD step per minibatch, and the topology is mutated *every
step*:

  1. Heavy-tailed unit delta Delta_n ~ +/- (1 + ceil(Pareto(alpha))): grow or
     shrink the hidden layer.
  2. Every `rewire_every` steps, Watts-Strogatz rewire a `rewire_frac` slice
     of alive recurrent edges (sever + reconnect to random partners).
  3. Growth-momentum accumulator over gradient norms fires a burst-grow when
     it crosses a threshold.

Regime signatures: consecutive-live-edge Jaccard sits well below 1.0, and
|Delta n_units| is heavy-tailed. Same summary schema as the pytorch impl.

JAX port notes:
  * Params are a pytree (dict of (W, b) pairs + a boolean rec_mask array).
  * The math (forward/loss/grad/adam-update) is jitted; each structural resize
    changes array shapes, so the bookkeeping is done in host numpy exactly as
    the pytorch impl does and the jax arrays are rebuilt afterwards. JIT
    recompiles on the new shapes — expected in this regime.
  * A warmup step runs each jitted fn once BEFORE the timed loop so XLA compile
    time isn't charged to a step. Phase marks block on their result so async
    dispatch is flushed and the timing is real (see common/jax PhaseTimer).
  * Adam state (m, v, t) is carried in the params pytree so it is jittable, and
    is resized alongside the weights on every structural op (mirrors the
    pytorch impl, which rebuilds the optimizer after each grow/shrink/burst so
    the moment buffers reset for the changed layer).

Note: Mackey-Glass is generated internally (no download), so there is no
`--synthetic` flag — the dataset is always deterministic from --seed.

Usage:
    uv run python 05_continuous_large_mackey_glass/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import math
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
# Mackey-Glass integration (numpy; identical to the pytorch impl)
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
# Sparse-recurrent net — host-numpy weight store + jax param pytree
# ---------------------------------------------------------------------------
#
# The pytorch impl keeps everything as torch tensors and mutates nn.Linear
# submodules in place. In JAX the jitted math wants immutable arrays of fixed
# shape, so we hold the mutable network state in host numpy (weights, biases,
# recurrent mask, Adam moments) and rebuild the jax pytree after each resize.
# This mirrors the pytorch "do the resize on host, rebuild the optimizer"
# pattern and keeps the per-step compute + structural cadence faithful.

class ReservoirState:
    """Host-numpy store for the sparse reservoir + its Adam moments."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int,
                 recur_density: float, rng_seed: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        g = np.random.default_rng(rng_seed)
        # Kaiming-uniform inits matching nn.Linear defaults.
        self.w_in = _lin_init(g, hidden, in_dim)
        self.b_in = np.zeros(hidden, dtype=np.float32)
        self.w_rec = _lin_init(g, hidden, hidden)
        self.b_rec = np.zeros(hidden, dtype=np.float32)
        self.w_out = _lin_init(g, out_dim, hidden)
        self.b_out = np.zeros(out_dim, dtype=np.float32)
        # Recurrent sparsity mask (no self-loops), spectral-radius scaling.
        mask = g.random((hidden, hidden)) < recur_density
        np.fill_diagonal(mask, False)
        self.rec_mask = mask
        self.w_rec *= 0.3
        self.w_rec *= self.rec_mask.astype(np.float32)
        self._new_adam()
        self.grows = 0
        self.shrinks = 0
        self.rewires = 0
        self.bursts = 0

    # --- adam moment buffers (reset on every structural op) --------------

    def _new_adam(self) -> None:
        self.m = {k: np.zeros_like(v) for k, v in self._weight_arrays().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._weight_arrays().items()}
        self.t = 0

    def _weight_arrays(self) -> dict:
        return {"w_in": self.w_in, "b_in": self.b_in,
                "w_rec": self.w_rec, "b_rec": self.b_rec,
                "w_out": self.w_out, "b_out": self.b_out}

    @property
    def hidden(self) -> int:
        return self.w_in.shape[0]

    # --- topology stats -------------------------------------------------

    def unit_count(self) -> int:
        return self.hidden + self.out_dim

    def edge_count(self) -> int:
        return int(self.w_in.size + self.rec_mask.sum() + self.w_out.size)

    def edge_set(self) -> set[tuple[int, int, int]]:
        out0, in0 = self.w_in.shape
        edges = {(0, r, c) for r in range(out0) for c in range(in0)}
        idx = np.argwhere(self.rec_mask)
        edges.update((1, int(r), int(c)) for r, c in idx)
        out2, in2 = self.w_out.shape
        edges.update((2, r, c) for r in range(out2) for c in range(in2))
        return edges

    # --- jax pytree bridge ----------------------------------------------

    def to_params(self) -> dict:
        """Build the jittable pytree from the host-numpy state."""
        return {
            "w_in": jnp.asarray(self.w_in), "b_in": jnp.asarray(self.b_in),
            "w_rec": jnp.asarray(self.w_rec), "b_rec": jnp.asarray(self.b_rec),
            "w_out": jnp.asarray(self.w_out), "b_out": jnp.asarray(self.b_out),
            "rec_mask": jnp.asarray(self.rec_mask.astype(np.float32)),
            "m": {k: jnp.asarray(v) for k, v in self.m.items()},
            "v": {k: jnp.asarray(v) for k, v in self.v.items()},
            "t": jnp.asarray(np.float32(self.t)),
        }

    def from_params(self, params: dict) -> None:
        """Pull an updated pytree back into host-numpy after a jitted step."""
        # copy: jax->numpy views are read-only, but rewire() mutates in place.
        self.w_in = np.array(params["w_in"])
        self.b_in = np.array(params["b_in"])
        self.w_rec = np.array(params["w_rec"])
        self.b_rec = np.array(params["b_rec"])
        self.w_out = np.array(params["w_out"])
        self.b_out = np.array(params["b_out"])
        self.m = {k: np.asarray(v) for k, v in params["m"].items()}
        self.v = {k: np.asarray(v) for k, v in params["v"].items()}
        self.t = int(params["t"])

    # --- structural ops (host numpy; mirrors pytorch _resize_to) --------

    def _resize_to(self, new_h: int, init_scale: float = 0.05) -> None:
        if new_h == self.hidden:
            return
        old_h = self.hidden
        g = np.random.default_rng()
        if new_h > old_h:
            d = new_h - old_h
            # l_in rows
            self.w_in = np.concatenate(
                [self.w_in, init_scale * g.standard_normal((d, self.in_dim)).astype(np.float32)], 0)
            self.b_in = np.concatenate([self.b_in, np.zeros(d, np.float32)])
            # l_rec rows then cols
            w_rec = np.concatenate(
                [self.w_rec, init_scale * g.standard_normal((d, old_h)).astype(np.float32)], 0)
            w_rec = np.concatenate(
                [w_rec, init_scale * g.standard_normal((new_h, d)).astype(np.float32)], 1)
            self.w_rec = w_rec
            self.b_rec = np.concatenate([self.b_rec, np.zeros(d, np.float32)])
            mask = np.concatenate(
                [self.rec_mask, np.zeros((d, old_h), dtype=bool)], 0)
            new_cols = g.random((new_h, d)) < 0.05
            mask = np.concatenate([mask, new_cols], 1)
            np.fill_diagonal(mask, False)
            self.rec_mask = mask
            # l_out cols
            self.w_out = np.concatenate(
                [self.w_out, init_scale * g.standard_normal((self.out_dim, d)).astype(np.float32)], 1)
        else:
            self.w_in = self.w_in[:new_h]
            self.b_in = self.b_in[:new_h]
            self.w_rec = self.w_rec[:new_h, :new_h]
            self.b_rec = self.b_rec[:new_h]
            self.rec_mask = self.rec_mask[:new_h, :new_h]
            self.w_out = self.w_out[:, :new_h]
        # b_out unchanged. Rebuild Adam moments (optimizer reset).
        self._new_adam()

    def grow(self, n: int, init_scale: float = 0.05) -> None:
        self._resize_to(self.hidden + n, init_scale=init_scale)
        self.grows += 1

    def shrink(self, n: int) -> None:
        target = max(8, self.hidden - n)
        self._resize_to(target)
        self.shrinks += 1

    def rewire(self, frac: float, rng: np.random.Generator) -> int:
        alive = np.argwhere(self.rec_mask)
        n_alive = alive.shape[0]
        if n_alive == 0:
            return 0
        n_rewire = max(1, int(frac * n_alive))
        sel = rng.choice(n_alive, size=n_rewire, replace=False)
        sel_idx = alive[sel]
        # sever
        self.rec_mask[sel_idx[:, 0], sel_idx[:, 1]] = False
        old_weights = self.w_rec[sel_idx[:, 0], sel_idx[:, 1]].copy()
        self.w_rec[sel_idx[:, 0], sel_idx[:, 1]] = 0.0
        # reconnect: single attempt per edge, drop invalid landings.
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
            self.w_rec[s, d] = w_val
            committed += 1
        self.rewires += 1
        return committed


def _lin_init(g: np.random.Generator, out_f: int, in_f: int) -> np.ndarray:
    lim = 1.0 / math.sqrt(in_f)
    return g.uniform(-lim, lim, size=(out_f, in_f)).astype(np.float32)


# ---------------------------------------------------------------------------
# Jitted math
# ---------------------------------------------------------------------------

def apply_reservoir(params, x):
    h = jnp.tanh(x @ params["w_in"].T + params["b_in"])
    rec = h @ (params["w_rec"] * params["rec_mask"]).T + params["b_rec"]
    h = jnp.tanh(h + rec)
    return h @ params["w_out"].T + params["b_out"]


@jax.jit
def forward(params, xb):
    return apply_reservoir(params, xb)


@jax.jit
def loss_from_pred(pred, yb):
    return jnp.mean((pred - yb) ** 2)          # mse (matches F.mse_loss)


def _loss(params, xb, yb):
    return jnp.mean((apply_reservoir(params, xb) - yb) ** 2)


_WEIGHT_KEYS = ("w_in", "b_in", "w_rec", "b_rec", "w_out", "b_out")


@jax.jit
def grad_fn(params, xb, yb):
    return jax.grad(_loss)(params, xb, yb)


@jax.jit
def grad_norm(grads):
    # sum of per-tensor L2 norms — matches sum(p.grad.norm()) in pytorch.
    return sum(jnp.sqrt(jnp.sum(grads[k] ** 2)) for k in _WEIGHT_KEYS)


def _make_adam(lr: float, beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8):
    @jax.jit
    def adam(params, grads):
        t = params["t"] + 1.0
        m = params["m"]
        v = params["v"]
        new_m, new_v, updates = {}, {}, {}
        for k in _WEIGHT_KEYS:
            gk = grads[k]
            mk = beta1 * m[k] + (1.0 - beta1) * gk
            vk = beta2 * v[k] + (1.0 - beta2) * (gk ** 2)
            mhat = mk / (1.0 - beta1 ** t)
            vhat = vk / (1.0 - beta2 ** t)
            new_m[k] = mk
            new_v[k] = vk
            updates[k] = params[k] - lr * mhat / (jnp.sqrt(vhat) + eps)
        out = dict(params)
        out.update(updates)
        out["m"] = new_m
        out["v"] = new_v
        out["t"] = t
        # Re-apply mask so severed recurrent weights stay zero (matches the
        # pytorch post-step `l_rec.weight.mul_(rec_mask)`).
        out["w_rec"] = out["w_rec"] * params["rec_mask"]
        return out
    return adam


@jax.jit
def eval_mse(params, x, y):
    return jnp.mean((apply_reservoir(params, x) - y) ** 2)


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(args) -> dict:
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    probe = MemoryProbe()
    probe.start()

    series = mackey_glass(n=args.series_len, tau=args.tau, seed=args.seed)
    series = (series - series.mean()) / (series.std() + 1e-6)

    X, Y = windowed(series, in_len=args.in_len, horizon=args.horizon)
    n = len(X)
    n_tr = int(0.7 * n); n_va = int(0.15 * n)
    Xtr, Ytr = jnp.asarray(X[:n_tr]), jnp.asarray(Y[:n_tr])
    Xva, Yva = jnp.asarray(X[n_tr:n_tr + n_va]), jnp.asarray(Y[n_tr:n_tr + n_va])
    Xte, Yte = jnp.asarray(X[n_tr + n_va:]), jnp.asarray(Y[n_tr + n_va:])
    Xtr_np = X[:n_tr]; Ytr_np = Y[:n_tr]
    probe.end_dataset()

    state = ReservoirState(in_dim=args.in_len, out_dim=1,
                           hidden=args.init_hidden,
                           recur_density=args.recur_density,
                           rng_seed=args.seed)
    params = state.to_params()
    adam = _make_adam(args.lr)
    probe.end_weights()

    hist_path, summary_path, plot_path = output_paths(args, "continuous_large_mg")
    log = StructuralLog(hist_path)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    rewire_every = max(1, args.rewire_every)
    print(f"[info] jax devices={jax.devices()}  N={n}  train={n_tr}  val={n_va}  "
          f"test={n - n_tr - n_va}  steps={max_steps}  init_hidden={args.init_hidden}  "
          f"recur_density={args.recur_density}")

    # Warmup: compile the jitted fns for the initial shapes off the clock.
    xb0 = Xtr[:args.batch]; yb0 = Ytr[:args.batch]
    p0 = forward(params, xb0); l0 = loss_from_pred(p0, yb0)
    g0 = grad_fn(params, xb0, yb0); gn0 = grad_norm(g0); u0 = adam(params, g0)
    jax.block_until_ready((p0, l0, g0, gn0, u0))

    growth_momentum = 0.0
    deltas_units: list[int] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        idx = rng.integers(0, n_tr, size=args.batch)
        xb = jnp.asarray(Xtr_np[idx]); yb = jnp.asarray(Ytr_np[idx])

        timer.tick()
        pred = forward(params, xb)
        timer.mark_forward(pred)
        loss = loss_from_pred(pred, yb)
        timer.mark_loss(loss)
        grads = grad_fn(params, xb, yb)
        timer.mark_backward(grads)
        gnorm = float(grad_norm(grads))
        params = adam(params, grads)
        timer.mark_update(params)

        units_before = state.unit_count()

        # Pull updated weights back to host so structural ops can resize them.
        state.from_params(params)

        # --- structural perturbation -------------------------------------
        # Heavy-tailed unit delta: sign +/-, magnitude 1 + ceil(Pareto(alpha)).
        did_grow = False
        did_shrink = False
        sign = 1 if rng.uniform() > 0.5 else -1
        mag = max(1, int(math.ceil(rng.pareto(args.pareto_alpha))))
        mag = min(mag, args.max_delta_per_step)
        if sign > 0 and state.hidden + mag <= args.max_hidden:
            state.grow(mag, init_scale=0.05)
            did_grow = True
        elif sign < 0 and state.hidden - mag >= args.min_hidden:
            state.shrink(mag)
            did_shrink = True

        # Periodic rewiring (Watts-Strogatz sever+reconnect).
        did_rewire = False
        if step % rewire_every == 0 and state.hidden >= 8:
            state.rewire(args.rewire_frac, rng)
            did_rewire = True

        # Growth-momentum burst.
        did_burst = False
        growth_momentum += gnorm
        if growth_momentum > args.momentum_threshold:
            burst = max(2, args.momentum_burst)
            if state.hidden + burst <= args.max_hidden:
                state.grow(burst)
                state.bursts += 1
                did_burst = True
            growth_momentum = 0.0

        # Phase marks for grow/prune. Sample every step even when idle so the
        # Welford accumulator sees a value each step (matches the rule).
        # grow phase covers grow + rewire-reconnect + burst; prune covers
        # shrink + rewire-sever.
        if did_grow or did_rewire or did_burst:
            timer.mark_grow(state.w_in)
        else:
            timer.mark_grow()
        if did_shrink or did_rewire:
            timer.mark_prune(state.w_in)
        else:
            timer.mark_prune()

        # If topology changed, rebuild the jax pytree (JIT recompiles on the
        # new shapes — expected in this regime).
        if did_grow or did_shrink or did_rewire or did_burst:
            params = state.to_params()

        timer.mark_reset()
        timer.step_done()

        deltas_units.append(state.unit_count() - units_before)

        if step % val_every == 0 or step == max_steps:
            vl = float(eval_mse(params, Xva, Yva))
            te = float(eval_mse(params, Xte, Yte))
            log.log(step, n_units=state.unit_count(),
                    n_edges=state.edge_count(),
                    edges=state.edge_set(),
                    val_loss=vl,
                    hidden=state.hidden,
                    grows=state.grows, shrinks=state.shrinks,
                    rewires=state.rewires, bursts=state.bursts,
                    delta_units=deltas_units[-1],
                    test_mse=te)
            if step % (val_every * 5) == 0 or step == max_steps:
                print(f"[step {step:>5d}] hidden={state.hidden:>4d}  "
                      f"edges={state.edge_count():>6d}  vl={vl:.4f}  "
                      f"jaccard={log.records[-1]['jaccard']:.3f}")

    wall = time.perf_counter() - t0
    log.flush()

    test_mse = float(eval_mse(params, Xte, Yte))

    abs_du = np.abs(deltas_units)
    nz_du = abs_du[abs_du > 0]
    summary = {
        "workload": "05_continuous_large_mg",
        "dataset": "mackey-glass",
        "tau": args.tau, "in_len": args.in_len, "horizon": args.horizon,
        "init_hidden": args.init_hidden, "recur_density": args.recur_density,
        "pareto_alpha": args.pareto_alpha,
        "rewire_every": args.rewire_every, "rewire_frac": args.rewire_frac,
        "max_steps": max_steps, "batch": args.batch,
        "wall_seconds": round(wall, 3),
        "grows": state.grows, "shrinks": state.shrinks,
        "rewires": state.rewires, "bursts": state.bursts,
        "hidden_final": state.hidden,
        "edges_final": state.edge_count(),
        "val_loss_initial": round(log.records[0]["val_loss"], 6),
        "val_loss_final": round(log.records[-1]["val_loss"], 6),
        "test_mse": round(test_mse, 6),
        "delta_units_p50_abs": int(np.percentile(abs_du, 50)) if len(abs_du) else 0,
        "delta_units_p95_abs": int(np.percentile(abs_du, 95)) if len(abs_du) else 0,
        "delta_units_max_abs": int(abs_du.max()) if len(abs_du) else 0,
        "delta_units_mean_nonzero": (float(nz_du.mean()) if len(nz_du) else 0.0),
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_mean": round(float(np.mean([r["jaccard"]
                                             for r in log.records])), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  grows={state.grows}  shrinks={state.shrinks}  "
          f"rewires={state.rewires}  hidden={state.hidden}  "
          f"test_mse={test_mse:.4f}  jaccard_mean={summary['jaccard_mean']:.3f}  "
          f"|du|_p95={summary['delta_units_p95_abs']}  "
          f"|du|_max={summary['delta_units_max_abs']}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
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
