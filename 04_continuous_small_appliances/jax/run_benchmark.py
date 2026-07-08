"""Workload 4 / 5 -- CONTINUOUS-SMALL regime, JAX port.

Mirrors 04_continuous_small_appliances/pytorch: a single-hidden-layer ReLU MLP
trained with sum-reduction MSE + plain SGD on the UCI Appliances streaming
regression task. Each step *maybe* splits the highest-activation-variance hidden
unit (a noisy, conservation-preserving copy) and *maybe* prunes the single
smallest-magnitude alive l1 edge. Net structural delta per step is at most +/-1
unit and a small number of edges, so the Jaccard-between-steps sits near 1.0.

This is the JAX reference impl for bench 04 -- same summary schema (metric +
phase + memory columns) as the pytorch/plastix/cpp/cuda impls.

JAX timing notes:
  * Params live as a pytree (W1, b1, W2, b2, mask). The math (forward / loss /
    grad / sgd) is jitted; ONE warmup step compiles every jitted fn off the
    clock before the timed loop.
  * Structural ops (split/prune) resize params, so they are done in host numpy
    -- exactly as the pytorch impl rebuilds nn.Linear layers -- and the jax
    arrays are rebuilt. A shape change triggers an XLA recompile; that is
    accepted for this initial port (faithfulness first).
  * Each phase mark blocks on that phase's result so async dispatch is flushed
    and the timing is real (see common/jax PhaseTimer).

Usage:
    uv run python 04_continuous_small_appliances/jax/run_benchmark.py --quick --no-plot
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
    download_if_missing,
    output_paths,
    write_summary_csv,
)

APPLIANCES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)


# --- data (numpy; mirrors the pytorch loader) ------------------------------

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


# --- host-side splittable/prunable MLP state --------------------------------
#
# Params carried as numpy arrays on the host so the grow/prune bookkeeping is a
# plain array edit (as the pytorch impl rebuilds layers). Jax arrays are rebuilt
# from these whenever the shape changes. The mask (l1 keep-mask) is folded into
# the jitted math via W1 * mask.

class HostMLP:
    def __init__(self, rng: np.random.Generator, in_dim: int, out_dim: int,
                 hidden: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Kaiming-ish uniform, matching nn.Linear default init scale.
        lim1 = 1.0 / (in_dim ** 0.5)
        self.W1 = rng.uniform(-lim1, lim1, (hidden, in_dim)).astype(np.float32)
        self.b1 = rng.uniform(-lim1, lim1, (hidden,)).astype(np.float32)
        lim2 = 1.0 / (hidden ** 0.5)
        self.W2 = rng.uniform(-lim2, lim2, (out_dim, hidden)).astype(np.float32)
        self.b2 = rng.uniform(-lim2, lim2, (out_dim,)).astype(np.float32)
        self.mask = np.ones((hidden, in_dim), dtype=bool)
        self.splits = 0
        self.prunes = 0

    @property
    def hidden(self) -> int:
        return self.W1.shape[0]

    def unit_count(self) -> int:
        return self.hidden + self.out_dim

    def edge_count(self) -> int:
        return int(self.mask.sum()) + self.W2.size

    def edge_set(self) -> set[tuple[int, int, int]]:
        rows, cols = np.nonzero(self.mask)
        edges = {(0, int(r), int(c)) for r, c in zip(rows, cols)}
        out_d, in_d = self.W2.shape
        edges.update((1, r, c) for r in range(out_d) for c in range(in_d))
        return edges

    def split_unit(self, idx: int, rng: np.random.Generator,
                   noise: float = 0.05) -> None:
        H = self.hidden
        new_row = self.W1[idx:idx + 1] + noise * rng.standard_normal(
            (1, self.in_dim)).astype(np.float32)
        self.W1 = np.concatenate([self.W1, new_row], axis=0)
        self.b1 = np.concatenate([self.b1, self.b1[idx:idx + 1].copy()])
        # Halve output col idx and copy to a new col (conservation-preserving).
        new_col = self.W2[:, idx:idx + 1] * 0.5
        self.W2[:, idx:idx + 1] = new_col
        self.W2 = np.concatenate([self.W2, new_col], axis=1)
        new_mask = np.ones((H + 1, self.in_dim), dtype=bool)
        new_mask[:H, :] = self.mask
        self.mask = new_mask
        self.splits += 1

    def kill_smallest_edge(self) -> bool:
        w = np.abs(self.W1) + (~self.mask).astype(np.float32) * 1e9
        i_min = int(w.argmin())
        v_min = w.flat[i_min]
        if v_min > 1e8:
            return False
        r, c = i_min // self.in_dim, i_min % self.in_dim
        self.mask[r, c] = False
        self.W1[r, c] = 0.0
        self.prunes += 1
        return True


# --- activation-variance tracking (host numpy) ------------------------------

class ActivationStats:
    def __init__(self, hidden: int, window: int, rng: np.random.Generator):
        self.window = window
        self.rng = rng
        self.buf = np.zeros((window, hidden), dtype=np.float32)
        self.idx = 0
        self.filled = 0

    def push(self, h: np.ndarray) -> None:
        if h.shape[1] != self.buf.shape[1]:
            self.buf = np.zeros((self.window, h.shape[1]), dtype=np.float32)
            self.idx = 0
            self.filled = 0
        self.buf[self.idx] = h.mean(0)
        self.idx = (self.idx + 1) % self.window
        self.filled = min(self.filled + 1, self.window)

    def hottest(self) -> int:
        if self.filled < 2:
            return int(self.rng.integers(self.buf.shape[1]))
        var = self.buf[:self.filled].var(0)
        return int(var.argmax())


# --- jitted math (params = (W1, b1, W2, b2, mask)) --------------------------

@jax.jit
def _hidden(params, xb):
    W1, b1, _W2, _b2, mask = params
    return jax.nn.relu(xb @ (W1 * mask).T + b1)


@jax.jit
def forward(params, xb):
    _W1, _b1, W2, b2, _mask = params
    h = _hidden(params, xb)
    return h @ W2.T + b2


@jax.jit
def loss_from_pred(pred, yb):
    return jnp.sum((pred - yb) ** 2)            # sum reduction (matches torch)


def _loss(params, xb, yb):
    W1, b1, W2, b2, mask = params
    h = jax.nn.relu(xb @ (W1 * mask).T + b1)
    pred = h @ W2.T + b2
    return jnp.sum((pred - yb) ** 2)


grad_fn = jax.jit(jax.grad(_loss))


def make_sgd(lr: float):
    @jax.jit
    def sgd(params, grads):
        W1, b1, W2, b2, mask = params
        gW1, gb1, gW2, gb2, _gmask = grads
        W1 = (W1 - lr * gW1) * mask     # re-apply keep-mask after the step
        b1 = b1 - lr * gb1
        W2 = W2 - lr * gW2
        b2 = b2 - lr * gb2
        return (W1, b1, W2, b2, mask)
    return sgd


@jax.jit
def eval_mse(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


def to_jax(m: HostMLP):
    return (jnp.asarray(m.W1), jnp.asarray(m.b1), jnp.asarray(m.W2),
            jnp.asarray(m.b2), jnp.asarray(m.mask, dtype=jnp.float32))


def run(args) -> dict:
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

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

    # Per-feature z-score over the first 10%.
    cut = max(int(0.1 * len(Xnp)), 512)
    mu = Xnp[:cut].mean(0); sd = Xnp[:cut].std(0)
    safe_sd = np.where(sd > 1e-3, sd, 1.0)
    Xnp = ((Xnp - mu) / safe_sd).astype(np.float32)
    y_mu = ynp[:cut].mean(); y_sd = ynp[:cut].std() + 1e-6
    ynp = ((ynp - y_mu) / y_sd).astype(np.float32)

    X = jnp.asarray(np.ascontiguousarray(Xnp))
    y = jnp.asarray(np.ascontiguousarray(ynp))[:, None]
    in_dim = X.shape[1]

    n_train = int(0.85 * len(X))
    X_test = X[n_train:]
    y_test = y[n_train:]
    probe.end_dataset()

    model = HostMLP(rng, in_dim, 1, hidden=args.init_hidden)
    params = to_jax(model)
    sgd = make_sgd(args.lr)
    probe.end_weights()
    act_stats = ActivationStats(model.hidden, window=args.var_window, rng=rng)

    hist_path, summary_path, _ = output_paths(args, "continuous_small_appliances")
    log = StructuralLog(hist_path)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    print(f"[info] jax devices={jax.devices()} in_dim={in_dim} "
          f"data={dataset_name} N={len(X)} steps={max_steps} "
          f"init_hidden={args.init_hidden}")

    # Warmup: force XLA compilation of every jitted fn off the clock.
    xb0, yb0 = X[:args.batch], y[:args.batch]
    p0 = forward(params, xb0)
    l0 = loss_from_pred(p0, yb0)
    g0 = grad_fn(params, xb0, yb0)
    _ = sgd(params, g0)
    _ = eval_mse(params, X_test, y_test)
    jax.block_until_ready((p0, l0, g0))

    deltas_units: list[int] = []
    deltas_edges: list[int] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        idx = (np.arange(args.batch) + (step - 1) * args.batch) % n_train
        xb = X[idx]; yb = y[idx]

        timer.tick()
        h = _hidden(params, xb)
        pred = forward(params, xb)
        timer.mark_forward(pred)
        loss = loss_from_pred(pred, yb)
        timer.mark_loss(loss)
        grads = grad_fn(params, xb, yb)
        timer.mark_backward(grads)
        params = sgd(params, grads)
        timer.mark_update(params)
        timer.step_done()

        # Sync the host param snapshot with the just-applied SGD update so the
        # structural bookkeeping (which lives in numpy) sees current weights.
        model.W1 = np.array(params[0])
        model.b1 = np.array(params[1])
        model.W2 = np.array(params[2])
        model.b2 = np.array(params[3])
        act_stats.push(np.asarray(h))

        units_before = model.unit_count()
        edges_before = model.edge_count()

        did_change = False
        # split?
        timer.tick()
        if rng.uniform() < args.p_split and model.hidden < args.max_hidden:
            hot = act_stats.hottest()
            model.split_unit(hot, rng, noise=args.split_noise)
            params = to_jax(model)
            did_change = True
        timer.mark_grow(params if did_change else None)
        # prune?
        pruned = False
        timer.tick()
        if rng.uniform() < args.p_prune and int(model.mask.sum()) > model.in_dim:
            if model.kill_smallest_edge():
                params = to_jax(model)
                did_change = True
                pruned = True
        timer.mark_prune(params if pruned else None)
        timer.mark_reset()

        deltas_units.append(model.unit_count() - units_before)
        deltas_edges.append(model.edge_count() - edges_before)

        if step % val_every == 0 or step == max_steps or did_change:
            vstart = (step * args.batch) % n_train
            vend = min(vstart + args.val_window, n_train)
            vl = float(eval_mse(params, X[vstart:vend], y[vstart:vend]))
            test_mse = float(eval_mse(params, X_test, y_test))
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
    test_mse_final = float(eval_mse(params, X_test, y_test))
    summary = {
        "workload": "04_continuous_small_appliances",
        "dataset": dataset_name,
        "in_dim": in_dim,
        "init_hidden": args.init_hidden,
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
          f"val_loss={summary['val_loss_final']:.4f} "
          f"test_mse={summary['test_mse']:.4f} "
          f"|du|_p99={summary['delta_units_p99_abs']} "
          f"jaccard_mean={summary['jaccard_mean']:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick, etc.
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
                   help="activation-variance window for hottest-unit pick")
    args = p.parse_args()

    summary = run(args)
    if summary["delta_units_max_abs"] > 1:
        print(f"[warn] |delta_units|_max = {summary['delta_units_max_abs']} > 1; "
              "regime invariant violated", file=sys.stderr)


if __name__ == "__main__":
    main()
