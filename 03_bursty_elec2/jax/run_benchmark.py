"""Workload 3 / 5 -- BURSTY (punctuated equilibrium) regime, JAX port.

Mirrors 03_bursty_elec2/pytorch: a streaming binary classifier on Elec2 that
sits structurally idle for many steps, then -- when a sliding window of
validation losses goes flat -- fires a burst event that adds hidden units +
connections, followed immediately by a magnitude-prune sweep to bound growth.

The model is a 3-Linear-layer ReLU MLP (in -> H -> H -> out) trained with
sum-reduction cross-entropy + plain SGD. Per-layer boolean weight masks track
pruned (dead) edges; the forward multiplies weights by their mask.

JAX timing notes:
  * Params are a JAX pytree (list of (W, b)); forward / loss / grad / update are
    jitted. A warmup step runs each jitted fn once BEFORE the timed loop so XLA
    compile time isn't charged to the first step.
  * A burst changes parameter shapes, which re-triggers JIT compilation -- that
    is expected. All structural bookkeeping (grow units/edges, prune, rebuild
    masks) is done in host numpy exactly like the pytorch impl; the jax param
    arrays are then rebuilt and the jitted fns recompile on the next step's new
    shapes.
  * Structural phases are sampled every step (mark_grow/mark_prune with no arg on
    non-structural steps, ~0ns), mirroring how the pytorch impl samples them.

Usage:
    uv run python 03_bursty_elec2/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
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


ELEC2_URL = (
    "https://raw.githubusercontent.com/scikit-multiflow/"
    "streaming-datasets/master/elec.csv"
)


# ---------------------------------------------------------------------------
# Data (numpy; mirrors the pytorch loader)
# ---------------------------------------------------------------------------

def load_elec2(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = data_dir / "elec2.csv"
    download_if_missing(ELEC2_URL, csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
    target_col = "class" if "class" in df.columns else df.columns[-1]
    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col]
    if y_raw.dtype == object:
        uniq = sorted(y_raw.unique())
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    else:
        y = y_raw.to_numpy(dtype=np.int64)
    return X, y


def synth_drift(n: int = 30_000, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic drifting binary classification stream. The decision
    hyperplane rotates over `n` steps so the network is forced to keep
    learning (and plateauing then growing) throughout."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    t = np.linspace(0, 4 * np.pi, n)
    for i in range(n):
        w = np.array([math.sin(t[i] + j) for j in range(dim)], dtype=np.float32)
        y[i] = int(X[i] @ w > 0)
    return X, y


# ---------------------------------------------------------------------------
# Growable MLP -- host (numpy) structural state + jax param arrays
# ---------------------------------------------------------------------------

class GrowableMLP:
    """MLP whose two hidden layers can grow. Weight arrays and masks live in
    host numpy so grow/prune are plain array surgery (exactly like the pytorch
    impl); `params()` materialises the current (W, b) list as a jax pytree.

    Layout: in_dim -> H -> H -> out_dim, three Linear layers.
    W[i] has shape (in_i, out_i) so forward is x @ W + b (jax convention).
    Masks are on W (same shape).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int, seed: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        dims = [(in_dim, hidden), (hidden, hidden), (hidden, out_dim)]
        self.W = []
        self.b = []
        for (di, do) in dims:
            lim = 1.0 / math.sqrt(di)  # matches nn.Linear default init scale
            self.W.append(rng.uniform(-lim, lim, size=(di, do)).astype(np.float32))
            self.b.append(np.zeros((do,), dtype=np.float32))
        self.masks = [np.ones_like(w, dtype=bool) for w in self.W]
        self.bursts = 0
        self.prunes = 0

    # --- jax param pytree ------------------------------------------------

    def params(self):
        return [(jnp.asarray(self.W[i] * self.masks[i]), jnp.asarray(self.b[i]))
                for i in range(3)]

    def set_params(self, params) -> None:
        """Pull updated (W, b) back from the jax pytree into host numpy, then
        re-apply masks so pruned edges stay dead."""
        for i, (W, b) in enumerate(params):
            self.W[i] = np.asarray(W, dtype=np.float32) * self.masks[i]
            self.b[i] = np.asarray(b, dtype=np.float32)

    # --- size / topology -------------------------------------------------

    def unit_count(self) -> int:
        # out_features of layer0 + layer1 + out_dim, matching pytorch.
        return self.W[0].shape[1] + self.W[1].shape[1] + self.out_dim

    def edge_count(self) -> int:
        return int(sum(m.sum() for m in self.masks))

    def edge_set(self) -> set[tuple[int, int, int]]:
        edges = set()
        for li, m in enumerate(self.masks):
            rows, cols = np.nonzero(m)
            edges.update((li, int(r), int(c)) for r, c in zip(rows, cols))
        return edges

    # --- growth ----------------------------------------------------------

    def grow(self, n_new: int, noise: float = 0.05) -> int:
        """Append n_new units to BOTH hidden layers. Mirrors pytorch grow()
        but with (in, out) weight layout. Returns new hidden size."""
        rng = np.random.default_rng()
        new_h = self.hidden + n_new
        # Layer 0: (in_dim, hidden) -> (in_dim, new_h): add columns (outputs).
        w0 = np.concatenate([
            self.W[0],
            (noise * rng.standard_normal((self.in_dim, n_new))).astype(np.float32),
        ], axis=1)
        b0 = np.concatenate([self.b[0], np.zeros(n_new, dtype=np.float32)])
        # Layer 1: (hidden, hidden) -> (new_h, new_h): add rows (inputs) then
        # columns (outputs).
        w1 = np.concatenate([
            self.W[1],
            (noise * rng.standard_normal((n_new, self.hidden))).astype(np.float32),
        ], axis=0)
        w1 = np.concatenate([
            w1,
            (noise * rng.standard_normal((new_h, n_new))).astype(np.float32),
        ], axis=1)
        b1 = np.concatenate([self.b[1], np.zeros(n_new, dtype=np.float32)])
        # Layer 2: (hidden, out_dim) -> (new_h, out_dim): add rows (inputs).
        w2 = np.concatenate([
            self.W[2],
            (noise * rng.standard_normal((n_new, self.out_dim))).astype(np.float32),
        ], axis=0)
        b2 = self.b[2].copy()

        # Expand masks: new edges all start alive; preserve old dead structure.
        m0 = np.ones_like(w0, dtype=bool)
        m0[:self.masks[0].shape[0], :self.masks[0].shape[1]] = self.masks[0]
        m1 = np.ones_like(w1, dtype=bool)
        m1[:self.masks[1].shape[0], :self.masks[1].shape[1]] = self.masks[1]
        m2 = np.ones_like(w2, dtype=bool)
        m2[:self.masks[2].shape[0], :self.masks[2].shape[1]] = self.masks[2]

        self.W = [w0, w1, w2]
        self.b = [b0, b1, b2]
        self.masks = [m0, m1, m2]
        self.hidden = new_h
        self.bursts += 1
        return new_h

    def magnitude_prune(self, prune_frac: float) -> int:
        """Kill `prune_frac` of currently-alive weights by smallest magnitude
        (global). Returns count killed. Bounds growth after a burst."""
        flat = np.concatenate([
            np.abs(self.W[i][self.masks[i]]).ravel() for i in range(3)
        ])
        k = int(prune_frac * flat.size)
        if k <= 0:
            return 0
        thresh = np.partition(flat, k - 1)[k - 1]  # k-th smallest (1-indexed)
        killed = 0
        for i in range(3):
            kill = (np.abs(self.W[i]) <= thresh) & self.masks[i]
            killed += int(kill.sum())
            self.masks[i] = self.masks[i] & ~kill
            self.W[i] = self.W[i] * self.masks[i]
        self.prunes += 1
        return killed


# ---------------------------------------------------------------------------
# Plateau detector
# ---------------------------------------------------------------------------

class PlateauDetector:
    def __init__(self, window: int, rel_tol: float, cooldown: int):
        self.win = deque(maxlen=window)
        self.window = window
        self.rel_tol = rel_tol
        self.cooldown = cooldown
        self.steps_since_burst = cooldown

    def update(self, loss: float) -> bool:
        self.win.append(float(loss))
        self.steps_since_burst += 1
        if len(self.win) < self.window:
            return False
        if self.steps_since_burst < self.cooldown:
            return False
        mu = sum(self.win) / len(self.win)
        if mu <= 0:
            return False
        std = (sum((x - mu) ** 2 for x in self.win) / len(self.win)) ** 0.5
        if std / mu < self.rel_tol:
            self.steps_since_burst = 0
            return True
        return False


# ---------------------------------------------------------------------------
# Model math (jax)
# ---------------------------------------------------------------------------

def apply_mlp(params, x):
    last = len(params) - 1
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i != last:
            x = jax.nn.relu(x)
    return x


def _ce_sum(logits, yb):
    # sum-reduction cross-entropy (matches torch reduction='sum').
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(logp[jnp.arange(logits.shape[0]), yb])


def _ce_mean(logits, yb):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(logp[jnp.arange(logits.shape[0]), yb])


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(args) -> dict:
    np.random.seed(args.seed)

    probe = MemoryProbe()
    probe.start()

    if args.synthetic:
        Xnp, ynp = synth_drift(n=args.max_steps * args.batch + 5000, dim=8,
                               seed=args.seed)
        dataset_name = "synthetic-drift"
    else:
        try:
            Xnp, ynp = load_elec2(args.data_dir)
            dataset_name = "elec2"
        except Exception as e:
            print(f"[warn] Elec2 download failed ({e}); using synthetic drift",
                  file=sys.stderr)
            Xnp, ynp = synth_drift(n=args.max_steps * args.batch + 5000, dim=8,
                                   seed=args.seed)
            dataset_name = "synthetic-drift-fallback"

    cut = max(int(0.1 * len(Xnp)), 256)
    mu = Xnp[:cut].mean(0); sd = Xnp[:cut].std(0)
    safe_sd = np.where(sd > 1e-3, sd, 1.0)
    Xnp = ((Xnp - mu) / safe_sd).astype(np.float32)
    in_dim = Xnp.shape[1]
    n_classes = int(ynp.max()) + 1

    X = jnp.asarray(Xnp)
    y = jnp.asarray(ynp)

    n_train = int(0.85 * len(X))
    X_test = X[n_train:]
    y_test = y[n_train:]
    probe.end_dataset()

    model = GrowableMLP(in_dim, n_classes, hidden=args.init_hidden, seed=args.seed)
    lr = args.lr
    params = model.params()
    probe.end_weights()

    # --- jitted fns (recompile after each structural change) -------------
    @jax.jit
    def forward(params, xb):
        return apply_mlp(params, xb)

    @jax.jit
    def loss_from_pred(logits, yb):
        return _ce_sum(logits, yb)

    def _loss(params, xb, yb):
        return _ce_sum(apply_mlp(params, xb), yb)

    grad_fn = jax.jit(jax.grad(_loss))

    @jax.jit
    def sgd(params, grads):
        return [(W - lr * gW, b - lr * gb)
                for (W, b), (gW, gb) in zip(params, grads)]

    @jax.jit
    def eval_ce_mean(params, x, yb):
        logits = apply_mlp(params, x)
        return _ce_mean(logits, yb)

    @jax.jit
    def eval_ce_acc(params, x, yb):
        logits = apply_mlp(params, x)
        ce = _ce_mean(logits, yb)
        acc = jnp.mean((jnp.argmax(logits, -1) == yb).astype(jnp.float32))
        return ce, acc

    hist_path, summary_path, plot_path = output_paths(args, "bursty_elec2")
    log = StructuralLog(hist_path)

    detector = PlateauDetector(window=args.plateau_window,
                               rel_tol=args.plateau_rel_tol,
                               cooldown=args.plateau_cooldown)

    max_steps = args.max_steps // (5 if args.quick else 1)
    val_every = max(1, args.val_every // (2 if args.quick else 1))
    plateau_start = min(int(0.1 * len(X)), n_train - args.val_window - 1)
    plateau_end = plateau_start + args.val_window
    X_plateau = X[plateau_start:plateau_end]
    y_plateau = y[plateau_start:plateau_end]
    print(f"[info] jax devices={jax.devices()}  in_dim={in_dim}  "
          f"classes={n_classes}  data={dataset_name}  N={len(X)}  "
          f"steps={max_steps}")

    # Warmup: force XLA compilation of every jitted fn off the clock.
    xb0 = X[:args.batch]; yb0 = y[:args.batch]
    p0 = forward(params, xb0); l0 = loss_from_pred(p0, yb0)
    g0 = grad_fn(params, xb0, yb0); _ = sgd(params, g0)
    _ = eval_ce_acc(params, X_plateau, y_plateau)
    jax.block_until_ready((p0, l0, g0))

    burst_log: list[dict] = []
    timer = PhaseTimer()
    t0 = time.perf_counter()
    correct = 0; seen = 0
    for step in range(1, max_steps + 1):
        idx = (np.arange(args.batch) + (step - 1) * args.batch) % n_train
        xb = X[idx]; yb = y[idx]

        timer.tick()
        logits = forward(params, xb)
        timer.mark_forward(logits)
        loss = loss_from_pred(logits, yb)
        timer.mark_loss(loss)
        grads = grad_fn(params, xb, yb)
        timer.mark_backward(grads)
        params = sgd(params, grads)
        timer.mark_update(params)

        # --- structural phases: sampled every step (mostly no-ops) --------
        did_burst = False
        should_burst = False
        vl = None
        if step % val_every == 0 or step == max_steps:
            plateau_loss = float(eval_ce_mean(params, X_plateau, y_plateau))
            vstart = (step * args.batch) % n_train
            vend = min(vstart + args.val_window, n_train)
            vl_arr, va_arr = eval_ce_acc(params, X[vstart:vend], y[vstart:vend])
            vl = float(vl_arr); va = float(va_arr)
            test_ce, test_acc = eval_ce_acc(params, X_test, y_test)
            test_loss = float(test_ce); test_acc = float(test_acc)
            should_burst = detector.update(plateau_loss)

        # Running train-acc tally on the current batch.
        correct += int(jnp.sum(jnp.argmax(logits, -1) == yb))
        seen += int(yb.shape[0])

        if should_burst:
            # Pull the latest weights back to host, then do structural surgery.
            model.set_params(params)
            n_new = max(1, int(args.burst_frac * model.hidden))
            old_h = model.hidden
            new_h = model.grow(n_new, noise=args.burst_noise)
            timer.mark_grow(new_h)
            killed = model.magnitude_prune(args.post_burst_prune_frac)
            timer.mark_prune(killed)
            params = model.params()  # rebuilt arrays; jitted fns recompile next step
            did_burst = True
            burst_log.append({"step": step, "old_h": old_h, "new_h": new_h,
                              "killed": killed, "val_loss": vl})
            print(f"[burst {len(burst_log):>2d}] step={step:>6d}  "
                  f"hidden {old_h}->{new_h}  killed={killed}  vl={vl:.4f}")
            correct = 0; seen = 0
        else:
            timer.mark_grow()
            timer.mark_prune()

        timer.mark_reset()
        timer.step_done()

        if vl is not None:
            extras = {
                "train_acc_run": correct / max(seen, 1),
                "val_acc": va,
                "plateau_loss": plateau_loss,
                "bursts": model.bursts, "prunes": model.prunes,
                "test_acc": test_acc,
                "test_loss": test_loss,
            }
            log.log(step, n_units=model.unit_count(),
                    n_edges=model.edge_count(),
                    edges=model.edge_set(),
                    val_loss=vl, **extras)

    wall = time.perf_counter() - t0
    log.flush()

    summary = {
        "workload": "03_bursty_elec2",
        "dataset": dataset_name,
        "in_dim": in_dim, "n_classes": n_classes,
        "init_hidden": args.init_hidden,
        "max_steps": max_steps, "batch": args.batch,
        "burst_frac": args.burst_frac,
        "post_burst_prune_frac": args.post_burst_prune_frac,
        "plateau_window": args.plateau_window,
        "plateau_rel_tol": args.plateau_rel_tol,
        "plateau_cooldown": args.plateau_cooldown,
        "wall_seconds": round(wall, 3),
        "bursts_fired": model.bursts,
        "prunes_fired": model.prunes,
        "hidden_final": model.hidden,
        "edges_final": model.edge_count(),
        "val_loss_initial": round(log.records[0]["val_loss"], 6),
        "val_loss_final": round(log.records[-1]["val_loss"], 6),
        "jaccard_min": round(min(r["jaccard"] for r in log.records), 6),
        "jaccard_max": round(max(r["jaccard"] for r in log.records), 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))
    print(f"[done] wall={wall:.1f}s  bursts={model.bursts}  "
          f"hidden_final={model.hidden}  edges={model.edge_count()}  "
          f"val_loss_final={summary['val_loss_final']:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick, etc.
    p.add_argument("--synthetic", action="store_true",
                   help="force synthetic drifting stream instead of Elec2")
    p.add_argument("--init-hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-window", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--burst-frac", type=float, default=0.15)
    p.add_argument("--burst-noise", type=float, default=0.05)
    p.add_argument("--post-burst-prune-frac", type=float, default=0.10)
    p.add_argument("--plateau-window", type=int, default=8,
                   help="number of validation snapshots in the plateau window")
    p.add_argument("--plateau-rel-tol", type=float, default=0.03,
                   help="std/mean threshold below which we count as flat")
    p.add_argument("--plateau-cooldown", type=int, default=4,
                   help="min validation snapshots between bursts")
    args = p.parse_args()

    summary = stream(args)
    if not args.quick and summary["bursts_fired"] == 0:
        print("[warn] no bursts fired; try smaller --plateau-rel-tol or "
              "longer --max-steps", file=sys.stderr)


if __name__ == "__main__":
    main()
