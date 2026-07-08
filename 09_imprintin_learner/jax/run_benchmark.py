"""Workload 9 -- imprinting-learner on the audio-prediction benchmark, JAX port.

Mirrors 09_imprintin_learner/pytorch: streaming TD(λ) over the 2500-dim binary
audio observation, with the same APBD reader, the same offline-discounted-return
metric (test_mse over the final 10% of steps), and the same replacing-trace
α/nnz-scaled TD update.

Unlike the pytorch reference -- which runs a *static* 2500-dim linear TD and
never actually grows -- this JAX port also exercises the structural dynamism the
bench is named for: it **generates pattern features at runtime** (conjunctions of
active observation bits, gated on activity, per the C++ imprinting_learner
generateFeatures/tenure/removeFeatures logic) and **removes idle low-weight
features**. That runtime growth is the whole point of this port: it is the
stress test for JAX/XLA's handling of dynamic shapes.

APPROACH: NAIVE HOST-NUMPY SURGERY + JAX-ARRAY REBUILD (same as bench 04's jax
port). The feature count D changes at runtime; the TD state (w, e) and the
per-step feature-activation matrix live in host numpy so grow/remove is a plain
array edit, and the jitted math is re-fed rebuilt jax arrays. **Every distinct D
triggers an XLA recompile of the jitted step fn.** Because feature generation is
(near-)continuous, D climbs monotonically for long stretches and XLA recompiles
on almost every growth step -- this is exactly the pathological case for XLA and
the practical cost is measured and reported below (grow-phase ns dwarfs the
actual TD compute).

An alternative that AVOIDS recompiles -- fixed-capacity padding: allocate D_max
feature slots up front, carry a boolean active-mask, and never change the array
shape (generation/removal flip mask bits) -- is discussed in the module verdict
but NOT implemented here; this port documents the naive-recompile cost that the
task asks to measure.

JAX timing notes:
  * A warmup step compiles the base (observation-only) jitted step off the clock.
  * Each grow/remove that changes D forces XLA to recompile the jitted step for
    the new shape; that recompile time lands in the `grow`/`prune` phase columns.
  * mark_forward/mark_loss/mark_backward wrap the TD forward/δ/update; mark_grow
    wraps feature generation, mark_prune wraps feature removal.

Usage (keep --quick SHORT; recompiles make it slow):
    uv run python 09_imprintin_learner/jax/run_benchmark.py --quick --no-plot
"""

from __future__ import annotations

import argparse
import struct
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
    plot_run,
    plot_test_curve,
    test_plot_path,
    write_summary_csv,
)


# Legacy v1 observation dimension. v1 files carry no obs_dim field; v2 files
# state their own dimension in the header (obs_dim = n_freq_bins * n_mag_bins).
LEGACY_OBS_DIM = 2500
HEADER_BYTES = 16          # base header: magic(4) + version(4) + n_steps(8)
MAGIC = b"APBD"
SUPPORTED_VERSIONS = (1, 2)


# ---------------------------------------------------------------------------
# APBD reader (identical to the pytorch impl / dataset.hpp)
# ---------------------------------------------------------------------------

def _resolve_dataset(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "audio_prediction" / "dataset.bin",
        data_dir / "audio" / "dataset.bin",
        data_dir / "dataset.bin",
        Path("09_imprintin_learner/cpp/examples/output/dataset.bin"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_apbd(path: Path, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, R): X shape (N, obs_dim) uint8 in {0,1}, R shape (N,) float32."""
    with path.open("rb") as f:
        head = f.read(HEADER_BYTES)
        if head[:4] != MAGIC:
            raise RuntimeError(f"Not an APBD file (bad magic): {path}")
        version, n_steps = struct.unpack("<IQ", head[4:HEADER_BYTES])
        if version not in SUPPORTED_VERSIONS:
            raise RuntimeError(f"Unsupported APBD version {version}")
        if version == 1:
            obs_dim = LEGACY_OBS_DIM
        else:  # v2: obs_dim follows the base header
            (obs_dim,) = struct.unpack("<Q", f.read(8))
        packed_bytes = (obs_dim + 7) // 8
        record_bytes = packed_bytes + 1
        n = min(max_steps, n_steps) if max_steps > 0 else n_steps
        body = np.frombuffer(f.read(n * record_bytes), dtype=np.uint8)
    body = body.reshape(n, record_bytes)
    obs_packed = body[:, :packed_bytes]
    rewards = body[:, packed_bytes].view(np.int8).astype(np.float32)
    X = np.unpackbits(obs_packed, axis=1, bitorder="little")[:, :obs_dim]
    return X, rewards


# ---------------------------------------------------------------------------
# Offline discounted return (same for every impl)
# ---------------------------------------------------------------------------

def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    G = np.zeros_like(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        G[t] = acc
    return G


# ---------------------------------------------------------------------------
# Jitted TD(λ) step (traced over feature count D -> recompiles when D changes)
# ---------------------------------------------------------------------------
#
# State per step t, over the current D-dim feature activation vector phi_t:
#
#   V_t     = w · phi_t
#   δ_t     = r_t + γ·V_t - V_{t-1}
#   e_t     = γ·λ·e_{t-1};  e_t[i] = 1 for active i   (replacing trace)
#   w_{t+1} = w_t + (α / nnz)·δ_t·e_t
#
# The whole step is jitted for the current D. D changing (feature grow/remove)
# retraces -> XLA recompile. That recompile is the cost this port measures.

@jax.jit
def td_forward(w, phi):
    return jnp.dot(w, phi)                                    # V_t


@jax.jit
def td_update(w, e, phi, decay, delta, alpha):
    e = e * decay
    e = jnp.where(phi > 0, 1.0, e)                            # replacing trace
    nnz = jnp.maximum(jnp.sum(phi), 1.0)
    step_alpha = alpha / nnz
    w = w + (step_alpha * delta) * e
    return w, e


# ---------------------------------------------------------------------------
# Fixed-capacity PADDED jitted step (shapes are constant -> compiles ONCE).
# ---------------------------------------------------------------------------
#
# `w`, `e`, `phi`, `active` are all fixed length C. Inactive slots are forced to
# contribute nothing by masking phi with `active` before any math, so the value,
# trace and weight update over the length-C arrays match what the naive path
# computes over its length-D arrays. Because C never changes, XLA traces the
# jitted step exactly once for the whole run (no recompiles on grow/remove).

@jax.jit
def td_forward_padded(w, phi, active):
    phi = phi * active                                       # dead slots -> 0
    return jnp.dot(w, phi)                                    # V_t


@jax.jit
def td_update_padded(w, e, phi, active, decay, delta, alpha):
    phi = phi * active                                       # dead slots -> 0
    e = e * decay
    e = jnp.where(phi > 0, 1.0, e)                            # replacing trace
    e = e * active                                           # keep dead slots 0
    nnz = jnp.maximum(jnp.sum(phi), 1.0)
    step_alpha = alpha / nnz
    w = w + (step_alpha * delta) * e
    w = w * active                                           # dead slots stay 0
    return w, e


# ---------------------------------------------------------------------------
# Host-side feature bookkeeping (numpy). Observation features first, then
# runtime-generated pattern features appended at the tail.
# ---------------------------------------------------------------------------

class FeatureSet:
    """Feature population as host numpy. Observation features [0, obs_dim) are
    permanent (always tenured). Pattern features are appended at the tail: each
    is (members: int array of observation indices, k_threshold: int); it fires
    when >= k_threshold of its members are active this step."""

    def __init__(self, obs_dim: int, capacity: int):
        self.obs_dim = obs_dim
        self.capacity = capacity
        # pattern definitions (parallel lists, index i -> pattern i)
        self.pat_members: list[np.ndarray] = []
        self.pat_kthr: list[int] = []
        self.generated = 0
        self.removed = 0

    @property
    def n_patterns(self) -> int:
        return len(self.pat_members)

    @property
    def dim(self) -> int:
        return self.obs_dim + self.n_patterns

    def phi(self, x_obs: np.ndarray) -> np.ndarray:
        """Build the D-dim activation vector for this step's observation bits."""
        if self.n_patterns == 0:
            return x_obs
        pats = np.empty(self.n_patterns, dtype=np.float32)
        for i, (mem, k) in enumerate(zip(self.pat_members, self.pat_kthr)):
            pats[i] = 1.0 if x_obs[mem].sum() >= k else 0.0
        return np.concatenate([x_obs, pats])

    def add_pattern(self, members: np.ndarray, fraction: float) -> None:
        k = max(1, int(np.ceil(fraction * len(members))))
        self.pat_members.append(members.astype(np.int64))
        self.pat_kthr.append(k)
        self.generated += 1

    def remove_pattern(self, local_idx: int) -> None:
        """Remove pattern `local_idx` (index into the pattern tail)."""
        del self.pat_members[local_idx]
        del self.pat_kthr[local_idx]
        self.removed += 1


class PaddedFeatureSet:
    """Fixed-capacity feature population. Slots [0, obs_dim) are the permanent
    observation features (always active); slots [obs_dim, capacity) are pattern
    slots that can be activated (feature generation) or deactivated (removal)
    without ever changing an array shape.

    Mirrors FeatureSet's generation/removal logic exactly -- same conjunctive
    pattern-feature definition (members + k_threshold, fire on >= k active) and
    same idle low-weight removal -- but keeps everything in fixed-length arrays:
    an `active` boolean mask of length C, and per-slot pattern definitions in
    slot-indexed lists. Generation writes into a free slot; removal clears one.
    """

    def __init__(self, obs_dim: int, capacity: int):
        if capacity <= obs_dim:
            raise SystemExit(
                f"[err] --capacity ({capacity}) must exceed obs_dim ({obs_dim}) "
                f"to leave room for pattern features.")
        self.obs_dim = obs_dim
        self.capacity = capacity
        # active mask over the full length-C population. Observation slots are
        # permanently active; pattern slots start inactive.
        self.active = np.zeros(capacity, dtype=np.float32)
        self.active[:obs_dim] = 1.0
        # per-slot pattern definitions (only meaningful for active pattern slots)
        self.pat_members: list[np.ndarray | None] = [None] * capacity
        self.pat_kthr = np.zeros(capacity, dtype=np.int64)
        # slot indices [obs_dim, capacity) currently free (inactive), as a stack
        self.free_slots: list[int] = list(range(capacity - 1, obs_dim - 1, -1))
        self.generated = 0
        self.removed = 0

    @property
    def n_patterns(self) -> int:
        return int(self.active[self.obs_dim:].sum())

    @property
    def dim(self) -> int:
        """Live feature count (obs + active patterns). For reporting only; the
        stored/computed shape is always `capacity`."""
        return self.obs_dim + self.n_patterns

    def phi(self, x_obs: np.ndarray) -> np.ndarray:
        """Build the fixed length-C activation vector for this step's obs bits.
        Inactive pattern slots are left at 0 (the jitted step masks them too)."""
        phi = np.zeros(self.capacity, dtype=np.float32)
        phi[:self.obs_dim] = x_obs
        for s in range(self.obs_dim, self.capacity):
            if self.active[s] and self.pat_members[s] is not None:
                mem = self.pat_members[s]
                phi[s] = 1.0 if x_obs[mem].sum() >= self.pat_kthr[s] else 0.0
        return phi

    def has_free_slot(self) -> bool:
        return len(self.free_slots) > 0

    def add_pattern(self, members: np.ndarray, fraction: float) -> int:
        """Activate a free pattern slot with a new conjunctive pattern; returns
        the absolute slot index (or -1 if capacity is exhausted)."""
        if not self.free_slots:
            return -1
        s = self.free_slots.pop()
        k = max(1, int(np.ceil(fraction * len(members))))
        self.pat_members[s] = members.astype(np.int64)
        self.pat_kthr[s] = k
        self.active[s] = 1.0
        self.generated += 1
        return s

    def remove_slot(self, s: int) -> None:
        """Deactivate pattern slot `s` and return it to the free pool."""
        self.active[s] = 0.0
        self.pat_members[s] = None
        self.pat_kthr[s] = 0
        self.free_slots.append(s)
        self.removed += 1


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_padded(args) -> dict:
    """Fixed-capacity padded variant. All feature arrays live at length C; grow/
    remove are index writes on a boolean active-mask, never shape changes, so the
    jitted step compiles ONCE (xla_recompiles ~= 1). Mirrors run()'s feature
    generation / removal logic exactly; only the storage strategy differs."""
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    probe = MemoryProbe()
    probe.start()

    path = _resolve_dataset(args.data_dir)
    if path is None:
        raise SystemExit(
            f"[err] audio-prediction dataset.bin not found under {args.data_dir}.\n"
            f"      Generate via 09_imprintin_learner/cpp/examples/"
            f"prepare-cpp.py, or pass --data-dir <dir-with-dataset.bin>."
        )

    max_steps = args.max_steps
    log_every = args.log_every
    if args.quick:
        max_steps = min(max_steps, args.quick_steps)
        log_every = min(log_every, 100)

    X, R = _load_apbd(path, max_steps)
    N, obs_dim = X.shape
    G = compute_returns(R, args.gamma)
    probe.end_dataset()

    print(f"[info] jax devices={jax.devices()}  dataset={path}  steps={N}  "
          f"obs_dim={obs_dim}  gamma={args.gamma}  alpha={args.alpha}  "
          f"lambda={args.lam}  quick={int(args.quick)}  mode=padded  "
          f"capacity={args.capacity}")

    hist_path, summary_path, plot_path = output_paths(args, "audio_imprinting")
    log = StructuralLog(hist_path)

    fs = PaddedFeatureSet(obs_dim, capacity=args.capacity)
    C = fs.capacity
    decay = args.gamma * args.lam

    # Fixed-capacity TD state: length-C arrays that never change shape. The
    # active mask (also length C) is carried on device; grow/remove edit it via
    # host writes and re-upload, but the *shape* is constant so no recompile.
    w_np = np.zeros(C, dtype=np.float32)
    e_np = np.zeros(C, dtype=np.float32)
    probe.end_weights()

    w_j = jnp.asarray(w_np)
    e_j = jnp.asarray(e_np)
    active_j = jnp.asarray(fs.active)

    # Warmup: compile the (single, shape-C) jitted step off the clock.
    phi0 = jnp.asarray(fs.phi(X[0].astype(np.float32)))
    _ = td_forward_padded(w_j, phi0, active_j)
    _w, _e = td_update_padded(w_j, e_j, phi0, active_j, decay, 0.0, args.alpha)
    jax.block_until_ready((_w, _e))

    predictions = np.zeros(N, dtype=np.float32)
    recompiles = 0
    prev_shape = int(w_j.shape[0])   # constant C; recompiles counts shape changes

    window_sse = 0.0
    window_cnt = 0
    epoch = 0

    timer = PhaseTimer()
    t0 = time.perf_counter()

    x0 = X[0].astype(np.float32)
    V_old = float(td_forward_padded(w_j, jnp.asarray(fs.phi(x0)), active_j))

    for t in range(N):
        x_obs = X[t].astype(np.float32)
        phi_np = fs.phi(x_obs)
        phi_j = jnp.asarray(phi_np)

        timer.tick()
        V_t = td_forward_padded(w_j, phi_j, active_j)
        timer.mark_forward(V_t)
        v_t_f = float(V_t)

        r_t_f = float(R[t])
        delta = r_t_f + args.gamma * v_t_f - V_old
        timer.mark_loss(V_t)  # delta computed on host; sync the (already-ready) V_t

        w_j, e_j = td_update_padded(
            w_j, e_j, phi_j, active_j, decay, delta, args.alpha)
        timer.mark_backward((w_j, e_j))
        timer.step_done()

        predictions[t] = v_t_f
        V_old = v_t_f

        err = v_t_f - float(G[t])
        window_sse += err * err
        window_cnt += 1

        # --- structural: remove idle low-weight patterns, then generate new ----
        grew = False
        pruned = False

        # PRUNE: deactivate pattern slots whose |weight| decayed below threshold.
        # Same logic as naive remove; here it is a masked index write (no shape
        # change), zeroing the slot's weight/eligibility and clearing its mask.
        timer.tick()
        if fs.n_patterns > 0 and (t + 1) % args.struct_every == 0:
            w_host = np.array(w_j)   # writable copy (jax arrays are read-only)
            e_host = np.array(e_j)
            pat_slots = np.nonzero(fs.active[obs_dim:] > 0)[0] + obs_dim
            idle = pat_slots[np.abs(w_host[pat_slots]) < args.idle_threshold]
            if len(idle) > 0:
                for s in idle:
                    s = int(s)
                    fs.remove_slot(s)
                    w_host[s] = 0.0
                    e_host[s] = 0.0
                w_j = jnp.asarray(w_host)
                e_j = jnp.asarray(e_host)
                active_j = jnp.asarray(fs.active)
                pruned = True
        timer.mark_prune((w_j, e_j) if pruned else None)

        # GROW: activate a free pattern slot with a conjunctive pattern feature
        # over currently-active observation bits (mirrors the naive grow gate:
        # sample n active members, fire on >= ceil(fraction*n)). Index write into
        # the free slot -> initializes its weight (already 0); NO shape change.
        timer.tick()
        active_obs = np.nonzero(x_obs > 0)[0]
        can_grow = (
            (t + 1) % args.struct_every == 0
            and fs.has_free_slot()
            and len(active_obs) >= args.pattern_min_conn
            and rng.uniform() < args.p_grow
        )
        if can_grow:
            n_members = int(rng.integers(
                args.pattern_min_conn,
                min(args.pattern_max_conn, len(active_obs)) + 1))
            members = rng.choice(active_obs, size=n_members, replace=False)
            s = fs.add_pattern(members, args.pattern_fraction)
            if s >= 0:
                w_host = np.array(w_j)    # writable copy
                w_host[s] = 0.0            # initialize new feature weight
                w_j = jnp.asarray(w_host)
                active_j = jnp.asarray(fs.active)
                grew = True
        timer.mark_grow((w_j, e_j) if grew else None)
        timer.mark_reset()

        # Shapes are constant, so this never increments (compile-once). Counted
        # the same way as the naive path for schema parity.
        cur_shape = int(w_j.shape[0])
        if cur_shape != prev_shape:
            recompiles += 1
            prev_shape = cur_shape

        if window_cnt >= log_every or t + 1 == N:
            window_mse = window_sse / window_cnt
            epoch += 1
            log.log(epoch, n_units=fs.dim, n_edges=fs.dim, edges=None,
                    val_loss=window_mse, train_loss=window_mse, train_step=t + 1,
                    test_mse=window_mse, n_features=fs.dim,
                    generated=fs.generated, removed=fs.removed)
            print(f"[ep {epoch}] step={t+1} D={fs.dim} (C={C}) "
                  f"gen={fs.generated} rem={fs.removed} "
                  f"window_mse={window_mse:.4f} recompiles={recompiles}")
            window_sse = 0.0
            window_cnt = 0

    wall = time.perf_counter() - t0
    log.flush()

    tail = max(1, N // 10)
    diff_tail = predictions[-tail:] - G[-tail:]
    test_mse = float((diff_tail * diff_tail).mean())
    full_mse = float(((predictions - G) ** 2).mean())

    summary = {
        "workload": "09_imprintin_learner",
        "dataset": "audio_prediction",
        "mode": "padded",
        "max_steps": N,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "lambda": args.lam,
        "wall_seconds": round(wall, 3),
        "val_mse_final": round(full_mse, 6),
        "test_mse": round(test_mse, 6),
        "metric_kind": "mse",
        "n_units": int(fs.dim),
        "n_edges": int(fs.dim),
        "obs_dim": int(obs_dim),
        "capacity": int(fs.capacity),
        "features_generated": int(fs.generated),
        "features_removed": int(fs.removed),
        "patterns_final": int(fs.n_patterns),
        "xla_recompiles": int(recompiles),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.2f}s  test_mse={test_mse:.6f}  "
          f"full_mse={full_mse:.6f}  D_final={fs.dim}  C={C}  "
          f"gen={fs.generated}  rem={fs.removed}  "
          f"recompiles={recompiles}  (step_count={timer.step_count})")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Audio Imprinting TD(λ) padded -- α={args.alpha} "
                       f"λ={args.lam} C={C}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "audio_imprinting")
        plot_test_curve(
            log.records, tpath,
            title=f"Audio Imprinting TD(λ) padded window MSE -- α={args.alpha}",
            metric_key="test_mse", ylabel="window MSE",
            higher_is_better=False,
        )
        print(f"[done] wrote {tpath}")
    return summary


def run(args) -> dict:
    if getattr(args, "padded", False):
        return run_padded(args)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    probe = MemoryProbe()
    probe.start()

    path = _resolve_dataset(args.data_dir)
    if path is None:
        raise SystemExit(
            f"[err] audio-prediction dataset.bin not found under {args.data_dir}.\n"
            f"      Generate via 09_imprintin_learner/cpp/examples/"
            f"prepare-cpp.py, or pass --data-dir <dir-with-dataset.bin>."
        )

    max_steps = args.max_steps
    log_every = args.log_every
    if args.quick:
        max_steps = min(max_steps, args.quick_steps)
        log_every = min(log_every, 100)

    X, R = _load_apbd(path, max_steps)
    N, obs_dim = X.shape
    G = compute_returns(R, args.gamma)
    probe.end_dataset()

    print(f"[info] jax devices={jax.devices()}  dataset={path}  steps={N}  "
          f"obs_dim={obs_dim}  gamma={args.gamma}  alpha={args.alpha}  "
          f"lambda={args.lam}  quick={int(args.quick)}")

    hist_path, summary_path, plot_path = output_paths(args, "audio_imprinting")
    log = StructuralLog(hist_path)

    fs = FeatureSet(obs_dim, capacity=args.capacity)
    decay = args.gamma * args.lam

    # TD state lives in host numpy so grow/remove is a plain array edit; jax
    # arrays are rebuilt whenever D changes (which forces an XLA recompile).
    w_np = np.zeros(obs_dim, dtype=np.float32)
    e_np = np.zeros(obs_dim, dtype=np.float32)
    probe.end_weights()

    # Warmup: compile the jitted step at the base (observation-only) D off clock.
    w_j = jnp.asarray(w_np)
    e_j = jnp.asarray(e_np)
    phi0 = jnp.asarray(fs.phi(X[0].astype(np.float32)))
    _ = td_forward(w_j, phi0)
    _w, _e = td_update(w_j, e_j, phi0, decay, 0.0, args.alpha)
    jax.block_until_ready((_w, _e))

    predictions = np.zeros(N, dtype=np.float32)
    recompiles = 0
    prev_dim = fs.dim

    window_sse = 0.0
    window_cnt = 0
    epoch = 0

    timer = PhaseTimer()
    t0 = time.perf_counter()

    x0 = X[0].astype(np.float32)
    V_old = float(td_forward(w_j, jnp.asarray(fs.phi(x0))))

    for t in range(N):
        x_obs = X[t].astype(np.float32)
        phi_np = fs.phi(x_obs)
        phi_j = jnp.asarray(phi_np)

        timer.tick()
        V_t = td_forward(w_j, phi_j)
        timer.mark_forward(V_t)
        v_t_f = float(V_t)

        r_t_f = float(R[t])
        delta = r_t_f + args.gamma * v_t_f - V_old
        timer.mark_loss(V_t)  # delta computed on host; sync the (already-ready) V_t

        w_j, e_j = td_update(w_j, e_j, phi_j, decay, delta, args.alpha)
        timer.mark_backward((w_j, e_j))
        timer.step_done()

        predictions[t] = v_t_f
        V_old = v_t_f

        err = v_t_f - float(G[t])
        window_sse += err * err
        window_cnt += 1

        # --- structural: remove idle low-weight patterns, then generate new ----
        grew = False
        pruned = False

        # PRUNE: drop pattern features whose |weight| decayed below threshold.
        timer.tick()
        if fs.n_patterns > 0 and (t + 1) % args.struct_every == 0:
            w_host = np.asarray(w_j)
            pat_w = np.abs(w_host[obs_dim:])
            # remove idle patterns (weight never grew past the idle threshold)
            idle = np.nonzero(pat_w < args.idle_threshold)[0]
            if len(idle) > 0:
                # remove highest-index-first so earlier indices stay valid
                keep = np.ones(fs.n_patterns, dtype=bool)
                keep[idle] = False
                for li in sorted(idle, reverse=True):
                    fs.remove_pattern(int(li))
                w_host = np.concatenate([w_host[:obs_dim], w_host[obs_dim:][keep]])
                e_host = np.concatenate([np.asarray(e_j)[:obs_dim],
                                         np.asarray(e_j)[obs_dim:][keep]])
                w_j = jnp.asarray(w_host)
                e_j = jnp.asarray(e_host)
                pruned = True
        timer.mark_prune((w_j, e_j) if pruned else None)

        # GROW: generate a conjunctive pattern feature over currently-active
        # observation bits (mirrors the C++ generateFeatures activity gate:
        # sample n active members, fire on >= ceil(fraction*n) of them).
        timer.tick()
        active_obs = np.nonzero(x_obs > 0)[0]
        can_grow = (
            (t + 1) % args.struct_every == 0
            and fs.dim < fs.capacity
            and len(active_obs) >= args.pattern_min_conn
            and rng.uniform() < args.p_grow
        )
        if can_grow:
            n_members = int(rng.integers(
                args.pattern_min_conn,
                min(args.pattern_max_conn, len(active_obs)) + 1))
            members = rng.choice(active_obs, size=n_members, replace=False)
            fs.add_pattern(members, args.pattern_fraction)
            # extend TD state by one zero slot for the new feature
            w_host = np.concatenate([np.asarray(w_j), np.zeros(1, np.float32)])
            e_host = np.concatenate([np.asarray(e_j), np.zeros(1, np.float32)])
            w_j = jnp.asarray(w_host)
            e_j = jnp.asarray(e_host)
            grew = True
        timer.mark_grow((w_j, e_j) if grew else None)
        timer.mark_reset()

        if fs.dim != prev_dim:
            recompiles += 1
            prev_dim = fs.dim

        if window_cnt >= log_every or t + 1 == N:
            window_mse = window_sse / window_cnt
            epoch += 1
            log.log(epoch, n_units=fs.dim, n_edges=fs.dim, edges=None,
                    val_loss=window_mse, train_loss=window_mse, train_step=t + 1,
                    test_mse=window_mse, n_features=fs.dim,
                    generated=fs.generated, removed=fs.removed)
            print(f"[ep {epoch}] step={t+1} D={fs.dim} "
                  f"gen={fs.generated} rem={fs.removed} "
                  f"window_mse={window_mse:.4f} recompiles={recompiles}")
            window_sse = 0.0
            window_cnt = 0

    wall = time.perf_counter() - t0
    log.flush()

    tail = max(1, N // 10)
    diff_tail = predictions[-tail:] - G[-tail:]
    test_mse = float((diff_tail * diff_tail).mean())
    full_mse = float(((predictions - G) ** 2).mean())

    summary = {
        "workload": "09_imprintin_learner",
        "dataset": "audio_prediction",
        "mode": "naive",
        "max_steps": N,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "lambda": args.lam,
        "wall_seconds": round(wall, 3),
        "val_mse_final": round(full_mse, 6),
        "test_mse": round(test_mse, 6),
        "metric_kind": "mse",
        "n_units": int(fs.dim),
        "n_edges": int(fs.dim),
        "obs_dim": int(obs_dim),
        "capacity": int(fs.capacity),
        "features_generated": int(fs.generated),
        "features_removed": int(fs.removed),
        "patterns_final": int(fs.n_patterns),
        "xla_recompiles": int(recompiles),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.2f}s  test_mse={test_mse:.6f}  "
          f"full_mse={full_mse:.6f}  D_final={fs.dim}  "
          f"gen={fs.generated}  rem={fs.removed}  "
          f"recompiles={recompiles}  (step_count={timer.step_count})")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Audio Imprinting TD(λ) -- α={args.alpha} λ={args.lam}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "audio_imprinting")
        plot_test_curve(
            log.records, tpath,
            title=f"Audio Imprinting TD(λ) window MSE -- α={args.alpha}",
            metric_key="test_mse", ylabel="window MSE",
            higher_is_better=False,
        )
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)   # provides --device (jax uses its own backend), --quick
    p.add_argument("--max-steps", type=int, default=20000,
                   help="cap the number of timesteps to run (0 = all)")
    p.add_argument("--quick-steps", type=int, default=400,
                   help="step cap under --quick (kept small: recompiles are slow)")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=3e-3)
    p.add_argument("--lambda", dest="lam", type=float, default=0.9)
    # --- storage strategy ---
    p.add_argument("--padded", action="store_true",
                   help="fixed-capacity padded mode: length-C arrays + active "
                        "mask, no shape changes, so the jitted step compiles "
                        "ONCE (default off = naive host-numpy surgery that "
                        "recompiles on every distinct feature count)")
    # --- runtime feature generation / removal ---
    p.add_argument("--capacity", type=int, default=8192,
                   help="max feature count / fixed padded array length C "
                        "(obs + generated pattern slots); sized above the "
                        "naive run's peak feature count")
    p.add_argument("--struct-every", type=int, default=1,
                   help="do a grow/remove attempt every this many steps")
    p.add_argument("--p-grow", type=float, default=0.5,
                   help="per-attempt probability of generating a pattern feature")
    p.add_argument("--pattern-min-conn", type=int, default=2)
    p.add_argument("--pattern-max-conn", type=int, default=8)
    p.add_argument("--pattern-fraction", type=float, default=0.5,
                   help="pattern fires when >= ceil(fraction*n_members) active")
    p.add_argument("--idle-threshold", type=float, default=1e-3,
                   help="remove a pattern feature whose |weight| stays below this")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
