"""plastax-side benchmark helpers — the plastax analogue of common/jax/common.py.

plastax is the JAX port of the Plastix plastic-network framework (the C++
`plastix::Network`). It fuses forward/loss/backward/update/prune/grow into ONE
jitted step (`plastax.make_step` / `plastax.Driver`), which is its throughput
win. But the native C++ bench times each `DoForwardPass` / `DoBackwardPass` /
`DoUpdateConn` separately, and the summary schema wants a per-phase breakdown,
so for an apples-to-apples comparison this helper drives the SAME plastax
phases *individually* (each its own jitted callable) and times each — exactly
mirroring the oracle's separate-pass structure. `build_phase_runners` returns
one jitted callable per present phase, in execution order, keyed by the
PhaseTimer phase name.

Everything framework-agnostic (PhaseTimer, MemoryProbe, CLI/CSV/paths,
device_bytes) is reused from common/jax/common.py — plastax IS jax underneath,
so the async-safe JAX PhaseTimer is exactly right.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Reuse the jax common (which itself re-exports the framework-agnostic pytorch
# helpers). Loaded by explicit path under a unique name so it doesn't collide
# with this module's own name.
_JAXC_PATH = Path(__file__).resolve().parents[1] / "jax" / "common.py"
_JAXC_NAME = "_plastix_jax_common"
_spec = importlib.util.spec_from_file_location(_JAXC_NAME, _JAXC_PATH)
_jaxc = importlib.util.module_from_spec(_spec)
sys.modules[_JAXC_NAME] = _jaxc
_spec.loader.exec_module(_jaxc)

MemoryProbe = _jaxc.MemoryProbe                 # noqa: F401  (re-exported)
PhaseTimer = _jaxc.PhaseTimer                   # noqa: F401
StructuralLog = _jaxc.StructuralLog             # noqa: F401
add_common_args = _jaxc.add_common_args         # noqa: F401
output_paths = _jaxc.output_paths               # noqa: F401
write_summary_csv = _jaxc.write_summary_csv     # noqa: F401
download_if_missing = _jaxc.download_if_missing  # noqa: F401
device_bytes = _jaxc.device_bytes               # noqa: F401
plot_run = _jaxc.plot_run                        # noqa: F401
test_plot_path = _jaxc.test_plot_path            # noqa: F401

import plastax as px  # noqa: E402
from plastax._types import (  # noqa: E402
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    TO_ID,
    WEIGHT,
    Propagation,
)
from plastax.phases import build_phases  # noqa: E402
from plastax.state import NetworkState, NetworkStatic  # noqa: E402
from plastax.topo import capacity_policy  # noqa: E402

# build_phases order (phases.py): forward, loss, backward, update_conn,
# prune_conn, add_conn, reset_global. Map each present slot to its PhaseTimer
# phase name so the timed loop can address them by name.
_PHASE_NAME = {
    "loss": "loss",
    "backward_pass": "backward",
    "update_conn": "update",
    "prune_conn": "prune",
    "add_conn": "grow",
    "reset_global": "reset",
}


def build_phase_runners(net, static, *, donate: bool = True) -> dict:
    """One jitted callable per present phase, keyed by PhaseTimer name
    ('forward','loss','backward','update','prune','grow','reset').

    'forward' has the per-step input scatter (StepInputs.inputs -> the input
    units' ACTIVATION) prepended, mirroring the oracle's DoForwardPass(input).
    Each callable is (state, StepInputs) -> (state, loss_contribution) and,
    with donate=True, donates the state pytree just like make_step's fused
    step, so per-phase timing reflects the same in-place update economics.
    """
    overflow_sink = [jnp.bool_(False)]
    phases = build_phases(net, static, overflow_sink=overflow_sink)

    names = ["forward"]
    if net.loss is not None:
        names.append("loss")
    if net.backward_pass is not None:
        names.append("backward")
    if net.update_conn is not None:
        names.append("update")
    if net.prune_conn is not None:
        names.append("prune")
    if net.add_conn is not None:
        names.append("grow")
    if net.reset_global is not None:
        names.append("reset")
    assert len(names) == len(phases), (names, len(phases))

    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)
    forward_phase = phases[0]

    def forward_with_scatter(state, step_inputs):
        activation = state.units[ACTIVATION.name].at[input_ids].set(step_inputs.inputs)
        state = dataclasses.replace(
            state, units={**state.units, ACTIVATION.name: activation}
        )
        return forward_phase(state, step_inputs)

    jit = (lambda fn: jax.jit(fn, donate_argnums=0)) if donate else jax.jit
    runners = {"forward": jit(forward_with_scatter)}
    for name, phase in zip(names[1:], phases[1:], strict=True):
        runners[name] = jit(phase)
    return runners


def measure_fused_step_ns(net, static, state, step_inputs, *, n: int = 500) -> float:
    """Mean ns/step of plastax's FUSED single-kernel step (make_step), the
    real production path: forward+loss+backward+update+prune+grow in ONE
    jitted call with one device sync per step, versus the phase-separated
    timing loop's one sync per phase. Reported alongside the phase breakdown
    so the comparison shows plastax's true per-step cost, not the
    measurement-induced per-phase sync overhead. Non-destructive: runs on a
    copy-free re-feed of `state` (make_step donates, so we thread the output
    forward and never touch the caller's `state` again)."""
    import time

    step = px.make_step(net, static)
    s = state
    r = step(s, step_inputs)  # warmup / compile
    jax.block_until_ready(r.state)
    s = r.state
    t0 = time.perf_counter_ns()
    for _ in range(n):
        r = step(s, step_inputs)
        s = r.state
    jax.block_until_ready(s)
    return (time.perf_counter_ns() - t0) / n


def run_phase_timed_step(runners: dict, state, step_inputs, timer) -> tuple:
    """Run one training step through the individual phase runners, marking each
    phase on the PhaseTimer. Returns (state, loss). Phases absent from
    `runners` are simply skipped (their PhaseTimer column stays 0)."""
    timer.tick()
    loss = jnp.float32(0.0)
    state, _ = runners["forward"](state, step_inputs)
    timer.mark_forward(state)
    if "loss" in runners:
        state, loss = runners["loss"](state, step_inputs)
        timer.mark_loss(loss)
    if "backward" in runners:
        state, _ = runners["backward"](state, step_inputs)
        timer.mark_backward(state)
    if "update" in runners:
        state, _ = runners["update"](state, step_inputs)
        timer.mark_update(state)
    if "prune" in runners:
        state, _ = runners["prune"](state, step_inputs)
        timer.mark_prune(state)
    if "grow" in runners:
        state, _ = runners["grow"](state, step_inputs)
        timer.mark_grow(state)
    if "reset" in runners:
        state, _ = runners["reset"](state, step_inputs)
        timer.mark_reset(state)
    timer.step_done()
    return state, loss


def build_pipeline_state(
    net,
    *,
    num_units: int,
    input_ids,
    output_ids,
    from_ids,
    to_ids,
    weights,
    activation_init=None,
    level_of=None,
    extra_conn_cols=None,
    extra_unit_cols=None,
    globals_=None,
    capacity: int | None = None,
):
    """Custom PIPELINE-mode builder for RECURRENT (cyclic) topologies — the
    plastax analogue of the native C++ ReservoirBuilder / NCPWiringBuilder.

    NetworkBuilder.finalize computes host-side longest-path levels
    (topo.initial_levels) and RAISES on any cycle, so the standard builder
    cannot construct a reservoir (07) or NCP wiring (06). In PIPELINE mode
    levels are only tags — the flat sweep ignores them — so this mirrors
    finalize's PIPELINE branch (one flat bucket, live edges sorted by
    (dead, to_id) for the sweep's indices_are_sorted=True contract, dead
    padding to capacity) while assigning levels manually (inputs 0, all
    other units 1, unless `level_of` overrides).

    from_ids/to_ids/weights are the LIVE edge list. extra_conn_cols /
    extra_unit_cols supply initial values for the net's extra columns
    (keyed by FieldSpec.name); anything omitted gets the spec default.
    Returns (NetworkStatic, NetworkState) ready for build_phase_runners /
    px.make_step, exactly like NetworkBuilder.from_topology.
    """
    unit_fields = (ACTIVATION, LEVEL, *net.extra_unit_fields)
    conn_fields = (FROM_ID, TO_ID, DEAD, WEIGHT, *net.extra_conn_fields)

    from_ids = np.asarray(from_ids, dtype=np.int32)
    to_ids = np.asarray(to_ids, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    n_edges = int(from_ids.shape[0])
    # (dead, to_id) order == live-first, sorted by to_id: exactly the layout
    # the PIPELINE forward sweep's indices_are_sorted=True segment-reduce needs.
    order = np.argsort(to_ids, kind="stable")

    cap = int(capacity) if capacity is not None else capacity_policy(n_edges)
    if cap < n_edges:
        raise ValueError(f"build_pipeline_state: capacity {cap} < n_edges {n_edges}")
    pad = cap - n_edges

    def _pad_live(live_vals, default, dtype):
        arr = np.concatenate(
            [np.asarray(live_vals, dtype=dtype), np.full((pad,), default, dtype=dtype)]
        )
        return jnp.asarray(arr, dtype=dtype)

    conns0: dict = {
        FROM_ID.name: _pad_live(from_ids[order], 0, np.int32),
        TO_ID.name: _pad_live(to_ids[order], 0, np.int32),
        WEIGHT.name: _pad_live(weights[order], 0.0, np.float32),
        DEAD.name: jnp.asarray(
            np.concatenate([np.zeros(n_edges, np.bool_), np.ones(pad, np.bool_)])
        ),
    }
    for spec in net.extra_conn_fields:
        if extra_conn_cols and spec.name in extra_conn_cols:
            live = np.asarray(extra_conn_cols[spec.name], dtype=spec.dtype)[order]
        else:
            live = np.full((n_edges,), spec.default, dtype=spec.dtype)
        conns0[spec.name] = _pad_live(live, spec.default, spec.dtype)

    if level_of is not None:
        levels = np.asarray(level_of, dtype=np.int32)
    else:
        levels = np.ones(num_units, dtype=np.int32)
        levels[np.asarray(input_ids, dtype=np.int32)] = 0

    units: dict = {
        ACTIVATION.name: (
            jnp.zeros((num_units,), jnp.float32)
            if activation_init is None
            else jnp.asarray(activation_init, dtype=jnp.float32)
        ),
        LEVEL.name: jnp.asarray(levels, dtype=jnp.int32),
    }
    for spec in net.extra_unit_fields:
        if extra_unit_cols and spec.name in extra_unit_cols:
            units[spec.name] = jnp.asarray(extra_unit_cols[spec.name], dtype=spec.dtype)
        else:
            units[spec.name] = jnp.full(
                (num_units,), np.asarray(spec.default), dtype=spec.dtype
            )

    static = NetworkStatic(
        num_units=int(num_units),
        propagation=Propagation.PIPELINE,
        unit_fields=unit_fields,
        conn_fields=conn_fields,
        level_capacities=(cap,),
        kahn_max_depth=None,
        input_ids=tuple(int(i) for i in input_ids),
        output_ids=tuple(int(i) for i in output_ids),
    )
    state = NetworkState(
        units=units,
        conns=(conns0,),
        globals_=globals_,
        needs_resort=jnp.bool_(False),
    )
    return static, state
