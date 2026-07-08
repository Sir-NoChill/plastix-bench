"""JAX-side benchmark helpers — mirror of common/pytorch/common.py.

Provides a JAX-aware `PhaseTimer` and a `MemoryProbe` so a JAX benchmark emits
the *identical* summary schema (phase + memory columns) as the pytorch/plastix/
cpp/cuda impls, and re-exports the shared CLI / summary / output helpers from the
pytorch common module (they are framework-agnostic — argparse + csv + paths).

The one JAX-specific wrinkle is timing: JAX dispatches asynchronously, so a naive
`time - time` around a call measures dispatch, not compute. `PhaseTimer.mark_*`
therefore accepts the value(s) produced by the phase and calls
`jax.block_until_ready(...)` on them before stamping the clock. Pass whatever the
phase returns (a DeviceArray, a pytree, or a tuple) — anything falsy/None is
treated as "already synced".

Benches 01-10 have JAX ports (`<bench>/jax/run_benchmark.py`) built on these
helpers; `jax` is a column in the phase/memory tables (bench_meta.FRAMEWORKS).
See docs/jax_expressibility.md for the per-bench findings.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

# Reuse the framework-agnostic pytorch common helpers (argparse/csv/paths/proc —
# no torch at module top-level). We load pytorch/common.py by explicit path under
# a unique module name rather than `from common import ...`: this module is ALSO
# named `common`, so a bare import would resolve to itself (circular). The module
# must be registered in sys.modules before exec so its @dataclass can resolve
# its own __module__.
_PYC_PATH = Path(__file__).resolve().parents[1] / "pytorch" / "common.py"
_PYC_NAME = "_plastix_pytorch_common"
_spec = importlib.util.spec_from_file_location(_PYC_NAME, _PYC_PATH)
_pyc = importlib.util.module_from_spec(_spec)
sys.modules[_PYC_NAME] = _pyc
_spec.loader.exec_module(_pyc)

StructuralLog = _pyc.StructuralLog              # noqa: F401  (re-exported)
add_common_args = _pyc.add_common_args          # noqa: F401
output_paths = _pyc.output_paths                # noqa: F401
read_vmrss_kb = _pyc.read_vmrss_kb              # noqa: F401
write_summary_csv = _pyc.write_summary_csv      # noqa: F401
download_if_missing = _pyc.download_if_missing  # noqa: F401
plot_run = _pyc.plot_run                        # noqa: F401
plot_test_curve = _pyc.plot_test_curve          # noqa: F401
test_plot_path = _pyc.test_plot_path            # noqa: F401

try:
    import jax
except ImportError:  # scaffolding: jax is an optional extra (uv sync --extra jax)
    jax = None


def _sync(value) -> None:
    """Block until an async JAX result is materialised, so timing is real."""
    if jax is not None and value is not None:
        jax.block_until_ready(value)


class _Welford:
    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def std(self) -> float:
        return (self.m2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0


class PhaseTimer:
    """Per-step phase-ns accumulator, JAX-async-safe. Same column output as the
    pytorch PhaseTimer. Call `tick()` at the top of the step, then
    `mark_<phase>(result)` after each phase, passing that phase's output so the
    async dispatch is flushed before the clock is read."""

    PHASES = ("forward", "loss", "backward", "update", "prune", "grow", "reset")

    def __init__(self) -> None:
        self._acc = {p: _Welford() for p in self.PHASES}
        self._steps = 0
        self._last = None

    def tick(self) -> None:
        self._last = time.perf_counter_ns()

    def _delta(self) -> int:
        now = time.perf_counter_ns()
        dt = now - (self._last if self._last is not None else now)
        self._last = now
        return dt

    def _mark(self, phase: str, result=None) -> None:
        _sync(result)
        self._acc[phase].add(self._delta())

    def mark_forward(self, r=None):  self._mark("forward", r)
    def mark_loss(self, r=None):     self._mark("loss", r)
    def mark_backward(self, r=None): self._mark("backward", r)
    def mark_update(self, r=None):   self._mark("update", r)
    def mark_prune(self, r=None):    self._mark("prune", r)
    def mark_grow(self, r=None):     self._mark("grow", r)
    def mark_reset(self, r=None):    self._mark("reset", r)

    def step_done(self) -> None:
        self._steps += 1

    @property
    def step_count(self) -> int:
        return self._steps

    def summary_fields(self, wall_seconds: float) -> dict:
        steps = max(self._steps, 1)
        step_ns_mean = (wall_seconds * 1e9) / steps
        accounted = sum(a.mean for a in self._acc.values())
        out = {"step_count": int(steps), "step_ns_mean": round(step_ns_mean, 3)}
        for p in self.PHASES:
            out[f"{p}_ns_mean"] = round(self._acc[p].mean, 3)
            out[f"{p}_ns_std"] = round(self._acc[p].std(), 3)
        out["other_ns_mean"] = round(max(0.0, step_ns_mean - accounted), 3)
        return out


class MemoryProbe:
    """RSS milestone breakdown — identical column output to the C++/pytorch
    probes. Host RSS milestones; when running on a JAX GPU device the `max`
    reported by the orchestrator poller is host RSS only, so for GPU runs also
    consult `device_bytes()` below for VRAM."""

    def __init__(self) -> None:
        self._overhead = 0
        self._dataset = 0
        self._weights = 0
        self._cursor = 0

    def start(self) -> None:
        self._overhead = self._cursor = read_vmrss_kb()

    def end_dataset(self) -> None:
        r = read_vmrss_kb(); self._dataset += r - self._cursor; self._cursor = r

    def end_weights(self) -> None:
        r = read_vmrss_kb(); self._weights += r - self._cursor; self._cursor = r

    def summary_fields(self) -> dict:
        return {
            "mem_overhead_kb": int(self._overhead),
            "mem_dataset_kb":  max(0, int(self._dataset)),
            "mem_weights_kb":  max(0, int(self._weights)),
        }


def device_bytes() -> int:
    """Best-effort current JAX-device memory in bytes (0 if unavailable / CPU).
    Handy to augment the host-RSS `max` for GPU runs in a real bench."""
    if jax is None:
        return 0
    try:
        stats = jax.devices()[0].memory_stats() or {}
        return int(stats.get("bytes_in_use", 0))
    except Exception:
        return 0
