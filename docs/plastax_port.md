# Porting plastix-bench to `plastax` (the JAX Plastix library)

Status report — the plastax (JAX port of the Plastix per-edge plastic-network
framework) benchmark port. Mirrors `docs/jax_expressibility.md` /
`docs/snn_expressibility.md`: what is expressible, what the numbers say, and
why some benches are out of scope.

## TL;DR

- **A reusable plastax bench harness exists** (`common/plastax/common.py`) and
  **`01_static_etth1/plastax/`** is a complete, GPU-validated port. Drop-in for
  the orchestrator (`<bench>/plastax/run_benchmark.py`, same summary schema).
- **Feasibility is the headline finding.** plastax v1 implements
  `AddConn`/`PruneConn` but **not `AddUnit`/`PruneUnit`** (runtime *unit*
  growth/deletion — an explicit v1 scope exclusion). Of the 13 benches (ex.
  `08_snn_shd`), **~7 are honestly portable**; **5 require unit growth** and
  are deferred; **1 has no native plastix impl** to compare against.
- **On static-dense (bench 01), plastax is ~2× slower than hand-written CUDA
  plastix but ~2× faster than CPU plastix**, with **no hand-written kernels** —
  it gets the GPU from XLA. Static-dense is the per-edge *worst case* (both
  plastix and plastax lose to a dense GEMM: `cpp` 8.4 s vs plastix 74–93 s).

## Feasibility triage (verified against each bench's plastix source)

`grep`-verified which native impls call `DoAddUnits`/`DoPruneUnits` (needs a
feature plastax v1 lacks) vs only `DoAddConnections`/`DoPruneConnections`.

| Bench | Acronym | Runtime structural ops | plastax v1 |
|---|---|---|---|
| 01_static_etth1 | Dense | none (static) | ✅ **ported** |
| 02_idempotent_imp | Sparse | `DoPruneConnections` (mask prune) | ✅ feasible |
| 06_ccwc_ncp | LTC-sine | none (static sparse, recurrent) | ✅ feasible |
| 07_esn_mackey_class | ESN | none (fixed reservoir) | ✅ feasible |
| 10_engineered_sparse_large_nn | XL-NN | `DoPruneConnections` (prune-only) | ✅ feasible |
| 12_churn | — | `GrowFanout`+`DoPruneConnections`, **no units** | ✅ feasible ★ |
| 14_churn_scaling | — | conn churn, **no units** | ✅ feasible ★ |
| 03_bursty_elec2 | Bursty | **`DoAddUnits`+`DoPruneUnits`** | ⛔ needs AddUnit |
| 04_continuous_small_appliances | Cont-S | **`DoAddUnits`+`DoPruneUnits`** | ⛔ needs AddUnit |
| 05_continuous_large_mackey_glass | Cont-L | **`DoAddUnits`+`DoPruneUnits`** | ⛔ needs AddUnit |
| 09_imprintin_learner | Imprint | **`DoAddUnits`+`DoPruneUnits`** | ⛔ needs AddUnit |
| 11_scaling_imprint | Scale | imprinting unit growth | ⛔ needs AddUnit |
| 13_depth_scaling | — | (no plastix impl — pytorch only) | — no oracle |

★ **12/14 are the dynamic-connection-growth showcases**: they grow *edges*
(`GrowFanout` = plastax's `AddConn`) without growing units, so plastax's
retrace contract (pre-allocated arenas + stable shapes) applies — the case
where naive JAX would hit XLA's recompile-on-shape-change wall
(`docs/jax_expressibility.md`: bench 09 ~160×). Only plastix+pytorch impls
exist for 12/14 (no jax/cpp), so the comparison there is plastax vs plastix.

## Bench 01 result (RTX 3060 Ti, 280 576 edges, synthetic ETTh1)

Same net as native (`hidden=256 depth=3`, 1352 units / 280 576 edges),
per-example SGD. plastax numbers are per-step so they compare regardless of
total step count; native numbers from the committed `_results/` archive.

| Impl | per-step | full-run wall (242 200 steps) | notes |
|---|---|---|---|
| plastax **fused** (`make_step`, 1 kernel/step) | **783 µs** | ~190 s (extrap.) | real production path, 49 MB VRAM |
| plastax phase-separated (per-`Do*` sync) | 1881 µs | ~455 s (extrap.) | matches plastix's separate-pass timing |
| native plastix **cuda** (hand kernels) | 383 µs | 93 s | GPU, hand-written CUDA |
| native plastix gpu | ~306 µs | 74 s | GPU |
| native plastix **cpu** | ~1647 µs | 399 s | CPU, per-edge |
| native `cpp` (OpenBLAS GEMM) | — | 8.4 s | dense matmul, minibatched |

**Reading it honestly:** plastax (auto-generated XLA) is ~2× slower than
hand-tuned CUDA plastix and ~2× *faster* than CPU plastix — a reasonable price
for zero hand-written kernels. Static-dense is the per-edge model's worst case;
neither plastix nor plastax should be headlined here (GEMM wins). plastax's
advantage lands on the **dynamic** benches (12/14), not this one.

The two plastax rows also quantify a measurement artifact: the phase-separated
loop pays one GPU sync *per phase* (fair to plastix's separate `DoForwardPass`/
`DoBackwardPass`/`DoUpdateConn`, but 4 syncs/step); the fused `make_step` — how
you actually use plastax — pays one, and is 2.4× faster.

## The harness (`common/plastax/common.py`)

- Reuses the framework-agnostic `PhaseTimer`/`MemoryProbe`/CLI/CSV helpers from
  `common/jax` (plastax *is* jax underneath — the async-safe timer is correct).
- `build_phase_runners(net, static)` → one jitted callable per present phase,
  keyed by PhaseTimer name, with the input scatter folded into `forward`. This
  drives plastax's phases individually so the per-phase breakdown matches the
  oracle's separate `Do*` calls.
- `measure_fused_step_ns(...)` → plastax's real one-kernel step, for a fair
  per-step-cost comparison.

## How to add a plastax bench (pattern from bench 01)

1. `FieldSpec` extra unit fields for anything the traits stash across phases
   (e.g. `PreAct` for `φ'(z)`, `GradPreAct` for dL/dz, `IsOutput` to select
   linear vs nonlinear units). `LossGrad` bridges loss→backward for output
   units (plastax has no framework `BackwardAcc` column — see
   `plastax/examples/mlp_xor.py`).
2. Trait classes: `ForwardPass`/`BackwardPass`/`Loss`/`UpdateConn`
   (+`PruneConn`/`AddConn` for dynamic). Backward `map`'s `src` arg is bound to
   the edge's *downstream* unit (framework reverses direction) — read the
   downstream dL/dz there.
3. `from_topology(sequential(input_units, dense…))` with a matching initializer
   (`variance_scaling(1/3,"fan_in","uniform")` == `nn.Linear` default).
4. Per-example loop through `run_phase_timed_step`; emit the summary via
   `PhaseTimer.summary_fields` + `MemoryProbe.summary_fields`.

## Running

```bash
# needs plastax + jax[cuda12] in the env (see below)
JAX_PLATFORMS=cuda python 01_static_etth1/plastax/run_benchmark.py \
    --synthetic --no-plot --hidden 256 --depth 3 --epochs 20
```

**Environment:** the orchestrator invokes `uv run python <impl>/run_benchmark.py`,
so plastax + `jax[cuda12]` must be in the plastix-bench uv env. To wire it:
add `plastax` (editable, `../plastax`) and `jax[cuda12]` to the suite's deps
(or a `plastax` extra kept in the default `uv run` env). Verified working with
`jax[cuda12]==0.11.0` on the RTX 3060 Ti (driver 610.57.04, CUDA 13.3);
plastax runs bit-identically on CPU and GPU.

## Narval (remote A100) — readiness

The SSH control-master (`~/.ssh/cm/narval`) is up and `mnt/` is sshfs-mounted
to `…:/home/achilibe/scratch/mnt`. `remote-work/` has build + sbatch scaffolds,
but those target the C++ plastix; a plastax run needs a Python/jax-cuda/plastax
venv module-loaded on Narval. Per the "test thoroughly locally first"
sequencing, remote runs should follow broader local coverage. When ready:
module-load Python + build a venv with `jax[cuda12]`+`plastax`, stage the bench
to `mnt/`, submit via sbatch (**≤5-min GPU wall**, `--mail-type=END,FAIL`, and
**poll result files over sshfs — do not loop on `squeue`**).

## Next steps (in priority order)

1. Port the remaining feasible static/prune benches: **02** (rich cpp/jax/cuda
   comparison), **10** (prune, large), **07** (ESN), **06** (LTC). Reuse the
   bench-01 pattern.
2. Port **12_churn** / **14_churn_scaling** — the `AddConn`/`GrowFanout`
   dynamic showcases (use `plastax.Driver`, which owns the overflow/resort
   retrace loop). This is where plastax's retrace contract should shine vs a
   naive-JAX reimplementation.
3. Decide on the AddUnit/PruneUnit-dependent benches (03/04/05/09/11): either
   a documented "infeasible in v1" note, or a plastax v2 `AddUnit` feature.
4. Wire plastax into the suite's uv env; run the full local GPU pass; then
   Narval scaling.
