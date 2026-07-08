# Expressing the benchmark networks in JAX

All ten benchmarks (**01–10**) have JAX ports at `<bench>/jax/run_benchmark.py`,
built on `common/jax/common.py` (a JAX-async-safe PhaseTimer + MemoryProbe) and
emitting the standard phase + memory schema, so `jax` is a full column in
`phase_table.csv` / `memory_table.csv`. Ports run on CPU (the installed jaxlib is
CPU-only; a `jax[cuda12]` swap would enable GPU but can clash with torch's CUDA).

## Headline finding

JAX splits cleanly into two regimes, drawn along the **static-vs-dynamic
topology** line — the same line the SNN study found, but JAX is on the *harsh*
side of it because it is **compiled (XLA)**, not eager:

- **Static-topology benches map beautifully.** Recurrence is `jax.lax.scan`,
  reservoirs are a scan + a closed-form `jnp.linalg.solve`, and even a
  surrogate-gradient SNN is a `jax.custom_jvp` spike inside a scan. These jit
  once and run fast.
- **Dynamic-structure benches hit the XLA recompile wall.** Every change to a
  neuron/edge count changes an array shape, which forces a fresh trace+compile of
  the jitted step. Compilation dwarfs execution, so throughput collapses.

## Per-benchmark verdict

| # | Bench | Shape | JAX mechanism | Verdict |
|---|-------|-------|---------------|---------|
| 01 | Dense | static MLP | jitted forward/grad/SGD | ✅ clean |
| 02 | Sparse | static + weight-mask prune | mask multiply; host-numpy threshold | ✅ clean (pruning = mask, no shape change) |
| 03 | Bursty | grow+prune | host surgery + rebuild → recompiles on each burst | ⚠️ works, recompiles per burst |
| 04 | Cont-S | per-step churn | host surgery every step | ❌ ~146 s quick — recompiles **every step** |
| 05 | Cont-L | heavy churn | host surgery every step | ⚠️ works but recompile-taxed |
| 06 | NCP | recurrent LTC | nested `lax.scan` over time + ODE unfolds; static NCP mask | ✅ clean (bwd≈75%: grad re-runs the scan) |
| 07 | ESN | reservoir + ridge | `lax.scan` states + closed-form `jnp.linalg.solve` | ✅ clean; "training" is one SPD solve, no autodiff |
| 08 | SNN | surrogate-grad spiking | LIF in `lax.scan`; spike via `jax.custom_jvp` | ✅ clean; JAX has no LIF layer so dynamics are hand-written (also: training can diverge to nan without grad clipping — a stability caveat, not an expressibility one) |
| 09 | Imprint | continuous growth | host surgery + rebuild | ❌ **the pathology** — see below |
| 10 | XL-NN | sparse pipeline + prune | `.at[dst].add(...)` scatter; host edge surgery | ⚠️ works; recompiles on the (periodic) structural events |

## The dynamic-shape pathology (bench 09, isolated)

Bench 09 grows features at runtime. Measured on the naive host-surgery port:
- balanced grow/remove (features oscillate) → **208 recompiles**, still cheap;
- **growth-only** (D increases every step) → **300 recompiles in 300 steps →
  ~42 s** vs ~0.26 s when D stays bounded. That is ~140 µs of real work per step
  ballooning to ~140 ms — a **~160× slowdown, essentially 100 % XLA
  compilation.**

The only way to make JAX competitive is to **abandon dynamic shapes**:
pre-allocate a fixed max-capacity array and mask inactive slots, trading
recompiles for wasted flops on padding and pushing all structural logic into
masks instead of array surgery.

## Cross-cutting notes

- **`grad` recomputes the forward.** JAX stores no activations, so `backward`
  (the `jax.grad` call) re-runs the forward pass — visible as backward ≫ forward
  in every gradient-trained jax row (Dense 41/23, NCP 75/17, SNN 45/28).
- **Closed-form learners** (07 ESN) show `backward`≈100 %: the whole "training"
  is a single `jnp.linalg.solve`, attributed to the backward bucket.
- **Warmup.** Every port runs one step off the clock to keep XLA's first-call
  compilation out of the timed loop (except where recompiles are the finding).

## GPU execution

With `jax[cuda12]` installed, jax runs on the RTX 5000 Ada. The orchestrator pins
the backend from `--device` via `JAX_PLATFORMS` (so a `cpu` pass really is CPU and
a `--device cuda` pass is GPU); the torch-based `snn`/`norse` impls use their
usual `--device cuda`. Representative `--quick` walls:

| bench | jax CPU | jax GPU | speedup |
|-------|--------:|--------:|--------:|
| 01 Dense (tiny MLP) | 4.4 s | 4.3 s | ~1× (overhead-bound) |
| 06 NCP (recurrent LTC) | 21.4 s | 1.2 s | **~18×** |
| 08 SNN (spiking BPTT) | 5.3 s | 0.7 s | **~7.5×** |

GPU pays off exactly where there's dense parallel compute to amortise kernel-launch
overhead (the recurrent LTC's ODE unfolds, the SNN's T-step BPTT); tiny models
(01) are launch-bound and see no gain. It does **not** rescue the dynamic-structure
benches — those are bottlenecked on host-side XLA *recompilation*, not device
throughput, so a GPU just compiles the churning graph on the GPU instead.

The torch-based SNN impls (`snn`/`norse`) also run on GPU but show no speedup at
this scale (small nets; the per-timestep kernel launches over the T-window
dominate) — GPU would help at larger hidden sizes / batch.

Run GPU passes with `--device cuda`, e.g.
`uv run python orchestrator.py run --impl jax,snn,norse --device cuda`.
The phase/memory table `jax` column is measured on **CPU** (consistent with the
other `runs_cpu` impls); GPU is reported here as a separate comparison. Note the
MemoryProbe measures **host RSS** only — for GPU jax the on-device VRAM is not
in the `max` (see `device_bytes()` in `common/jax/common.py`).

## Takeaway for the paper

Both external studies (SNN and JAX) converge on the same message from opposite
sides: the cost of runtime structural change is set by the **execution model**.
Eager frameworks (PyTorch, snnTorch, Norse) express it naturally but forgo
compilation/acceleration; compiled frameworks (JAX/XLA) accelerate the static
kernel but pay a catastrophic recompile tax on every structural edit. Plastix's
contribution is doing **both at once** — structural adaptation at step frequency
*with* a compiled/GPU execution path — which is exactly the quadrant neither
baseline family occupies.
