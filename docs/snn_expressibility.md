# Expressing the benchmark networks in an SNN framework

We ported benchmarks **01–05** to spiking neural networks in **snnTorch** (all
five) and **Norse** (01–02), as first-class per-bench impls
(`<bench>/{snn,norse}/run_benchmark.py`) that emit the standard phase + memory
schema and show up as `snn` / `norse` columns in the summary tables.

## Headline finding

The prior hypothesis was that the **dynamic-structure** benches (03/04/05, which
grow and prune neurons at runtime) would **not** be expressible in an SNN
framework. That turned out to be **wrong** — all five are expressible and
faithful. The reason is specific and worth stating:

> An SNN library like snnTorch / Norse is **eager PyTorch**, and its LIF neurons
> carry **no persistent per-neuron state across steps** — the membrane is
> re-initialised at the top of every forward (`init_leaky()` / `state=None`). So
> a runtime grow/prune touches *only* the `nn.Linear` weight tensors, exactly
> like the ANN PyTorch impls — there is no neuron state to resize, and (unlike
> JAX) no graph to recompile. Spiking neurogenesis is arguably *cleaner* here
> than in a jitted framework.

So the expressibility boundary for these networks is **not** ANN-vs-SNN; it's
**eager-vs-compiled**. Dynamic structure is natural in eager frameworks (PyTorch,
snnTorch, Norse) and painful in compiled ones (JAX/XLA recompiles on every shape
change — see the 04 jax port taking ~146 s vs ~16 s for 04 snn).

## Per-benchmark verdict

| # | Bench | Task | snnTorch | Norse | Notes |
|---|-------|------|----------|-------|-------|
| 01 | Dense | regression (ETTh1) | ✅ faithful | ✅ faithful | continuous output read from a **non-spiking leaky integrator** membrane (snnTorch `reset_mechanism="none"` / Norse `LICell`), averaged over the T-step window |
| 02 | Sparse | IMP classifier | ✅ faithful | ✅ faithful | pruning = 0/1 weight mask + gradient masking (identical to ANN); class logits read from **accumulated membrane**, not spike counts (hard spike counts give no cold-start gradient) |
| 03 | Bursty | grow+prune classifier | ✅ faithful | ⧗ same pattern | burst-add / magnitude-prune done as host-side `Linear` surgery; stateless LIF ⇒ no state to reconcile |
| 04 | Cont-S | per-step churn regression | ✅ faithful | ⧗ same pattern | structure changes **every step**; eager rebuild is a cheap realloc (no per-step XLA recompile, unlike jax) |
| 05 | Cont-L | heavy churn regression | ✅ faithful | ⧗ same pattern | full grow/shrink/rewire/burst cadence preserved via host surgery |

✅ built + verified · ⧗ not built for Norse, but ports identically via the same
host-surgery pattern demonstrated in snnTorch (available on request).

## The real costs (what *is* different about SNNs here)

1. **Time-unrolled BPTT dominates.** Every `forward` runs the network for `T`
   spiking timesteps and `backward` is surrogate-gradient BPTT through all `T`.
   This makes forward+backward the overwhelming share of every SNN step
   (visible in `phase_table.csv`: Dense snn ≈ 45% fwd + 40% bwd) and inflates
   wall time several-fold vs the ANN impls — orthogonal to the structural
   mechanism.
2. **Regression needs a membrane readout.** Spike *counts* can't emit a smooth
   continuous target, so the output layer is a non-spiking leaky integrator and
   the prediction is its (mean) membrane voltage. Standard SNN idiom; works
   fine for 01/04/05.
3. **Classification cold-start.** Hard spike-count logits are all-zero from
   random init ⇒ flat cross-entropy gradient ⇒ stuck at chance. Reading the
   accumulated output membrane as logits fixes it (used in 02).
4. **Norse integrator form.** Norse's default `LIFCell` is a two-stage
   (synapse+membrane) dt-scaled Euler integrator that won't spike from vanilla
   init. Reproducing snnTorch's simple `v[t] = β·v + I` requires the single-state
   `LIFBoxCell`/`LIBoxCell` variants plus an input rescale. snnTorch trades this
   for a one-line `beta`; Norse trades it for explicit control.

## How they're wired in

- Impls: `01..05/snn/run_benchmark.py` (snnTorch), `01..02/norse/run_benchmark.py`
  (Norse). Torch-based, so they reuse `common/pytorch` (PhaseTimer, MemoryProbe).
- Registered in `orchestrator.py` (`IMPL_ORDER`/colour/label, `--device`/`--no-plot`
  gating) and in the default `impls` list in the justfile.
- `snn` / `norse` columns added to `bench_meta.FRAMEWORKS`, so `just tables`
  emits them in `phase_table.csv` / `memory_table.csv` (blank where no impl
  exists). Each has a `--timesteps` knob for the spike window `T`.

## Takeaway for the paper

The SNN ports reinforce the framework thesis from the *opposite* direction: the
difficulty of runtime structural change is a property of the **execution model**
(compiled vs eager), not of the neuron model. Plastix delivers eager-style
structural freedom *with* a compiled/GPU execution path — the combination that
neither the jitted (JAX) nor the eager-but-unaccelerated (PyTorch/snnTorch/Norse)
baselines provide at once.
