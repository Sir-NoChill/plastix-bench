# Forward-View / Return-Prediction Analysis

Analysis of why the imprinting learner's GVF prediction on the Audio Prediction
Benchmark currently looks like a **spike-and-decay** at each chord, instead of
the **held / ramping return** we want, and what we expect to fix it.

Driver: `examples/il_audio_pred.cpp` (harness) + `examples/plot_il.py` (overlay).
Learner: `il::ImprintingLearner` (`include/imprinting/imprinting_learner.hpp`,
`src/imprinting_learner.cpp`) over the SOA `FeatureArena`, updated by the
vendored SwiftTD (`third_party/swifttd`).

## What we want vs. what we get

The benchmark delivers a scalar reward (−1 / 0 / +1) **3–5 s after** each chord
(the "No"/"Yes" sounds). The discounted return at time *t* is
`G_t = Σ_k γ^k R_{t+k+1}`. As *t* approaches a reward it should ramp toward that
reward's (discounted) value and jump at delivery — the blue "Return" curve in the
thesis figure. A learner with a transient representation can't ramp smoothly, so
the *best learnable prediction* (thesis, pink) is a **step that holds the
discounted value across the cue→reward gap**, then jumps.

Our prediction instead **spikes at the chord onset and decays back to ~0** before
the reward arrives. The sign is correct (up for +1 chords, down for −1, ~flat for
the neutral chord), so the cue identity is learned — but the value does not hold
or ramp across the gap.

## Timing (the numbers that matter)

- sample rate 16384 Hz, step = 640 samples → **25.6 steps/s**
- chord sound ≈ 0.84 s ≈ **21 steps** (the only steps observation features fire)
- reward delay 3–5 s ≈ **77–128 steps** after the chord
- silent gap (sound end → reward) ≈ **55–107 steps** with no chord features active

## Root cause: the eligibility-trace horizon is ~10× too short

`V_t = Σ_i w_i · φ_i(t)` is a correct discounted-return estimator; the TD fixed
point *is* the return. The forward view (λ-return) is realized online by the
eligibility trace, which **decays by `γ·λ` per step**
(`SwiftTDBinaryFeatures::Step`: `z[i] = gamma*lambda*z[i]`).

With the current `γ = 0.99`, `λ = 0.9`:

| quantity | value |
|---|---|
| `γλ` | **0.891** |
| effective horizon `1/(1−γλ)` | **≈ 9 steps** |
| trace surviving to the chord, `(γλ)^100` | **≈ 1×10⁻⁵** |
| reward delay | **77–128 steps** |

When the reward fires its large TD error δ, the eligibility trace has **already
decayed to ~0** for every feature active more than ~30–40 steps earlier. So the
chord features (and the first ~⅔ of the gap) receive **essentially no credit for
the reward and never learn to predict it.** The value can only be elevated within
the short trace window near where features fire — exactly the spike-then-collapse.
The forward view is "looking back" ~10 steps when the task needs ~100.

For comparison, the credit reaching across a 100-step gap:

| λ (with γ=0.99) | `γλ` | horizon `1/(1−γλ)` | `(γλ)^100` |
|---|---|---|---|
| 0.9 (current) | 0.891 | ~9 | ~1e-5 |
| 0.99 | 0.980 | ~50 | ~0.13 |
| 0.999 | 0.989 | ~91 | ~0.33 |
| 1.0 | 0.990 | ~100 | ~0.37 |

SwiftTD is explicitly designed to bound the rate of learning so that TD(λ) stays
stable **even at λ=1**, so we can push λ toward 1 without divergence.

## Two necessary conditions (and where we fail each)

A held/ramping return needs **both**:

1. **A representation active across the gap.** `V` is nonzero only where features
   fire. Observations fire ~0.8 s; memory units blip for `window` steps; nothing
   reliably spans the whole gap. *Partial fail* — addressed by tiling memory
   delays across the gap (`memory_delay_*` ranges).
2. **Those features must learn the discounted value.** This needs credit to reach
   them from the reward, i.e. `γλ ≈ 1`. *Hard fail* at λ=0.9.

This is why the earlier **memory-delay sweep (64/130/200) failed to produce a
ramp**: we generated memory units tiling the gap, but at λ=0.9 they got ≈0 credit
when the reward arrived, so they never learned a weight, and unlearned active
features contribute 0 to `V`. **λ was the missing ingredient, not the delay range
alone** — the two must work together.

## Aggravating factor: removal

Removal (`epsilon_z`) culls a near-reward memory unit after it fires once (it goes
`Idle`, `z` decays below `e^β·εᶻ`) before it can accumulate weight across the
~158 events. So the bridging representation never stabilizes long enough to train.
(Note: delayed memory units are already protected *while armed* — see below — but
not after they finish firing.)

## Fixes already applied (prerequisites, not sufficient)

These were needed just to make memory features *exist and function*; they do not
by themselves produce the ramp:

- **Armed-memory protection** (`FeatureArena::isMemoryArmed`): removal no longer
  culls a delayed memory unit during its delay/window countdown, so it survives
  to fire. (Regression test: `ImprintingLearner.MemoryNotRemovedWhileArmed`.)
- **Interleaved generation**: pattern/memory are generated with a random type per
  slot, so patterns (forced-active, budget-consuming) no longer starve memory
  generation under a tight τ budget.

## Recommended next experiment + expectations

**Sweep λ ∈ {0.9, 0.99, 0.999, 1.0}** at memory delay ≈ [1, 130] (tile the full
gap), window ≈ [1, 5], keeping the rest of the thesis HP
(`alpha=3e-3`, `tenure_track=3e-4`, `tenure=0.01`, `epsilon_z=0.01`, `eta=0.1`,
`gamma=0.99`, `k_pattern=k_memory=10`, pattern fractions {0.6,0.7,0.8,0.9}),
and re-plot the last 180 s.

Expectations:

- **λ=0.9 (baseline):** unchanged spike-and-decay (trace can't reach the gap).
- **λ=0.99 (~50-step horizon):** partial — the value should start holding and
  ramping over the *second half* of the gap (the part within the trace horizon);
  still collapses near the chord.
- **λ=0.999 / 1.0 (~90–100-step horizon):** the gap memory units finally receive
  reward credit, so the value should **hold and ramp across the whole gap** —
  qualitatively the thesis's held/ramping return, peaking at reward delivery,
  rather than decaying to 0.

If λ→1 alone does **not** produce the ramp, the next suspects (in order):

1. **Representation density/persistence** — too few memory units tiling the gap,
   or windows too short to keep something active at every gap step. Raise
   `k_memory`, widen windows, or relax removal (`epsilon_z`) so bridging units
   persist long enough to train.
2. **Removal timing** — protect a generated unit for a minimum age (grace period)
   before it's removal-eligible, so near-reward units survive across events.
3. **Step-size budget** — with 50-hot inputs and `alpha=3e-3`, base
   `τ ≈ 0.15 > eta=0.1`, so generation is throttled until β decays; the gap may
   simply be under-populated early in the run.

## Reproduce

```sh
# build (per-generation sampling currently enabled in build/default)
cmake --build build/default

# run + plot (λ via a future IL_LAMBDA override or edited MakeHyperParams)
IL_MEMORY_DELAY_MIN=1 IL_MEMORY_DELAY_MAX=130 IL_MEMORY_WINDOW_MAX=5 \
  ./build/default/il_audio_pred examples/output/dataset.bin 0 /tmp/il_pred.csv
examples/venv/bin/python examples/plot_il.py --predictions /tmp/il_pred.csv \
  --data-dir examples/output --time-range 3420 3600 \
  --output examples/output/plots/il_lambda_test.png
```

(λ is not yet env-overridable; add an `IL_LAMBDA` override in
`MakeHyperParams()` or edit `hp.lambda` to run the sweep.)
