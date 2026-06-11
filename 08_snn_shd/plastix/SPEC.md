# 06 sparse-snn-shd — design spec

A sparse, online-learning spiking neural network on the Spiking
Heidelberg Digits dataset, written entirely against Plastix's five
declarative per-step phases. The Python counterpart for diff is
[`snn-shd/`][py-snn]; the goal of this benchmark is to show
the same task running on Plastix in a shape the framework's design
actually rewards.

[py-snn]: ../../snn-shd/

## What is the framework actually good at?

Three properties of the workload have to be true for Plastix to be
competitive, since per-connection scalar iteration is its bottleneck
(measured ~5–20 M conn-ops/sec/core on `01_static_etth1`):

1. **Static, sparse spatial connectivity.** The connection count is what
   the per-step cost is proportional to. Dense fully-connected layers
   (input→hidden ≈ 179 k synapses for SHD) blow up that cost.
2. **No per-timestep state explosion.** BPTT requires storing T copies
   of every per-unit field; Plastix's per-unit fields are scalar. So
   learning has to be online (local in time): one Plastix `DoStep` per
   network timestep, state lives in O(1) traces per unit / synapse.
3. **Compute that fits Map / Combine / Apply.** A LIF integration step
   and a surrogate-gradient backward pass both decompose this way.

This spec hits all three by choosing **e-prop** as the learning rule and
**local-fan-in sparse connectivity** for the input → hidden layer.

## Network topology

```
input (700, level 0)
  │
  │  K-sparse fan-in:
  │   each hidden unit j picks K=32 input channels uniformly at
  │   construction time and never changes them. Edge count
  │   becomes K · H = 32 · 256 = 8 192 instead of 179 200.
  ▼
hidden (256 LIF, level 1)
  │
  │  dense fan-in (256 · 20 = 5 120 edges)
  ▼
readout (20 non-spiking integrators, level 2)
```

**Total live connections: ~13 312** (vs ~184 320 dense). At Plastix's
per-conn ceiling, the dense FFN epoch is ~4 hours and the sparse one is
~17 minutes on a single core, ignoring any further sparsity gains from
zero spikes. The framework's `Propagation::Topological` mode applies; no
structural mutation phases are needed.

`MaxLevels = 1024` is irrelevant here (depth = 2). Allocator capacities:

- `UnitCapacity = 4096`  (700 + 256 + 20 = 976 units)
- `ConnCapacity = 16384` (rounded up from 13 312)

## Host-side time loop

Plastix's `DoStep` is per-spatial-step, not per-temporal-step. The host
runs the time loop and arms the loss only on the final tick:

```cpp
for each training example (Xex of shape (T, 700), yex):
    ResetPerExampleState(Net);     // membrane, logits, eligibility, ...

    for (size_t t = 0; t < T; ++t) {
        Globals.IsFinalStep = (t == T - 1);
        std::span<const float> target_span =
            Globals.IsFinalStep ? OneHot(yex) : std::span<const float>{};
        Net.DoStep(Xex[t], target_span);   // empty targets short-circuit Loss
    }
```

`ResetPerExampleState` zeroes `MembraneTag`, `LogitAccTag`,
`LearningSignalTag`, and `EligibilityTag` across every live unit /
connection. Cost is O(units + conns), negligible vs the forward sweep.

## Per-unit fields

Beyond the framework's core (`ActivationTag`, `ForwardAccTag`,
`BackwardAccTag`, `LevelTag`, `PrunedTag`):

| Tag | Type | Purpose |
|---|---|---|
| `MembraneTag` | float | LIF membrane potential `V[t]` |
| `PreActTag` | float | `V[t] - threshold` saved at Forward Apply for the surrogate gradient |
| `LogitAccTag` | float | Running sum of readout membrane over T (the logit) |
| `LearningSignalTag` | float | E-prop's `L_j[t]`, written by Backward Apply |
| `PrevSpikeTag` | float | `spk[t-1]` for subtract-reset (`mem ← βmem + I − thr·prev`) |
| `BetaTag` | float | Per-unit membrane decay (initialised to 0.9; learnable later) |
| `IsOutputTag` | bool | Readout units (no spiking, no reset) |
| `IsHiddenTag` | bool | Hidden LIF units (the only ones whose Apply spikes) |

Input units (level 0) carry only `ActivationTag` and are written by the
host before each `DoForwardPass`; their Apply runs in the unit-loop
inside Forward but is short-circuited by their `Level == 0` (the
dispatcher already skips `i < NumInput`).

## Per-connection fields

| Tag | Type | Purpose |
|---|---|---|
| `WeightTag` | float | Synaptic weight (Xavier-initialised) |
| `EligibilityTag` | float | E-prop's `e_ij[t] = β_e · e_ij[t-1] + spk_pre · ψ_post` |

## Global state

```cpp
struct EpropGlobals {
    bool   IsFinalStep   = false;       // gates Loss
    float  Lr            = 1e-3f;       // learning rate
    float  BetaTrace     = 0.9f;        // eligibility-trace decay
    float  SurrogateSlope = 25.0f;      // fast-sigmoid slope
    float  Threshold     = 1.0f;        // LIF threshold (constant)
    float  Loss          = 0.0f;        // reporting only; reset each step
};
```

`IsFinalStep` and friends are statics-on-the-policy in the existing
framework idiom (see `03_bursty_elec2.cpp::BurstAddUnit::Armed`), or a
real `GlobalState` struct — either works.

## Mapping to the five phases

For the SNN, the per-`DoStep` ordering inside Plastix is exactly:

```
DoForwardPass → DoCalculateLoss → DoBackwardPass → DoUpdateUnit → DoUpdateConn
```

The remaining phases (`PruneUnit/Conn`, `AddUnit/Conn`, `ResetGlobal`)
are left at their `NoX` defaults and compile out via `if constexpr`.

### Phase 1 — `Forward`

LIF integration. Recovers the snnTorch dynamics exactly:

```cpp
struct LifForward {
  using Accumulator = float;

  PLASTIX_HD static float Map(auto &U, size_t /*self*/, size_t SrcId,
                              auto &C, size_t ConnId, auto &) {
    // spk_pre[t] is the source unit's current activation field.
    return GetWeight(C, ConnId) * GetActivation(U, SrcId);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float I) {
    bool   IsOut = GetField<IsOutputTag>(U, Id);
    float  Beta  = GetField<BetaTag>(U, Id);
    float  Mem   = GetField<MembraneTag>(U, Id);
    float  Prev  = GetField<PrevSpikeTag>(U, Id);

    if (IsOut) {
      // No-reset integrator.
      Mem = Beta * Mem + I;
      GetField<MembraneTag>(U, Id) = Mem;
      GetField<LogitAccTag>(U, Id) += Mem;
      GetField<PreActTag>(U, Id)   = Mem - G.Threshold;
      GetActivation(U, Id)         = 0.0f;          // doesn't propagate further
    } else {
      // Subtract-reset LIF.
      Mem = Beta * Mem + I - G.Threshold * Prev;
      float Z = Mem - G.Threshold;
      float Spk = (Z >= 0.0f) ? 1.0f : 0.0f;
      GetField<MembraneTag>(U, Id) = Mem;
      GetField<PreActTag>(U, Id)   = Z;
      GetField<PrevSpikeTag>(U, Id) = Spk;
      GetActivation(U, Id)         = Spk;
    }
  }
};
```

Note: input-unit activations are written by the host before each step
and the Forward Apply skips them (`Level == 0`, dispatcher already
guards). The same `LifForward` policy serves all non-input units; the
`IsOutputTag` branch picks integrator vs spiking behavior.

### Phase 2 — `Loss`

Custom loss policy. Fires every step but no-ops unless
`Globals.IsFinalStep` is true and the target span is non-empty.

```cpp
struct EpropLoss {
  template <typename UA>
  static void CalculateLoss(UA &U, UnitRange Out,
                            std::span<const float> Target,
                            auto &G) {
    if (!G.IsFinalStep || Target.empty()) return;

    // 1. Read logits = LogitAcc on output units, compute softmax.
    float MaxL = -1e30f;
    for (size_t I = Out.Begin; I < Out.End; ++I)
      MaxL = std::max(MaxL, GetField<LogitAccTag>(U, I));
    float Z = 0.0f;
    for (size_t I = Out.Begin; I < Out.End; ++I)
      Z += std::exp(GetField<LogitAccTag>(U, I) - MaxL);

    // 2. dL/dlogit = softmax - one_hot.  Staged in BackwardAcc.
    G.Loss = 0.0f;
    for (size_t I = Out.Begin, K = 0; I < Out.End; ++I, ++K) {
      float P = std::exp(GetField<LogitAccTag>(U, I) - MaxL) / Z;
      float Y = Target[K];
      GetBackwardAcc(U, I) = P - Y;
      if (Y > 0) G.Loss += -std::log(P + 1e-30f);
    }
  }
};
```

The framework already short-circuits `DoCalculateLoss` when `Target` is
empty, so the host gating (passing `{}` for non-final timesteps) is
enough by itself; the `IsFinalStep` check is belt-and-braces.

### Phase 3 — `Backward`

Spatial reverse sweep. For the readout layer, the learning signal is the
staged `dL/dlogit`; for the hidden layer, it is the upstream gradient
projected through the (currently shared) weights and multiplied by the
surrogate `ψ(z) = 1 / (1 + |slope · z|)²`.

```cpp
struct LifBackward {
  using Accumulator = float;

  PLASTIX_HD static float Map(auto &U, size_t /*src*/, size_t ToId,
                              auto &C, size_t ConnId, auto &) {
    return GetWeight(C, ConnId) * GetField<LearningSignalTag>(U, ToId);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float Up) {
    bool IsOut = GetField<IsOutputTag>(U, Id);
    if (IsOut) {
      // Identity surrogate: integrator, no spike non-linearity.
      // (BackwardAcc was staged by Loss earlier this step.)
      GetField<LearningSignalTag>(U, Id) = GetBackwardAcc(U, Id);
    } else {
      float Z = GetField<PreActTag>(U, Id);
      float Den = 1.0f + G.SurrogateSlope * std::fabs(Z);
      float Psi = 1.0f / (Den * Den);
      GetField<LearningSignalTag>(U, Id) = Up * Psi;
    }
  }
};
```

This is the same shape as the `mlp-xor` example's `SigmoidBackwardPass`;
e-prop differs only in *what* the per-unit signal means (a learning
signal rather than a pre-activation gradient), and in skipping BPTT.

### Phase 4 — `UpdateUnit`

A no-op for the minimum-viable e-prop variant. **Used** if (and only if)
we extend to filtered learning signals (low-pass on `L_j`) or adaptive
threshold neurons, both of which need a per-unit slow-trace decay each
step. Left as `NoUpdateUnit` for the first pass; promoting it later is
zero-cost because it compiles out.

### Phase 5 — `UpdateConn`

The heart of e-prop. Eligibility trace lives in the incoming sweep
(needs `pre_spike` and `surrogate(post)`); the outgoing sweep is unused
and pays the framework's known no-op cost (~30 % of `UpdateConn` per the
earlier audit on `01_static_etth1`).

```cpp
struct EpropUpdateConn {
  PLASTIX_HD static void UpdateIncomingConnection(
      auto &U, size_t DstId, size_t SrcId, auto &C, size_t ConnId, auto &G)
  {
    float PreSpk = GetActivation(U, SrcId);                    // 0 or 1
    float Z      = GetField<PreActTag>(U, DstId);
    float Den    = 1.0f + G.SurrogateSlope * std::fabs(Z);
    float Psi    = 1.0f / (Den * Den);                         // ψ_post
    float &E     = GetField<EligibilityTag>(C, ConnId);
    E = G.BetaTrace * E + PreSpk * Psi;

    float L = GetField<LearningSignalTag>(U, DstId);
    if (L != 0.0f) {
      GetWeight(C, ConnId) -= G.Lr * L * E;
    }
  }

  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};
```

On non-final timesteps `L == 0` for every unit (Loss didn't fire), so
the multiplication is skipped — eligibility accumulates, weights stay
put. On the final timestep `L != 0` and the accumulated trace is
consumed.

### Phases not used

`PruneUnit`, `PruneConn`, `AddUnit`, `AddConn`, `ResetGlobal`: all left
at their `NoX` defaults. Connectivity is static after construction.
`if constexpr` removes the corresponding loops from `DoStep`.

## Construction

Two `LayerBuilder`s:

```cpp
struct LocalSparseLayer {
  size_t Width;
  size_t FanIn;
  uint64_t Seed;
  // Built-in K-sparse fan-in: for each new hidden unit, draw FanIn
  // distinct input ids uniformly without replacement, allocate one
  // connection each, set Weight via Xavier scaling sqrt(2/FanIn).
  UnitRange operator()(auto &UA, auto &CA, UnitRange Prev) const;
};

using FCOut = plastix::FullyConnected<UniformInit, MarkOutput>;

Network<SparseSnnTraits> Net(/*input dim*/ 700,
                              LocalSparseLayer{256, 32, seed},
                              FCOut{20, /*init=*/..., MarkOutput{}});
```

`MarkOutput` flips `IsOutputTag = true` on each readout unit so
`LifForward::Apply` takes the integrator branch. Hidden units default to
`IsOutputTag = false` and pick up the spiking branch.

## Data integration

SHD HDF5 isn't in the Plastix C++ side's repertoire today. The C++ side
already reads CSV via `bench::ReadCsv`; we'll do the analogue for raw
spike events:

**Path of least resistance:** extend `snn-shd/data.py`'s
`_materialise` to also dump the binned float32 tensors to a flat
little-endian `.bin` file:

```
header (24 bytes):  uint32 magic = 0x53484430 ('SHD0')
                    uint32 n_samples
                    uint32 n_bins
                    uint32 n_channels
                    uint32 dtype  (0 = float32)
                    uint32 reserved
body:               n_samples × n_bins × n_channels float32  (binarized {0,1})
                    n_samples int64                          (labels)
```

The C++ side loads this once with a 30-line `LoadBinned()` helper in the
benchmark `.cpp`. Cache path: `data/SHD_cache/{train,test}_n{n_bins}.plxbin`.
Python side writes it lazily on its next run.

## Time loop (full C++ skeleton)

```cpp
auto T = H.NBins, B = H.Batch /* but Plastix is per-example, ignore */;
auto N_OUT = H.NumClasses;
EpropGlobals G;
G.Lr = H.Lr; G.BetaTrace = H.BetaTrace; G.SurrogateSlope = H.SurrSlope;

for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    shuffle(perm);
    for (size_t ex : perm) {
        ResetPerExample(Net);                      // O(units+conns) wipe
        auto OneHot = OneHotVector(Ytr[ex], N_OUT);
        for (size_t t = 0; t < T; ++t) {
            G.IsFinalStep = (t + 1 == T);
            std::span<const float> Tgt = G.IsFinalStep
                ? std::span<const float>{OneHot}
                : std::span<const float>{};
            Net.DoStep(Xtr[ex][t], Tgt);
        }
    }
    // Eval over val + test: same loop but no UpdateConn — easiest way is
    // to set G.Lr = 0 temporarily during the eval window (the multiply
    // by 0 makes the update inert; the eligibility still accumulates
    // harmlessly because it's reset at the start of each example).
}
```

## Reset cost

`ResetPerExample` runs in O(units + conns) per example, T × per epoch.
For SHD's ~976 units + 13 k conns × 7 340 examples × T=50 timesteps, the
reset is ~5 G ops/epoch. Per-step DoStep is 4 × |conns| ≈ 50 k ops; T
steps × 7 340 examples = 18 G ops/epoch. The reset is ~25 % of the
training cost. Worth measuring before optimising; if it bites, the
allocator's `Gather` machinery could in principle bulk-zero a field
slice.

## Outputs

Same as the other Plastix benchmarks:

```
traditional-plastix/results/sparse_snn_shd[_tag].history.jsonl
traditional-plastix/results/sparse_snn_shd[_tag].summary.csv
traditional-plastix/csv/sparse_snn_shd[_tag].test.csv
traditional-plastix/plots/sparse_snn_shd[_tag].test.png
```

Summary CSV columns: `workload, dataset, n_in, n_hid, n_out, fan_in,
n_bins, n_conns, epochs, batch, lr, beta_trace, surrogate_slope,
wall_seconds, val_acc_best, test_acc, test_acc_shuffled, ablation_drop,
firing_rate_final, seed`.

The time-shuffle ablation that anchors `snn-shd/` is
re-runnable here too: shuffle `Xtr[ex]` along axis 0 at evaluation time.
Expected drop is the headline temporal-coding diagnostic.

## Acceptance criteria

1. Builds cleanly into the existing `traditional-plastix/CMakeLists.txt`
   target list (just add `06-sparse-snn-shd` to `TRADITIONAL_BENCHES`).
2. `--quick` smoke runs in under 60 s on one CPU core (e.g. `--n-bins 30
   --epochs 2 --n-per-class 32 --fan-in 16 --n-hid 64`).
3. A full training run (`--epochs 30 --n-bins 50`) clears **≥ 55 % test
   accuracy** on SHD. Lower than the Python dense FFN (≥ 65 %) because
   the K-sparse fan-in throws away input information; tunable up by
   increasing `--fan-in`.
4. The time-shuffle ablation drop is **≥ 0.15** absolute (same
   directionality as `snn-shd/`'s 0.35 — temporal coding
   genuinely matters).
5. Per-step wall-clock at full config is **≥ 10× faster than a
   dense-fan-in variant** (`--fan-in 700`) of the same network, confirming
   the sparsity exploitation is real.

## Why this is the *right* benchmark to add

It exercises five framework features that the existing five benchmarks
do not exercise together:

- A non-trivial `ForwardPass` with **state across `DoStep` invocations**
  (membrane carries between timesteps via `MembraneTag`).
- A `Loss` that is **conditionally a no-op** (one of the regimes where
  the optional `if (Target.empty()) return;` short-circuit really matters).
- A `BackwardPass` whose Apply uses **per-unit saved state** (`PreActTag`)
  written by an earlier phase of the same step — same trick as `mlp-xor`'s
  `GradPreActTag`, but at a longer trip length (Forward → ... → Backward).
- An `UpdateConn` whose **incoming sweep maintains a slow trace** (the
  IDBD-in-SwiftTD pattern, applied to e-prop's eligibility).
- A **K-sparse static topology** built with a custom `LayerBuilder` —
  the existing benchmarks use `FullyConnected` for everything.

If this benchmark works end-to-end, every per-step primitive the
framework exposes has at least one regression test for *spiking,
time-extended* workloads — the regime that the framework was supposed to
own but currently has no example for.

## Open questions before implementation

These don't block the spec but they're decisions I'd want the user to
confirm before coding:

- **Threshold:** fixed at 1.0 for parity with snnTorch defaults, or
  per-unit `ThresholdTag` (would enable adaptive-threshold neurons later)?
- **Eligibility reset:** wipe per-example, or let it run continuously?
  The Python reference resets per-example (membrane is reset too); the
  online-learning literature is split. Pick per-example for parity.
- **Reporting cadence:** evaluate val/test once per epoch (matches the
  other benchmarks) or every K examples? Once-per-epoch is the cheap
  default; eval over 2 264 examples × T=50 forward passes ≈ ~6× a
  training pass over one batch.
- **Random-feedback alignment for the hidden learning signal:** vanilla
  e-prop uses the symmetric (transposed) readout weights for the
  feedback path. Asymmetric random feedback (Lillicrap 2016) decouples
  forward and feedback weights and avoids the weight-transport problem;
  it costs nothing in Plastix terms but adds a `FeedbackTag` per
  readout-to-hidden connection. Default: symmetric (simpler, fewer
  fields).

If these are fine as drafted, the implementation is one C++ file
(~400–500 lines) plus a 30-line extension to `snn-shd/data.py` to dump
the `.plxbin` cache. Most of the C++ is dataset loading and the
per-example reset; the policy structs themselves are <60 lines combined.
