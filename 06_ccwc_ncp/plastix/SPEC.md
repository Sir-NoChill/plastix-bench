# 07 ccwc-ncp — Compact C. elegans-Wired Controller on Plastix

A sparse, fixed-topology, continuous-time recurrent network (a Liquid
Time-Constant net over an `AutoNCP`-style four-tier wiring), trained
online with **e-prop** instead of BPTT. The PyTorch counterpart for
output-shape parity is [`ccwc/`][py-ccwc]; this benchmark is
the "Plastix-native" rendering of the same idea — one Plastix `DoStep`
per LTC integration step, all state lives in O(1) per-unit / per-conn
fields, no temporal tape.

[py-ccwc]: ../../ccwc/

## Why this fits Plastix (and why the BPTT variant would not)

The Plastix design constraints from `AGENTS.md` line up with three
properties of this workload:

1. **Static, sparse spatial connectivity.** AutoNCP-style wiring has
   tens of neurons and a few hundred edges — the per-conn-op cost is
   the right denominator.
2. **No per-timestep state explosion.** e-prop replaces BPTT with a
   running eligibility trace on each connection plus a per-unit
   learning signal on each unit. Both are scalar (O(1) per element);
   one Plastix `DoStep` is one network timestep.
3. **Map / Combine / Apply decomposition.** The LTC semi-implicit
   Euler step is two reductions (numerator and denominator) over the
   incoming connections, then a per-unit fused update — exactly the
   PassPolicy shape. The backward sweep is the standard "weight times
   downstream signal" projection, same as `mlp-xor`.

What we deliberately do **not** attempt: BPTT through the ODE solver.
A T-step tape would require a per-unit `std::array<float, TMAX>` field
plus host orchestration of T backward `DoStep`s; that violates per-step
locality and dilutes the kernels the paper is selling. e-prop with a
symmetric (transposed-weight) feedback path gives a one-step backward
sweep that fits the framework natively.

## Topology

```
inputs (2, level 0)                           ← Plastix input units; written by host each step
  │
  │  dense fan-out into the sensory layer
  ▼
sensory  (Ns LTC neurons, level 1)
  │
  │  sparse fan-out (K_sparse=4 per dst)
  ▼
inter    (Ni LTC neurons, level 1)
  │
  │  sparse fan-out (K_sparse=4)
  ▼
command  (Nc LTC neurons, level 1)
  │       ↺  recurrent edges (each command unit picks K_rec=4 other command
  │            units uniformly; written so the destination's pre-state was
  │            set on the previous DoStep — Pipeline-mode propagation handles
  │            this naturally)
  ▼
motor    (Nm LTC neurons, level 1, Nm = OutputDim)
  ↑
  │  motor → command feedback edges (each motor unit picks K_fb=2 command
  │  units), giving the network the closed-loop shape the original NCP has
```

All non-input units sit at the same level. Under `Propagation::Pipeline`
the level number is informational only — the forward sweep walks every
live connection once per `DoStep`, reading source activations as they
stand at the *start* of the step (i.e. the previous timestep's value).
That is precisely the discrete RNN unroll we want; recurrent and
feed-forward edges are treated identically.

`MaxLevels = 1024` is irrelevant. Allocator capacities scale with the
unit / edge counts at construction time:

- `UnitCapacity = 256`  (two-digit neuron counts, plus headroom)
- `ConnCapacity = 4096` (a few hundred edges for `--units 32`, room for sweeps)

## Per-unit fields

Beyond the framework core (`ActivationTag`, `ForwardAccTag`,
`BackwardAccTag`, `LevelTag`):

| Tag | Type | Purpose |
|---|---|---|
| `TauTag`           | float | Membrane time constant τ_i (fixed in first pass; could be made learnable) |
| `KindTag`          | uint8 | One of {Sensory=0, Inter=1, Command=2, Motor=3}; lets Apply branch on role |
| `LearningSignalTag`| float | E-prop's per-unit signal `L_i[t]`, written by BackwardPass::Apply |
| `LastDenomTag`     | float | Saved denom `(1 + Δt(1/τ + Σf_ij))` of the LTC step; needed by eligibility |
| `LastFsumTag`      | float | Saved `Σ f_ij` (used only for diagnostics; could be folded into LastDenom) |

Forward's `Accumulator` is a 2-field struct `{ Num, Denom }`, with `Combine`
adding component-wise. `ForwardAccTag` on the unit carries this struct.

## Per-connection fields

| Tag | Type | Purpose |
|---|---|---|
| `WeightTag`        | float | Synaptic strength w_ij. Learned. |
| `GammaTag`         | float | Sigmoid gain γ_ij of the synapse activation f_ij. Fixed at init. |
| `MuTag`            | float | Sigmoid bias μ_ij of the synapse activation f_ij. Fixed at init. |
| `EligibilityTag`   | float | e_ij[t] = β·e_ij[t-1] + (Δt/D_i)·f_ij·(1 − sign(w)·x_i_new). Phase-1 update. |

The synapse activation is `f_ij = σ(γ_ij · (x_src − μ_ij))`, bounded in
(0, 1). The signed scalar `w_ij` carries excitation/inhibition; the
denominator uses `|w_ij|·f_ij` so the LTC state stays bounded.

## Global state

```cpp
struct CcwcGlobals {
  float Dt           = 0.1f;       // integration step
  float Lr           = 1e-2f;      // weight learning rate
  float BetaTrace    = 0.9f;       // eligibility decay
  bool  IsFinalStep  = false;      // gates Loss (we treat every step as final
                                   //   for sine seq-to-seq regression; psmnist
                                   //   would set this only at t == T-1)
  float Loss         = 0.0f;       // reporting only
  uint32_t Step      = 0;          // monotonic step counter (host-managed)
};
```

## Mapping to the policy slots

```
DoForwardPass → DoCalculateLoss → DoBackwardPass → DoUpdateConn → DoResetGlobal
```

All other phases (`UpdateUnit`, `PruneUnit`, `PruneConn`, `AddUnit`,
`AddConn`) stay at their `NoX` sentinel defaults and `if constexpr`
out of the step entirely.

### Phase 1 — `ForwardPass` (tanh leaky RNN, LTC-lite)

The benchmark was originally drafted with the full LTC semi-implicit
Euler step (`x_new = (x + Δt·num) / (1 + Δt·(1/τ + denom))` with
sigmoid synapses and signed `w_ij` numerator vs. `|w_ij|` denominator).
That formulation is *implementable* in Plastix — it exercises a 2-field
`Accumulator` struct — but the `|w_ij|·f_ij` denominator dominates when
weights are small and collapses every hidden unit to ≈0, leaving the
network stuck at the constant-zero predictor (test MSE ≈ 0.5 on
unit-amplitude sine targets, no learning).

The shipped version uses the discrete LTC-lite update Hasani 2021
reduces to in their appendix — a leaky tanh RNN where α plays τ's role:

```cpp
struct LtcAcc { float Drive = 0.0f; };  // single-field struct kept so
                                        //   the framework's struct-Accumulator
                                        //   path still gets exercised
struct LtcForward {
  using Accumulator = LtcAcc;

  // Synaptic drive: w_ij · tanh(x_src). Signed, bounded, well-conditioned.
  static LtcAcc Map(auto &U, size_t, size_t Src, auto &C, size_t Cid, auto &) {
    return { GetWeight(C, Cid) * std::tanh(GetActivation(U, Src)) };
  }
  static LtcAcc Combine(LtcAcc A, LtcAcc B) { return { A.Drive + B.Drive }; }

  // x_new = tanh(α·x_prev + Σ w_ij·tanh(x_src)),  α = exp(-Δt/τ).
  static void Apply(auto &U, size_t Id, auto &, LtcAcc Acc) {
    float Tau   = GetField<TauTag>(U, Id);
    float Alpha = std::exp(-Hyper::Dt / Tau);
    float Xprev = GetActivation(U, Id);
    float Pre   = Alpha * Xprev + Acc.Drive;
    GetActivation(U, Id) = std::tanh(Pre);
    GetField<LastDenomTag>(U, Id) = Alpha;       // saved for e-prop
  }
};
```

Pipeline mode reads source activations as they stood at the *start* of
this `DoStep`; that's the explicit-RNN semantics we want.

Note: `GammaTag` and `MuTag` (the per-conn sigmoid gain / bias from the
true LTC formulation) are still on every connection — set to `(γ=1, μ=0)`
at construction time, never read by the forward pass. Kept so a future
revision can re-enable the sigmoid synapse without an SOA layout change.

### Phase 2 — `Loss` (MSE on motor activations)

Sine task: at every step, MSE against `(target_sin, target_cos)`.
psMNIST (future): gated on `G.IsFinalStep`.

The built-in `MSELoss` stages `dL/dActivation_i = pred_i − target_i`
into `BackwardAcc` on the motor units. The motor `UnitRange` is the
final span returned by the layer builder, so `Network::OutputRange`
points to it automatically — no custom loss policy needed.

### Phase 3 — `BackwardPass` (spatial learning-signal sweep)

```cpp
struct LtcBackward {
  using Accumulator = float;

  // Project the downstream learning signal back through the connection
  // weight AND the destination's tanh sensitivity φ'(z_Dst) = (1 − x_Dst²).
  // Both factors are required to keep the e-prop gradient unbiased w.r.t.
  // the one-step truncation — see derivation in this file's discussion.
  static float Map(auto &U, size_t /*Src*/, size_t Dst, auto &C,
                   size_t Cid, auto &) {
    float W    = GetWeight(C, Cid);
    float Xdst = GetActivation(U, Dst);
    float PhiP = 1.0f - Xdst * Xdst;
    return W * PhiP * GetField<LearningSignalTag>(U, Dst);
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Up) {
    uint8_t K = GetField<KindTag>(U, Id);
    // Motor units: fresh dL/dActivation staged by the loss this step.
    // Other units: the projected sum (one step lagged — Map read the
    // previous step's L; that is the standard e-prop "truncation
    // depth = 1" approximation).
    GetField<LearningSignalTag>(U, Id) =
        (K == /*Motor*/3) ? GetBackwardAcc(U, Id) : Up;
  }
};
```

One Pipeline backward sweep per `DoStep` — the one-step approximation
that distinguishes e-prop from BPTT.

### Phase 4 — `UpdateConn` (e-prop, phase 1 only)

For the tanh-leaky-RNN forward in use, the per-step sensitivity of
`x_new_i` to `w_ij` is `φ'(z_i) · tanh(x_src)`. The eligibility trace
accumulates that with β decay, and the weight step is `Lr · L · e`
clipped to a fixed per-step delta cap and the weight cap `WMax`. Both
clips are there to keep the noisy one-step e-prop gradient from
overshooting under the recurrent command-layer dynamics; they kick in
rarely once the network is in a sensible weight regime.

```cpp
struct EpropUpdate {
  static void UpdateIncomingConnection(auto &U, size_t Dst, size_t Src,
                                       auto &C, size_t Cid, auto &) {
    float Hpre     = std::tanh(GetActivation(U, Src));     // pre activation
    float Xnew     = GetActivation(U, Dst);
    float PhiPrime = 1.0f - Xnew * Xnew;                   // tanh derivative
    float Sens     = PhiPrime * Hpre;

    float &E = GetField<EligibilityTag>(C, Cid);
    E = Hyper::BetaTrace * E + Sens;

    float L     = GetField<LearningSignalTag>(U, Dst);
    float Delta = Hyper::Lr * L * E;
    Delta = std::clamp(Delta, -Hyper::ClipDelta, Hyper::ClipDelta);
    float Wnew = std::clamp(GetWeight(C, Cid) - Delta,
                            -Hyper::WMax, Hyper::WMax);
    GetWeight(C, Cid) = Wnew;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                       auto &, size_t, auto &) {}
};
```

The outgoing sweep stays empty; the framework still pays a no-op pass
over the connection arena (a known cost — see the 02 IMP audit).

### Phases not used

`UpdateUnit`, `PruneUnit/Conn`, `AddUnit/Conn`, `ResetGlobal`: all
`NoX`. Per-example state reset (zeroing `ActivationTag`, `EligibilityTag`,
`LearningSignalTag`, `LastDenomTag`) is done host-side between
sequences in O(units + conns).

## Construction

A custom LayerBuilder builds the entire NCP graph in one call (it can't
be decomposed into stacked `FullyConnected`s because of the recurrent
edges):

```cpp
struct NCPWiringBuilder {
  size_t Units;     // total NCP unit count (Ns + Ni + Nc + Nm)
  size_t Motors;    // = Nm = output dim
  size_t KSparse;   // fan-in for inter and command layers
  size_t KRec;      // fan-in for command-command recurrence
  size_t KFb;       // fan-in for motor->command feedback
  uint64_t Seed;
  UnitRange operator()(auto &UA, auto &CA, UnitRange Inputs) const;
};

Network<CcwcTraits> Net(InputDim,
                        NCPWiringBuilder{Units, OutputDim, 4, 4, 2, seed});
```

The builder:

1. Computes the AutoNCP partition (sensory ≈ N/3, motor = OutputDim,
   the rest split between inter and command).
2. Allocates all NCP units at level 1, tags them with `KindTag`,
   initialises τ uniformly in [0.5, 1.5], and stamps a default γ_ij,
   μ_ij on each connection it creates.
3. Builds the inter-layer edges (dense input→sensory, then sparse
   sensory→inter, inter→command, command→motor) and the recurrent /
   feedback loops (command→command, motor→command).
4. Returns the motor `UnitRange` so `Network::OutputRange` points at
   the right units for `MSELoss` and `GetOutput()`.

## Host time-loop

```cpp
for each training sequence (Xex of shape (T, F), Yex of shape (T, M)):
    ResetPerExample(Net);            // O(units + conns) wipe of recurrent state

    float SeqLoss = 0;
    for (size_t t = 0; t < T; ++t) {
        G.IsFinalStep = (t + 1 == T);   // gates Loss for classification tasks
        Net.DoStep(Xex[t], Yex[t]);     // sine: target each step; psmnist: only at T-1
        SeqLoss += G.Loss;
    }
```

`ResetPerExample` zeroes activations on non-input units, eligibility on
every connection, learning signals on every unit, and the saved
denominator / fsum fields.

## Output schema (parity with the rest of the suite)

```
traditional-plastix/results/ccwc_ncp[_tag].history.jsonl
traditional-plastix/results/ccwc_ncp[_tag].summary.csv
traditional-plastix/csv/ccwc_ncp[_tag].test.csv
```

Summary CSV columns:

```
workload, dataset, task, n_in, n_out, n_units, n_edges,
epochs, batch, lr, dt, beta_trace,
wall_seconds, val_loss_final, test_loss, test_metric, seed
```

(`test_metric` is final-step MSE for sine; would be accuracy if the
psMNIST extension lands.)

## Acceptance criteria

1. Builds into `traditional-plastix/CMakeLists.txt` by adding
   `07-ccwc-ncp` to `TRADITIONAL_BENCHES`. ✓
2. `--quick` sine smoke runs end-to-end in under 30 s on one CPU core
   (current `--quick` defaults complete in ≈15 ms). ✓
3. Train MSE on the sine task decreases meaningfully from the
   untrained baseline. Current numbers at `--train-seqs 256 --seq-len 32
   --epochs 12 --lr 5e-3`: ep 0 untrained val_mse ≈ 0.79, ep 1
   val_mse ≈ 0.22, settles in ≈ 0.17–0.20 thereafter — a ~4× drop from
   the constant-zero baseline (which would sit at ≈ 0.5). e-prop's
   one-step gradient is noisy, so the curve has the SwiftTD-shaped
   variance rather than a strictly monotonic descent. ✓
4. Network sparsity is non-trivial: 164 live edges across 32 NCP
   neurons at the default `--units 32 --k-sparse 4 --k-rec 4 --k-fb 2`,
   well under units² = 1024. ✓

## What this benchmark exercises that the existing suite does not

- A `ForwardPass::Accumulator` that is a **multi-field struct** with a
  user-defined `Combine`. The other Plastix benchmarks all use scalar
  float accumulators.
- **Pipeline propagation with recurrent edges.** `pipeline-fcc`
  demonstrates Pipeline mode on a strict DAG; this benchmark is the
  first that exercises Pipeline's actual point — handling cycles
  without a topology sort.
- A `BackwardPass` whose **output is the per-unit learning signal**,
  not a dL/dz. The semantic is e-prop's; the mechanics are the same
  Plastix backward sweep, which is the diagnostic worth.
- An **eligibility-driven `UpdateConn`** — the SwiftTD trace pattern,
  applied to a continuous-time RNN rather than TD(λ).
