// Workload 7 — Compact C. elegans-Wired Controller, Plastix translation.
//
// Sparse, fixed-topology Liquid Time-Constant network on an AutoNCP-style
// 4-tier wiring (sensory → inter → command → motor + command recurrence +
// motor → command feedback), trained online with e-prop instead of BPTT.
// PyTorch counterpart for output-shape parity is ccwc/ model A.
//
// Plastix policy mapping (see SPEC.md in this directory for the full story):
//
//   ForwardPass   custom    LTC semi-implicit Euler step; Accumulator is the
//                           2-field struct { Num, Denom } so one Map / Combine
//                           pass carries both reductions.
//   Loss          MSELoss   built-in; motor activations vs target each step
//                           (sine task is sequence-to-sequence regression).
//   BackwardPass  custom    one Pipeline backward sweep produces the per-unit
//                           e-prop learning signal L_i = (Kind==Motor) ?
//                           BackwardAcc : projected Σ w_ij · L_j.
//   UpdateConn    custom    phase 1 only — eligibility trace and weight step.
//                           Phase 2 left empty (framework pays the no-op
//                           sweep, same as snn-shd / mlp-xor).
//   All other phases NoX. Topology is fixed at construction.
//
// The sine task here is the same noisy-sine smoke task the Python ccwc port
// uses for its first-passing check — train MSE must decrease across epochs.

#include "plastix/common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Tags + accumulator
// ---------------------------------------------------------------------------

struct TauTag {};
struct KindTag {};
struct LearningSignalTag {};
struct LastDenomTag {};
struct LastFsumTag {};

struct GammaTag {};
struct MuTag {};
struct EligibilityTag {};

// AutoNCP role for each non-input unit. Kept as uint8_t so the SOA row is
// compact and KindTag fits one byte.
enum class Kind : uint8_t {
  Sensory = 0,
  Inter   = 1,
  Command = 2,
  Motor   = 3,
};

// Forward accumulator. We started with the LTC semi-implicit Euler
// formulation (numerator + denominator reductions), but the |w|·f
// denominator collapses every unit toward zero under sparse small weights
// — the network never escapes the constant-zero predictor (MSE ≈ 0.5 on
// unit-amplitude sin/cos). A scalar accumulator on top of a
// leaky-tanh RNN (the discrete LTC-lite the LTC paper itself reduces to
// in its appendix) recovers the right gradient flow without sacrificing
// the continuous-time interpretation — α plays τ's role and the synaptic
// drive Σ w_ij · tanh(x_src) plays Σ S_ij · (A_ij − x_i).
//
// Keeping the struct around (even with just one field) means the
// framework's Accumulator-as-struct path is still exercised — that was
// part of the point of this benchmark, see SPEC.md.
struct LtcAcc {
  float Drive = 0.0f;
};

// ---------------------------------------------------------------------------
// Hyperparameters — kept as static members so the CLI can rewrite them once
// at startup, then the policy structs read them in their lambdas. The same
// pattern is used by 02_idempotent_imp.cpp (`UpdateConn::Lr`).
// ---------------------------------------------------------------------------

// Network GlobalState: held in managed memory and staged from the host via
// Net::Global(), so the (possibly device-side) policies read it through their
// Globals handle rather than from host-side statics, which device code cannot
// read.
struct Hyper {
  float Dt = 0.1f;
  float Lr = 1e-3f;
  float BetaTrace = 0.9f;
  // Per-step weight-delta clip. e-prop's |L · e| can spike when the
  // recurrent command layer drives a sigmoid into saturation; clipping
  // bounds the explosion without changing the steady-state behaviour.
  float ClipDelta = 0.1f;
  // Hard cap on |w|. Same purpose, complementary axis.
  float WMax = 5.0f;
};

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

struct LtcForward {
  using Accumulator = LtcAcc;

  // Synaptic drive into the destination unit. The synapse activation is
  // tanh(γ·(x_src − μ)) — bounded and signed, so a positive w can either
  // excite or inhibit depending on the source's sign. Wired here as
  // tanh(x_src) (γ=1, μ=0) to keep the gradient computation symmetric
  // and well-conditioned.
  PLASTIX_HD static LtcAcc Map(auto &U, size_t /*Dst*/, size_t Src,
                               auto &C, size_t Cid, auto &) {
    float Xs = plastix::GetActivation(U, Src);
    float W  = plastix::GetWeight(C, Cid);
    return LtcAcc{W * std::tanh(Xs)};
  }

  PLASTIX_HD static LtcAcc Combine(LtcAcc A, LtcAcc B) {
    return LtcAcc{A.Drive + B.Drive};
  }

  // x_new = tanh(α · x_prev + Σ w_ij · tanh(x_src_prev)), α ∈ (0, 1)
  // derived from τ as α = exp(−Δt / τ). The sine-task targets sit in
  // [−1, 1] (sin/cos), so a tanh-saturated readout is in the right
  // range. A linear motor unit would let the recurrent motor→command
  // feedback drive unbounded activations within a few steps.
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, LtcAcc Acc) {
    float Tau = plastix::GetField<TauTag>(U, Id);
    if (Tau <= 0.0f) Tau = 1.0f;
    float Alpha = std::exp(-G.Dt / Tau);
    float Xprev = plastix::GetActivation(U, Id);
    float Pre   = Alpha * Xprev + Acc.Drive;
    float Xnew  = std::tanh(Pre);
    plastix::GetActivation(U, Id)         = Xnew;
    plastix::GetField<LastDenomTag>(U, Id) = Alpha;        // saved for e-prop
    plastix::GetField<LastFsumTag>(U, Id)  = Acc.Drive;    // diagnostic
  }
};

struct LtcBackward {
  using Accumulator = float;

  // Project the downstream learning signal back through the connection
  // weight, *including* the destination's tanh sensitivity φ'(z_Dst) =
  // (1 − x_Dst²). The same Jacobian factor appears in the eligibility
  // trace later — Sens carries it for the source-side multiplication;
  // here we carry it for the destination-side propagation. Symmetric
  // feedback (random-feedback alignment is a future extension).
  PLASTIX_HD static float Map(auto &U, size_t /*Src*/, size_t Dst,
                              auto &C, size_t Cid, auto &) {
    float W    = plastix::GetWeight(C, Cid);
    float Xdst = plastix::GetActivation(U, Dst);
    float PhiP = 1.0f - Xdst * Xdst;
    return W * PhiP * plastix::GetField<LearningSignalTag>(U, Dst);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Up) {
    auto K = plastix::GetField<KindTag>(U, Id);
    // Motor: fresh dL/dActivation staged by the loss this step.
    // Everyone else: the projected sum from outgoing edges, which is
    // one step lagged (Map reads last step's LearningSignalTag) — that
    // is the standard e-prop "truncation depth = 1" approximation.
    // Inputs land here too in Pipeline mode (the dispatcher walks from
    // id 0); their KindTag stays at the default 0 (Sensory), value
    // unread elsewhere — harmless.
    float Signal = (K == static_cast<uint8_t>(Kind::Motor))
                       ? plastix::GetBackwardAcc(U, Id)
                       : Up;
    plastix::GetField<LearningSignalTag>(U, Id) = Signal;
  }
};

struct EpropUpdate {
  // E-prop phase-1 update. The forward unit is
  //   x_new = φ(α · x_prev + Σ w_ij · tanh(x_src_prev)),
  // with φ = tanh for hidden, φ = id for motor. The instantaneous
  // sensitivity of x_new to w_ij is
  //   ∂x_new/∂w_ij = φ'(z_i) · tanh(x_src),
  // and the trace accumulates this with β decay across steps.
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t Dst,
                                                  size_t Src, auto &C,
                                                  size_t Cid, auto &G) {
    using namespace plastix;
    float Xs   = GetActivation(U, Src);
    float Hpre = std::tanh(Xs);                  // pre-synaptic activation
    float Xnew = GetActivation(U, Dst);
    // φ'(z) = 1 − x_new² for the tanh activation used by every non-input
    // unit (motor included — see the comment in LtcForward::Apply).
    float PhiPrime = 1.0f - Xnew * Xnew;
    float Sens = PhiPrime * Hpre;

    float &E = GetField<EligibilityTag>(C, Cid);
    E = G.BetaTrace * E + Sens;

    float L = GetField<LearningSignalTag>(U, Dst);
    float Delta = G.Lr * L * E;
    if (Delta >  G.ClipDelta) Delta =  G.ClipDelta;
    if (Delta < -G.ClipDelta) Delta = -G.ClipDelta;
    float Wnew = GetWeight(C, Cid) - Delta;
    if (Wnew >  G.WMax) Wnew =  G.WMax;
    if (Wnew < -G.WMax) Wnew = -G.WMax;
    GetWeight(C, Cid) = Wnew;
  }

  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

struct CcwcTraits : plastix::DefaultNetworkTraits<> {
  using GlobalState  = Hyper;
  using ForwardPass  = LtcForward;
  using BackwardPass = LtcBackward;
  using Loss         = plastix::MSELoss;
  using UpdateConn   = EpropUpdate;

  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;

  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<TauTag,             float>,
      plastix::alloc::SOAField<KindTag,            uint8_t>,
      plastix::alloc::SOAField<LearningSignalTag,  float>,
      plastix::alloc::SOAField<LastDenomTag,       float>,
      plastix::alloc::SOAField<LastFsumTag,        float>>;

  using ExtraConnFields = plastix::ConnFieldList<
      plastix::alloc::SOAField<plastix::WeightTag, float>,
      plastix::alloc::SOAField<GammaTag,           float>,
      plastix::alloc::SOAField<MuTag,              float>,
      plastix::alloc::SOAField<EligibilityTag,     float>>;

  // ~tens of NCP neurons, hundreds-to-low-thousands of edges. Headroom for
  // a 64-unit sweep; bump if you push --units higher.
  static constexpr size_t UnitCapacity = 256;
  static constexpr size_t ConnCapacity = 4096;
};
static_assert(plastix::NetworkTraits<CcwcTraits>);

using Net = plastix::Network<CcwcTraits>;

// ---------------------------------------------------------------------------
// Custom LayerBuilder: builds the entire NCP wiring in one call.
// The graph has recurrent and feedback edges, so it can't be decomposed
// into stacked plastix::FullyConnected layers.
// ---------------------------------------------------------------------------

struct NCPWiringBuilder {
  size_t Units;     // total NCP neurons (sensory + inter + command + motor)
  size_t Motors;    // = output dim
  size_t KSparse;   // fan-in for inter and command layers
  size_t KRec;      // fan-in for command-command recurrence
  size_t KFb;       // fan-in for motor → command feedback
  uint64_t Seed;
  float TauMin = 0.5f;
  float TauMax = 1.5f;
  float WInit  = 0.5f;

  // AutoNCP-style partition. Heuristic chosen to mirror the ncps Python
  // library: sensory ≈ N/3, motor pinned to output dim, the rest split
  // between inter and command. Falls back gracefully on small N.
  struct Partition {
    size_t Ns, Ni, Nc, Nm;
  };

  Partition ComputePartition() const {
    Partition P{};
    P.Nm = Motors;
    size_t Rest = (Units > Motors) ? (Units - Motors) : 0;
    P.Ns = std::max<size_t>(1, Rest / 3);
    size_t AfterSens = (Rest > P.Ns) ? (Rest - P.Ns) : 0;
    P.Ni = AfterSens / 2;
    P.Nc = AfterSens - P.Ni;
    if (P.Nc == 0) {
      P.Nc = 1;
      if (P.Ni > 0)
        --P.Ni;
    }
    return P;
  }

  template <typename UA, typename CA>
  plastix::UnitRange operator()(UA &Ua, CA &Ca,
                                plastix::UnitRange Inputs) const {
    using namespace plastix;
    auto P = ComputePartition();
    uint16_t SrcLevel = GetLevel(Ua, Inputs.Begin);
    uint16_t Layer1   = SrcLevel + 1;

    std::mt19937_64 Rng(Seed);
    std::uniform_real_distribution<float> Tau(TauMin, TauMax);
    std::uniform_real_distribution<float> SignedW(-WInit, WInit);

    // Allocate units in the order sensory → inter → command → motor so the
    // returned UnitRange points at the motor block (Network::OutputRange).
    auto AllocBlock = [&](size_t N, Kind K) -> UnitRange {
      UnitRange R{Ua.AllocateMany(N)};
      for (auto Id : R.Ids()) {
        GetLevel(Ua, Id)              = Layer1;
        GetField<TauTag>(Ua, Id)      = Tau(Rng);
        GetField<KindTag>(Ua, Id)     = static_cast<uint8_t>(K);
        GetActivation(Ua, Id)         = 0.0f;
      }
      return R;
    };

    auto Sensory = AllocBlock(P.Ns, Kind::Sensory);
    auto Inter   = AllocBlock(P.Ni, Kind::Inter);
    auto Command = AllocBlock(P.Nc, Kind::Command);
    auto MotorR  = AllocBlock(P.Nm, Kind::Motor);

    auto MakeEdge = [&](size_t SrcId, size_t DstId) {
      auto Cid = Ca.Allocate();
      GetField<FromIdTag>(Ca, Cid)   = plastix::GlobalUnitId{static_cast<uint32_t>(SrcId)};
      GetField<ToIdTag>(Ca, Cid)     = plastix::GlobalUnitId{static_cast<uint32_t>(DstId)};
      GetField<SrcLevelTag>(Ca, Cid) = GetLevel(Ua, SrcId);
      GetWeight(Ca, Cid)             = SignedW(Rng);
      GetField<GammaTag>(Ca, Cid)    = 1.0f;
      GetField<MuTag>(Ca, Cid)       = 0.0f;
      GetField<EligibilityTag>(Ca, Cid) = 0.0f;
    };

    // Sample K distinct ids from `Pool`, excluding `Forbidden` if present.
    auto SampleK = [&](plastix::UnitRange Pool, size_t K,
                       size_t Forbidden) -> std::vector<size_t> {
      std::vector<size_t> Ids;
      Ids.reserve(Pool.Size());
      for (auto Id : Pool.Ids())
        if (Id != Forbidden)
          Ids.push_back(Id);
      if (K >= Ids.size())
        return Ids;
      std::shuffle(Ids.begin(), Ids.end(), Rng);
      Ids.resize(K);
      return Ids;
    };

    // (1) Inputs → Sensory: dense (small input count, gives every sensory
    // neuron access to the full input vector).
    for (auto Src : Inputs.Ids())
      for (auto Dst : Sensory.Ids())
        MakeEdge(Src, Dst);

    // (2) Sensory → Inter: each inter unit picks KSparse sensory sources.
    if (P.Ni > 0 && P.Ns > 0)
      for (auto Dst : Inter.Ids())
        for (auto Src : SampleK(Sensory, KSparse, /*forbidden*/ ~size_t{0}))
          MakeEdge(Src, Dst);

    // (3) Inter → Command: each command unit picks KSparse inter sources.
    // If Inter is empty, fall through directly from sensory.
    if (P.Nc > 0) {
      if (P.Ni > 0)
        for (auto Dst : Command.Ids())
          for (auto Src : SampleK(Inter, KSparse, ~size_t{0}))
            MakeEdge(Src, Dst);
      else
        for (auto Dst : Command.Ids())
          for (auto Src : SampleK(Sensory, KSparse, ~size_t{0}))
            MakeEdge(Src, Dst);
    }

    // (4) Command → Command: recurrent fan-in. Each command unit picks
    // KRec other command units. Self-loops excluded.
    if (P.Nc > 1)
      for (auto Dst : Command.Ids())
        for (auto Src : SampleK(Command, KRec, /*forbidden self*/ Dst))
          MakeEdge(Src, Dst);

    // (5) Command → Motor: dense (every motor unit reads every command).
    if (P.Nc > 0)
      for (auto Dst : MotorR.Ids())
        for (auto Src : Command.Ids())
          MakeEdge(Src, Dst);

    // (6) Motor → Command: feedback. Each motor unit picks KFb command
    // units to project back to.
    if (P.Nc > 0 && KFb > 0)
      for (auto Src : MotorR.Ids())
        for (auto Dst : SampleK(Command, KFb, ~size_t{0}))
          MakeEdge(Src, Dst);

    return MotorR;
  }
};

// ---------------------------------------------------------------------------
// Per-sequence reset — zero recurrent state, eligibility, learning signal.
// O(units + conns). Called between training sequences.
// ---------------------------------------------------------------------------

static void ResetPerSequence(Net &N, size_t NumInput) {
  auto &Ua = N.GetUnitAlloc();
  auto &Ca = N.GetConnAlloc();
  size_t Nu = Ua.Size();
  for (size_t Id = NumInput; Id < Nu; ++Id) {
    plastix::GetActivation(Ua, Id) = 0.0f;
    plastix::GetField<LearningSignalTag>(Ua, Id) = 0.0f;
    plastix::GetField<LastDenomTag>(Ua, Id) = 1.0f;
    plastix::GetField<LastFsumTag>(Ua, Id) = 0.0f;
    plastix::GetForwardAcc(Ua, Id) = LtcAcc{};
    plastix::GetBackwardAcc(Ua, Id) = 0.0f;
  }
  size_t Nc = Ca.Size();
  for (size_t Cid = 0; Cid < Nc; ++Cid) {
    if (plastix::GetField<plastix::DeadTag>(Ca, Cid))
      continue;
    plastix::GetField<EligibilityTag>(Ca, Cid) = 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Noisy-sine dataset — same generator shape as ccwc/data.py so
// the numbers are interpretable side-by-side.
// ---------------------------------------------------------------------------

struct SineDataset {
  // Each sample: shape (T, 2) input + (T, 2) target.
  std::vector<std::vector<std::array<float, 2>>> X;
  std::vector<std::vector<std::array<float, 2>>> Y;
};

static SineDataset MakeSine(size_t N, size_t SeqLen, float NoiseStd,
                            uint64_t Seed) {
  SineDataset D;
  D.X.resize(N);
  D.Y.resize(N);
  std::mt19937_64 Rng(Seed);
  std::uniform_real_distribution<float> Freq(0.5f, 2.0f);
  std::uniform_real_distribution<float> Phase(0.0f, 6.28318530718f);
  std::normal_distribution<float> Noise(0.0f, NoiseStd);
  for (size_t I = 0; I < N; ++I) {
    float F = Freq(Rng);
    float P = Phase(Rng);
    D.X[I].resize(SeqLen);
    D.Y[I].resize(SeqLen);
    // t in [0, 2π] over SeqLen+1 samples; input is the first SeqLen noisy
    // pairs, target is the *next* clean pair.
    for (size_t T = 0; T < SeqLen; ++T) {
      float A0 = F * (6.28318530718f * static_cast<float>(T) / SeqLen) + P;
      float A1 = F * (6.28318530718f * static_cast<float>(T + 1) / SeqLen) + P;
      D.X[I][T][0] = std::sin(A0) + Noise(Rng);
      D.X[I][T][1] = std::cos(A0) + Noise(Rng);
      D.Y[I][T][0] = std::sin(A1);
      D.Y[I][T][1] = std::cos(A1);
    }
  }
  return D;
}

// ---------------------------------------------------------------------------
// Evaluation — mean per-step MSE over a held-out set. No weight updates.
// We achieve "no update" by zeroing Lr in main() during the eval window.
// ---------------------------------------------------------------------------

static double EvalMse(Net &N, const SineDataset &D, size_t NumInput) {
  double SumSq = 0.0;
  size_t Count = 0;
  std::array<float, 2> InBuf{};
  std::array<float, 2> TgtBuf{};
  for (size_t I = 0; I < D.X.size(); ++I) {
    ResetPerSequence(N, NumInput);
    const auto &Xseq = D.X[I];
    const auto &Yseq = D.Y[I];
    for (size_t T = 0; T < Xseq.size(); ++T) {
      InBuf  = Xseq[T];
      TgtBuf = Yseq[T];
      N.DoStep(std::span<const float>(InBuf.data(), 2),
               std::span<const float>(TgtBuf.data(), 2));
      auto Out = N.GetOutput();
      // MSE = mean (pred - target)^2 per output dim.
      float D0 = Out[0] - TgtBuf[0];
      float D1 = Out[1] - TgtBuf[1];
      SumSq += static_cast<double>(D0 * D0 + D1 * D1);
      Count += 2;
    }
  }
  return Count == 0 ? 0.0 : SumSq / Count;
}

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  struct HP {
    size_t Units      = 32;
    size_t OutputDim  = 2;
    size_t KSparse    = 4;
    size_t KRec       = 4;
    size_t KFb        = 2;
    size_t SeqLen     = 64;  // match cpp/pytorch/jax so the sine task is identical
    size_t TrainSeqs  = 256;
    size_t ValSeqs    = 64;
    size_t TestSeqs   = 64;
    size_t Epochs     = 20;
    float  NoiseStd   = 0.1f;
    float  Dt         = 0.1f;
    float  Lr         = 1e-3f;
    float  BetaTrace  = 0.9f;
    float  ClipDelta  = 0.1f;
    float  WMax       = 5.0f;
  } H;

  H.Units      = static_cast<size_t>(Args.GetInt("units", H.Units));
  H.SeqLen     = static_cast<size_t>(Args.GetInt("seq-len", H.SeqLen));
  H.TrainSeqs  = static_cast<size_t>(Args.GetInt("train-seqs", H.TrainSeqs));
  H.ValSeqs    = static_cast<size_t>(Args.GetInt("val-seqs", H.ValSeqs));
  H.TestSeqs   = static_cast<size_t>(Args.GetInt("test-seqs", H.TestSeqs));
  H.Epochs     = static_cast<size_t>(Args.GetInt("epochs", H.Epochs));
  H.KSparse    = static_cast<size_t>(Args.GetInt("k-sparse", H.KSparse));
  H.KRec       = static_cast<size_t>(Args.GetInt("k-rec", H.KRec));
  H.KFb        = static_cast<size_t>(Args.GetInt("k-fb", H.KFb));
  H.NoiseStd   = Args.GetFloat("noise-std", H.NoiseStd);
  H.Dt         = Args.GetFloat("dt", H.Dt);
  H.Lr         = Args.GetFloat("lr", H.Lr);
  H.BetaTrace  = Args.GetFloat("beta-trace", H.BetaTrace);
  H.ClipDelta  = Args.GetFloat("clip-delta", H.ClipDelta);
  H.WMax       = Args.GetFloat("w-max",      H.WMax);
  if (Args.Quick) {
    H.Epochs     = std::max<size_t>(1, H.Epochs / 4);
    H.TrainSeqs  = std::max<size_t>(8, H.TrainSeqs / 4);
    H.ValSeqs    = std::max<size_t>(4, H.ValSeqs / 2);
    H.TestSeqs   = std::max<size_t>(4, H.TestSeqs / 2);
    H.SeqLen     = std::max<size_t>(16, H.SeqLen / 2);
  }

  bench::MemoryProbe MP;
  MP.Start();

  // Build the network. Sine task is regression with 2 input dims, 2 output.
  uint64_t Seed = static_cast<uint64_t>(Args.Seed) * 7919ull + 13ull;
  auto N = std::make_unique<Net>(
      /*InputDim=*/2,
      NCPWiringBuilder{H.Units, H.OutputDim, H.KSparse, H.KRec, H.KFb, Seed});
  MP.EndWeights();

  // Stage runtime hyperparameters into the managed GlobalState; the policies
  // read them on host or device through their Globals handle.
  N->Global().Dt        = H.Dt;
  N->Global().Lr        = H.Lr;
  N->Global().BetaTrace = H.BetaTrace;
  N->Global().ClipDelta = H.ClipDelta;
  N->Global().WMax      = H.WMax;

  size_t NumInput = 2;
  size_t NumUnits = N->GetUnitAlloc().Size();
  size_t NumEdges = bench::LiveEdgeCount(N->GetConnAlloc());
  std::cout << "[info] units=" << NumUnits - NumInput
            << " edges=" << NumEdges
            << " (capacity unit=" << CcwcTraits::UnitCapacity
            << " conn=" << CcwcTraits::ConnCapacity << ")\n";
  std::cout << "[info] task=sine seq_len=" << H.SeqLen
            << " train=" << H.TrainSeqs << " val=" << H.ValSeqs
            << " test=" << H.TestSeqs << " epochs=" << H.Epochs
            << " dt=" << H.Dt << " lr=" << H.Lr
            << " beta=" << H.BetaTrace << "\n";

  // Datasets — synthetic and seeded so the run is reproducible without
  // network access (matches the other 0X benchmarks under --synthetic).
  auto Train = MakeSine(H.TrainSeqs, H.SeqLen, H.NoiseStd,
                        Seed ^ 0xa1a1ull);
  auto Val   = MakeSine(H.ValSeqs,   H.SeqLen, H.NoiseStd,
                        Seed ^ 0xb2b2ull);
  auto Test  = MakeSine(H.TestSeqs,  H.SeqLen, H.NoiseStd,
                        Seed ^ 0xc3c3ull);
  MP.EndDataset();

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "ccwc_ncp");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_mse");

  // Initial point (untrained). Eval must run with Lr=0 because DoStep
  // still calls UpdateConn — leaving Lr at its training value here would
  // silently train on the val + test sets before epoch 1 begins.
  float SavedLrInit = N->Global().Lr;
  N->Global().Lr = 0.0f;
  double InitVal  = EvalMse(*N, Val, NumInput);
  double InitTest = EvalMse(*N, Test, NumInput);
  N->Global().Lr = SavedLrInit;
  auto Edges0 = bench::LiveEdgeSet(N->GetConnAlloc());
  Log.Log(0, NumUnits, NumEdges, &Edges0, &InitVal,
          {{"epoch", 0.0},
           {"train_mse", 0.0},
           {"val_mse", InitVal},
           {"test_mse", InitTest}});
  TestCsv.Add(0, InitTest, NumUnits, NumEdges);

  std::cout << "[ep   0] val_mse=" << InitVal
            << "  test_mse=" << InitTest << "\n";

  // Training: per sequence, reset state, walk T DoSteps with the target at
  // every step. e-prop updates accrue inside DoStep via UpdateConn.
  std::vector<size_t> Perm(H.TrainSeqs);
  for (size_t I = 0; I < Perm.size(); ++I)
    Perm[I] = I;
  std::mt19937 ShuffleRng(static_cast<uint32_t>(Args.Seed));
  bench::PhaseTimer Timer;

  auto T0 = std::chrono::steady_clock::now();
  std::array<float, 2> InBuf{};
  std::array<float, 2> TgtBuf{};
  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), ShuffleRng);
    double TrainSumSq = 0.0;
    size_t TrainCount = 0;
    for (size_t Sid : Perm) {
      ResetPerSequence(*N, NumInput);
      const auto &Xseq = Train.X[Sid];
      const auto &Yseq = Train.Y[Sid];
      for (size_t T = 0; T < Xseq.size(); ++T) {
        InBuf  = Xseq[T];
        TgtBuf = Yseq[T];
        Timer.Tick();
        N->DoForwardPass(std::span<const float>(InBuf.data(), 2));
        Timer.MarkForward();
        N->DoCalculateLoss(std::span<const float>(TgtBuf.data(), 2));
        Timer.MarkLoss();
        N->DoBackwardPass();
        Timer.MarkBackward();
        N->DoUpdateUnitState();
        N->DoUpdateConnectionState();
        Timer.MarkUpdate();
        N->DoResetGlobalState();
        Timer.MarkReset();
        Timer.StepDone();
        auto Out = N->GetOutput();
        float D0 = Out[0] - TgtBuf[0];
        float D1 = Out[1] - TgtBuf[1];
        TrainSumSq += static_cast<double>(D0 * D0 + D1 * D1);
        TrainCount += 2;
      }
    }
    double TrainMse = TrainCount == 0 ? 0.0 : TrainSumSq / TrainCount;

    // Eval: temporarily zero Lr so DoStep's UpdateConn becomes a no-op
    // multiplication. Eligibility still accumulates but is wiped at the
    // next ResetPerSequence — same trick as 06's snn-shd plan.
    float SavedLr = N->Global().Lr;
    N->Global().Lr = 0.0f;
    double ValMse  = EvalMse(*N, Val, NumInput);
    double TestMse = EvalMse(*N, Test, NumInput);
    N->Global().Lr = SavedLr;

    auto Edges = bench::LiveEdgeSet(N->GetConnAlloc());
    Log.Log(Ep, NumUnits, NumEdges, &Edges, &ValMse,
            {{"epoch", static_cast<double>(Ep)},
             {"train_mse", TrainMse},
             {"val_mse",   ValMse},
             {"test_mse",  TestMse}});
    TestCsv.Add(Ep, TestMse, NumUnits, NumEdges);
    std::cout << "[ep " << std::setw(3) << Ep << "] "
              << "train_mse=" << TrainMse
              << "  val_mse=" << ValMse
              << "  test_mse=" << TestMse << "\n";
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  // Final test pass (best-state would require a snapshot mechanism we don't
  // have — report the last-epoch number, mirroring 02_idempotent_imp).
  N->Global().Lr = 0.0f;
  double FinalTest = EvalMse(*N, Test, NumInput);

  // ------------------------------------------------------------------
  // Auxiliary CSVs consumed by traditional-plastix/plot_ccwc_ncp.py:
  //   *.pred.csv     — pred vs target trace on one held-out test sequence.
  //   *.topology.csv — partition + edge list, used to draw the NCP
  //                    wiring schematic in the paper-style figure.
  // ------------------------------------------------------------------
  std::string Suffix = Args.Tag.empty() ? "" : "_" + Args.Tag;
  auto CsvDir = Args.OutDir.parent_path() / "csv";
  std::filesystem::create_directories(CsvDir);

  // Predictions on test sequence 0. Reset, then walk forward (Lr already 0).
  {
    auto PredPath = CsvDir / ("ccwc_ncp" + Suffix + ".pred.csv");
    std::ofstream Pf(PredPath);
    Pf << "t,target_sin,target_cos,pred_sin,pred_cos,noisy_sin,noisy_cos\n";
    ResetPerSequence(*N, NumInput);
    const auto &Xseq = Test.X[0];
    const auto &Yseq = Test.Y[0];
    for (size_t T = 0; T < Xseq.size(); ++T) {
      InBuf  = Xseq[T];
      TgtBuf = Yseq[T];
      N->DoStep(std::span<const float>(InBuf.data(), 2),
                std::span<const float>(TgtBuf.data(), 2));
      auto Out = N->GetOutput();
      Pf << T << ',' << TgtBuf[0] << ',' << TgtBuf[1] << ',' << Out[0]
         << ',' << Out[1] << ',' << InBuf[0] << ',' << InBuf[1] << '\n';
    }
    std::cout << "[done] wrote " << PredPath << "\n";
  }

  // Topology dump: one header row with the partition counts, then one row
  // per live edge. KindTag is exported alongside so the plotter can colour
  // edges by source/dest role (sensory/inter/command/motor).
  {
    auto TopoPath = CsvDir / ("ccwc_ncp" + Suffix + ".topology.csv");
    std::ofstream Tf(TopoPath);
    auto Part = NCPWiringBuilder{H.Units, H.OutputDim, H.KSparse, H.KRec,
                                  H.KFb, Seed}.ComputePartition();
    Tf << "# partition n_input=" << NumInput
       << " n_sensory=" << Part.Ns << " n_inter=" << Part.Ni
       << " n_command=" << Part.Nc << " n_motor=" << Part.Nm << '\n';
    Tf << "from_id,to_id,from_kind,to_kind,weight\n";
    auto &Ca = N->GetConnAlloc();
    auto &Ua = N->GetUnitAlloc();
    for (size_t Cid = 0; Cid < Ca.Size(); ++Cid) {
      if (plastix::GetField<plastix::DeadTag>(Ca, Cid))
        continue;
      uint32_t F = plastix::GetField<plastix::FromIdTag>(Ca, Cid).Value;
      uint32_t T = plastix::GetField<plastix::ToIdTag>(Ca, Cid).Value;
      // From-side kind: inputs are at level 0 and have no KindTag set;
      // tag them with a sentinel 255 ("Input") so the plotter can colour
      // them distinctly.
      uint8_t Fk = (F < NumInput)
                       ? uint8_t{255}
                       : plastix::GetField<KindTag>(Ua, F);
      uint8_t Tk = plastix::GetField<KindTag>(Ua, T);
      Tf << F << ',' << T << ',' << static_cast<int>(Fk) << ','
         << static_cast<int>(Tk) << ',' << plastix::GetWeight(Ca, Cid)
         << '\n';
    }
    std::cout << "[done] wrote " << TopoPath << "\n";
  }

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "ccwc_ncp"));

  bench::SummaryWriter S;
  S.Set("workload", std::string{"06_ccwc_ncp"});
  S.Set("dataset", std::string{"synthetic-sine"});
  S.Set("task",    std::string{"sine"});
  S.Set("n_in",    static_cast<int>(NumInput));
  S.Set("n_out",   static_cast<int>(H.OutputDim));
  S.Set("n_units", static_cast<int>(NumUnits - NumInput));
  S.Set("n_edges", static_cast<int>(NumEdges));
  S.Set("seq_len", static_cast<int>(H.SeqLen));
  S.Set("epochs",  static_cast<int>(H.Epochs));
  S.Set("lr",      H.Lr);
  S.Set("dt",      H.Dt);
  S.Set("beta_trace", H.BetaTrace);
  S.Set("k_sparse", static_cast<int>(H.KSparse));
  S.Set("k_rec",    static_cast<int>(H.KRec));
  S.Set("k_fb",     static_cast<int>(H.KFb));
  S.Set("wall_seconds", Wall);
  S.Set("val_mse_final",
        Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss);
  S.Set("test_mse", FinalTest);
  S.Set("seed",     Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  final_test_mse=" << FinalTest
            << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << ", "
            << bench::TestCsvPath(Args, "ccwc_ncp") << "\n";
  return 0;
}
