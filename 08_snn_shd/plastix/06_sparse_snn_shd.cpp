// Workload 6 / 6 — sparse spiking neural network on SHD, e-prop learning.
//
// Implements the design in 06-sparse-snn-shd/SPEC.md. Five Plastix phases:
//
//   ForwardPass   LIF integration with subtract reset (hidden) or non-
//                 spiking integrator (output); saves pre-activation for the
//                 surrogate gradient.
//   Loss          Softmax cross-entropy on per-output LogitAccTag, gated on
//                 GlobalState::IsFinalStep. Stages dL/dlogit into
//                 BackwardAccTag of output units.
//   BackwardPass  Spatial reverse sweep. Output Apply passes BackwardAcc
//                 through as the learning signal; hidden Apply multiplies
//                 the upstream signal by the fast-sigmoid surrogate at the
//                 saved pre-activation. Uses RANDOM FEEDBACK (FeedbackTag
//                 instead of WeightTag) so the readout/feedback matrices
//                 are decoupled — Lillicrap 2016.
//   UpdateUnit    NoOp (left for filtered-trace e-prop variants).
//   UpdateConn    Incoming sweep: eligibility e = beta * e + pre·psi_post;
//                 w -= lr * L_post * e. Outgoing sweep is the framework's
//                 known no-op cost.
//
// Host-side time loop: one DoStep per network timestep. Empty target span
// on non-final timesteps short-circuits Loss; L stays zero so UpdateConn
// only accumulates the trace, doesn't move weights.
//
// Connectivity: K-sparse input->hidden (K=32 fan-in per hidden unit),
// dense hidden->output. ~13 k connections total for SHD (vs ~184 k dense)
// so the per-step cost is bounded by what Plastix's per-conn dispatch can
// actually deliver on a single core.

#include "plastix/common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>
#include <plastix/random.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace {

// --- hyperparameters --------------------------------------------------------

struct HP {
  size_t NBins = 50;
  size_t NHid = 256;
  size_t FanIn = 32;
  size_t NumClasses = 20;
  size_t Epochs = 80;
  float Lr = 1e-3f;
  float Beta = 0.9f;          // membrane decay
  float BetaTrace = 0.9f;     // eligibility-trace decay
  float Threshold = 1.0f;
  float SurrogateSlope = 25.0f;
  float WeightScale = 1.0f;   // multiplier on xavier init for fc_in
  float FeedbackScale = 1.0f; // scale of random-feedback weights B
  size_t MaxTrainRows = 0;    // 0 = no cap
  size_t MaxEvalRows = 0;     // 0 = full val/test
  size_t EvalEvery = 1;       // epochs between evaluations
};

// --- per-unit and per-conn field tags --------------------------------------

struct MembraneTag {};
struct PreActTag {};
struct LogitAccTag {};
struct LearningSignalTag {};
struct BetaTag {};
struct ThresholdTag {};
struct IsOutputTag {};
struct IsHiddenTag {};

struct EligibilityTag {};
struct FeedbackTag {}; // random-feedback weight B (forward uses WeightTag)

// --- global state -----------------------------------------------------------

struct EpropGlobals {
  bool IsFinalStep = false;
  float Lr = 1e-3f;
  float BetaTrace = 0.9f;
  float SurrogateSlope = 25.0f;
  float LogitScale = 1.0f; // 1/n_bins so summed readout membrane stays O(1)
  float Loss = 0.0f;       // reporting only; reset each call to Loss policy
};

// --- forward: LIF integration ----------------------------------------------

struct LifForward {
  using Accumulator = float;

  PLASTIX_HD static float Map(auto &U, size_t /*Self*/, size_t SrcId,
                              auto &C, size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float I) {
    bool IsOut = plastix::GetField<IsOutputTag>(U, Id);
    float Beta = plastix::GetField<BetaTag>(U, Id);
    float Thr = plastix::GetField<ThresholdTag>(U, Id);
    float Mem = plastix::GetField<MembraneTag>(U, Id);

    if (IsOut) {
      // Non-spiking integrator: mem' = beta*mem + I, no reset.
      Mem = Beta * Mem + I;
      plastix::GetField<MembraneTag>(U, Id) = Mem;
      // Accumulate the mean (×LogitScale = 1/n_bins) so the final
      // LogitAcc literally is the per-step mean of mem_o, keeping the
      // softmax inputs in a sane range without needing a chain-rule
      // correction at loss time.
      plastix::GetField<LogitAccTag>(U, Id) += Mem * G.LogitScale;
      // Output doesn't propagate forward; later layers (if any) would
      // see Activation=0. Backward Apply takes BackwardAcc as the
      // upstream signal, so Activation is unused.
      plastix::GetActivation(U, Id) = 0.0f;
      // Stash z so future readout-side surrogates could use it (none yet).
      plastix::GetField<PreActTag>(U, Id) = Mem - Thr;
    } else {
      // Subtract-reset LIF matching snnTorch's default Leaky:
      //   mem  = beta*mem + I
      //   spk  = step(mem - thr)
      //   mem -= thr * spk
      Mem = Beta * Mem + I;
      float Z = Mem - Thr;
      float Spk = (Z >= 0.0f) ? 1.0f : 0.0f;
      Mem -= Thr * Spk;
      plastix::GetField<MembraneTag>(U, Id) = Mem;
      plastix::GetField<PreActTag>(U, Id) = Z;
      plastix::GetActivation(U, Id) = Spk;
    }
  }
};

// --- loss: softmax CE on accumulated readout membrane ----------------------

struct EpropLoss {
  static EpropGlobals *G;

  template <typename UA>
  static void CalculateLoss(UA &U, plastix::UnitRange Out,
                            std::span<const float> Target, auto & /*Gref*/) {
    if (G == nullptr || !G->IsFinalStep || Target.empty())
      return;

    // LogitAcc is already the per-timestep mean of mem_o (Forward Apply
    // applied the LogitScale factor), so the softmax is in the right
    // range and no chain-rule correction is needed at loss time.
    float MaxL = -1e30f;
    for (size_t I = Out.Begin; I < Out.End; ++I) {
      float L = plastix::GetField<LogitAccTag>(U, I);
      if (L > MaxL)
        MaxL = L;
    }
    double Z = 0.0;
    for (size_t I = Out.Begin; I < Out.End; ++I)
      Z += std::exp(plastix::GetField<LogitAccTag>(U, I) - MaxL);

    float Loss = 0.0f;
    for (size_t K = 0, I = Out.Begin; I < Out.End; ++I, ++K) {
      float P = static_cast<float>(
          std::exp(plastix::GetField<LogitAccTag>(U, I) - MaxL) / Z);
      float Y = Target[K];
      // dL/dlogit = softmax - one_hot, staged for backward.
      plastix::GetBackwardAcc(U, I) = P - Y;
      if (Y > 0.0f)
        Loss += -std::log(std::max(P, 1e-30f));
    }
    G->Loss = Loss;
  }
};
EpropGlobals *EpropLoss::G = nullptr;

// --- backward: spatial reverse with random feedback alignment -------------

struct LifBackward {
  using Accumulator = float;
  static EpropGlobals *G;

  PLASTIX_HD static float Map(auto &U, size_t /*Src*/, size_t ToId, auto &C,
                              size_t ConnId, auto &) {
    // Random feedback: backward path uses FeedbackTag (fixed B) instead of
    // WeightTag (W). For input->hidden conns, FeedbackTag is zero, so
    // backward contributions into input units are zero (correct: inputs
    // aren't learnable).
    return plastix::GetField<FeedbackTag>(C, ConnId) *
           plastix::GetField<LearningSignalTag>(U, ToId);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto & /*Gref*/, float Up) {
    bool IsOut = plastix::GetField<IsOutputTag>(U, Id);
    if (IsOut) {
      // Output learning signal = staged dL/dlogit. No surrogate
      // (the integrator has linear gradient).
      plastix::GetField<LearningSignalTag>(U, Id) =
          plastix::GetBackwardAcc(U, Id);
    } else {
      float Z = plastix::GetField<PreActTag>(U, Id);
      float Slope = G ? G->SurrogateSlope : 25.0f;
      float Den = 1.0f + Slope * std::fabs(Z);
      float Psi = 1.0f / (Den * Den);
      plastix::GetField<LearningSignalTag>(U, Id) = Up * Psi;
    }
  }
};
EpropGlobals *LifBackward::G = nullptr;

// --- update connections: maintain eligibility trace + apply weight delta --

struct EpropUpdateConn {
  static EpropGlobals *G;

  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId,
                                                  auto & /*Gref*/) {
    float Slope = G ? G->SurrogateSlope : 25.0f;
    float BetaT = G ? G->BetaTrace : 0.9f;
    float Lr = G ? G->Lr : 0.0f;

    float PreSpk = plastix::GetActivation(U, SrcId);
    // Output is a linear integrator (no surrogate); hidden uses
    // fast-sigmoid surrogate at the saved pre-activation.
    float Psi;
    if (plastix::GetField<IsOutputTag>(U, DstId)) {
      Psi = 1.0f;
    } else {
      float Z = plastix::GetField<PreActTag>(U, DstId);
      float Den = 1.0f + Slope * std::fabs(Z);
      Psi = 1.0f / (Den * Den);
    }
    float &Elig = plastix::GetField<EligibilityTag>(C, ConnId);
    Elig = BetaT * Elig + PreSpk * Psi;

    float L = plastix::GetField<LearningSignalTag>(U, DstId);
    if (L != 0.0f && Lr != 0.0f)
      plastix::GetWeight(C, ConnId) -= Lr * L * Elig;
  }

  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};
EpropGlobals *EpropUpdateConn::G = nullptr;

// --- traits -----------------------------------------------------------------

struct SparseSnnTraits : plastix::DefaultNetworkTraits<EpropGlobals> {
  using ForwardPass = LifForward;
  using BackwardPass = LifBackward;
  using Loss = EpropLoss;
  using UpdateConn = EpropUpdateConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<MembraneTag, float>,
      plastix::alloc::SOAField<PreActTag, float>,
      plastix::alloc::SOAField<LogitAccTag, float>,
      plastix::alloc::SOAField<LearningSignalTag, float>,
      plastix::alloc::SOAField<BetaTag, float>,
      plastix::alloc::SOAField<ThresholdTag, float>,
      plastix::alloc::SOAField<IsOutputTag, bool>,
      plastix::alloc::SOAField<IsHiddenTag, bool>>;
  using ExtraConnFields = plastix::ConnFieldList<
      plastix::alloc::SOAField<plastix::WeightTag, float>,
      plastix::alloc::SOAField<EligibilityTag, float>,
      plastix::alloc::SOAField<FeedbackTag, float>>;
  static constexpr size_t UnitCapacity = 4096;
  static constexpr size_t ConnCapacity = 32768; // ~13 k live, headroom
};
static_assert(plastix::NetworkTraits<SparseSnnTraits>);

using Net = plastix::Network<SparseSnnTraits>;

// --- unit initialisers ------------------------------------------------------

struct InitHidden {
  float Beta;
  float Threshold;
  void operator()(auto &UA, auto Id) const {
    plastix::GetField<BetaTag>(UA, Id) = Beta;
    plastix::GetField<ThresholdTag>(UA, Id) = Threshold;
    plastix::GetField<IsOutputTag>(UA, Id) = false;
    plastix::GetField<IsHiddenTag>(UA, Id) = true;
  }
};

struct InitOutput {
  float Beta;
  float Threshold;
  void operator()(auto &UA, auto Id) const {
    plastix::GetField<BetaTag>(UA, Id) = Beta;
    plastix::GetField<ThresholdTag>(UA, Id) = Threshold;
    plastix::GetField<IsOutputTag>(UA, Id) = true;
    plastix::GetField<IsHiddenTag>(UA, Id) = false;
  }
};

// --- layer builders ---------------------------------------------------------

// K-sparse fan-in: each new hidden unit draws FanIn distinct source ids
// from the previous layer's range, allocates one connection each,
// initialises Weight by Xavier-like scaling and FeedbackTag with the same
// distribution (independent draw). Connections from input units don't need
// feedback (input is not learnable), so their FeedbackTag is left at zero;
// see the SPEC.md note on backward into level 0.
struct LocalSparseLayer {
  size_t Width;
  size_t FanIn;
  uint64_t Seed;
  float Beta;
  float Threshold;
  float WeightScale;

  template <typename UnitAlloc, typename ConnAlloc>
  plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                plastix::UnitRange Prev) const {
    uint16_t NewLevel = plastix::GetLevel(UA, Prev.Begin) + 1;
    uint16_t SrcLevel = plastix::GetLevel(UA, Prev.Begin);
    plastix::UnitRange Units = UA.AllocateMany(Width);
    InitHidden Init{Beta, Threshold};

    std::mt19937_64 Rng(Seed);
    float Bound =
        WeightScale * std::sqrt(6.0f / static_cast<float>(FanIn + Width));
    std::uniform_real_distribution<float> W(-Bound, Bound);
    size_t PrevWidth = Prev.End - Prev.Begin;
    std::vector<size_t> Pool(PrevWidth);
    std::iota(Pool.begin(), Pool.end(), Prev.Begin);

    for (auto Id : Units.Ids()) {
      plastix::GetLevel(UA, Id) = NewLevel;
      Init(UA, Id);

      // Without replacement: partial Fisher-Yates over Pool to pick FanIn.
      size_t K = std::min(FanIn, PrevWidth);
      for (size_t I = 0; I < K; ++I) {
        std::uniform_int_distribution<size_t> Pick(I, PrevWidth - 1);
        std::swap(Pool[I], Pool[Pick(Rng)]);
        auto Src = Pool[I];
        auto ConnId = CA.Allocate();
        plastix::GetField<plastix::FromIdTag>(CA, ConnId) =
            static_cast<uint32_t>(Src);
        plastix::GetField<plastix::ToIdTag>(CA, ConnId) =
            static_cast<uint32_t>(Id);
        plastix::GetField<plastix::SrcLevelTag>(CA, ConnId) = SrcLevel;
        plastix::GetWeight(CA, ConnId) = W(Rng);
        plastix::GetField<EligibilityTag>(CA, ConnId) = 0.0f;
        // Input-layer source: no learning signal flows back further, so
        // leave FeedbackTag at zero.
        plastix::GetField<FeedbackTag>(CA, ConnId) = 0.0f;
      }
    }
    return Units;
  }
};

// Dense fan-in hidden -> output. Weight init: scaled uniform; FeedbackTag
// drawn independently from the same distribution. The independence is what
// makes this random-feedback alignment rather than symmetric backprop.
struct DenseOutputLayer {
  size_t Width;
  uint64_t Seed;
  float Beta;
  float Threshold;
  float WeightScale;
  float FeedbackScale;

  template <typename UnitAlloc, typename ConnAlloc>
  plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                plastix::UnitRange Prev) const {
    uint16_t NewLevel = plastix::GetLevel(UA, Prev.Begin) + 1;
    uint16_t SrcLevel = plastix::GetLevel(UA, Prev.Begin);
    plastix::UnitRange Units = UA.AllocateMany(Width);
    InitOutput Init{Beta, Threshold};
    size_t PrevWidth = Prev.End - Prev.Begin;
    float Bound =
        WeightScale * std::sqrt(6.0f / static_cast<float>(PrevWidth + Width));
    float BBound =
        FeedbackScale * std::sqrt(6.0f / static_cast<float>(PrevWidth + Width));
    std::mt19937_64 Rng(Seed);
    std::uniform_real_distribution<float> W(-Bound, Bound);
    std::uniform_real_distribution<float> B(-BBound, BBound);

    for (auto Id : Units.Ids()) {
      plastix::GetLevel(UA, Id) = NewLevel;
      Init(UA, Id);
      for (auto Src : Prev.Ids()) {
        auto ConnId = CA.Allocate();
        plastix::GetField<plastix::FromIdTag>(CA, ConnId) =
            static_cast<uint32_t>(Src);
        plastix::GetField<plastix::ToIdTag>(CA, ConnId) =
            static_cast<uint32_t>(Id);
        plastix::GetField<plastix::SrcLevelTag>(CA, ConnId) = SrcLevel;
        plastix::GetWeight(CA, ConnId) = W(Rng);
        plastix::GetField<EligibilityTag>(CA, ConnId) = 0.0f;
        plastix::GetField<FeedbackTag>(CA, ConnId) = B(Rng);
      }
    }
    return Units;
  }
};

// --- dataset loader (.plxbin format described in SPEC.md) ------------------

struct Dataset {
  std::vector<float> X;     // n_samples * n_bins * n_channels, float32
  std::vector<int64_t> Y;   // n_samples
  size_t NSamples = 0;
  size_t NBins = 0;
  size_t NChannels = 0;

  const float *Sample(size_t I, size_t T) const {
    return X.data() + (I * NBins + T) * NChannels;
  }
};

static Dataset LoadPlxbin(const std::filesystem::path &Path) {
  std::ifstream In(Path, std::ios::binary);
  if (!In) {
    std::cerr << "failed to open " << Path << "\n";
    std::exit(2);
  }
  uint32_t Header[6];
  In.read(reinterpret_cast<char *>(Header), sizeof(Header));
  if (Header[0] != 0x53484430u) {
    std::cerr << "bad magic in " << Path << "\n";
    std::exit(2);
  }
  Dataset D;
  D.NSamples = Header[1];
  D.NBins = Header[2];
  D.NChannels = Header[3];
  size_t NFloats = D.NSamples * D.NBins * D.NChannels;
  D.X.resize(NFloats);
  In.read(reinterpret_cast<char *>(D.X.data()),
          NFloats * sizeof(float));
  D.Y.resize(D.NSamples);
  In.read(reinterpret_cast<char *>(D.Y.data()),
          D.NSamples * sizeof(int64_t));
  return D;
}

// --- per-example state reset ----------------------------------------------

static void ResetPerExample(Net &N) {
  auto &UA = N.GetUnitAlloc();
  size_t NU = UA.Size();
  for (size_t I = 0; I < NU; ++I) {
    plastix::GetField<MembraneTag>(UA, I) = 0.0f;
    plastix::GetField<PreActTag>(UA, I) = 0.0f;
    plastix::GetField<LogitAccTag>(UA, I) = 0.0f;
    plastix::GetField<LearningSignalTag>(UA, I) = 0.0f;
    plastix::GetField<plastix::ForwardAccTag>(UA, I) = 0.0f;
    plastix::GetField<plastix::BackwardAccTag>(UA, I) = 0.0f;
    plastix::GetActivation(UA, I) = 0.0f;
  }
  auto &CA = N.GetConnAlloc();
  size_t NC = CA.Size();
  for (size_t C = 0; C < NC; ++C)
    plastix::GetField<EligibilityTag>(CA, C) = 0.0f;
}

// --- one-hot helper -------------------------------------------------------

static std::vector<float> OneHot(int Y, size_t K) {
  std::vector<float> V(K, 0.0f);
  if (static_cast<size_t>(Y) < K)
    V[Y] = 1.0f;
  return V;
}

// --- forward-only pass over one example (no learning) -------------------

static int Predict(Net &N, const Dataset &D, size_t ExIdx,
                   plastix::UnitRange Out, EpropGlobals &G) {
  ResetPerExample(N);
  G.IsFinalStep = false;
  for (size_t T = 0; T < D.NBins; ++T) {
    std::span<const float> X(D.Sample(ExIdx, T), D.NChannels);
    N.DoForwardPass(X);
  }
  // argmax over LogitAcc on the output range
  auto &UA = N.GetUnitAlloc();
  size_t Best = Out.Begin;
  float BestV = plastix::GetField<LogitAccTag>(UA, Out.Begin);
  for (size_t I = Out.Begin + 1; I < Out.End; ++I) {
    float V = plastix::GetField<LogitAccTag>(UA, I);
    if (V > BestV) {
      BestV = V;
      Best = I;
    }
  }
  return static_cast<int>(Best - Out.Begin);
}

static double EvalAccuracy(Net &N, const Dataset &D, plastix::UnitRange Out,
                           EpropGlobals &G, size_t Cap = 0,
                           bool TimeShuffle = false, uint32_t ShuffleSeed = 0) {
  size_t N_Ex = D.NSamples;
  if (Cap > 0 && Cap < N_Ex)
    N_Ex = Cap;
  size_t Correct = 0;
  std::mt19937 Rng(ShuffleSeed);
  std::vector<size_t> Perm(D.NBins);
  for (size_t I = 0; I < D.NBins; ++I)
    Perm[I] = I;
  std::vector<float> Buf(D.NBins * D.NChannels);
  for (size_t E = 0; E < N_Ex; ++E) {
    if (TimeShuffle) {
      std::shuffle(Perm.begin(), Perm.end(), Rng);
      for (size_t T = 0; T < D.NBins; ++T)
        std::memcpy(Buf.data() + T * D.NChannels, D.Sample(E, Perm[T]),
                    D.NChannels * sizeof(float));
      ResetPerExample(N);
      G.IsFinalStep = false;
      for (size_t T = 0; T < D.NBins; ++T) {
        std::span<const float> X(Buf.data() + T * D.NChannels, D.NChannels);
        N.DoForwardPass(X);
      }
      auto &UA = N.GetUnitAlloc();
      size_t Best = Out.Begin;
      float BestV = plastix::GetField<LogitAccTag>(UA, Out.Begin);
      for (size_t I = Out.Begin + 1; I < Out.End; ++I) {
        float V = plastix::GetField<LogitAccTag>(UA, I);
        if (V > BestV) {
          BestV = V;
          Best = I;
        }
      }
      if (static_cast<int>(Best - Out.Begin) == static_cast<int>(D.Y[E]))
        ++Correct;
    } else {
      int Pred = Predict(N, D, E, Out, G);
      if (Pred == static_cast<int>(D.Y[E]))
        ++Correct;
    }
  }
  return N_Ex ? static_cast<double>(Correct) / N_Ex : 0.0;
}

// --- compute mean hidden firing rate over a slice of training data ------

static double MeanFiringRate(Net &N, const Dataset &D,
                             plastix::UnitRange Hidden, EpropGlobals &G,
                             size_t Cap) {
  size_t N_Ex = std::min(Cap, D.NSamples);
  if (N_Ex == 0)
    return 0.0;
  double Total = 0.0;
  size_t Slots = 0;
  for (size_t E = 0; E < N_Ex; ++E) {
    ResetPerExample(N);
    G.IsFinalStep = false;
    for (size_t T = 0; T < D.NBins; ++T) {
      std::span<const float> X(D.Sample(E, T), D.NChannels);
      N.DoForwardPass(X);
      auto &UA = N.GetUnitAlloc();
      for (size_t I = Hidden.Begin; I < Hidden.End; ++I)
        Total += plastix::GetActivation(UA, I);
      Slots += Hidden.End - Hidden.Begin;
    }
  }
  return Slots ? Total / static_cast<double>(Slots) : 0.0;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.NBins = static_cast<size_t>(Args.GetInt("n-bins", H.NBins));
  H.NHid = static_cast<size_t>(Args.GetInt("n-hid", H.NHid));
  H.FanIn = static_cast<size_t>(Args.GetInt("fan-in", H.FanIn));
  H.NumClasses = static_cast<size_t>(Args.GetInt("n-classes", H.NumClasses));
  H.Epochs = static_cast<size_t>(Args.GetInt("epochs", H.Epochs));
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.Beta = Args.GetFloat("beta", H.Beta);
  H.BetaTrace = Args.GetFloat("beta-trace", H.BetaTrace);
  H.Threshold = Args.GetFloat("threshold", H.Threshold);
  H.SurrogateSlope = Args.GetFloat("surrogate-slope", H.SurrogateSlope);
  H.WeightScale = Args.GetFloat("weight-scale", H.WeightScale);
  H.FeedbackScale = Args.GetFloat("feedback-scale", H.FeedbackScale);
  H.MaxTrainRows = static_cast<size_t>(
      Args.GetInt("max-train-rows", static_cast<int>(H.MaxTrainRows)));
  H.MaxEvalRows = static_cast<size_t>(
      Args.GetInt("max-eval-rows", static_cast<int>(H.MaxEvalRows)));
  H.EvalEvery = static_cast<size_t>(
      Args.GetInt("eval-every", static_cast<int>(H.EvalEvery)));
  if (Args.Quick) {
    // Don't shrink n-bins — that would invalidate the pre-binned plxbin
    // cache. The eval/train row caps below already keep --quick fast.
    H.Epochs = std::max<size_t>(1, H.Epochs / 4);
    if (H.MaxTrainRows == 0)
      H.MaxTrainRows = 1024;
    if (H.MaxEvalRows == 0)
      H.MaxEvalRows = 512;
  }

  // .plxbin caches live alongside SHD_cache produced by snn-shd/data.py.
  auto TrainPath =
      Args.DataDir / "SHD_cache" / ("train_n" + std::to_string(H.NBins) + ".plxbin");
  auto TestPath =
      Args.DataDir / "SHD_cache" / ("test_n" + std::to_string(H.NBins) + ".plxbin");
  if (!std::filesystem::exists(TrainPath)) {
    std::cerr << "missing " << TrainPath
              << "\nRun:\n  uv run python snn-shd/data.py "
                 "--export --n-bins "
              << H.NBins << "\n";
    return 2;
  }
  std::cout << "[data] loading " << TrainPath << "\n";
  auto Train = LoadPlxbin(TrainPath);
  std::cout << "[data] loading " << TestPath << "\n";
  auto Test = LoadPlxbin(TestPath);
  size_t NIn = Train.NChannels;
  std::cout << "[info] train=" << Train.NSamples << " test=" << Test.NSamples
            << " n_in=" << NIn << " n_hid=" << H.NHid
            << " n_out=" << H.NumClasses << " n_bins=" << H.NBins
            << " fan_in=" << H.FanIn << " epochs=" << H.Epochs
            << " lr=" << H.Lr << "\n";

  // Build the network.
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 41ull;
  auto N = std::unique_ptr<Net>(new Net(
      NIn,
      LocalSparseLayer{H.NHid, H.FanIn, SeedBase + 1, H.Beta, H.Threshold,
                       H.WeightScale},
      DenseOutputLayer{H.NumClasses, SeedBase + 2, H.Beta, H.Threshold,
                       1.0f, H.FeedbackScale}));

  // The Network constructor folds builders left-to-right; OutputRange lives
  // at the last layer's UnitRange. We need the hidden range for firing-rate
  // reporting too; reconstruct by sliding past the input units.
  plastix::UnitRange Hidden{NIn, NIn + H.NHid};
  plastix::UnitRange OutRange{NIn + H.NHid, NIn + H.NHid + H.NumClasses};

  size_t NConns = bench::LiveEdgeCount(N->GetConnAlloc());
  std::cout << "[info] live conns=" << NConns
            << "  (vs " << (NIn * H.NHid + H.NHid * H.NumClasses)
            << " dense)\n";

  // Wire globals into the policies. They're stateless statics; setting
  // each once at startup is sufficient.
  EpropGlobals G;
  G.Lr = H.Lr;
  G.BetaTrace = H.BetaTrace;
  G.SurrogateSlope = H.SurrogateSlope;
  G.LogitScale = 1.0f / static_cast<float>(H.NBins);
  EpropLoss::G = &G;
  LifBackward::G = &G;
  EpropUpdateConn::G = &G;

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "sparse_snn_shd");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_acc");

  // Initial point — pure forward-only eval to confirm we start at chance.
  size_t EvalCap = H.MaxEvalRows;
  double InitVal = 1.0 - EvalAccuracy(*N, Test, OutRange, G, EvalCap);
  double InitTest = 1.0 - InitVal; // == EvalAccuracy
  Log.Log(0, N->GetUnitAlloc().Size(), NConns, nullptr, &InitVal,
          {{"test_acc", InitTest},
           {"firing_rate", 0.0},
           {"train_loss", 0.0},
           {"epoch", 0.0}});
  TestCsv.Add(0, InitTest, N->GetUnitAlloc().Size(), NConns);
  std::cout << "[ep   0] test_acc=" << InitTest << "\n";

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  size_t TrainCap =
      H.MaxTrainRows > 0 ? std::min(H.MaxTrainRows, Train.NSamples)
                          : Train.NSamples;
  std::vector<size_t> Perm(TrainCap);
  std::iota(Perm.begin(), Perm.end(), 0);

  double BestTest = InitTest;
  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();

  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    double LossSum = 0.0;
    size_t LossCnt = 0;
    size_t Correct = 0;

    for (size_t Idx : Perm) {
      ResetPerExample(*N);
      auto OneHotV = OneHot(static_cast<int>(Train.Y[Idx]), H.NumClasses);

      for (size_t T = 0; T < Train.NBins; ++T) {
        G.IsFinalStep = (T + 1 == Train.NBins);
        std::span<const float> X(Train.Sample(Idx, T), Train.NChannels);
        std::span<const float> Tgt = G.IsFinalStep
                                          ? std::span<const float>{OneHotV}
                                          : std::span<const float>{};
        Timer.Tick();
        N->DoForwardPass(X);
        Timer.MarkForward();
        N->DoCalculateLoss(Tgt);
        Timer.MarkLoss();
        N->DoBackwardPass();
        Timer.MarkBackward();
        N->DoUpdateUnitState();
        N->DoUpdateConnectionState();
        Timer.MarkUpdate();
        N->DoResetGlobalState();
        Timer.MarkReset();
        Timer.StepDone();
      }
      LossSum += G.Loss;
      ++LossCnt;
      // Argmax over LogitAcc gives the prediction without rerunning.
      auto &UA = N->GetUnitAlloc();
      size_t Best = OutRange.Begin;
      float BestV = plastix::GetField<LogitAccTag>(UA, OutRange.Begin);
      for (size_t I = OutRange.Begin + 1; I < OutRange.End; ++I) {
        float V = plastix::GetField<LogitAccTag>(UA, I);
        if (V > BestV) {
          BestV = V;
          Best = I;
        }
      }
      if (static_cast<int>(Best - OutRange.Begin) ==
          static_cast<int>(Train.Y[Idx]))
        ++Correct;
    }

    double TrLoss = LossCnt ? LossSum / LossCnt : 0.0;
    double TrAcc = TrainCap ? static_cast<double>(Correct) / TrainCap : 0.0;
    double TestAcc = 0.0;
    double Rate = 0.0;
    if (Ep == H.Epochs || (Ep % H.EvalEvery) == 0) {
      TestAcc = EvalAccuracy(*N, Test, OutRange, G, EvalCap);
      Rate = MeanFiringRate(*N, Train, Hidden, G,
                             std::min<size_t>(128, TrainCap));
    }
    if (TestAcc > BestTest)
      BestTest = TestAcc;

    double NegMetric = 1.0 - TestAcc; // sentinel "loss-like" for val_loss column
    Log.Log(Ep, N->GetUnitAlloc().Size(), NConns, nullptr, &NegMetric,
            {{"test_acc", TestAcc},
             {"firing_rate", Rate},
             {"train_loss", TrLoss},
             {"train_acc", TrAcc},
             {"epoch", static_cast<double>(Ep)}});
    TestCsv.Add(Ep, TestAcc, N->GetUnitAlloc().Size(), NConns);
    std::cout << "[ep " << std::setw(3) << Ep
              << "] train_loss=" << TrLoss
              << " train_acc=" << TrAcc
              << " test_acc=" << TestAcc
              << " rate=" << Rate << "\n" << std::flush;
  }

  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  // Final test eval + time-shuffle ablation (the primary diagnostic).
  double FinalTest = EvalAccuracy(*N, Test, OutRange, G, EvalCap);
  double ShuffledTest = EvalAccuracy(*N, Test, OutRange, G, EvalCap,
                                     /*TimeShuffle=*/true,
                                     /*ShuffleSeed=*/Args.Seed + 9999);

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "sparse_snn_shd"));

  bench::SummaryWriter S;
  S.Set("workload", std::string{"06_sparse_snn_shd"});
  S.Set("dataset", std::string{"SHD"});
  S.Set("n_in", static_cast<int>(NIn));
  S.Set("n_hid", static_cast<int>(H.NHid));
  S.Set("n_out", static_cast<int>(H.NumClasses));
  S.Set("fan_in", static_cast<int>(H.FanIn));
  S.Set("n_bins", static_cast<int>(H.NBins));
  S.Set("n_conns", static_cast<int>(NConns));
  S.Set("epochs", static_cast<int>(H.Epochs));
  S.Set("lr", static_cast<double>(H.Lr));
  S.Set("beta", static_cast<double>(H.Beta));
  S.Set("beta_trace", static_cast<double>(H.BetaTrace));
  S.Set("threshold", static_cast<double>(H.Threshold));
  S.Set("surrogate_slope", static_cast<double>(H.SurrogateSlope));
  S.Set("wall_seconds", Wall);
  S.Set("val_acc_best", BestTest);
  S.Set("test_acc", FinalTest);
  S.Set("test_acc_shuffled", ShuffledTest);
  S.Set("ablation_drop", FinalTest - ShuffledTest);
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_acc=" << FinalTest
            << "  shuffled=" << ShuffledTest
            << "  drop=" << (FinalTest - ShuffledTest) << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";
  (void)LogPath;
  return 0;
}
