// Workload 9 — imprinting-learner on the audio-prediction benchmark, Plastix
// translation.
//
// Mirrors 09_imprintin_learner/cpp/09_imprintin_learner.cpp:
// streaming TD prediction over the APBD dataset (2500-dim binary observation +
// {-1,0,+1} reward per step). We adapt the policy bundle from
// examples/imprinting-learner/imprinting_learner.cpp — the algorithm is
// untouched, only the harness changes (input dim, dataset, per-step MSE vs.
// the discounted return rather than a 5-state random walk).
//
// Plastix policy mapping (matches the algorithm in examples/imprinting-learner):
//
//   ForwardPass        custom    pattern (k/n threshold), memory (delay/window),
//                                output (linear sum)
//   Loss               custom    computes δ = r + γ·v - v_old, exposed via Globals
//   UpdateConn         custom    SwiftTD-style two-phase update on output weights
//   AddUnit / AddConn  custom    spawns pattern/memory features bounded by τ < η
//   ResetGlobal        custom    rolls v -> v_old; resets per-step reductions
//
// Sizing: 2500 input units, 1 output, with headroom for ~10× generated units
// (capacity 16384 / 524288) under the τ < η gate that throttles growth.

#include "plastix/common.hpp"
#include "../cpp/examples/dataset.hpp"

#include <plastix/alloc.hpp>
#include <plastix/device_atomics.hpp>
#include <plastix/device_rng.hpp>
#include <plastix/macros.hpp>
#include <plastix/plastix.hpp>
#include <plastix/reverse_adjacency.hpp>
#include <plastix/traits.hpp>
#include <plastix/unit_state.hpp>

#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace {

// --- per-connection state ----------------------------------------------------
// (plastix::WeightTag is reused for w_i.)
struct ZTag {};
struct ZDeltaTag {};
struct DeltaWTag {};
struct HTag {};
struct HOldTag {};
struct HTempTag {};
struct ZBarTag {};
struct PTag {};
struct BetaTag {}; // beta = log(alpha)

// --- per-feature state -------------------------------------------------------
enum UnitKind : uint8_t { UK_Pattern, UK_Memory, UK_Output };
enum TenureStatus : uint8_t { TS_Idle, TS_TenureTrack, TS_Tenured };
struct UKindTag {};
struct UFlagsTags {};
struct Pulse {};
struct Delay {};
struct ActThreshold {};
struct Tenure {};
struct InDegreeTag {}; // per-unit incoming-edge count (for on-device forward)

// Atomic-or-plain global accumulate: kernels run many threads into the shared
// GlobalState, so on-device these reductions must be atomic; on host the same
// policy code runs serially and a plain += is correct (and faster).
// Global reductions over edges (G.Tau/G.B/G.VDelta) all target one scalar; use
// the warp-aggregated atomic to avoid the per-edge serialization that profiling
// showed was ~99% of the on-device update phase.
PLASTIX_HD void AtomicAddGlobal(float &Dst, float Val) {
  plastix::WarpAtomicAdd(Dst, Val);
}

struct UnitFlags {
  bool InputsAdded = false;
  bool OutputsAdded = false;
};

namespace hp {
constexpr static float Gamma = 0.99f;
constexpr static float Lambda = 0.9f;
constexpr static float Eta = 0.1f;
constexpr static float EtaMin = 1e-7f;
constexpr static float Decay = 0.999f;
constexpr static float MetaStepSize = 1e-3f;
constexpr static float AlphaInit = 3e-3f;

constexpr static float TenureThreshold = 0.01f;
constexpr static float TenureTrackThreshold = 3e-4f;
constexpr static int MaxGenerationsPerStep = 5;
}; // namespace hp

struct ImprintingLearnerGlobals {
  std::mt19937 Rng{42};

  float V = 0.0f;
  float VOld = 0.0f;
  float Delta = 0.0f;
  float VDeltaPrev = 0.0f;
  float VDelta = 0.0f;
  float Tau = 0.0f;
  float B = 0.0f;

  int GenerationLeft = hp::MaxGenerationsPerStep;
  bool WasActive = false;
};

struct ImprintingLearnerOutputUnitInit {
  void operator()(auto &UA, auto Id) const {
    plastix::GetField<UKindTag>(UA, Id) = UK_Output;
  }
};

struct ImprintingLearnerConnInit {
  void operator()(auto &CA, auto ConnId) const {
    plastix::GetField<BetaTag>(CA, ConnId) = std::log(hp::AlphaInit);
  }
};

struct ImprintingLearnerForward {
  // float accumulator (= Σ w·act) so the GPU forward path (which requires
  // Accumulator==float) activates. The per-unit connection count that the old
  // struct accumulator carried is precomputed once into the InDegree unit
  // field (via the reverse-adjacency), so the pattern threshold (avg = Σ/n)
  // still works on-device.
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float Activation) {
    UnitKind Kind = plastix::GetField<UKindTag>(U, Id);
    if (Kind == UK_Output) {
      plastix::GetActivation(U, Id) = Activation;
      G.V = Activation;
    } else if (Kind == UK_Pattern) {
      auto Threshold = plastix::GetField<ActThreshold>(U, Id);
      uint32_t Nc = plastix::GetField<InDegreeTag>(U, Id);
      int Act = Nc > 0 ? static_cast<int>((Activation / Nc) > Threshold) : 0;
      G.WasActive |= (Act != 0); // benign: all writers store true
      plastix::GetActivation(U, Id) = static_cast<float>(Act);
    } else {
      uint32_t &P = plastix::GetField<Pulse>(U, Id);
      uint32_t D = plastix::GetField<Delay>(U, Id);
      plastix::GetActivation(U, Id) = static_cast<float>(P & 0x1u);
      P = (P >> 1);
      if (Activation != 0) {
        G.WasActive = true;
        P |= 0x1u << D;
      }
    }
  }
};

struct ImprintingLearnerLoss {
  static void CalculateLoss(auto &, plastix::UnitRange,
                            std::span<const float> Targets,
                            ImprintingLearnerGlobals &G) {
    float Reward = Targets[0];
    G.Delta = Reward + hp::Gamma * G.V - G.VOld;
    G.VDeltaPrev = G.VDelta;
    G.VDelta = 0.0f;
    G.B = 0.0f;
  }
};

inline TenureStatus DetermineTenureState(float Weight) {
  float A = std::fabs(Weight);
  if (A > hp::TenureThreshold)
    return TS_Tenured;
  if (A > hp::TenureTrackThreshold)
    return TS_TenureTrack;
  return TS_Idle;
}

struct ImprintingLearnerUnitUpdate {
  PLASTIX_HD static void Update(auto &UA, size_t Id, auto &) {
    if (plastix::GetField<UKindTag>(UA, Id) == UK_Pattern)
      plastix::GetField<UFlagsTags>(UA, Id).InputsAdded = true;
  }
};

struct ImprintingLearnerConnUpdate {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                       size_t SrcId,
                                       auto &C, size_t ConnId,
                                       ImprintingLearnerGlobals &G) {
    using namespace plastix;
    if (GetField<UKindTag>(U, DstId) != UK_Output)
      return;

    const float F = GetActivation(U, SrcId);

    float &W = GetWeight(C, ConnId);
    float &Z = GetField<ZTag>(C, ConnId);
    float &Zd = GetField<ZDeltaTag>(C, ConnId);
    float &Dw = GetField<DeltaWTag>(C, ConnId);
    float &H = GetField<HTag>(C, ConnId);
    float &HOld = GetField<HOldTag>(C, ConnId);
    float &HTemp = GetField<HTempTag>(C, ConnId);
    float &Beta = GetField<BetaTag>(C, ConnId);
    float &ZBar = GetField<ZBarTag>(C, ConnId);
    float &P = GetField<PTag>(C, ConnId);

    Dw = G.Delta * Z - Zd * G.VDeltaPrev;
    W += Dw;
    GetField<Tenure>(U, SrcId) = DetermineTenureState(W);

    Beta += hp::MetaStepSize / std::exp(Beta) * (G.Delta - G.VDeltaPrev) * P;
    float ExpBeta = std::exp(Beta);
    if (ExpBeta > hp::Eta || std::isinf(ExpBeta)) {
      Beta = std::log(hp::Eta);
      ExpBeta = hp::Eta;
    }
    if (ExpBeta < hp::EtaMin) {
      Beta = std::log(hp::EtaMin);
      ExpBeta = hp::EtaMin;
    }

    HOld = H;
    H = HTemp + G.Delta * ZBar - Zd * G.VDeltaPrev;
    HTemp = H;
    Zd = 0.0f;

    const float TraceDecay = hp::Gamma * hp::Lambda;
    Z *= TraceDecay;
    P *= TraceDecay;
    ZBar *= TraceDecay;

    AtomicAddGlobal(G.Tau, ExpBeta * F * F);
    AtomicAddGlobal(G.B, Z * F);
  }

  PLASTIX_HD static void UpdateOutgoingConnection(auto &U, size_t SrcId,
                                       size_t DstId,
                                       auto &C, size_t ConnId,
                                       ImprintingLearnerGlobals &G) {
    using namespace plastix;
    if (GetField<UKindTag>(U, DstId) != UK_Output)
      return;
    const float F = GetActivation(U, SrcId);

    float &Z = GetField<ZTag>(C, ConnId);
    float &Zd = GetField<ZDeltaTag>(C, ConnId);
    float &Dw = GetField<DeltaWTag>(C, ConnId);
    float &H = GetField<HTag>(C, ConnId);
    float &HOld = GetField<HOldTag>(C, ConnId);
    float &HTemp = GetField<HTempTag>(C, ConnId);
    float &Beta = GetField<BetaTag>(C, ConnId);
    float &ZBar = GetField<ZBarTag>(C, ConnId);
    float &P = GetField<PTag>(C, ConnId);

    AtomicAddGlobal(G.VDelta, Dw * F);

    float Multiplier = 1.0f;
    if (G.Tau > 0.0f && hp::Eta / G.Tau < 1.0f)
      Multiplier = hp::Eta / G.Tau;
    Zd = Multiplier * std::exp(Beta) * F;

    Z += Zd * (1.0f - G.B);
    P += HOld * F;
    ZBar += Zd * (1.0f - G.B - ZBar * F);
    HTemp = H - HOld * F * (Z - Zd) - H * Zd * F;

    if (G.Tau > hp::Eta) {
      HTemp = 0.0f;
      H = 0.0f;
      HOld = 0.0f;
      ZBar = 0.0f;
      Beta += std::log(hp::Decay) * F * F;
    }
  }
};

struct ImprintingLearnerAddUnit {
  static std::optional<int16_t> AddUnit(auto & /*U*/, size_t /*Id*/,
                                        ImprintingLearnerGlobals &G) {
    if (!G.GenerationLeft || !G.WasActive ||
        G.Tau + hp::AlphaInit > hp::Eta)
      return std::nullopt;
    --G.GenerationLeft;
    G.Tau += hp::AlphaInit;
    return 0;
  }

  static void InitUnit(auto &U, size_t Id, size_t /*Parent*/,
                       ImprintingLearnerGlobals &G) {
    auto Sampled =
        std::bernoulli_distribution{0.5}(G.Rng) ? UK_Pattern : UK_Memory;
    plastix::GetField<UKindTag>(U, Id) = Sampled;
    if (Sampled == UK_Pattern) {
      constexpr float Thresholds[] = {0.6f, 0.7f, 0.8f, 0.9f};
      float Threshold =
          Thresholds[std::uniform_int_distribution<int>{0, 3}(G.Rng)];
      plastix::GetField<ActThreshold>(U, Id) = Threshold;
      plastix::GetActivation(U, Id) = 1.0f;
    } else {
      int D = std::uniform_int_distribution<int>{1, 20}(G.Rng);
      plastix::GetField<Delay>(U, Id) = D;
      auto &P = plastix::GetField<Pulse>(U, Id);
      P = P | 0x1u << D;
    }
  }
};

struct ImprintingLearnerAddConn {
  static bool ShouldAddIncomingConnection(auto &U, size_t Self,
                                          size_t Candidate,
                                          ImprintingLearnerGlobals &G) {
    UnitKind SelfKind = plastix::GetField<UKindTag>(U, Self);
    if (SelfKind == UK_Output)
      return HandleConnsToOutput(U, Candidate);

    UnitFlags &Flags = plastix::GetField<UFlagsTags>(U, Self);
    if (Flags.InputsAdded)
      return false;

    UnitKind CandidateKind = plastix::GetField<UKindTag>(U, Candidate);
    if (CandidateKind == UK_Output)
      return false;

    bool CandidateWasActive = plastix::GetActivation(U, Candidate) != 0;
    bool CandidateWasTenured =
        plastix::GetField<Tenure>(U, Candidate) == TS_Tenured;
    if (!CandidateWasActive || !CandidateWasTenured)
      return false;

    if (SelfKind == UK_Memory) {
      if (std::bernoulli_distribution{0.2}(G.Rng)) {
        Flags.InputsAdded = true;
        return true;
      }
      return false;
    }
    return SelfKind == UK_Pattern;
  }

  static bool HandleConnsToOutput(auto &U, size_t Candidate) {
    UnitFlags &CandidateFlags = plastix::GetField<UFlagsTags>(U, Candidate);
    if (!CandidateFlags.OutputsAdded) {
      CandidateFlags.OutputsAdded = true;
      return true;
    }
    return false;
  }

  static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                          ImprintingLearnerGlobals &) {
    return false;
  }

  static void InitConnection(auto &UA, size_t, size_t To, auto &CA, size_t Conn,
                             ImprintingLearnerGlobals &) {
    UnitKind ToKind = plastix::GetField<UKindTag>(UA, To);
    plastix::GetWeight(CA, Conn) = (ToKind == UK_Output) ? 0.0f : 1.0f;
  }
};

struct ImprintingLearnerResetGlobal {
  static void Reset(auto &G) {
    G.VOld = G.V;
    G.GenerationLeft = hp::MaxGenerationsPerStep;
    G.WasActive = false;
    G.Tau = 0.0f;
  }
};

struct ImprintingLearnerTraits
    : plastix::DefaultNetworkTraits<ImprintingLearnerGlobals> {
  using ForwardPass = ImprintingLearnerForward;
  using Loss = ImprintingLearnerLoss;
  using UpdateUnit = ImprintingLearnerUnitUpdate;
  using UpdateConn = ImprintingLearnerConnUpdate;
  using ResetGlobal = ImprintingLearnerResetGlobal;
  using AddUnit = ImprintingLearnerAddUnit;
  using AddConn = ImprintingLearnerAddConn;
  static constexpr bool KernelizeUpdate = true;  // forward+update on-device
  static constexpr bool KernelizeAdd = false;    // structural host (early-out)
  // Per-unit reduction forward over the reverse-adjacency CSR — avoids the
  // per-edge atomicAdd contention when the output unit has ~N incoming edges.
  static constexpr bool ReverseAdjForward = false;

  using ExtraConnFields = plastix::ConnFieldList<
      plastix::alloc::SOAField<plastix::WeightTag, float>,
      plastix::alloc::SOAField<ZTag, float>,
      plastix::alloc::SOAField<ZDeltaTag, float>,
      plastix::alloc::SOAField<DeltaWTag, float>,
      plastix::alloc::SOAField<HTag, float>,
      plastix::alloc::SOAField<HOldTag, float>,
      plastix::alloc::SOAField<HTempTag, float>,
      plastix::alloc::SOAField<ZBarTag, float>,
      plastix::alloc::SOAField<PTag, float>,
      plastix::alloc::SOAField<BetaTag, float>>;

  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<UKindTag, UnitKind>,
                             plastix::alloc::SOAField<UFlagsTags, UnitFlags>,
                             plastix::alloc::SOAField<Pulse, uint32_t>,
                             plastix::alloc::SOAField<Delay, uint16_t>,
                             plastix::alloc::SOAField<ActThreshold, float>,
                             plastix::alloc::SOAField<Tenure, TenureStatus>,
                             plastix::alloc::SOAField<InDegreeTag, uint32_t>>;

  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
  // 2500 input observations + headroom for generated patterns/memories
  // (gated by Tau < Eta so growth is throttled in practice). Override at
  // compile time for large-scale experiments, e.g. the ~1M-neuron sweep:
  //   -DIL_UNIT_CAPACITY=1100000 -DIL_CONN_CAPACITY=1200000
#ifndef IL_UNIT_CAPACITY
#define IL_UNIT_CAPACITY 16384
#endif
#ifndef IL_CONN_CAPACITY
#define IL_CONN_CAPACITY 524288
#endif
  static constexpr size_t UnitCapacity = IL_UNIT_CAPACITY;
  static constexpr size_t ConnCapacity = IL_CONN_CAPACITY;
};

static_assert(plastix::NetworkTraits<ImprintingLearnerTraits>);

using ImprintingLearner = plastix::Network<ImprintingLearnerTraits>;

// --- harness --------------------------------------------------------------

struct HP {
  // Matches 09_imprintin_learner.py and the raw-C++ default
  // so cross-impl trajectories are directly comparable.
  size_t MaxSteps = 20000;
  size_t LogEvery = 500;
};

static std::filesystem::path
ResolveDataset(const bench::CliArgs &Args) {
  auto It = Args.Extras.find("dataset");
  if (It != Args.Extras.end())
    return It->second;
  std::vector<std::filesystem::path> Candidates = {
      Args.DataDir / "audio_prediction" / "dataset.bin",
      Args.DataDir / "audio" / "dataset.bin",
      Args.DataDir / "dataset.bin",
      "09_imprintin_learner/cpp/examples/output/dataset.bin",
  };
  for (const auto &C : Candidates)
    if (std::filesystem::exists(C))
      return C;
  return {};
}

static std::vector<float> ComputeReturns(const std::vector<int> &Rewards,
                                         float Gamma) {
  std::vector<float> G(Rewards.size(), 0.0f);
  if (Rewards.empty())
    return G;
  G.back() = static_cast<float>(Rewards.back());
  for (ptrdiff_t I = static_cast<ptrdiff_t>(Rewards.size()) - 2; I >= 0; --I)
    G[I] = static_cast<float>(Rewards[I]) + Gamma * G[I + 1];
  return G;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.LogEvery = static_cast<size_t>(Args.GetInt("log-every", H.LogEvery));
  if (Args.Quick) {
    H.MaxSteps = std::min<size_t>(H.MaxSteps, 400);
    H.LogEvery = std::min<size_t>(H.LogEvery, 100);
  }

  bench::MemoryProbe MP;
  MP.Start();

  auto DatasetPath = ResolveDataset(Args);
  if (DatasetPath.empty() || !std::filesystem::exists(DatasetPath)) {
    std::cerr << "[err] audio-prediction dataset.bin not found.\n";
    return 2;
  }

  audio_pred::Dataset DS(DatasetPath);
  size_t N = std::min<size_t>(H.MaxSteps, DS.Size());

  std::cout << "[info] dataset=" << DatasetPath.string() << " ("
            << DS.Size() << " steps, using " << N << ")"
            << " quick=" << (Args.Quick ? 1 : 0) << "\n";

  std::vector<int> Rewards(N);
  for (size_t T = 0; T < N; ++T)
    Rewards[T] = DS[T].Reward();
  auto Returns = ComputeReturns(Rewards, hp::Gamma);
  MP.EndDataset();

  using FC = plastix::FullyConnected<ImprintingLearnerConnInit,
                                     ImprintingLearnerOutputUnitInit>;
  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<UFlagsTags>(U, Id).InputsAdded = true;
    plastix::GetField<UFlagsTags>(U, Id).OutputsAdded = true;
    plastix::GetField<Tenure>(U, Id) = TS_Tenured;
  };

  ImprintingLearner Net(DS.ObservationDim(), InputInit,
                        FC{1, ImprintingLearnerConnInit{},
                           ImprintingLearnerOutputUnitInit{}});

  // Precompute per-unit in-degree (the connection count the forward Apply needs
  // for the pattern threshold) via the reverse-adjacency, and stash it in the
  // InDegree unit field so the on-device forward can read it. Static here (no
  // steady-state growth on this workload); rebuild after structural changes if
  // growth is enabled.
  {
    auto &UA = Net.GetUnitAlloc();
    auto &CA = Net.GetConnAlloc();
    size_t NU = UA.Size();
    std::vector<uint32_t> Offsets(NU + 1), Incoming(CA.Size()), WritePos(NU);
    plastix::BuildReverseAdjacency(CA, NU, Offsets.data(), Incoming.data(),
                                   WritePos.data());
    for (size_t I = 0; I < NU; ++I)
      plastix::GetField<InDegreeTag>(UA, I) = Offsets[I + 1] - Offsets[I];
  }
  MP.EndWeights();

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "audio_imprinting");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_mse");

  // Seed v_old by running a forward pass over the first observation so the
  // first δ isn't contaminated by VOld=0.
  std::vector<float> Features(DS.ObservationDim(), 0.0f);
  {
    audio_pred::StepView S0 = DS[0];
    for (size_t I = 0; I < DS.ObservationDim(); ++I)
      Features[I] = S0.Test(I) ? 1.0f : 0.0f;
    Net.DoForwardPass(Features);
    Net.DoResetGlobalState();
  }

  std::array<float, 1> RewardBuf{0.0f};
  double WindowSse = 0.0;
  size_t WindowCnt = 0;
  size_t EpochIdx = 0;
  double GlobalSse = 0.0;
  std::vector<float> Predictions(N, 0.0f);

  // Phase-level timing via the shared PhaseTimer. `backward` here is just
  // DoBackwardPass; the per-edge weight update lives in `update`
  // (DoUpdateUnitState + DoUpdateConnectionState). `structural` bundles
  // Prune+Add (units + conns), `reset` is DoResetGlobalState.
  bench::PhaseTimer Timer;

  // Fine-grained structural breakdown (each plastix Do* syncs the device, so
  // wall time around each call is a valid per-op measurement). Diagnostic for
  // the 09 "why is structural so dominant" analysis.
  double PruneUnitsNs = 0, PruneConnsNs = 0, AddUnitsNs = 0, AddConnsNs = 0;
  auto DurNs = [](auto A, auto B) {
    return std::chrono::duration<double, std::nano>(B - A).count();
  };

  auto T0 = std::chrono::steady_clock::now();
  for (size_t T = 0; T < N; ++T) {
    audio_pred::StepView Step = DS[T];
    for (size_t I = 0; I < DS.ObservationDim(); ++I)
      Features[I] = Step.Test(I) ? 1.0f : 0.0f;
    RewardBuf[0] = static_cast<float>(Step.Reward());

    Timer.Tick();
    Net.DoForwardPass(Features);
    Timer.MarkForward();
    Net.DoCalculateLoss(RewardBuf);
    Timer.MarkLoss();
    Net.DoBackwardPass();
    Timer.MarkBackward();
    Net.DoUpdateUnitState();
    Net.DoUpdateConnectionState();
    Timer.MarkUpdate();
    auto Sa = std::chrono::steady_clock::now();
    Net.DoPruneUnits();
    auto Sb = std::chrono::steady_clock::now();
    Net.DoPruneConnections();
    auto Sc = std::chrono::steady_clock::now();
    Timer.MarkPrune();
    Net.DoAddUnits();
    auto Sd = std::chrono::steady_clock::now();
    Net.DoAddConnections();
    auto Se = std::chrono::steady_clock::now();
    PruneUnitsNs += DurNs(Sa, Sb);
    PruneConnsNs += DurNs(Sb, Sc);
    AddUnitsNs += DurNs(Sc, Sd);
    AddConnsNs += DurNs(Sd, Se);
    Timer.MarkGrow();
    Net.DoResetGlobalState();
    Timer.MarkReset();
    Timer.StepDone();

    float V = Net.GetOutput()[0];
    Predictions[T] = V;

    double Err = static_cast<double>(V) - static_cast<double>(Returns[T]);
    WindowSse += Err * Err;
    GlobalSse += Err * Err;
    ++WindowCnt;

    if (WindowCnt >= H.LogEvery || T + 1 == N) {
      double WindowMse = WindowCnt > 0 ? WindowSse / WindowCnt : 0.0;
      ++EpochIdx;
      auto Ed = bench::LiveEdgeSet(Net.GetConnAlloc());
      Log.Log(EpochIdx, Net.GetUnitAlloc().Size(),
              bench::LiveEdgeCount(Net.GetConnAlloc()), &Ed, &WindowMse,
              {{"train_loss", WindowMse},
               {"epoch", static_cast<double>(EpochIdx)},
               {"step", static_cast<double>(T + 1)},
               {"test_mse", WindowMse}});
      TestCsv.Add(EpochIdx, WindowMse, Net.GetUnitAlloc().Size(),
                  bench::LiveEdgeCount(Net.GetConnAlloc()));
      std::cout << "[ep " << EpochIdx << "] step=" << (T + 1)
                << " window_mse=" << WindowMse
                << " units=" << Net.GetUnitAlloc().Size()
                << " edges=" << bench::LiveEdgeCount(Net.GetConnAlloc())
                << "\n";
      WindowSse = 0.0;
      WindowCnt = 0;
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  size_t TailStart = N - std::max<size_t>(1, N / 10);
  double TailSse = 0.0;
  for (size_t T = TailStart; T < N; ++T) {
    double E = static_cast<double>(Predictions[T]) -
               static_cast<double>(Returns[T]);
    TailSse += E * E;
  }
  double TestMse = TailSse / std::max<size_t>(1, N - TailStart);
  double FullMse = N > 0 ? GlobalSse / static_cast<double>(N) : 0.0;

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "audio_imprinting"));

  bench::SummaryWriter S;
  S.Set("workload", std::string{"09_imprintin_learner"});
  S.Set("dataset", std::string{"audio_prediction"});
  S.Set("max_steps", static_cast<int>(N));
  S.Set("gamma", static_cast<double>(hp::Gamma));
  S.Set("alpha_init", static_cast<double>(hp::AlphaInit));
  S.Set("eta", static_cast<double>(hp::Eta));
  S.Set("wall_seconds", Wall);
  S.Set("val_mse_final", FullMse);
  S.Set("test_mse", TestMse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<int>(Net.GetUnitAlloc().Size()));
  S.Set("n_edges",
        static_cast<int>(bench::LiveEdgeCount(Net.GetConnAlloc())));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TestMse
            << "  full_mse=" << FullMse
            << "  units=" << Net.GetUnitAlloc().Size()
            << "  edges=" << bench::LiveEdgeCount(Net.GetConnAlloc()) << "\n";
  std::cout << "[phase] step_count=" << Timer.StepCount()
            << " (see summary csv for per-phase mean/std)\n";
  {
    double S = static_cast<double>(N > 0 ? N : 1);
    std::cout << "[phase-detail] structural ns/step  "
              << "prune_units=" << (PruneUnitsNs / S)
              << "  prune_conns=" << (PruneConnsNs / S)
              << "  add_units=" << (AddUnitsNs / S)
              << "  add_conns=" << (AddConnsNs / S) << "\n";
  }
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
