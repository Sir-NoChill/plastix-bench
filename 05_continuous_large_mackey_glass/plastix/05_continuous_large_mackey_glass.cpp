// Workload 5 / 5 — CONTINUOUS-LARGE regime, Plastix translation.
//
// Mirrors 05_continuous_large_mackey_glass.py. The Python
// version trains a sparse "reservoir-like" recurrent network on the
// Mackey-Glass chaotic series with three sources of structural change:
//   1. Heavy-tailed per-step unit grows/shrinks drawn from a Pareto.
//   2. Watts-Strogatz-style rewires of a random fraction of edges every
//      R steps.
//   3. Growth-momentum bursts triggered by an accumulator over recent
//      gradient norms.
//
// Plastix does not currently support real recurrence (Topological mode
// requires a DAG), so the C++ translation runs a plain feedforward MLP
// (input → hidden → output) with the *same structural mutation profile*
// applied to the input → hidden connections. The regime signature that
// the workload exists to validate — heavy-tailed |Δn_units|, low Jaccard
// between consecutive live-edge sets — is preserved.
//
// Plastix policy mapping (matches the bottom-of-file comment in the
// Python source):
//
//   ForwardPass        custom    Tanh hidden, linear output
//   BackwardPass       custom    backprop through Tanh
//   Loss               MSELoss
//   UpdateConn         custom    plain SGD on WeightTag
//   PruneUnit          custom    fires while UnitsToKill > 0 (shrink draw)
//   PruneConn          custom    fires while EdgeKillBudget > 0 (rewire cut)
//   AddUnit            custom    fires while UnitsToAdd > 0 (grow draw)
//   AddConn            custom    wires new units + adds random rewire edges
//   ResetGlobal        NoX
//
// All structural arming happens via static members on each policy struct
// because Network owns its GlobalState by value with no public setter;
// see the README for the trade-off.

#include "plastix/common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <random>

namespace {

struct HP {
  // Parameters mirror 05_continuous_large_mackey_glass.py and
  // the raw-C++ translation so cross-impl plots compare like-for-like.
  size_t SeriesLen = 5000;
  size_t Tau = 17;
  size_t InLen = 32;
  size_t Horizon = 1;
  size_t InitHidden = 128;
  size_t MinHidden = 16;
  size_t MaxHidden = 512;
  size_t MaxSteps = 1000;
  size_t ValEvery = 25;
  size_t RewireEvery = 3;
  float RewireFrac = 0.25f;
  float ParetoAlpha = 1.5f;
  size_t MaxDeltaPerStep = 20;
  float Lr = 5e-3f;
};

struct PreActTag {};
struct GradPreActTag {};
struct IsOutputTag {};
struct IsNewTag {};

struct Forward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Z) {
    plastix::GetField<PreActTag>(U, Id) = Z;
    plastix::GetActivation(U, Id) = plastix::GetField<IsOutputTag>(U, Id)
                                        ? plastix::math::Linear(Z)
                                        : plastix::math::Tanh(Z);
  }
};

struct Backward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t ToId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) *
           plastix::GetField<GradPreActTag>(U, ToId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    bool IsOut = plastix::GetField<IsOutputTag>(U, Id);
    float DLDA = IsOut ? plastix::GetBackwardAcc(U, Id) : Accumulated;
    float Z = plastix::GetField<PreActTag>(U, Id);
    float DPhiDz = IsOut ? plastix::math::LinearGrad(Z)
                         : plastix::math::TanhGradFromPreact(Z);
    plastix::GetField<GradPreActTag>(U, Id) = DLDA * DPhiDz;
  }
};

// Runtime-tunable / host-staged parameters and per-burst budgets read (and
// decremented) by the policies. Held in the network's GlobalState (managed
// memory) — set from the host via Net::Global() — because device code cannot
// read host-side static members. The structural phases run on the host here
// (Kernelize*=false), so the budget decrements over G are well-defined.
struct MgGlobals {
  float Lr = 5e-3f;
  uint16_t HiddenLevel = 1;
  int ShrinkBudget = 0;          // PruneUnit
  int RewireBudget = 0;          // PruneConn
  uint64_t RewireSeed = 0;       // PruneConn
  float RewireBias = 0.0f;       // PruneConn sample rate per call
  int AddBudget = 0;             // MgAddUnit
  uint64_t AddConnSeed = 0;      // MgAddConn
  bool RewireAddEnabled = false; // MgAddConn
  float RewireDensity = 0.0f;    // MgAddConn
};

struct UpdateConn {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    float Grad = plastix::GetField<GradPreActTag>(U, DstId);
    float A = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= G.Lr * Grad * A;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

// PruneUnit — fires while ShrinkBudget > 0. Filters to hidden-level units
// only. Decrements the static budget on each fire.
struct PruneUnit {
  PLASTIX_HD static bool ShouldPrune(auto &UA, size_t Id, auto &G) {
    if (G.ShrinkBudget <= 0)
      return false;
    if (plastix::GetLevel(UA, Id) != G.HiddenLevel)
      return false;
    --G.ShrinkBudget;
    return true;
  }
};

// PruneConn — used for rewires. Fires while RewireBudget > 0 with a
// per-conn coin flip seeded by ConnId so the result is deterministic but
// scattered across the connection space.
struct PruneConn {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &,
                                     size_t ConnId, auto &G) {
    if (G.RewireBudget <= 0)
      return false;
    if (plastix::Bernoulli(G.RewireSeed, static_cast<uint64_t>(ConnId),
                           G.RewireBias)) {
      --G.RewireBudget;
      return true;
    }
    return false;
  }
};

// MgAddUnit — same single-shot growth pattern as workload 04, but with a
// multi-fire budget instead of one-at-a-time. Heavy-tail Pareto magnitudes
// from the host become a one-step burst of variable size.
struct MgAddUnit {
  PLASTIX_HD static std::optional<int16_t> AddUnit(auto &UA, size_t ParentId,
                                                    auto &G) {
    if (G.AddBudget <= 0)
      return std::nullopt;
    if (plastix::GetLevel(UA, ParentId) != G.HiddenLevel)
      return std::nullopt;
    --G.AddBudget;
    return int16_t{0};
  }
  PLASTIX_HD static void InitUnit(auto &UA, size_t NewId, size_t /*Parent*/,
                                  auto &) {
    plastix::GetActivation(UA, NewId) = 0.0f;
    plastix::GetField<PreActTag>(UA, NewId) = 0.0f;
    plastix::GetField<GradPreActTag>(UA, NewId) = 0.0f;
    plastix::GetField<IsOutputTag>(UA, NewId) = false;
    plastix::GetField<IsNewTag>(UA, NewId) = true;
  }
};

// MgAddConn — two simultaneous duties:
//   - Wire every freshly-added hidden unit to all inputs and the output
//     (same pattern as workload 04).
//   - When RewireAddEnabled is true, also propose a small number of new
//     random input -> hidden edges to compensate for the rewire's prune.
struct MgAddConn {
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &UA, size_t SelfId,
                                                     size_t CandidateId,
                                                     auto &G) {
    bool SelfNew = plastix::GetField<IsNewTag>(UA, SelfId);
    bool SelfOut = plastix::GetField<IsOutputTag>(UA, SelfId);
    bool CandNew = plastix::GetField<IsNewTag>(UA, CandidateId);
    if (SelfNew && plastix::GetLevel(UA, CandidateId) <
                       plastix::GetLevel(UA, SelfId))
      return true;
    if (SelfOut && CandNew)
      return true;
    if (G.RewireAddEnabled &&
        plastix::GetLevel(UA, CandidateId) <
            plastix::GetLevel(UA, SelfId) &&
        plastix::GetLevel(UA, SelfId) == G.HiddenLevel) {
      uint64_t Counter = (static_cast<uint64_t>(SelfId) << 32) |
                         static_cast<uint64_t>(CandidateId);
      return plastix::Bernoulli(G.AddConnSeed ^ 0xDEADBEEFull, Counter,
                                G.RewireDensity);
    }
    return false;
  }
  PLASTIX_HD static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static void InitConnection(auto &, size_t /*From*/, size_t /*To*/,
                                        auto &CA, size_t ConnId, auto &G) {
    uint64_t Counter = static_cast<uint64_t>(ConnId);
    plastix::GetWeight(CA, ConnId) =
        plastix::UniformReal(G.AddConnSeed, Counter, -0.05f, 0.05f);
  }
};

struct MgTraits : plastix::DefaultNetworkTraits<> {
  using GlobalState = MgGlobals;
  using ForwardPass = Forward;
  using BackwardPass = Backward;
  using Loss = plastix::MSELoss;
  using UpdateConn = ::UpdateConn;
  using PruneUnit = ::PruneUnit;
  using PruneConn = ::PruneConn;
  using AddUnit = ::MgAddUnit;
  using AddConn = ::MgAddConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<PreActTag, float>,
      plastix::alloc::SOAField<GradPreActTag, float>,
      plastix::alloc::SOAField<IsOutputTag, bool>,
      plastix::alloc::SOAField<IsNewTag, bool>>;
  static constexpr uint16_t Neighbourhood = 1;
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizeAdd = false;
  static constexpr bool KernelizePrune = false;
  // Capacity for the largest configuration (InLen=32, MaxHidden=512). Worst
  // case: ~32*512 input + 512*1 output + recurrent rewires + structural slack.
  // 4096 units is enough for in+hidden+out; bump conns above the default
  // 16384 cap to leave headroom for the dense-recurrent burst case.
  static constexpr size_t UnitCapacity = 4096;
  static constexpr size_t ConnCapacity = 65536;
};
static_assert(plastix::NetworkTraits<MgTraits>);

using Net = plastix::Network<MgTraits>;

struct UniformInit {
  uint64_t Seed;
  float Limit;
  void operator()(auto &CA, auto Id) const {
    plastix::GetWeight(CA, Id) =
        plastix::UniformReal(Seed, static_cast<uint64_t>(Id), -Limit, Limit);
  }
};
struct MarkOutput {
  void operator()(auto &UA, auto Id) const {
    plastix::GetField<IsOutputTag>(UA, Id) = true;
  }
};

using FCHidden = plastix::FullyConnected<UniformInit>;
using FCOut = plastix::FullyConnected<UniformInit, MarkOutput>;

// --- Mackey-Glass series --------------------------------------------------

static std::vector<float> MackeyGlass(size_t N, size_t Tau, uint32_t Seed) {
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> Noise(0.0f, 0.01f);
  std::vector<double> Buf(Tau * 2 + 16, 1.2);
  for (auto &V : Buf)
    V += Noise(Rng);
  std::vector<float> Out(N);
  double Cur = Buf.back();
  double H = 0.1;
  double Beta = 0.2, Gamma = 0.1;
  size_t BurnSteps = Tau * 20;
  for (size_t I = 0; I < BurnSteps + N; ++I) {
    double Delayed =
        Buf[Buf.size() >= Tau ? Buf.size() - Tau : 0];
    double Dx = Beta * Delayed / (1.0 + std::pow(Delayed, 10.0)) - Gamma * Cur;
    Cur += H * Dx;
    Buf.push_back(Cur);
    if (Buf.size() > Tau * 2 + 16)
      Buf.erase(Buf.begin());
    if (I >= BurnSteps)
      Out[I - BurnSteps] = static_cast<float>(Cur);
  }
  return Out;
}

static void Windowed(const std::vector<float> &S, size_t InLen, size_t Horizon,
                     std::vector<std::vector<float>> &X,
                     std::vector<float> &Y) {
  if (S.size() < InLen + Horizon)
    return;
  size_t N = S.size() - InLen - Horizon + 1;
  X.assign(N, std::vector<float>(InLen));
  Y.assign(N, 0.0f);
  for (size_t I = 0; I < N; ++I) {
    for (size_t J = 0; J < InLen; ++J)
      X[I][J] = S[I + J];
    Y[I] = S[I + InLen + Horizon - 1];
  }
}

static void Standardise1D(std::vector<float> &S, size_t Cut) {
  double M = 0.0, V = 0.0;
  size_t N = std::min(Cut, S.size());
  for (size_t I = 0; I < N; ++I)
    M += S[I];
  M /= N;
  for (size_t I = 0; I < N; ++I)
    V += (S[I] - M) * (S[I] - M);
  double Sd = std::sqrt(V / N) + 1e-6;
  for (auto &X : S)
    X = static_cast<float>((X - M) / Sd);
}

static double EvalMse(Net &N,
                      const std::vector<std::vector<float>> &X,
                      const std::vector<float> &Y, size_t Begin, size_t End) {
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = Begin; I < End && I < X.size(); ++I) {
    N.DoForwardPass(X[I]);
    auto Out = N.GetOutput();
    double D = static_cast<double>(Out[0]) - static_cast<double>(Y[I]);
    Sum += D * D;
    ++Cnt;
  }
  return Cnt ? Sum / Cnt : 0.0;
}

static void ClearIsNew(Net &N) {
  auto &UA = N.GetUnitAlloc();
  for (size_t I = 0; I < UA.Size(); ++I)
    plastix::GetField<IsNewTag>(UA, I) = false;
}

static size_t CountHidden(Net &N) {
  size_t H = 0;
  auto &UA = N.GetUnitAlloc();
  for (size_t I = 0; I < UA.Size(); ++I)
    if (plastix::GetLevel(UA, I) == N.Global().HiddenLevel &&
        !plastix::GetField<plastix::PrunedTag>(UA, I))
      ++H;
  return H;
}

// Sample magnitude ceil(1 + Pareto(alpha)), capped at MaxDelta. Pareto from
// inverse-CDF on a uniform sample; alpha smaller = heavier tail.
static int ParetoMag(std::mt19937 &Rng, float Alpha, int MaxDelta) {
  std::uniform_real_distribution<float> U(0.0f, 1.0f);
  float V = U(Rng);
  if (V < 1e-6f)
    V = 1e-6f;
  float Pareto = std::pow(1.0f / V, 1.0f / Alpha) - 1.0f;
  int Mag = static_cast<int>(std::ceil(Pareto));
  if (Mag < 1)
    Mag = 1;
  if (Mag > MaxDelta)
    Mag = MaxDelta;
  return Mag;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  HP H;
  H.SeriesLen =
      static_cast<size_t>(Args.GetInt("series-len", H.SeriesLen));
  H.Tau = static_cast<size_t>(Args.GetInt("tau", H.Tau));
  H.InLen = static_cast<size_t>(Args.GetInt("in-len", H.InLen));
  H.Horizon = static_cast<size_t>(Args.GetInt("horizon", H.Horizon));
  H.InitHidden =
      static_cast<size_t>(Args.GetInt("init-hidden", H.InitHidden));
  H.MinHidden =
      static_cast<size_t>(Args.GetInt("min-hidden", H.MinHidden));
  H.MaxHidden =
      static_cast<size_t>(Args.GetInt("max-hidden", H.MaxHidden));
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.RewireEvery =
      static_cast<size_t>(Args.GetInt("rewire-every", H.RewireEvery));
  H.RewireFrac = Args.GetFloat("rewire-frac", H.RewireFrac);
  H.ParetoAlpha = Args.GetFloat("pareto-alpha", H.ParetoAlpha);
  H.MaxDeltaPerStep = static_cast<size_t>(
      Args.GetInt("max-delta-per-step",
                  static_cast<int>(H.MaxDeltaPerStep)));
  H.Lr = Args.GetFloat("lr", H.Lr);
  if (Args.Quick)
    H.MaxSteps = std::max<size_t>(200, H.MaxSteps / 5);

  auto Series =
      MackeyGlass(H.SeriesLen, H.Tau, static_cast<uint32_t>(Args.Seed));
  Standardise1D(Series, Series.size() / 2);
  std::vector<std::vector<float>> X;
  std::vector<float> Y;
  Windowed(Series, H.InLen, H.Horizon, X, Y);
  size_t NTr = static_cast<size_t>(0.7f * X.size());
  size_t NVa = static_cast<size_t>(0.15f * X.size());
  std::cout << "[info] series=" << Series.size() << " N=" << X.size()
            << " train=" << NTr << " val=" << NVa << " init_h=" << H.InitHidden
            << " max_steps=" << H.MaxSteps << "\n";

  float Limit = std::sqrt(6.0f / static_cast<float>(H.InLen + H.InitHidden));
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 31ull;
  auto N = std::unique_ptr<Net>(new Net(
      H.InLen, FCHidden{H.InitHidden, UniformInit{SeedBase + 1, Limit}},
      FCOut{1, UniformInit{SeedBase + 2, Limit}, MarkOutput{}}));
  // Stage runtime params into the managed GlobalState (read by the policies).
  N->Global().Lr = H.Lr;
  N->Global().AddConnSeed = static_cast<uint64_t>(Args.Seed) * 1000ull + 29ull;
  N->Global().RewireSeed = N->Global().AddConnSeed ^ 0xC0FFEEull;
  N->Global().RewireDensity = H.RewireFrac;

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::uniform_real_distribution<float> Coin(0.0f, 1.0f);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "continuous_large_mg");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_mse");
  auto Edges = bench::LiveEdgeSet(N->GetConnAlloc());
  double InitVal = EvalMse(*N, X, Y, NTr, NTr + NVa);
  double InitTest = EvalMse(*N, X, Y, NTr + NVa, X.size());
  Log.Log(0, N->GetUnitAlloc().Size(),
          bench::LiveEdgeCount(N->GetConnAlloc()), &Edges, &InitVal,
          {{"hidden", static_cast<double>(H.InitHidden)},
           {"test_mse", InitTest}});
  TestCsv.Add(0, InitTest, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()));

  size_t Grows = 0, Shrinks = 0, Rewires = 0;
  std::vector<int> DeltaUnits;
  bench::PhaseTimer Timer;

  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    ClearIsNew(*N);
    size_t HBefore = CountHidden(*N);

    // Heavy-tailed delta draw. Sign +/- chosen uniformly; magnitude is
    // ceil(Pareto). The grow / shrink direction sets which budget is armed.
    int Mag = ParetoMag(Rng, H.ParetoAlpha,
                        static_cast<int>(H.MaxDeltaPerStep));
    bool GrowDir = Coin(Rng) > 0.5f;
    if (GrowDir && HBefore + Mag <= H.MaxHidden) {
      N->Global().AddBudget = Mag;
      N->Global().ShrinkBudget = 0;
    } else if (!GrowDir && HBefore > H.MinHidden + Mag) {
      N->Global().AddBudget = 0;
      N->Global().ShrinkBudget = Mag;
    } else {
      N->Global().AddBudget = 0;
      N->Global().ShrinkBudget = 0;
    }

    // Rewires fire every `rewire_every` steps.
    bool DoRewire = (Step % H.RewireEvery) == 0;
    if (DoRewire) {
      size_t Alive = bench::LiveEdgeCount(N->GetConnAlloc());
      int Budget =
          std::max<int>(1, static_cast<int>(H.RewireFrac * Alive));
      N->Global().RewireBudget = Budget;
      N->Global().RewireBias = H.RewireFrac;
      N->Global().RewireAddEnabled = true;
      ++Rewires;
    } else {
      N->Global().RewireBudget = 0;
      N->Global().RewireBias = 0.0f;
      N->Global().RewireAddEnabled = false;
    }

    size_t I = (Step - 1) % NTr;
    float Ytgt[1] = {Y[I]};

    size_t UnitsBefore = N->GetUnitAlloc().Size();

    Timer.Tick();
    N->DoForwardPass(X[I]);
    Timer.MarkForward();
    N->DoCalculateLoss(Ytgt);
    Timer.MarkLoss();
    N->DoBackwardPass();
    Timer.MarkBackward();
    N->DoUpdateUnitState();
    N->DoUpdateConnectionState();
    Timer.MarkUpdate();
    N->DoPruneUnits();
    N->DoPruneConnections();
    N->DoAddUnits();
    N->DoAddConnections();
    Timer.MarkStructural();
    N->DoResetGlobalState();
    Timer.MarkReset();
    Timer.StepDone();

    N->Global().ShrinkBudget = 0;
    N->Global().RewireBudget = 0;
    N->Global().AddBudget = 0;
    N->Global().RewireAddEnabled = false;

    size_t HAfter = CountHidden(*N);
    int DU = static_cast<int>(HAfter) - static_cast<int>(HBefore);
    DeltaUnits.push_back(DU);
    if (DU > 0)
      ++Grows;
    if (DU < 0)
      ++Shrinks;

    (void)UnitsBefore;

    if (Step % H.ValEvery == 0 || Step == H.MaxSteps) {
      double VL = EvalMse(*N, X, Y, NTr, NTr + NVa);
      double TestMseStep = EvalMse(*N, X, Y, NTr + NVa, X.size());
      auto Ed = bench::LiveEdgeSet(N->GetConnAlloc());
      Log.Log(Step, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()), &Ed, &VL,
              {{"hidden", static_cast<double>(HAfter)},
               {"grows", static_cast<double>(Grows)},
               {"shrinks", static_cast<double>(Shrinks)},
               {"rewires", static_cast<double>(Rewires)},
               {"delta_units", static_cast<double>(DU)},
               {"test_mse", TestMseStep}});
      TestCsv.Add(Step, TestMseStep, N->GetUnitAlloc().Size(),
                  bench::LiveEdgeCount(N->GetConnAlloc()));
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();
  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "continuous_large_mg"));

  double TestMse = EvalMse(*N, X, Y, NTr + NVa, X.size());

  bench::SummaryWriter S;
  S.Set("workload", std::string{"05_continuous_large_mg"});
  S.Set("dataset", std::string{"mackey-glass"});
  S.Set("tau", static_cast<int>(H.Tau));
  S.Set("in_len", static_cast<int>(H.InLen));
  S.Set("horizon", static_cast<int>(H.Horizon));
  S.Set("init_hidden", static_cast<int>(H.InitHidden));
  S.Set("pareto_alpha", static_cast<double>(H.ParetoAlpha));
  S.Set("rewire_every", static_cast<int>(H.RewireEvery));
  S.Set("rewire_frac", static_cast<double>(H.RewireFrac));
  S.Set("max_steps", static_cast<int>(H.MaxSteps));
  S.Set("wall_seconds", Wall);
  S.Set("grows", static_cast<int>(Grows));
  S.Set("shrinks", static_cast<int>(Shrinks));
  S.Set("rewires", static_cast<int>(Rewires));
  S.Set("hidden_final", static_cast<int>(CountHidden(*N)));
  S.Set("edges_final",
        static_cast<int>(bench::LiveEdgeCount(N->GetConnAlloc())));
  S.Set("val_loss_initial", Log.Records().front().ValLoss);
  S.Set("val_loss_final", Log.Records().back().ValLoss);
  S.Set("test_mse", TestMse);

  std::vector<int> AbsDu;
  AbsDu.reserve(DeltaUnits.size());
  int MaxAbsDu = 0;
  for (int V : DeltaUnits) {
    int Av = V < 0 ? -V : V;
    AbsDu.push_back(Av);
    if (Av > MaxAbsDu)
      MaxAbsDu = Av;
  }
  auto Percentile = [&](double P) {
    if (AbsDu.empty())
      return 0;
    auto Tmp = AbsDu;
    size_t K = static_cast<size_t>(P * Tmp.size());
    if (K >= Tmp.size())
      K = Tmp.size() - 1;
    std::nth_element(Tmp.begin(), Tmp.begin() + K, Tmp.end());
    return Tmp[K];
  };
  S.Set("delta_units_p50_abs", Percentile(0.50));
  S.Set("delta_units_p95_abs", Percentile(0.95));
  S.Set("delta_units_max_abs", MaxAbsDu);

  float JMin = 1.0f, JSum = 0.0f;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JSum += R.Jaccard;
  }
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_mean",
        static_cast<double>(JSum) /
            std::max<size_t>(Log.Records().size(), 1));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s grows=" << Grows
            << " shrinks=" << Shrinks << " rewires=" << Rewires
            << " hidden_final=" << CountHidden(*N) << " test_mse=" << TestMse
            << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
