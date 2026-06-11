// Workload 3 / 5 — BURSTY (punctuated equilibrium) regime, Plastix translation.
//
// Mirrors 03_bursty_elec2.py. Streaming classifier that sits
// structurally idle for many training steps; when a sliding window of
// validation losses goes flat, the host stages a "burst" that adds several
// hidden units and their attendant connections, immediately followed by a
// magnitude-based prune sweep that trims the smallest weights to keep total
// size bounded. The visible regime: long flat stretches in unit_count
// interrupted by sharp jumps, with matching small dips in val loss.
//
// Plastix policy mapping (matches the comment block at the bottom of the
// Python source):
//
//   ForwardPass        custom    ReLU hidden, linear output (logits)
//   BackwardPass       custom    backprop through ReLU
//   Loss               SoftmaxCrossEntropyLoss
//   UpdateUnit         custom    clears the per-unit IsNew flag after a burst
//   UpdateConn         custom    plain SGD on WeightTag
//   PruneConn          custom    ShouldPrune := Armed && |w| <= Threshold
//   AddUnit            custom    fires while UnitsRemaining > 0 (host-armed)
//   AddConn            custom    wires new units to input + output layers
//   ResetGlobal        NoX
//
// "Arming" pattern: Network owns its GlobalState by value with no public
// getter. To stage a burst from the host we therefore route the burst
// parameters through static members on each policy struct — the same
// trick traditional-plastix/02_idempotent_imp.cpp uses for its prune
// threshold. The static lives entirely inside the benchmark .cpp; no
// framework change is required.

#include "common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <random>

namespace {

struct HP {
  // Sizing matches the PyTorch reference (03_bursty_elec2.py):
  // 3000 steps, init_hidden=32, plateau rel-tol 3%, burst widens BOTH
  // hidden layers by 15%, post-burst prune trims 10% of alive weights.
  size_t MaxSteps = 3000;
  size_t InitHidden = 32;
  size_t InDim = 8; // Elec2 actually has 6 features; overridden after load
  size_t NumClasses = 2;
  size_t ValEvery = 25;
  size_t PlateauWindow = 8;
  size_t PlateauCooldown = 4;
  float PlateauRelTol = 0.03f;
  float BurstFrac = 0.15f;
  float PostBurstPruneFrac = 0.10f;
  float Lr = 1e-3f;
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
                                        : plastix::math::ReLU(Z);
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
                         : plastix::math::ReLUGradFromPreact(Z);
    plastix::GetField<GradPreActTag>(U, Id) = DLDA * DPhiDz;
  }
};

struct UpdateConn {
  static float Lr;
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &) {
    float Grad = plastix::GetField<GradPreActTag>(U, DstId);
    float A = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= Lr * Grad * A;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};
float UpdateConn::Lr = 5e-3f;

// UpdateUnit runs before AddUnit in DoStep ordering, so clearing IsNew here
// only affects units that were added in *previous* steps — units inserted
// later in *this* step (by AddUnit::InitUnit) still see IsNew=true when
// AddConn queries them seconds later.
struct UpdateUnit {
  PLASTIX_HD static void Update(auto &UA, size_t Id, auto &) {
    plastix::GetField<IsNewTag>(UA, Id) = false;
  }
};

struct PruneConn {
  static bool Armed;
  static float Threshold;
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &C,
                                     size_t ConnId, auto &) {
    if (!Armed)
      return false;
    return std::abs(plastix::GetWeight(C, ConnId)) <= Threshold;
  }
};
bool PruneConn::Armed = false;
float PruneConn::Threshold = 0.0f;

// BurstAddUnit — fires `BudgetL1` times for parents at hidden level 1 and
// `BudgetL2` times for parents at hidden level 2, matching the PyTorch
// reference's behaviour of widening BOTH hidden layers by the same `n_new`
// on each burst. Returns offset 0 to keep the new unit at its parent's
// level. The struct has a different name from its `AddUnit` member
// function so the compiler doesn't read `AddUnit::AddUnit` as a
// constructor.
struct BurstAddUnit {
  static int BudgetL1;
  static int BudgetL2;
  static uint16_t HiddenL1;
  static uint16_t HiddenL2;
  PLASTIX_HD static std::optional<int16_t> AddUnit(auto &UA, size_t ParentId,
                                                    auto &) {
    uint16_t Lvl = plastix::GetLevel(UA, ParentId);
    if (Lvl == HiddenL1 && BudgetL1 > 0) {
      --BudgetL1;
      return int16_t{0};
    }
    if (Lvl == HiddenL2 && BudgetL2 > 0) {
      --BudgetL2;
      return int16_t{0};
    }
    return std::nullopt;
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
int BurstAddUnit::BudgetL1 = 0;
int BurstAddUnit::BudgetL2 = 0;
uint16_t BurstAddUnit::HiddenL1 = 1;
uint16_t BurstAddUnit::HiddenL2 = 2;

// BurstAddConn — wires every freshly-allocated unit fully into its
// adjacent levels (matches PyTorch's `grow()` which widens layers densely).
// Approves any forward edge (CandLvl < SelfLvl) where at least one
// endpoint carries IsNewTag. New units therefore gain edges to/from every
// neighbour on both sides in a single burst step.
struct BurstAddConn {
  static uint64_t Seed;
  static bool Enabled;
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &UA, size_t SelfId,
                                                     size_t CandidateId,
                                                     auto &) {
    if (!Enabled)
      return false;
    uint16_t SelfLvl = plastix::GetLevel(UA, SelfId);
    uint16_t CandLvl = plastix::GetLevel(UA, CandidateId);
    if (CandLvl >= SelfLvl)
      return false; // only propose forward edges
    bool SelfNew = plastix::GetField<IsNewTag>(UA, SelfId);
    bool CandNew = plastix::GetField<IsNewTag>(UA, CandidateId);
    return SelfNew || CandNew;
  }
  PLASTIX_HD static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static void InitConnection(auto &, size_t /*From*/, size_t /*To*/,
                                        auto &CA, size_t ConnId, auto &) {
    uint64_t Counter = static_cast<uint64_t>(ConnId);
    // Small Gaussian-ish init via uniform[-0.05, 0.05], mirroring PyTorch's
    // `noise=0.05 * randn` magnitude.
    plastix::GetWeight(CA, ConnId) =
        plastix::UniformReal(Seed ^ 0xC3C3ull, Counter, -0.05f, 0.05f);
  }
};
uint64_t BurstAddConn::Seed = 0;
bool BurstAddConn::Enabled = false;

struct BurstyTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = Forward;
  using BackwardPass = Backward;
  using Loss = plastix::SoftmaxCrossEntropyLoss;
  using UpdateUnit = ::UpdateUnit;
  using UpdateConn = ::UpdateConn;
  using PruneConn = ::PruneConn;
  using AddUnit = ::BurstAddUnit;
  using AddConn = ::BurstAddConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<PreActTag, float>,
      plastix::alloc::SOAField<GradPreActTag, float>,
      plastix::alloc::SOAField<IsOutputTag, bool>,
      plastix::alloc::SOAField<IsNewTag, bool>>;
  static constexpr uint16_t Neighbourhood = 1;
  // Headroom for a two-hidden-layer net that grows by ~15% per burst over
  // 30 bursts. Initial: 6 input + 32 + 32 + 2 = 72 units, plus growth to
  // ~80 per hidden layer caps at ~200 total. Edges scale to ~10k.
  static constexpr size_t UnitCapacity = 1024;
  static constexpr size_t ConnCapacity = 65536;
  // Burst arming lives in non-atomic statics — keep on host.
  static constexpr bool KernelizeAdd = false;
  static constexpr bool KernelizeUpdate = false;
};
static_assert(plastix::NetworkTraits<BurstyTraits>);

using Net = plastix::Network<BurstyTraits>;

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

// --- data --------------------------------------------------------------------

static void SynthDrift(size_t N, size_t Dim, uint32_t Seed,
                       std::vector<std::vector<float>> &X,
                       std::vector<int> &Y) {
  constexpr float Pi = 3.14159265358979323846f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> G(0.0f, 1.0f);
  X.assign(N, std::vector<float>(Dim));
  Y.assign(N, 0);
  for (size_t I = 0; I < N; ++I) {
    float Tt = 4.0f * Pi * static_cast<float>(I) / N;
    float Dot = 0.0f;
    for (size_t J = 0; J < Dim; ++J) {
      X[I][J] = G(Rng);
      Dot += X[I][J] * std::sin(Tt + static_cast<float>(J));
    }
    Y[I] = Dot > 0.0f ? 1 : 0;
  }
}

static bool LoadElec2(const std::filesystem::path &Dir,
                      std::vector<std::vector<float>> &X,
                      std::vector<int> &Y, size_t MaxRows) {
  auto Path = Dir / "elec2.csv";
  if (!std::filesystem::exists(Path))
    return false;
  auto Csv = bench::ReadCsv(Path);
  if (Csv.Header.empty())
    return false;
  int TargetIdx = -1;
  for (size_t I = 0; I < Csv.Header.size(); ++I)
    if (Csv.Header[I] == "class") {
      TargetIdx = static_cast<int>(I);
      break;
    }
  if (TargetIdx < 0)
    TargetIdx = static_cast<int>(Csv.Header.size()) - 1;
  std::vector<int> FeatIdx;
  for (size_t I = 0; I < Csv.Header.size(); ++I)
    if (static_cast<int>(I) != TargetIdx)
      FeatIdx.push_back(static_cast<int>(I));
  X.clear();
  Y.clear();
  size_t N = std::min(MaxRows, Csv.Rows.size());
  X.reserve(N);
  Y.reserve(N);
  for (size_t I = 0; I < N; ++I) {
    const auto &R = Csv.Rows[I];
    if (R.size() <= static_cast<size_t>(TargetIdx))
      continue;
    std::vector<float> F;
    F.reserve(FeatIdx.size());
    bool Ok = true;
    for (int J : FeatIdx) {
      try {
        F.push_back(std::stof(R[J]));
      } catch (...) {
        Ok = false;
        break;
      }
    }
    if (!Ok)
      continue;
    int Lbl = 0;
    try {
      Lbl = std::stoi(R[TargetIdx]);
    } catch (...) {
      Lbl = (R[TargetIdx] == "UP") ? 1 : 0;
    }
    X.push_back(std::move(F));
    Y.push_back(Lbl);
  }
  if (X.empty())
    return false;
  std::cerr << "[data] loaded Elec2 (" << X.size() << " rows, "
            << X[0].size() << " features)\n";
  return true;
}

// --- helpers ---------------------------------------------------------------

static double EvalLoss(Net &N, const std::vector<std::vector<float>> &X,
                       const std::vector<int> &Y, size_t Begin, size_t End) {
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = Begin; I < End && I < X.size(); ++I) {
    N.DoForwardPass(X[I]);
    auto Out = N.GetOutput();
    float Max = -1e30f;
    for (float V : Out)
      Max = std::max(Max, V);
    double Z = 0.0;
    for (float V : Out)
      Z += std::exp(V - Max);
    double Lp = (Out[Y[I]] - Max) - std::log(Z);
    Sum += -Lp;
    ++Cnt;
  }
  return Cnt ? Sum / Cnt : 0.0;
}

static double EvalAcc(Net &N, const std::vector<std::vector<float>> &X,
                      const std::vector<int> &Y, size_t Begin, size_t End) {
  size_t Correct = 0, Cnt = 0;
  for (size_t I = Begin; I < End && I < X.size(); ++I) {
    N.DoForwardPass(X[I]);
    auto Out = N.GetOutput();
    size_t Arg = 0;
    for (size_t J = 1; J < Out.size(); ++J)
      if (Out[J] > Out[Arg])
        Arg = J;
    if (static_cast<int>(Arg) == Y[I])
      ++Correct;
    ++Cnt;
  }
  return Cnt ? static_cast<double>(Correct) / Cnt : 0.0;
}

class PlateauDetector {
public:
  PlateauDetector(size_t W, float RelTol, size_t Cooldown)
      : W_(W), RelTol_(RelTol), Cooldown_(Cooldown), Since_(Cooldown) {}
  bool Update(float Loss) {
    Buf_.push_back(Loss);
    if (Buf_.size() > W_)
      Buf_.pop_front();
    ++Since_;
    if (Buf_.size() < W_)
      return false;
    if (Since_ < Cooldown_)
      return false;
    double M = 0.0;
    for (float V : Buf_)
      M += V;
    M /= Buf_.size();
    if (M <= 0.0)
      return false;
    double V = 0.0;
    for (float X : Buf_)
      V += (X - M) * (X - M);
    double Sd = std::sqrt(V / Buf_.size());
    if (Sd / M < RelTol_) {
      Since_ = 0;
      return true;
    }
    return false;
  }

private:
  size_t W_;
  float RelTol_;
  size_t Cooldown_;
  size_t Since_;
  std::deque<float> Buf_;
};

// Compute magnitude threshold across all alive connections and arm the
// PruneConn policy. The host then calls N->DoPruneConnections() to fire the
// trim deterministically as a single phase, decoupled from training steps.
template <typename N> static size_t MagnitudePrune(N &Net, float Frac) {
  auto &CA = Net.GetConnAlloc();
  std::vector<float> Alive;
  Alive.reserve(CA.Size());
  for (size_t C = 0; C < CA.Size(); ++C)
    if (!plastix::GetField<plastix::DeadTag>(CA, C))
      Alive.push_back(std::abs(plastix::GetWeight(CA, C)));
  if (Alive.empty())
    return 0;
  size_t K = std::max<size_t>(1, static_cast<size_t>(Frac * Alive.size()));
  K = std::min(K, Alive.size());
  std::nth_element(Alive.begin(), Alive.begin() + K - 1, Alive.end());
  size_t Before = bench::LiveEdgeCount(CA);
  PruneConn::Threshold = Alive[K - 1];
  PruneConn::Armed = true;
  Net.DoPruneConnections();
  PruneConn::Armed = false;
  size_t After = bench::LiveEdgeCount(CA);
  return Before - After;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  HP H;
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.InitHidden =
      static_cast<size_t>(Args.GetInt("init-hidden", H.InitHidden));
  H.InDim = static_cast<size_t>(Args.GetInt("in-dim", H.InDim));
  H.ValEvery = static_cast<size_t>(Args.GetInt("val-every", H.ValEvery));
  H.BurstFrac = Args.GetFloat("burst-frac", H.BurstFrac);
  H.PostBurstPruneFrac =
      Args.GetFloat("post-burst-prune-frac", H.PostBurstPruneFrac);
  H.PlateauWindow =
      static_cast<size_t>(Args.GetInt("plateau-window", H.PlateauWindow));
  H.PlateauCooldown =
      static_cast<size_t>(Args.GetInt("plateau-cooldown", H.PlateauCooldown));
  H.PlateauRelTol = Args.GetFloat("plateau-rel-tol", H.PlateauRelTol);
  H.Lr = Args.GetFloat("lr", H.Lr);
  if (Args.Quick)
    H.MaxSteps = std::max<size_t>(200, H.MaxSteps / 5);
  UpdateConn::Lr = H.Lr;
  BurstAddConn::Seed = static_cast<uint64_t>(Args.Seed) * 1000ull + 17ull;

  std::vector<std::vector<float>> X;
  std::vector<int> Y;
  std::string DatasetName = "synthetic-drift";
  if (!Args.Synthetic && LoadElec2(Args.DataDir, X, Y, 10'000)) {
    H.InDim = X[0].size();
    DatasetName = "elec2";
  } else {
    SynthDrift(std::max<size_t>(H.MaxSteps + 1024, 5000), H.InDim,
               static_cast<uint32_t>(Args.Seed), X, Y);
  }
  bench::Standardise(X, std::max<size_t>(256, X.size() / 10));

  std::cout << "[info] dataset=" << DatasetName << " InDim=" << H.InDim
            << " N=" << X.size() << " MaxSteps=" << H.MaxSteps
            << " InitHidden=" << H.InitHidden << "\n";

  float Limit = std::sqrt(6.0f / static_cast<float>(H.InDim + H.InitHidden));
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 13ull;
  // Two hidden layers (matches PyTorch's GrowableMLP shape in→H→H→out).
  auto N = std::unique_ptr<Net>(new Net(
      H.InDim,
      FCHidden{H.InitHidden, UniformInit{SeedBase + 1, Limit}},
      FCHidden{H.InitHidden, UniformInit{SeedBase + 2, Limit}},
      FCOut{H.NumClasses, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));

  // Carve final 15% as held-out test set. The training cursor wraps within
  // [0, NTrain); the plateau slice stays inside the training portion too,
  // so the test slice is never seen during training.
  size_t NTrain = static_cast<size_t>(0.85 * X.size());
  size_t TestStart = NTrain;
  size_t TestEnd = X.size();
  size_t PlateauStart =
      std::min<size_t>(NTrain / 10, NTrain > 512 ? NTrain - 512 : 0);
  size_t PlateauEnd = std::min(PlateauStart + 256, NTrain);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "bursty_elec2");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_acc");
  auto Edges = bench::LiveEdgeSet(N->GetConnAlloc());
  double InitVal = EvalLoss(*N, X, Y, PlateauStart, PlateauEnd);
  double InitTestAcc = EvalAcc(*N, X, Y, TestStart, TestEnd);
  Log.Log(0, N->GetUnitAlloc().Size(),
          bench::LiveEdgeCount(N->GetConnAlloc()), &Edges, &InitVal,
          {{"bursts", 0.0}, {"prunes", 0.0}, {"test_acc", InitTestAcc}});
  TestCsv.Add(0, InitTestAcc, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()));

  PlateauDetector PD(H.PlateauWindow, H.PlateauRelTol, H.PlateauCooldown);
  size_t Bursts = 0, Prunes = 0;
  size_t CursorI = 0;
  bench::PhaseTimer Timer;

  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    size_t I = CursorI;
    CursorI = (CursorI + 1) % NTrain;

    auto Tgt = std::vector<float>(H.NumClasses, 0.0f);
    Tgt[Y[I]] = 1.0f;
    Timer.Tick();
    N->DoForwardPass(X[I]);
    Timer.MarkForward();
    N->DoCalculateLoss(Tgt);
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

    if (Step % H.ValEvery == 0 || Step == H.MaxSteps) {
      double VL = EvalLoss(*N, X, Y, PlateauStart, PlateauEnd);
      double TestAcc = EvalAcc(*N, X, Y, TestStart, TestEnd);
      bool ShouldBurst = PD.Update(static_cast<float>(VL));
      auto Ed = bench::LiveEdgeSet(N->GetConnAlloc());
      Log.Log(Step, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()), &Ed, &VL,
              {{"bursts", static_cast<double>(Bursts)},
               {"prunes", static_cast<double>(Prunes)},
               {"test_acc", TestAcc}});
      TestCsv.Add(Step, TestAcc, N->GetUnitAlloc().Size(),
                  bench::LiveEdgeCount(N->GetConnAlloc()));

      if (ShouldBurst) {
        // Count current hidden width on layer 1 (PyTorch grows both layers
        // by the same n_new; we size the burst off layer 1's current
        // width).
        size_t OldH1 = 0;
        auto &UA = N->GetUnitAlloc();
        for (size_t U = 0; U < UA.Size(); ++U)
          if (plastix::GetLevel(UA, U) == BurstAddUnit::HiddenL1)
            ++OldH1;
        int NNew = std::max(
            1, static_cast<int>(H.BurstFrac * static_cast<float>(OldH1)));

        // Arm AddUnit (both hidden layers) + AddConn, run a single DoStep
        // with the same training pair to let the framework fire Add* in
        // phase order.
        BurstAddUnit::BudgetL1 = NNew;
        BurstAddUnit::BudgetL2 = NNew;
        BurstAddConn::Enabled = true;
        N->DoStep(X[I], Tgt);
        BurstAddConn::Enabled = false;
        BurstAddUnit::BudgetL1 = 0;
        BurstAddUnit::BudgetL2 = 0;

        size_t Killed = MagnitudePrune(*N, H.PostBurstPruneFrac);
        ++Bursts;
        if (Killed > 0)
          ++Prunes;
        std::cout << "[burst " << Bursts << "] step=" << Step
                  << " new_hidden=" << NNew << " killed=" << Killed
                  << " edges=" << bench::LiveEdgeCount(N->GetConnAlloc())
                  << "\n";
      }
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "bursty_elec2"));
  bench::SummaryWriter S;
  S.Set("workload", std::string{"03_bursty_elec2"});
  S.Set("dataset", DatasetName);
  S.Set("in_dim", static_cast<int>(H.InDim));
  S.Set("init_hidden", static_cast<int>(H.InitHidden));
  S.Set("max_steps", static_cast<int>(H.MaxSteps));
  S.Set("burst_frac", static_cast<double>(H.BurstFrac));
  S.Set("post_burst_prune_frac",
        static_cast<double>(H.PostBurstPruneFrac));
  S.Set("plateau_window", static_cast<int>(H.PlateauWindow));
  S.Set("plateau_rel_tol", static_cast<double>(H.PlateauRelTol));
  S.Set("wall_seconds", Wall);
  S.Set("bursts_fired", static_cast<int>(Bursts));
  S.Set("prunes_fired", static_cast<int>(Prunes));
  size_t HiddenFinal = 0;
  for (size_t U = 0; U < N->GetUnitAlloc().Size(); ++U) {
    uint16_t Lvl = plastix::GetLevel(N->GetUnitAlloc(), U);
    if (Lvl == BurstAddUnit::HiddenL1 || Lvl == BurstAddUnit::HiddenL2)
      ++HiddenFinal;
  }
  S.Set("hidden_final", static_cast<int>(HiddenFinal));
  S.Set("edges_final",
        static_cast<int>(bench::LiveEdgeCount(N->GetConnAlloc())));
  S.Set("val_loss_initial", Log.Records().front().ValLoss);
  S.Set("val_loss_final", Log.Records().back().ValLoss);
  float JMin = 1.0f, JMax = 1.0f;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JMax = std::max(JMax, R.Jaccard);
  }
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_max", static_cast<double>(JMax));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s bursts=" << Bursts
            << " hidden_final=" << HiddenFinal << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
