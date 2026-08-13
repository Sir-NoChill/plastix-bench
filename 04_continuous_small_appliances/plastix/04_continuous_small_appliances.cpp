// Workload 4 / 5 — CONTINUOUS-SMALL regime, Plastix translation.
//
// Mirrors 04_continuous_small_appliances.py. Each training step:
//   1. SGD update on one minibatch from the stream.
//   2. With probability p_split, spawn a new hidden unit (noisy copy of the
//      one with the largest recent activation variance) and wire it to
//      every input and the output.
//   3. With probability p_prune, kill the single smallest-magnitude alive
//      edge.
//
// |Δn_units| per step ≤ 1, |Δn_edges| per step is small. The Jaccard
// similarity between consecutive live-edge sets stays very close to 1.0 —
// the regime is a slow random walk over architecture space.
//
// Plastix policy mapping (matches the bottom-of-file comment in the
// Python source):
//
//   ForwardPass        custom    ReLU hidden, linear output
//   BackwardPass       custom    backprop through ReLU
//   Loss               MSELoss
//   UpdateUnit         custom    per-unit EMA of activation^2; tracks the
//                                argmax (hottest) into a static.
//   UpdateConn         custom    SGD step + tracks the argmin |w| into a
//                                static so PruneConn can fire deterministically.
//   PruneConn          custom    Kills the global-argmin conn when armed.
//   AddUnit            custom    Spawns one new hidden unit when armed,
//                                slot relative to the hottest unit.
//   AddConn            custom    Wires the new unit to every input and the
//                                output (single phase, via the policy's
//                                neighbourhood sweep).
//   ResetGlobal        NoX

#include "plastix/common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>

namespace {

struct HP {
  // Sizing matches the PyTorch reference
  // (04_continuous_small_appliances.py): 3000 steps,
  // init_hidden=32, max_hidden=256, per-step p_split / p_prune = 0.5.
  size_t MaxSteps = 3000;
  size_t InitHidden = 32;
  size_t MaxHidden = 256;
  size_t InDim = 25;
  size_t ValEvery = 20;
  size_t ValWindow = 256;
  float PSplit = 0.5f;
  float PPrune = 0.5f;
  // PyTorch's Adam used 3e-3 with batch=32; per-example SGD on a wide
  // hidden layer + heavy structural churn is unstable at that rate.
  // 1e-3 is a stable per-example step that still trains.
  float Lr = 1e-3f;
  float VarEmaAlpha = 0.1f;
};

struct PreActTag {};
struct GradPreActTag {};
struct IsOutputTag {};
struct ActVarTag {}; // EMA of activation^2, the "hotness" proxy
// Set in SplitAddUnit::InitUnit, cleared by the host at the start of every
// step; AddConn reads it to gate input/output wiring on freshly-added units.
struct IsNewTag {};
// Per-unit live-incoming-degree, recomputed every step. We use it to guard
// the per-step prune from killing the last input edge of a hidden unit —
// that would orphan the unit and Kahn's BFS would relabel it to level 0,
// dropping it out of the network's effective topology.
struct LiveInDegreeTag {};

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

// Runtime-tunable parameters plus the cross-phase argmax/argmin scratch that
// the policies write and read within a DoStep. Held in the network's
// GlobalState (managed memory) — staged/reset from the host via Net::Global()
// — because device code cannot read host-side static members. The structural
// phases run on the host here (Kernelize*=false), so the sequential argmax/
// argmin reductions over G are well-defined.
struct SplitGlobals {
  // UpdateUnit
  float Alpha = 0.1f;
  uint16_t HiddenLevel = 1;
  int ArgmaxId = -1;
  float ArgmaxVar = -1.0f;
  // UpdateConn
  float Lr = 5e-3f;
  int ArgminConnId = -1;
  float ArgminAbs = std::numeric_limits<float>::infinity();
  // PruneConn / SplitAddUnit arming
  bool PruneArmed = false;
  bool AddArmed = false;
  // SplitAddConn
  uint64_t AddConnSeed = 0;
};

// UpdateUnit — EMA of activation^2 per unit, and global argmax tracker.
// G.ArgmaxId / G.ArgmaxVar are reset by the host before DoStep, then AddUnit
// reads G.ArgmaxId at the AddUnit phase to pick the parent.
struct UpdateUnit {
  PLASTIX_HD static void Update(auto &UA, size_t Id, auto &G) {
    // Clear the per-unit live-in-degree so UpdateConn can re-accumulate it
    // below. UpdateUnit fires before UpdateConn in DoStep ordering.
    plastix::GetField<LiveInDegreeTag>(UA, Id) = 0;
    if (plastix::GetLevel(UA, Id) != G.HiddenLevel)
      return;
    float A = plastix::GetActivation(UA, Id);
    float &V = plastix::GetField<ActVarTag>(UA, Id);
    V = (1.0f - G.Alpha) * V + G.Alpha * A * A;
    if (V > G.ArgmaxVar) {
      G.ArgmaxVar = V;
      G.ArgmaxId = static_cast<int>(Id);
    }
  }
};

// UpdateConn — SGD on weights + tracks argmin |w| for the per-step prune.
// Two-phase split inside UpdateConn:
//   - UpdateIncomingConnection runs first across all live edges; we use it
//     to count per-destination LiveInDegree and to apply the SGD weight
//     update.
//   - UpdateOutgoingConnection runs second; by the time it fires, every
//     destination's LiveInDegree is final, so we can scan for the argmin
//     |w| edge while skipping those that are the *last* incoming edge of
//     their destination. PruneConn fires later in the same DoStep and
//     reads the resulting argmin id.
struct UpdateConn {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    ++plastix::GetField<LiveInDegreeTag>(U, DstId);
    float Grad = plastix::GetField<GradPreActTag>(U, DstId);
    float A = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= G.Lr * Grad * A;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &U, size_t /*SrcId*/,
                                                  size_t DstId, auto &C,
                                                  size_t ConnId, auto &G) {
    // Don't even propose a connection for pruning if it's the only live
    // incoming edge of its destination — that's the orphan guard.
    if (plastix::GetField<LiveInDegreeTag>(U, DstId) <= 1)
      return;
    float Abs = std::abs(plastix::GetWeight(C, ConnId));
    if (Abs < G.ArgminAbs) {
      G.ArgminAbs = Abs;
      G.ArgminConnId = static_cast<int>(ConnId);
    }
  }
};

// PruneConn — fires only when armed; reads the argmin id written by
// UpdateConn earlier in this same DoStep. Phase ordering inside DoStep is
// Forward -> Loss -> Backward -> UpdateUnit -> UpdateConn -> PruneUnit ->
// PruneConn -> ... so the static is always fresh when ShouldPrune fires.
struct PruneConn {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &,
                                     size_t ConnId, auto &G) {
    return G.PruneArmed && static_cast<int>(ConnId) == G.ArgminConnId;
  }
};

// SplitAddUnit — spawns one new hidden unit. Filter: ParentId must equal
// the argmax-variance unit picked by UpdateUnit. The static Armed flag is
// reset right after AddUnits fires so only one spawn happens per step.
struct SplitAddUnit {
  PLASTIX_HD static std::optional<int16_t> AddUnit(auto &UA, size_t ParentId,
                                                    auto &G) {
    if (!G.AddArmed)
      return std::nullopt;
    if (plastix::GetLevel(UA, ParentId) != G.HiddenLevel)
      return std::nullopt;
    if (static_cast<int>(ParentId) != G.ArgmaxId)
      return std::nullopt;
    G.AddArmed = false; // single-shot
    return int16_t{0};
  }
  PLASTIX_HD static void InitUnit(auto &UA, size_t NewId, size_t /*Parent*/,
                                  auto &) {
    plastix::GetActivation(UA, NewId) = 0.0f;
    plastix::GetField<PreActTag>(UA, NewId) = 0.0f;
    plastix::GetField<GradPreActTag>(UA, NewId) = 0.0f;
    plastix::GetField<IsOutputTag>(UA, NewId) = false;
    plastix::GetField<ActVarTag>(UA, NewId) = 0.0f;
    plastix::GetField<IsNewTag>(UA, NewId) = true;
  }
};

// SplitAddConn — wires the freshly-allocated unit to every input (level 0)
// and to every output (level >= HiddenLevel+1). The framework's
// neighbourhood sweep iterates pairs by level distance, so Neighbourhood=1
// reaches input -> hidden and hidden -> output. Uses the same per-step
// IsNewTag pattern as workload 03.
struct SplitAddConn {
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &UA, size_t SelfId,
                                                     size_t CandidateId,
                                                     auto &) {
    bool SelfNew = plastix::GetField<IsNewTag>(UA, SelfId);
    bool SelfOut = plastix::GetField<IsOutputTag>(UA, SelfId);
    bool CandNew = plastix::GetField<IsNewTag>(UA, CandidateId);
    // input -> new hidden (always — wire to every input).
    if (SelfNew && plastix::GetLevel(UA, CandidateId) <
                       plastix::GetLevel(UA, SelfId))
      return true;
    // new hidden -> output.
    if (SelfOut && CandNew)
      return true;
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

struct SplitTraits : plastix::DefaultNetworkTraits<> {
  using GlobalState = SplitGlobals;
  using ForwardPass = Forward;
  using BackwardPass = Backward;
  using Loss = plastix::MSELoss;
  using UpdateUnit = ::UpdateUnit;
  using UpdateConn = ::UpdateConn;
  using PruneConn = ::PruneConn;
  using AddUnit = ::SplitAddUnit;
  using AddConn = ::SplitAddConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<PreActTag, float>,
      plastix::alloc::SOAField<GradPreActTag, float>,
      plastix::alloc::SOAField<IsOutputTag, bool>,
      plastix::alloc::SOAField<IsNewTag, bool>,
      plastix::alloc::SOAField<ActVarTag, float>,
      plastix::alloc::SOAField<LiveInDegreeTag, uint32_t>>;
  static constexpr uint16_t Neighbourhood = 1;
  // PyTorch defaults: 27 input features (UCI Appliances), max_hidden=256,
  // 1 output. Initial edges ~ 27*32 + 32 = 896. Peak after splits to
  // max_hidden=256: 27*256 + 256 = 7168 edges. We size the conn alloc
  // bigger to cover the prune-add churn slack.
  static constexpr size_t UnitCapacity = 1024;
  static constexpr size_t ConnCapacity = 65536;
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizeAdd = false;
  static constexpr bool KernelizePrune = false;
};
static_assert(plastix::NetworkTraits<SplitTraits>);

using Net = plastix::Network<SplitTraits>;

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

// --- data ----------------------------------------------------------------

static void SynthSlowDrift(size_t N, size_t Dim, uint32_t Seed,
                           std::vector<std::vector<float>> &X,
                           std::vector<float> &Y) {
  constexpr float Pi = 3.14159265358979323846f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> G(0.0f, 1.0f);
  X.assign(N, std::vector<float>(Dim));
  Y.assign(N, 0.0f);
  for (size_t I = 0; I < N; ++I) {
    float Tt = 2.0f * Pi * static_cast<float>(I) / N;
    float Sum = 0.0f;
    for (size_t J = 0; J < Dim; ++J) {
      X[I][J] = G(Rng);
      float Coef = std::sin(Tt + 0.3f * static_cast<float>(J));
      Sum += X[I][J] * Coef;
    }
    Y[I] = Sum + 0.1f * G(Rng);
  }
}

static bool LoadAppliances(const std::filesystem::path &Dir,
                           std::vector<std::vector<float>> &X,
                           std::vector<float> &Y, size_t MaxRows) {
  auto Path = Dir / "energydata_complete.csv";
  if (!std::filesystem::exists(Path))
    return false;
  auto Csv = bench::ReadCsv(Path);
  if (Csv.Header.empty())
    return false;
  int TargetIdx = -1, DateIdx = -1;
  for (size_t I = 0; I < Csv.Header.size(); ++I) {
    if (Csv.Header[I] == "Appliances")
      TargetIdx = static_cast<int>(I);
    if (Csv.Header[I] == "date")
      DateIdx = static_cast<int>(I);
  }
  if (TargetIdx < 0)
    return false;
  std::vector<int> FeatIdx;
  for (size_t I = 0; I < Csv.Header.size(); ++I)
    if (static_cast<int>(I) != TargetIdx && static_cast<int>(I) != DateIdx)
      FeatIdx.push_back(static_cast<int>(I));
  size_t N = std::min(MaxRows, Csv.Rows.size());
  X.clear();
  Y.clear();
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
    try {
      Y.push_back(std::stof(R[TargetIdx]));
    } catch (...) {
      continue;
    }
    X.push_back(std::move(F));
  }
  if (X.empty())
    return false;
  std::cerr << "[data] loaded UCI Appliances (" << X.size() << " rows, "
            << X[0].size() << " features)\n";
  return true;
}

static void StandardiseY(std::vector<float> &Y, size_t Cut) {
  double M = 0.0, V = 0.0;
  size_t N = std::min(Cut, Y.size());
  for (size_t I = 0; I < N; ++I)
    M += Y[I];
  M /= N;
  for (size_t I = 0; I < N; ++I)
    V += (Y[I] - M) * (Y[I] - M);
  double Sd = std::sqrt(V / N) + 1e-6;
  for (auto &V0 : Y)
    V0 = static_cast<float>((V0 - M) / Sd);
}

// Walks the unit allocator and clears IsNewTag on every hidden unit. Called
// every step to "consume" the flag so AddConn only fires on the truly-fresh
// units in subsequent steps. UpdateUnit can't help here because it runs
// before AddUnit in DoStep ordering — we'd be clearing the flag we just
// set. Do it from the host between steps instead.
static void ClearIsNew(Net &N) {
  auto &UA = N.GetUnitAlloc();
  for (size_t I = 0; I < UA.Size(); ++I)
    plastix::GetField<IsNewTag>(UA, I) = false;
}

// PyTorch's `split_unit(idx)` is function-preserving:
//   - The new unit's incoming weights = noisy copy of `idx`'s incoming weights.
//   - The new unit's outgoing weights = HALF of `idx`'s outgoing weights.
//   - The original `idx`'s outgoing weights are HALVED.
// Result: immediately after the split the network computes the same function
// (each output unit receives the original contribution split across two
// inputs, the sum of which equals the pre-split value).
//
// In the Plastix translation, AddConn::InitConnection initialises every new
// edge to a small uniform random value — *not* function-preserving. We patch
// that up in the host between DoStep calls by walking the connection alloc
// and overwriting the new unit's edge weights from the parent's. The new
// unit is the most recently allocated `IsNewTag=true` unit; we identify the
// parent via the UpdateUnit::ArgmaxId reduction that drove this step's
// SplitAddUnit.
static void RepairSplitWeights(Net &N, size_t ParentId, size_t NewId,
                               std::mt19937 &Rng, float Noise) {
  auto &CA = N.GetConnAlloc();
  auto &UA = N.GetUnitAlloc();
  std::normal_distribution<float> Gauss(0.0f, Noise);

  // Pass 1: collect parent's incoming edges keyed by source id, halve any
  // outgoing edges (parent -> downstream).
  std::unordered_map<uint32_t, float> ParentIncoming;
  std::unordered_map<uint32_t, float> ParentOutgoingHalved;
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (plastix::GetField<plastix::DeadTag>(CA, C))
      continue;
    uint32_t F = plastix::GetField<plastix::FromIdTag>(CA, C).Value;
    uint32_t T = plastix::GetField<plastix::ToIdTag>(CA, C).Value;
    if (T == ParentId) {
      ParentIncoming[F] = plastix::GetWeight(CA, C);
    } else if (F == ParentId) {
      float W = plastix::GetWeight(CA, C);
      ParentOutgoingHalved[T] = W * 0.5f;
      plastix::GetWeight(CA, C) = W * 0.5f;
    }
  }

  // Pass 2: write the new unit's incoming/outgoing edges to match.
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (plastix::GetField<plastix::DeadTag>(CA, C))
      continue;
    uint32_t F = plastix::GetField<plastix::FromIdTag>(CA, C).Value;
    uint32_t T = plastix::GetField<plastix::ToIdTag>(CA, C).Value;
    if (T == NewId) {
      auto It = ParentIncoming.find(F);
      if (It != ParentIncoming.end())
        plastix::GetWeight(CA, C) = It->second + Gauss(Rng);
    } else if (F == NewId) {
      auto It = ParentOutgoingHalved.find(T);
      if (It != ParentOutgoingHalved.end())
        plastix::GetWeight(CA, C) = It->second;
    }
  }
  (void)UA;
}

// Mark the unit that will be added in this step's AddUnit phase with
// IsNewTag=true. The framework's InitUnit hook *can* set the flag, but
// flag-setting in InitUnit only affects future AddConn queries from the
// rolling neighbourhood-window walk — which is exactly when we need it.
// So InitUnit sets the flag and ClearIsNew runs at the *start* of the
// host's step (before DoStep).

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

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  HP H;
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.InitHidden =
      static_cast<size_t>(Args.GetInt("init-hidden", H.InitHidden));
  H.MaxHidden = static_cast<size_t>(Args.GetInt("max-hidden", H.MaxHidden));
  H.InDim = static_cast<size_t>(Args.GetInt("in-dim", H.InDim));
  H.ValEvery = static_cast<size_t>(Args.GetInt("val-every", H.ValEvery));
  H.PSplit = Args.GetFloat("p-split", H.PSplit);
  H.PPrune = Args.GetFloat("p-prune", H.PPrune);
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.VarEmaAlpha = Args.GetFloat("var-ema-alpha", H.VarEmaAlpha);
  if (Args.Quick)
    H.MaxSteps = std::max<size_t>(600, H.MaxSteps / 5);

  bench::MemoryProbe MP;
  MP.Start();

  std::vector<std::vector<float>> X;
  std::vector<float> Y;
  std::string DatasetName = "synthetic-slow-drift";
  if (!Args.Synthetic && LoadAppliances(Args.DataDir, X, Y, 10'000)) {
    H.InDim = X[0].size();
    DatasetName = "uci-appliances";
  } else {
    SynthSlowDrift(std::max<size_t>(H.MaxSteps + 1024, 5000), H.InDim,
                   static_cast<uint32_t>(Args.Seed), X, Y);
  }
  bench::Standardise(X, std::max<size_t>(256, X.size() / 10));
  StandardiseY(Y, std::max<size_t>(256, Y.size() / 10));
  MP.EndDataset();

  std::cout << "[info] dataset=" << DatasetName << " InDim=" << H.InDim
            << " N=" << X.size() << " MaxSteps=" << H.MaxSteps
            << " InitHidden=" << H.InitHidden << " MaxHidden=" << H.MaxHidden
            << "\n";

  float Limit = std::sqrt(6.0f / static_cast<float>(H.InDim + H.InitHidden));
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 23ull;
  auto N = std::unique_ptr<Net>(new Net(
      H.InDim, FCHidden{H.InitHidden, UniformInit{SeedBase + 1, Limit}},
      FCOut{1, UniformInit{SeedBase + 2, Limit}, MarkOutput{}}));
  MP.EndWeights();
  // Stage runtime params into the managed GlobalState (read by the policies).
  N->Global().Lr = H.Lr;
  N->Global().Alpha = H.VarEmaAlpha;
  N->Global().AddConnSeed = static_cast<uint64_t>(Args.Seed) * 1000ull + 19ull;

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::uniform_real_distribution<float> Coin(0.0f, 1.0f);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "continuous_small_appliances");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_mse");
  auto Edges = bench::LiveEdgeSet(N->GetConnAlloc());
  // Carve final 15% as held-out test set; training cursor wraps inside
  // [0, NTrain). The val slice stays inside the training portion too.
  size_t NTrain = static_cast<size_t>(0.85 * X.size());
  size_t TestStart = NTrain;
  size_t TestEnd = X.size();
  size_t ValStart =
      std::min<size_t>(static_cast<size_t>(NTrain / 10), NTrain);
  size_t ValEnd = std::min(ValStart + H.ValWindow, NTrain);
  double InitVal = EvalMse(*N, X, Y, ValStart, ValEnd);
  double InitTest = EvalMse(*N, X, Y, TestStart, TestEnd);
  Log.Log(0, N->GetUnitAlloc().Size(),
          bench::LiveEdgeCount(N->GetConnAlloc()), &Edges, &InitVal,
          {{"hidden", static_cast<double>(H.InitHidden)},
           {"splits", 0.0},
           {"prunes", 0.0},
           {"test_mse", InitTest}});
  TestCsv.Add(0, InitTest, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()));

  size_t Splits = 0, Prunes = 0;
  std::vector<int> DeltaUnits;
  std::vector<int> DeltaEdges;
  size_t CursorI = 0;
  bench::PhaseTimer Timer;

  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    // Host-side per-step setup. Clear the per-step argmax/argmin reductions
    // and the structural-mutation arms; the policy methods will rewrite
    // them during the step.
    N->Global().ArgmaxId = -1;
    N->Global().ArgmaxVar = -1.0f;
    N->Global().ArgminConnId = -1;
    N->Global().ArgminAbs = std::numeric_limits<float>::infinity();

    ClearIsNew(*N);

    // Count current hidden width to enforce MaxHidden ceiling without a
    // framework hook.
    size_t HiddenWidth = 0;
    auto &UA = N->GetUnitAlloc();
    for (size_t U = 0; U < UA.Size(); ++U)
      if (plastix::GetLevel(UA, U) == N->Global().HiddenLevel)
        ++HiddenWidth;

    bool WantSplit =
        Coin(Rng) < H.PSplit && HiddenWidth < H.MaxHidden;
    // Conservative prune: never below a per-unit floor of incoming edges,
    // since killing the last incoming edge of a hidden unit orphans it and
    // Kahn's BFS drops it back to level 0 — the level we use to count
    // "hidden" disappears under us. We approximate the per-unit floor by
    // a network-wide one: at least 1.5x hidden_width edges have to stay
    // alive across the run.
    bool WantPrune = Coin(Rng) < H.PPrune;
    N->Global().AddArmed = WantSplit;
    N->Global().PruneArmed = WantPrune;

    size_t I = CursorI;
    CursorI = (CursorI + 1) % NTrain;
    float Ytgt[1] = {Y[I]};

    size_t UnitsBefore = N->GetUnitAlloc().Size();
    size_t EdgesBefore = bench::LiveEdgeCount(N->GetConnAlloc());

    // The framework runs phases in this order:
    //   Forward → Loss → Backward → UpdateUnit → UpdateConn → PruneUnit →
    //   PruneConn → Compact → AddUnit → AddConn → Resort → ResetGlobal.
    // We want PruneConn's TargetConnId to be the argmin from this step's
    // UpdateConn — so we need to set TargetConnId *between* UpdateConn
    // and PruneConn. We can't intercept mid-DoStep, so we instead stage
    // the prune across two DoStep calls: this step's UpdateConn writes
    // the argmin id, the *next* step's PruneConn fires on it. That's
    // the same "stage now, fire next" trick used in 03_bursty.
    //
    // For correctness we therefore keep TargetConnId static and only
    // mutate Armed each step; the host sets TargetConnId after the
    // DoStep returns (just below).

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
    Timer.MarkPrune();
    N->DoAddUnits();
    N->DoAddConnections();
    Timer.MarkGrow();
    N->DoResetGlobalState();
    Timer.MarkReset();
    Timer.StepDone();

    size_t UnitsAfter = N->GetUnitAlloc().Size();
    size_t EdgesAfter = bench::LiveEdgeCount(N->GetConnAlloc());
    int DU = static_cast<int>(UnitsAfter) - static_cast<int>(UnitsBefore);
    int DE = static_cast<int>(EdgesAfter) - static_cast<int>(EdgesBefore);

    // Function-preserving split fixup: AddConn::InitConnection initialises
    // every new edge to a small uniform random value, which is *not*
    // function-preserving (the network's output jumps with each split,
    // forcing SGD to re-adapt). PyTorch's reference splits the hottest
    // unit by halving its outgoing weights and giving the new unit a
    // noisy copy of the parent's incoming + the halved outgoing. We do
    // the same fixup here in host code now that the new unit and its
    // edges exist in the SOA arenas.
    if (DU > 0 && N->Global().ArgmaxId >= 0) {
      size_t Parent = static_cast<size_t>(N->Global().ArgmaxId);
      // The new unit is at the previously-allocated count.
      size_t NewId = UnitsBefore;
      RepairSplitWeights(*N, Parent, NewId, Rng, /*Noise=*/0.05f);
    }
    DeltaUnits.push_back(DU);
    DeltaEdges.push_back(DE);
    if (DU > 0)
      ++Splits;
    if (DE < 0)
      ++Prunes;

    bool DidChange = DU != 0 || DE != 0;
    if (Step % H.ValEvery == 0 || Step == H.MaxSteps || DidChange) {
      double VL = EvalMse(*N, X, Y, ValStart, ValEnd);
      double TestMse = EvalMse(*N, X, Y, TestStart, TestEnd);
      auto Ed = bench::LiveEdgeSet(N->GetConnAlloc());
      Log.Log(Step, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()), &Ed, &VL,
              {{"hidden", static_cast<double>(HiddenWidth)},
               {"splits", static_cast<double>(Splits)},
               {"prunes", static_cast<double>(Prunes)},
               {"delta_units", static_cast<double>(DU)},
               {"delta_edges", static_cast<double>(DE)},
               {"test_mse", TestMse}});
      TestCsv.Add(Step, TestMse, N->GetUnitAlloc().Size(),
                  bench::LiveEdgeCount(N->GetConnAlloc()));
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "continuous_small_appliances"));
  bench::SummaryWriter S;
  S.Set("workload", std::string{"04_continuous_small_appliances"});
  S.Set("dataset", DatasetName);
  S.Set("in_dim", static_cast<int>(H.InDim));
  S.Set("init_hidden", static_cast<int>(H.InitHidden));
  S.Set("max_steps", static_cast<int>(H.MaxSteps));
  S.Set("p_split", static_cast<double>(H.PSplit));
  S.Set("p_prune", static_cast<double>(H.PPrune));
  S.Set("wall_seconds", Wall);
  S.Set("splits_fired", static_cast<int>(Splits));
  S.Set("prunes_fired", static_cast<int>(Prunes));
  size_t HiddenFinal = 0;
  for (size_t U = 0; U < N->GetUnitAlloc().Size(); ++U)
    if (plastix::GetLevel(N->GetUnitAlloc(), U) == N->Global().HiddenLevel)
      ++HiddenFinal;
  S.Set("hidden_final", static_cast<int>(HiddenFinal));
  S.Set("edges_final",
        static_cast<int>(bench::LiveEdgeCount(N->GetConnAlloc())));
  S.Set("val_loss_initial", Log.Records().front().ValLoss);
  S.Set("val_loss_final", Log.Records().back().ValLoss);

  std::vector<int> AbsDu;
  AbsDu.reserve(DeltaUnits.size());
  int MaxAbsDu = 0;
  for (int V : DeltaUnits) {
    int Av = V < 0 ? -V : V;
    AbsDu.push_back(Av);
    MaxAbsDu = std::max(MaxAbsDu, Av);
  }
  S.Set("delta_units_max_abs", MaxAbsDu);
  S.Set("delta_units_p99_abs", AbsDu.empty() ? 0 : [&]() {
    auto Tmp = AbsDu;
    size_t K = static_cast<size_t>(0.99 * Tmp.size());
    if (K >= Tmp.size())
      K = Tmp.size() - 1;
    std::nth_element(Tmp.begin(), Tmp.begin() + K, Tmp.end());
    return Tmp[K];
  }());
  float JMin = 1.0f, JSum = 0.0f;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JSum += R.Jaccard;
  }
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_mean", static_cast<double>(JSum) /
                            std::max<size_t>(Log.Records().size(), 1));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s splits=" << Splits
            << " prunes=" << Prunes << " hidden_final=" << HiddenFinal
            << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
