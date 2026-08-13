// Workload 10 — engineered sparse large NN, Plastix translation.
//
// Streaming regression over a deep (~1000-layer), sparse, irregular DAG that
// grows and shrinks while running in Pipeline propagation. The initial network
// and data stream are loaded from the shared `topology.bin` (see gen.py /
// README.md), so all three impls (plastix / cpp / pytorch) run the same
// workload. Growth/shrink is parametric and driven by the README's shared LCG
// so the schedule matches across impls.
//
// Plastix policy mapping (matches the README's per-step algorithm):
//
//   ForwardPass   custom   Map = w * act[src]; Combine = +; Apply = tanh(sum)
//                          for hidden units, identity for the output unit
//                          (and stash sum into G.V). This is the README's
//                          step-2 "single layer-advance" — exactly Plastix's
//                          Pipeline forward (one sweep over live edges, one
//                          Apply per non-input unit).
//   Loss          custom   G.Delta = target - G.V.
//   UpdateConn    custom   per-edge eligibility trace E:
//                            E = DECAY*E + act[src];  w += LR * G.Delta * E.
//                          Uniform over every live edge (no per-edge gate).
//   ResetGlobal   custom   clears the per-step delta.
//
// Growth/shrink is driven from the harness via direct allocator manipulation
// (AddUnit/AddConn/Prune policies are NOT used): every GROW_EVERY steps after
// WARMUP we Allocate() new hidden units + their edges, and mark PRUNE_EDGES
// live edges Dead. The per-step Pipeline compute is what is measured; exact
// topology match with the other impls is secondary to it actually growing /
// shrinking on the specified cadence (README §"Growth / shrink").
//
// Host build only (KernelizeUpdate / KernelizeAdd = false) — this is a CPU
// comparison.

#include "plastix/common.hpp"

#include <plastix/alloc.hpp>
#include <plastix/conn.hpp>
#include <plastix/layers.hpp>
#include <plastix/macros.hpp>
#include <plastix/plastix.hpp>
#include <plastix/traits.hpp>
#include <plastix/unit_state.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

// --- per-connection state ----------------------------------------------------
// plastix::WeightTag carries w; ETag is the eligibility trace.
struct ETag {};

// --- per-unit state ----------------------------------------------------------
// IsOutputTag flags the single output unit so Apply uses the linear (identity)
// activation; every other non-input unit uses tanh.
struct IsOutputTag {};

namespace hp {
constexpr float LR = 0.01f;
constexpr float Decay = 0.9f; // = gamma*lambda
} // namespace hp

struct Globals {
  float V = 0.0f;     // output activation (yhat) from the last forward
  float Delta = 0.0f; // target - yhat
};

// --- ForwardPass: README step 2 (Pipeline layer-advance) ---------------------
struct Forward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float Sum) {
    if (plastix::GetField<IsOutputTag>(U, Id)) {
      float Out = std::tanh(Sum); // bounded output (keeps metric finite over
      plastix::GetActivation(U, Id) = Out; // long horizons; not meant to converge)
      G.V = Out;
    } else {
      plastix::GetActivation(U, Id) = std::tanh(Sum); // hidden
    }
  }
};

// --- Loss: README step 3 -----------------------------------------------------
struct LossPolicy {
  static void CalculateLoss(auto &, plastix::UnitRange,
                            std::span<const float> Targets, Globals &G) {
    G.Delta = Targets[0] - G.V;
  }
};

// --- UpdateConn: README step 4 (TD(λ)-style trace, uniform over edges) -------
struct ConnUpdate {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t /*DstId*/,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, Globals &G) {
    using namespace plastix;
    float &W = GetWeight(C, ConnId);
    float &E = GetField<ETag>(C, ConnId);
    E = hp::Decay * E + GetActivation(U, SrcId);
    W += hp::LR * G.Delta * E;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &,
                                                  size_t, Globals &) {}
};

struct ResetGlobalPolicy {
  static void Reset(auto &G) { G.Delta = 0.0f; }
};

struct Traits : plastix::DefaultNetworkTraits<Globals> {
  using ForwardPass = Forward;
  using Loss = LossPolicy;
  using UpdateConn = ConnUpdate;
  using ResetGlobal = ResetGlobalPolicy;
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizeAdd = false;
  static constexpr bool ReverseAdjForward = false;

  using ExtraConnFields = plastix::ConnFieldList<
      plastix::alloc::SOAField<plastix::WeightTag, float>,
      plastix::alloc::SOAField<ETag, float>>;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<IsOutputTag, uint8_t>>;

  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;

  // Initial network: ~4000 units / ~24k edges. Growth caps at MAX_UNITS=6000
  // and steadily adds edges (net +9 edges per grow event, minus 8 prunes), so
  // generous capacity headroom. Override at compile time if the generator is
  // run with larger parameters.
#ifndef ESLN_UNIT_CAPACITY
#define ESLN_UNIT_CAPACITY 16384
#endif
#ifndef ESLN_CONN_CAPACITY
#define ESLN_CONN_CAPACITY 262144
#endif
  static constexpr size_t UnitCapacity = ESLN_UNIT_CAPACITY;
  static constexpr size_t ConnCapacity = ESLN_CONN_CAPACITY;
};

static_assert(plastix::NetworkTraits<Traits>);
using Net = plastix::Network<Traits>;

// --- shared LCG (README §"Shared LCG") --------------------------------------
struct Lcg {
  uint64_t State;
  explicit Lcg(uint64_t Seed) : State(Seed) {}
  uint32_t Next() {
    State = State * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(State >> 33); // 31-bit
  }
};

// --- topology.bin reader -----------------------------------------------------
struct Topology {
  uint32_t NIn = 0, NUnits = 0, NEdges = 0, NSteps = 0, OutputId = 0;
  std::vector<uint32_t> Layer;            // [NUnits]
  std::vector<std::pair<uint32_t, uint32_t>> Edges; // [NEdges] (src,dst)
  std::vector<float> Data;                // [NSteps * (NIn+1)] x..., target
};

template <typename T> static T ReadPod(std::ifstream &In) {
  T V{};
  In.read(reinterpret_cast<char *>(&V), sizeof(T));
  return V;
}

static bool LoadTopology(const std::filesystem::path &Path, Topology &Topo) {
  std::ifstream In(Path, std::ios::binary);
  if (!In)
    return false;
  char Magic[4];
  In.read(Magic, 4);
  if (std::memcmp(Magic, "ESLN", 4) != 0) {
    std::cerr << "[err] bad magic in " << Path << "\n";
    return false;
  }
  uint32_t Version = ReadPod<uint32_t>(In);
  (void)Version;
  Topo.NIn = ReadPod<uint32_t>(In);
  Topo.NUnits = ReadPod<uint32_t>(In);
  Topo.NEdges = ReadPod<uint32_t>(In);
  Topo.NSteps = ReadPod<uint32_t>(In);
  Topo.OutputId = ReadPod<uint32_t>(In);

  Topo.Layer.resize(Topo.NUnits);
  In.read(reinterpret_cast<char *>(Topo.Layer.data()),
          static_cast<std::streamsize>(Topo.NUnits * sizeof(uint32_t)));

  Topo.Edges.resize(Topo.NEdges);
  for (uint32_t I = 0; I < Topo.NEdges; ++I) {
    uint32_t Src = ReadPod<uint32_t>(In);
    uint32_t Dst = ReadPod<uint32_t>(In);
    Topo.Edges[I] = {Src, Dst};
  }

  size_t RecW = static_cast<size_t>(Topo.NIn) + 1;
  Topo.Data.resize(static_cast<size_t>(Topo.NSteps) * RecW);
  In.read(reinterpret_cast<char *>(Topo.Data.data()),
          static_cast<std::streamsize>(Topo.Data.size() * sizeof(float)));
  return static_cast<bool>(In);
}

static std::filesystem::path ResolveTopology(const bench::CliArgs &Args) {
  std::vector<std::filesystem::path> Candidates = {
      Args.DataDir / "topology.bin",
      "10_engineered_sparse_large_nn/topology.bin",
  };
  for (const auto &C : Candidates)
    if (std::filesystem::exists(C))
      return C;
  return {};
}

// --- LayerBuilder: build the explicit DAG from the edge list -----------------
// Receives the input UnitRange (already allocated by the Network ctor), then
// allocates every remaining unit (hidden + output) with its layer + output
// flag, and adds each edge as a directed connection with weight 0. Returns the
// single-element output range so Network::GetOutput points at it.
struct TopologyBuilder {
  const Topology *Topo;

  template <typename UnitAlloc, typename ConnAlloc>
  PLASTIX_HOST plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                             plastix::UnitRange Inputs) const {
    using namespace plastix;
    const Topology &T = *Topo;
    // Inputs [0, NIn) are layer 0 already; set them explicitly for clarity.
    for (size_t I = 0; I < T.NIn; ++I) {
      GetLevel(UA, I) = static_cast<uint16_t>(T.Layer[I]);
      GetField<IsOutputTag>(UA, I) = 0;
    }
    (void)Inputs;
    // Allocate the remaining units [NIn, NUnits) in id order so unit id ==
    // file index (the edge list and output_id reference these ids directly).
    for (size_t Id = T.NIn; Id < T.NUnits; ++Id) {
      auto New = UA.Allocate();
      GetLevel(UA, New) = static_cast<uint16_t>(T.Layer[Id]);
      GetField<IsOutputTag>(UA, New) = (New == T.OutputId) ? 1 : 0;
    }
    // Add every edge with weight 0 and zero eligibility trace (default-init).
    for (const auto &[Src, Dst] : T.Edges) {
      auto ConnId = CA.Allocate();
      GetField<FromIdTag>(CA, ConnId) = plastix::GlobalUnitId{Src};
      GetField<ToIdTag>(CA, ConnId) = plastix::GlobalUnitId{Dst};
      GetField<SrcLevelTag>(CA, ConnId) =
          static_cast<uint16_t>(T.Layer[Src]);
      GetWeight(CA, ConnId) = 0.0f;
    }
    return UnitRange{T.OutputId, T.OutputId + 1};
  }
};

// --- growth / shrink params (README) ----------------------------------------
namespace gs {
constexpr uint32_t Warmup = 1000;
constexpr uint32_t GrowEvery = 500;
constexpr uint32_t GrowUnits = 4;
constexpr uint32_t MaxUnits = 6000;
constexpr uint32_t Fanin = 4;
constexpr uint32_t PruneEdges = 8;
} // namespace gs

struct HP {
  size_t MaxSteps = 10000;
  size_t LogEvery = 500;
};

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.LogEvery = static_cast<size_t>(Args.GetInt("log-every", H.LogEvery));
  if (Args.Quick) {
    H.MaxSteps = std::min<size_t>(H.MaxSteps, 2000);
    H.LogEvery = std::min<size_t>(H.LogEvery, 200);
  }

  bench::MemoryProbe MP;
  MP.Start();

  auto TopoPath = ResolveTopology(Args);
  if (TopoPath.empty()) {
    std::cerr << "[err] topology.bin not found (looked in --data-dir and "
                 "10_engineered_sparse_large_nn/).\n";
    return 2;
  }

  Topology Topo;
  if (!LoadTopology(TopoPath, Topo)) {
    std::cerr << "[err] failed to read " << TopoPath << "\n";
    return 2;
  }
  MP.EndDataset();

  size_t N = std::min<size_t>(H.MaxSteps, Topo.NSteps);
  size_t RecW = static_cast<size_t>(Topo.NIn) + 1;

  std::cout << "[info] topology=" << TopoPath.string() << " units="
            << Topo.NUnits << " edges=" << Topo.NEdges << " steps="
            << Topo.NSteps << " using=" << N << " output_id=" << Topo.OutputId
            << " quick=" << (Args.Quick ? 1 : 0) << "\n";

  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<IsOutputTag>(U, Id) = 0;
  };
  Net Network(Topo.NIn, InputInit, TopologyBuilder{&Topo});

  auto &UA = Network.GetUnitAlloc();
  auto &CA = Network.GetConnAlloc();

  // Host-side live-edge index in insertion order (initial edges first, then
  // growth edges) so prune indices line up with the README's live_edge_count.
  // Each entry is the connection-allocator slot id; an entry is removed from
  // this list when its edge is pruned (so the list == current live edges).
  std::vector<size_t> LiveEdges;
  LiveEdges.reserve(Topo.NEdges + 4096);
  for (size_t C = 0; C < CA.Size(); ++C)
    LiveEdges.push_back(C);
  MP.EndWeights();

  Lcg Rng(0x9E3779B97F4A7C15ull);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "engineered_sparse_large_nn");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_mse");

  std::array<float, 1> TargetBuf{0.0f};
  std::vector<float> Features(Topo.NIn, 0.0f);
  std::vector<float> Deltas(N, 0.0f);

  double WindowSse = 0.0;
  size_t WindowCnt = 0;
  size_t EpochIdx = 0;

  bench::PhaseTimer Timer;

  auto T0 = std::chrono::steady_clock::now();
  for (size_t T = 0; T < N; ++T) {
    const float *Rec = &Topo.Data[T * RecW];
    for (uint32_t I = 0; I < Topo.NIn; ++I)
      Features[I] = Rec[I];
    TargetBuf[0] = Rec[Topo.NIn];

    Timer.Tick();
    Network.DoForwardPass(Features); // README step 2
    Timer.MarkForward();
    Network.DoCalculateLoss(TargetBuf); // README step 3
    Timer.MarkLoss();
    Timer.MarkBackward(); // no backward phase
    Network.DoUpdateConnectionState(); // README step 4
    Timer.MarkUpdate();

    // --- Growth / shrink (README §"Growth / shrink") ------------------------
    // Direct allocator manipulation on the GROW_EVERY cadence after WARMUP.
    if (T >= gs::Warmup && ((T - gs::Warmup) % gs::GrowEvery == 0)) {
      // Grow GROW_UNITS hidden units (capped at MAX_UNITS).
      uint32_t MaxHiddenLayer = 0;
      for (size_t I = Topo.NIn; I < UA.Size(); ++I)
        MaxHiddenLayer = std::max<uint32_t>(MaxHiddenLayer, plastix::GetLevel(UA, I));
      for (uint32_t G = 0; G < gs::GrowUnits; ++G) {
        if (UA.Size() >= gs::MaxUnits)
          break;
        uint32_t NewLayer =
            1u + (MaxHiddenLayer > 1 ? Rng.Next() % MaxHiddenLayer : 0u);
        auto NewId = UA.Allocate();
        if (NewId == static_cast<size_t>(-1))
          break;
        plastix::GetLevel(UA, NewId) = static_cast<uint16_t>(
            std::min<uint32_t>(NewLayer, plastix::MaxLevels - 1));
        plastix::GetField<IsOutputTag>(UA, NewId) = 0;
        // FANIN incoming edges from distinct earlier units.
        uint32_t Added = 0, Attempts = 0;
        std::vector<uint32_t> Used;
        while (Added < gs::Fanin && Attempts < 64) {
          ++Attempts;
          uint32_t Src = Rng.Next() % Topo.OutputId;
          if (plastix::GetLevel(UA, Src) >= NewLayer)
            continue;
          if (std::find(Used.begin(), Used.end(), Src) != Used.end())
            continue;
          Used.push_back(Src);
          auto C = CA.Allocate();
          if (C == static_cast<size_t>(-1))
            break;
          plastix::GetField<plastix::FromIdTag>(CA, C) = plastix::GlobalUnitId{Src};
          plastix::GetField<plastix::ToIdTag>(CA, C) =
              plastix::GlobalUnitId{static_cast<uint32_t>(NewId)};
          plastix::GetField<plastix::SrcLevelTag>(CA, C) =
              plastix::GetLevel(UA, Src);
          plastix::GetWeight(CA, C) = 0.0f;
          LiveEdges.push_back(C);
          ++Added;
        }
        // One edge (newunit -> output).
        auto Co = CA.Allocate();
        if (Co != static_cast<size_t>(-1)) {
          plastix::GetField<plastix::FromIdTag>(CA, Co) =
              plastix::GlobalUnitId{static_cast<uint32_t>(NewId)};
          plastix::GetField<plastix::ToIdTag>(CA, Co) = plastix::GlobalUnitId{Topo.OutputId};
          plastix::GetField<plastix::SrcLevelTag>(CA, Co) =
              plastix::GetLevel(UA, NewId);
          plastix::GetWeight(CA, Co) = 0.0f;
          LiveEdges.push_back(Co);
        }
      }
      // Shrink: prune PRUNE_EDGES edges (skip edges into the output).
      for (uint32_t P = 0; P < gs::PruneEdges; ++P) {
        if (LiveEdges.empty())
          break;
        uint32_t Idx = Rng.Next() % static_cast<uint32_t>(LiveEdges.size());
        size_t C = LiveEdges[Idx];
        if (plastix::GetField<plastix::ToIdTag>(CA, C) == plastix::GlobalUnitId{Topo.OutputId})
          continue; // keep the output connected
        plastix::GetField<plastix::DeadTag>(CA, C) = true;
        LiveEdges[Idx] = LiveEdges.back();
        LiveEdges.pop_back();
      }
    }
    Timer.MarkPrune(); // bench only removes edges; no growth phase
    // Capture delta before ResetGlobal clears it.
    float Delta = Network.Global().Delta;
    Network.DoResetGlobalState();
    Timer.MarkReset();
    Timer.StepDone();

    Deltas[T] = Delta;
    double Err = static_cast<double>(Delta);
    WindowSse += Err * Err;
    ++WindowCnt;

    if (WindowCnt >= H.LogEvery || T + 1 == N) {
      double WindowMse = WindowCnt > 0 ? WindowSse / WindowCnt : 0.0;
      ++EpochIdx;
      size_t NEdges = bench::LiveEdgeCount(CA);
      Log.Log(EpochIdx, UA.Size(), NEdges, nullptr, &WindowMse,
              {{"train_loss", WindowMse},
               {"epoch", static_cast<double>(EpochIdx)},
               {"step", static_cast<double>(T + 1)},
               {"test_mse", WindowMse},
               {"n_units", static_cast<double>(UA.Size())},
               {"n_edges", static_cast<double>(NEdges)}});
      TestCsv.Add(T + 1, WindowMse, UA.Size(), NEdges);
      std::cout << "[ep " << EpochIdx << "] step=" << (T + 1)
                << " window_mse=" << WindowMse << " units=" << UA.Size()
                << " edges=" << NEdges << "\n";
      WindowSse = 0.0;
      WindowCnt = 0;
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  // test_mse = mean delta^2 over the last 10% of steps (README §Metric).
  size_t TailStart = N - std::max<size_t>(1, N / 10);
  double TailSse = 0.0;
  for (size_t T = TailStart; T < N; ++T)
    TailSse += static_cast<double>(Deltas[T]) * static_cast<double>(Deltas[T]);
  double TestMse = TailSse / std::max<size_t>(1, N - TailStart);

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "engineered_sparse_large_nn"));

  bench::SummaryWriter S;
  S.Set("workload", std::string{"10_engineered_sparse_large_nn"});
  S.Set("max_steps", static_cast<int>(N));
  S.Set("wall_seconds", Wall);
  S.Set("test_mse", TestMse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<int>(UA.Size()));
  S.Set("n_edges", static_cast<int>(bench::LiveEdgeCount(CA)));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TestMse
            << "  units=" << UA.Size() << "  edges=" << bench::LiveEdgeCount(CA)
            << "\n";
  std::cout << "[phase] step_count=" << Timer.StepCount()
            << " (see summary csv for per-phase mean/std)\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
