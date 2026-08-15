// Workload 13 — DEPTH scaling sweep, Plastix.
//
// Sibling of 11_scaling_imprint, but the independent variable is network DEPTH,
// not size. Bench 11 pins every non-input unit to level 1 (a 2-level network:
// inputs -> everything), so its "pipeline" is a single layer advance and the
// longest input->output chain is one hop. Here we partition the (N - NIn)
// non-input units into `--depth L` sequential layers at levels 1..L, each unit
// wired from `--fanin` random units in the previous layer. The longest chain
// from an input to the output unit is therefore L edges (L+1 nodes), and the
// Pipeline forward pass processes L level-batches in dependency order.
//
// Same per-step compute as bench 11 (Forward + TD(lambda) edge update), fixed
// topology (no runtime growth), N held constant while L is swept, so edge count
// stays ~ (N - NIn) * fanin and the measurement isolates the effect of depth.
//
//   ForwardPass   Map = w*act[src]; Combine = +; Apply = tanh(sum)
//   Loss          G.Delta = target - yhat
//   UpdateConn    per-edge trace E = DECAY*E + act[src]; w += LR*Delta*E
//   ResetGlobal   clears the per-step delta

#include "plastix/common.hpp"

#include <plastix/alloc.hpp>
#include <plastix/conn.hpp>
#include <plastix/layers.hpp>
#include <plastix/macros.hpp>
#include <plastix/math.hpp>
#include <plastix/plastix.hpp>
#include <plastix/traits.hpp>
#include <plastix/unit_state.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct ETag {};        // per-connection eligibility trace
struct IsOutputTag {}; // flags the single linear-output unit

namespace hp {
constexpr float LR = 0.01f;
constexpr float Decay = 0.9f;
} // namespace hp

struct Globals {
  float V = 0.0f;     // output activation from the last forward
  float Delta = 0.0f; // target - yhat
};

struct Forward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float Sum) {
    float Out = plastix::math::Tanh(Sum);
    plastix::GetActivation(U, Id) = Out;
    if (plastix::GetField<IsOutputTag>(U, Id))
      G.V = Out;
  }
};

struct LossPolicy {
  static void CalculateLoss(auto &, plastix::UnitRange,
                            std::span<const float> Targets, Globals &G) {
    G.Delta = Targets[0] - G.V;
  }
};

struct ConnUpdate {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t, size_t SrcId,
                                                  auto &C, size_t ConnId,
                                                  Globals &G) {
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

#ifndef SCALE_UNIT_CAPACITY
#define SCALE_UNIT_CAPACITY 12000000
#endif
#ifndef SCALE_CONN_CAPACITY
#define SCALE_CONN_CAPACITY 60000000
#endif
  static constexpr size_t UnitCapacity = SCALE_UNIT_CAPACITY;
  static constexpr size_t ConnCapacity = SCALE_CONN_CAPACITY;
};

static_assert(plastix::NetworkTraits<Traits>);
using Net = plastix::Network<Traits>;

// Two-shard policy for the multi-GPU variant: level 0 (input units) on
// shard 0, every deeper level on shard 1. Same policy bench 01 uses; the
// plastix library routes Pipeline+sharded through the Topological sharded
// dispatchers (level-serialized shortcut, see plastix.hpp).
struct TwoShardOnLevel1 {
  static constexpr uint16_t NumShards = 2;
  static constexpr bool IsContiguousByLevel = true;
  static constexpr plastix::ShardId Assign(uint32_t, uint16_t Level) {
    return plastix::ShardId{Level == 0 ? uint16_t{0} : uint16_t{1}};
  }
};

struct TraitsSharded : Traits {
  using Sharding = TwoShardOnLevel1;
};
static_assert(plastix::NetworkTraits<TraitsSharded>);

using NetSharded = plastix::Network<TraitsSharded>;

struct Lcg {
  uint64_t State;
  explicit Lcg(uint64_t Seed) : State(Seed ? Seed : 0x9E3779B97F4A7C15ull) {}
  uint32_t Next() {
    State = State * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(State >> 33);
  }
  float Unit() { return static_cast<float>(Next()) / 2147483648.0f; }
};

// Layered sparse DAG: NIn inputs at level 0, then the remaining units split
// across `Layers` sequential levels 1..Layers. Each unit draws `Fanin` distinct
// sources from the immediately preceding layer, so the only path from input to
// output runs through every level -> depth = Layers.
struct DepthBuilder {
  size_t N;
  uint32_t NIn;
  uint32_t Fanin;
  uint32_t Layers;
  uint64_t Seed;

  template <typename UnitAlloc, typename ConnAlloc>
  PLASTIX_HOST plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                             plastix::UnitRange Inputs) const {
    using namespace plastix;
    Lcg Rng(Seed);
    const uint32_t OutId = static_cast<uint32_t>(N - 1);
    const size_t Hidden = N - NIn;
    const uint32_t L = std::max<uint32_t>(1u, Layers);

    size_t PrevBegin = Inputs.Begin;
    size_t PrevEnd = Inputs.End;
    uint16_t PrevLevel = 0;
    size_t Made = 0;

    for (uint32_t Lay = 1; Lay <= L; ++Lay) {
      const size_t Remaining = Hidden - Made;
      const uint32_t LayersLeft = L - Lay + 1;
      size_t LayerSize = (Lay == L) ? Remaining : Remaining / LayersLeft;
      if (LayerSize == 0)
        LayerSize = 1;
      const size_t Begin = NIn + Made;
      const size_t End = Begin + LayerSize;
      const uint16_t Level = static_cast<uint16_t>(Lay);
      const size_t PrevSize = PrevEnd - PrevBegin;

      for (size_t Id = Begin; Id < End && Id < N; ++Id) {
        auto New = UA.Allocate();
        GetLevel(UA, New) = Level;
        GetField<IsOutputTag>(UA, New) = (New == OutId) ? 1 : 0;

        const uint32_t Want =
            std::min<uint32_t>(std::min<uint32_t>(Fanin, 16u),
                               static_cast<uint32_t>(PrevSize));
        std::array<uint32_t, 16> Used{};
        uint32_t Added = 0, Attempts = 0;
        while (Added < Want && Attempts < Want * 8u + 8u) {
          ++Attempts;
          const uint32_t Off = Rng.Next() % static_cast<uint32_t>(PrevSize);
          const uint32_t Src = static_cast<uint32_t>(PrevBegin + Off);
          bool Dup = false;
          for (uint32_t K = 0; K < Added; ++K)
            if (Used[K] == Src) {
              Dup = true;
              break;
            }
          if (Dup)
            continue;
          Used[Added] = Src;
          auto C = CA.Allocate();
          GetField<FromIdTag>(CA, C) = plastix::GlobalUnitId{Src};
          GetField<ToIdTag>(CA, C) = plastix::GlobalUnitId{static_cast<uint32_t>(New)};
          GetField<SrcLevelTag>(CA, C) = PrevLevel;
          GetWeight(CA, C) = Rng.Unit() * 0.02f - 0.01f;
          ++Added;
        }
      }
      Made += (End - Begin);
      PrevBegin = Begin;
      PrevEnd = End;
      PrevLevel = Level;
    }
    return UnitRange{OutId, OutId + 1};
  }
};

// Templated body: instantiated with Net (single-device) or NetSharded
// (2 shards, MultiDeviceExecutor). MultiDevice=true attaches the
// MultiDeviceExecutor after construction.
template <typename NetT>
static int RunBench(bench::CliArgs &Args, size_t Neurons, uint32_t NIn,
                    uint32_t Fanin, uint32_t Depth, size_t Steps,
                    bool MultiDevice) {
  bench::MemoryProbe MP;
  MP.Start();
  MP.EndDataset();

  std::cout << "[info] neurons=" << Neurons << " inputs=" << NIn
            << " fanin=" << Fanin << " depth=" << Depth << " steps=" << Steps
            << " MultiDevice=" << (MultiDevice ? "yes" : "no") << "\n";

  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<IsOutputTag>(U, Id) = 0;
  };
  NetT Network(NIn, InputInit,
               DepthBuilder{Neurons, NIn, Fanin, Depth,
                            0x1234ull + Args.Seed});
  MP.EndWeights();

#ifdef PLASTIX_HAS_CUDA
  if (MultiDevice) {
    Network.SetExecutor(std::make_unique<plastix::MultiDeviceExecutor>(2));
    std::cout << "[info] executor: MultiDeviceExecutor(2)\n";
  }
#else
  if (MultiDevice) {
    std::cerr << "[fatal] --multi-device requires PLASTIX_HAS_CUDA\n";
    return 2;
  }
#endif

  auto &UA = Network.GetUnitAlloc();
  auto &CA = Network.GetConnAlloc();
  const size_t NEdges = bench::LiveEdgeCount(CA);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "depth_scaling");
  bench::StructuralLog Log(HistPath);

  Lcg Rng(0xABCDEFull + Args.Seed);
  std::vector<float> Features(NIn, 0.0f);
  std::array<float, 1> TargetBuf{0.0f};
  double Sse = 0.0;

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t T = 0; T < Steps; ++T) {
    float Active = 0.0f;
    for (uint32_t I = 0; I < NIn; ++I) {
      float Bit = (Rng.Next() & 7u) == 0u ? 1.0f : 0.0f;
      Features[I] = Bit;
      Active += Bit;
    }
    TargetBuf[0] = Active / static_cast<float>(NIn);

    Timer.Tick();
    Network.DoForwardPass(Features);
    Timer.MarkForward();
    Network.DoCalculateLoss(TargetBuf);
    Timer.MarkLoss();
    Timer.MarkBackward();
    Network.DoUpdateConnectionState();
    Timer.MarkUpdate();
    Timer.MarkPrune();
    Timer.MarkGrow();
    // compression-branch API: Network::Global() is private; read the output
    // activation instead (yhat == G.V, so target - yhat == the loss delta).
    float Delta = TargetBuf[0] - Network.GetOutput()[0];
    Network.DoResetGlobalState();
    Timer.MarkReset();
    Timer.StepDone();

    Sse += static_cast<double>(Delta) * static_cast<double>(Delta);
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();
  double Mse = Sse / static_cast<double>(std::max<size_t>(1, Steps));

  Log.Log(1, UA.Size(), NEdges, nullptr, &Mse,
          {{"test_mse", Mse},
           {"n_units", static_cast<double>(UA.Size())},
           {"n_edges", static_cast<double>(NEdges)},
           {"depth", static_cast<double>(Depth)},
           {"step", static_cast<double>(Steps)}});
  Log.Flush();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"13_depth_scaling"});
  S.Set("neurons", static_cast<long long>(Neurons));
  S.Set("depth", static_cast<int>(Depth));
  S.Set("max_steps", static_cast<int>(Steps));
  S.Set("wall_seconds", Wall);
  S.Set("test_mse", Mse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<long long>(UA.Size()));
  S.Set("n_edges", static_cast<long long>(NEdges));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  units=" << UA.Size()
            << "  edges=" << NEdges << "  depth=" << Depth << "  step_ns="
            << (Wall * 1e9 / static_cast<double>(std::max<size_t>(1, Steps)))
            << "\n";
  (void)LogPath;
  return 0;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  size_t Neurons = static_cast<size_t>(Args.GetInt("neurons", 100000));
  uint32_t NIn = static_cast<uint32_t>(Args.GetInt("inputs", 64));
  uint32_t Fanin = static_cast<uint32_t>(Args.GetInt("fanin", 4));
  uint32_t Depth = static_cast<uint32_t>(Args.GetInt("depth", 8));
  size_t Steps = static_cast<size_t>(Args.GetInt("steps", 200));
  if (Args.Quick) {
    Neurons = std::min<size_t>(Neurons, 20000);
    Steps = std::min<size_t>(Steps, 100);
  }
  Neurons = std::max<size_t>(Neurons, NIn + Depth + 1);
  if (Neurons > Traits::UnitCapacity) {
    std::cerr << "[err] neurons=" << Neurons << " exceeds compile-time "
              << "UnitCapacity=" << Traits::UnitCapacity
              << " (rebuild with -DSCALE_UNIT_CAPACITY=...)\n";
    return 2;
  }

  const bool MultiDevice = Args.GetBool("multi-device", false);
  if (MultiDevice) {
    return RunBench<NetSharded>(Args, Neurons, NIn, Fanin, Depth, Steps, true);
  }
  return RunBench<Net>(Args, Neurons, NIn, Fanin, Depth, Steps, false);
}
