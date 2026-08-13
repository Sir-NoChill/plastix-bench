// Workload 11 — imprinting-style scaling sweep, Plastix.
//
// A purpose-built micro-benchmark for the memory/compute SCALING study: build a
// sparse imprinting-style network of EXACTLY `--neurons N` units (N swept from
// ~10k to ~10M by the scaling.py driver), run a fixed number of pipeline steps,
// and report per-step wall time + the RSS memory breakdown. There is NO runtime
// growth here (the size is the independent variable), and no dataset file — the
// topology + the sparse binary input stream are generated in-process so any N
// is exact and reproducible from the seed.
//
// Engineered in the style of workload 10 (same pipeline Forward + TD(λ) edge
// update), so the plastix-vs-pytorch comparison is apples-to-apples. The single
// binary is compiled once at a large compile-time capacity; mmap MAP_NORESERVE
// means the unused capacity costs no physical memory, so `--neurons` simply
// Allocate()s up to N at runtime.
//
//   ForwardPass   Map = w*act[src]; Combine = +; Apply = tanh(sum) (linear out)
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

  // Compiled once at a large capacity; mmap MAP_NORESERVE makes the headroom
  // free until written. `--neurons` allocates up to UnitCapacity at runtime.
  // Override at compile time for the very top of the sweep, e.g.
  //   -DSCALE_UNIT_CAPACITY=12000000 -DSCALE_CONN_CAPACITY=60000000
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

struct Lcg {
  uint64_t State;
  explicit Lcg(uint64_t Seed) : State(Seed ? Seed : 0x9E3779B97F4A7C15ull) {}
  uint32_t Next() {
    State = State * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(State >> 33);
  }
  float Unit() { return static_cast<float>(Next()) / 2147483648.0f; }
};

// Generated sparse DAG: NIn inputs at level 0, then (N-NIn) units each wired
// from Fanin distinct earlier ids (src < dst, so the pipeline advances one
// layer/step). The last id is the linear output unit.
struct ScaleBuilder {
  size_t N;       // total units
  uint32_t NIn;   // input units
  uint32_t Fanin; // incoming edges per non-input unit
  uint64_t Seed;

  template <typename UnitAlloc, typename ConnAlloc>
  PLASTIX_HOST plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                             plastix::UnitRange) const {
    using namespace plastix;
    Lcg Rng(Seed);
    const uint32_t OutId = static_cast<uint32_t>(N - 1);
    for (size_t I = 0; I < NIn; ++I) {
      GetLevel(UA, I) = 0;
      GetField<IsOutputTag>(UA, I) = 0;
    }
    std::vector<uint32_t> Used;
    Used.reserve(Fanin);
    for (size_t Id = NIn; Id < N; ++Id) {
      auto New = UA.Allocate();
      GetLevel(UA, New) = 1;
      GetField<IsOutputTag>(UA, New) = (New == OutId) ? 1 : 0;
      // Fanin distinct sources from [0, Id).
      Used.clear();
      uint32_t Added = 0, Attempts = 0;
      uint32_t Want = std::min<uint32_t>(Fanin, static_cast<uint32_t>(Id));
      while (Added < Want && Attempts < Want * 8u + 8u) {
        ++Attempts;
        uint32_t Src = Rng.Next() % static_cast<uint32_t>(Id);
        if (std::find(Used.begin(), Used.end(), Src) != Used.end())
          continue;
        Used.push_back(Src);
        auto C = CA.Allocate();
        GetField<FromIdTag>(CA, C) = plastix::GlobalUnitId{Src};
        GetField<ToIdTag>(CA, C) = plastix::GlobalUnitId{static_cast<uint32_t>(New)};
        GetField<SrcLevelTag>(CA, C) = GetLevel(UA, Src);
        GetWeight(CA, C) = Rng.Unit() * 0.02f - 0.01f;
        ++Added;
      }
    }
    return UnitRange{OutId, OutId + 1};
  }
};

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  size_t Neurons = static_cast<size_t>(Args.GetInt("neurons", 10000));
  uint32_t NIn = static_cast<uint32_t>(Args.GetInt("inputs", 64));
  uint32_t Fanin = static_cast<uint32_t>(Args.GetInt("fanin", 4));
  size_t Steps = static_cast<size_t>(Args.GetInt("steps", 500));
  if (Args.Quick) {
    Neurons = std::min<size_t>(Neurons, 20000);
    Steps = std::min<size_t>(Steps, 100);
  }
  Neurons = std::max<size_t>(Neurons, NIn + 2);
  if (Neurons > Traits::UnitCapacity) {
    std::cerr << "[err] neurons=" << Neurons << " exceeds compile-time "
              << "UnitCapacity=" << Traits::UnitCapacity
              << " (rebuild with -DSCALE_UNIT_CAPACITY=...)\n";
    return 2;
  }

  bench::MemoryProbe MP;
  MP.Start();
  // No external dataset — the input stream is generated per step below.
  MP.EndDataset();

  std::cout << "[info] neurons=" << Neurons << " inputs=" << NIn
            << " fanin=" << Fanin << " steps=" << Steps << "\n";

  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<IsOutputTag>(U, Id) = 0;
  };
  Net Network(NIn, InputInit, ScaleBuilder{Neurons, NIn, Fanin, 0x1234ull + Args.Seed});
  MP.EndWeights();

  auto &UA = Network.GetUnitAlloc();
  auto &CA = Network.GetConnAlloc();
  const size_t NEdges = bench::LiveEdgeCount(CA);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "scaling_imprint");
  bench::StructuralLog Log(HistPath);

  Lcg Rng(0xABCDEFull + Args.Seed);
  std::vector<float> Features(NIn, 0.0f);
  std::array<float, 1> TargetBuf{0.0f};
  double Sse = 0.0;

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t T = 0; T < Steps; ++T) {
    // Sparse binary input (imprinting-style); target = mean of active inputs.
    float Active = 0.0f;
    for (uint32_t I = 0; I < NIn; ++I) {
      float Bit = (Rng.Next() & 7u) == 0u ? 1.0f : 0.0f; // ~1/8 sparsity
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
    // activation instead. yhat == G.V, so target - yhat == the loss delta the
    // update policy consumed internally this step.
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
           {"step", static_cast<double>(Steps)}});
  Log.Flush();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"11_scaling_imprint"});
  S.Set("neurons", static_cast<long long>(Neurons));
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
            << "  edges=" << NEdges << "  step_ns="
            << (Wall * 1e9 / static_cast<double>(std::max<size_t>(1, Steps)))
            << "\n";
  (void)LogPath;
  return 0;
}
