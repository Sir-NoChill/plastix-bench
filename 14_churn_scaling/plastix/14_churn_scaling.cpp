// Workload 14 — combined per-step CHURN at scale, Plastix.
//
// Puts benches 12 (runtime topology change) and 13 (depth) together and makes
// the scaling real: a layered DAG of `--neurons` units at `--depth` levels runs
// a full per-step loop that mutates structure every step —
//
//   forward -> loss -> update-conn -> prune-conn -> compact -> add-conn
//
// prune drops a ~1/PRUNE_DENOM fraction of edges (stateless hash of the
// post-compaction ConnId, so a different set each step); add grows `--grow-k`
// forward edges per unit via the O(N*k) sampled path (GlobalState::GrowFanout).
// We report mean per-step wall time and the live-edge trajectory, swept over N.
//
// Demonstrates that with the linear-add fix, Plastix sustains O(N+E) per-step
// cost under continuous structural churn — the regime bench 11 (fixed topology)
// could not exercise.

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
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef SAMPLE_K
#define SAMPLE_K 4
#endif
#ifndef PRUNE_DENOM
#define PRUNE_DENOM 2
#endif

namespace {

struct ETag {};
struct IsOutputTag {};
namespace hp {
constexpr float LR = 0.01f, Decay = 0.9f;
}

struct Globals {
  float V = 0.0f, Delta = 0.0f;
  uint32_t GrowFanout = SAMPLE_K; // read by cpu::DoAddConnections
};

PLASTIX_HD uint64_t Mix64(uint64_t X) {
  X += 0x9E3779B97F4A7C15ull;
  X = (X ^ (X >> 30)) * 0xBF58476D1CE4E5B9ull;
  X = (X ^ (X >> 27)) * 0x94D049BB133111EBull;
  return X ^ (X >> 31);
}

struct Forward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float S) {
    float O = plastix::math::Tanh(S);
    plastix::GetActivation(U, Id) = O;
    if (plastix::GetField<IsOutputTag>(U, Id))
      G.V = O;
  }
};
struct LossPolicy {
  static void CalculateLoss(auto &, plastix::UnitRange,
                            std::span<const float> T, Globals &G) {
    G.Delta = T[0] - G.V;
  }
};
struct ConnUpdate {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t, size_t SrcId,
                                                  auto &C, size_t ConnId,
                                                  Globals &G) {
    float &W = plastix::GetWeight(C, ConnId);
    float &E = plastix::GetField<ETag>(C, ConnId);
    E = hp::Decay * E + plastix::GetActivation(U, SrcId);
    W += hp::LR * G.Delta * E;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &,
                                                  size_t, Globals &) {}
};
struct ResetGlobalPolicy {
  static void Reset(auto &G) { G.Delta = 0.0f; }
};
struct AddConn {
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static bool ShouldAddOutgoingConnection(auto &U, size_t Self,
                                                     size_t Cand, auto &) {
    return plastix::GetLevel(U, Cand) ==
           static_cast<uint16_t>(plastix::GetLevel(U, Self) + 1);
  }
  PLASTIX_HD static void InitConnection(auto &, size_t, size_t, auto &C,
                                        size_t ConnId, auto &) {
    plastix::GetWeight(C, ConnId) = 0.0f;
    plastix::GetField<ETag>(C, ConnId) = 0.0f;
  }
};
struct PruneConn {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &,
                                     size_t ConnId, auto &) {
    return (Mix64(ConnId) % PRUNE_DENOM) == 0; // ~1/PRUNE_DENOM per step
  }
};

struct Traits : plastix::DefaultNetworkTraits<Globals> {
  using ForwardPass = Forward;
  using Loss = LossPolicy;
  using UpdateConn = ConnUpdate;
  using ResetGlobal = ResetGlobalPolicy;
  using AddConn = ::AddConn;
  using PruneConn = ::PruneConn;
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizePrune = false;
  static constexpr bool KernelizeAdd = false;
  static constexpr uint16_t Neighbourhood = 1;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
  using ExtraConnFields = plastix::ConnFieldList<
      plastix::alloc::SOAField<plastix::WeightTag, float>,
      plastix::alloc::SOAField<ETag, float>>;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<IsOutputTag, uint8_t>>;
#ifndef SCALE_UNIT_CAPACITY
#define SCALE_UNIT_CAPACITY 3500000
#endif
#ifndef SCALE_CONN_CAPACITY
#define SCALE_CONN_CAPACITY 80000000
#endif
  static constexpr size_t UnitCapacity = SCALE_UNIT_CAPACITY;
  static constexpr size_t ConnCapacity = SCALE_CONN_CAPACITY;
};
using Net = plastix::Network<Traits>;

struct Lcg {
  uint64_t State;
  explicit Lcg(uint64_t S) : State(S ? S : 0x9E3779B97F4A7C15ull) {}
  uint32_t Next() {
    State = State * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(State >> 33);
  }
  float Unit() { return static_cast<float>(Next()) / 2147483648.0f; }
};

struct DepthBuilder {
  size_t N;
  uint32_t NIn, Fanin, Layers;
  uint64_t Seed;
  template <typename UA, typename CA>
  PLASTIX_HOST plastix::UnitRange operator()(UA &U, CA &C,
                                             plastix::UnitRange Inputs) const {
    using namespace plastix;
    Lcg Rng(Seed);
    const uint32_t OutId = static_cast<uint32_t>(N - 1);
    const size_t Hidden = N - NIn;
    const uint32_t L = std::max<uint32_t>(1u, Layers);
    size_t PrevBegin = Inputs.Begin, PrevEnd = Inputs.End, Made = 0;
    uint16_t PrevLevel = 0;
    for (uint32_t Lay = 1; Lay <= L; ++Lay) {
      const size_t Remaining = Hidden - Made;
      size_t Sz = (Lay == L) ? Remaining : Remaining / (L - Lay + 1);
      if (Sz == 0)
        Sz = 1;
      const size_t Begin = NIn + Made, End = Begin + Sz;
      const uint16_t Level = static_cast<uint16_t>(Lay);
      const size_t PrevSize = PrevEnd - PrevBegin;
      for (size_t Id = Begin; Id < End && Id < N; ++Id) {
        auto New = U.Allocate();
        GetLevel(U, New) = Level;
        GetField<IsOutputTag>(U, New) = (New == OutId) ? 1 : 0;
        const uint32_t Want = std::min<uint32_t>(
            std::min<uint32_t>(Fanin, 16u), static_cast<uint32_t>(PrevSize));
        std::array<uint32_t, 16> Used{};
        uint32_t Added = 0, Att = 0;
        while (Added < Want && Att < Want * 8u + 8u) {
          ++Att;
          const uint32_t Src =
              static_cast<uint32_t>(PrevBegin + Rng.Next() % PrevSize);
          bool Dup = false;
          for (uint32_t K = 0; K < Added; ++K)
            if (Used[K] == Src) {
              Dup = true;
              break;
            }
          if (Dup)
            continue;
          Used[Added] = Src;
          auto Cn = C.Allocate();
          GetField<FromIdTag>(C, Cn) = plastix::GlobalUnitId{Src};
          GetField<ToIdTag>(C, Cn) = plastix::GlobalUnitId{static_cast<uint32_t>(New)};
          GetField<SrcLevelTag>(C, Cn) = PrevLevel;
          GetWeight(C, Cn) = Rng.Unit() * 0.02f - 0.01f;
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

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  size_t Neurons = static_cast<size_t>(Args.GetInt("neurons", 100000));
  uint32_t NIn = static_cast<uint32_t>(Args.GetInt("inputs", 64));
  uint32_t Fanin = static_cast<uint32_t>(Args.GetInt("fanin", 4));
  uint32_t Depth = static_cast<uint32_t>(Args.GetInt("depth", 4));
  size_t Steps = static_cast<size_t>(Args.GetInt("steps", 20));
  Neurons = std::max<size_t>(Neurons, NIn + Depth + 1);

  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<IsOutputTag>(U, Id) = 0;
  };
  Net Network(NIn, InputInit,
              DepthBuilder{Neurons, NIn, Fanin, Depth, 0x1234ull + Args.Seed});

  std::vector<float> Features(NIn, 0.0f);
  std::array<float, 1> TargetBuf{0.0f};
  Lcg Rng(0xABCDEFull + Args.Seed);

  double SumNs = 0.0;
  double PhFwd = 0, PhUpd = 0, PhPrune = 0, PhCompact = 0, PhAdd = 0;
  size_t Counted = 0, MinE = SIZE_MAX, MaxE = 0;
  auto Now = [] { return std::chrono::steady_clock::now(); };
  auto Ns = [](auto A, auto B) {
    return std::chrono::duration<double, std::nano>(B - A).count();
  };
  for (size_t T = 0; T < Steps; ++T) {
    float Active = 0.0f;
    for (uint32_t I = 0; I < NIn; ++I) {
      float B = (Rng.Next() & 7u) == 0u ? 1.0f : 0.0f;
      Features[I] = B;
      Active += B;
    }
    TargetBuf[0] = Active / static_cast<float>(NIn);

    auto T0 = Now();
    Network.DoForwardPass(Features);
    auto T1 = Now();
    Network.DoCalculateLoss(TargetBuf);
    Network.DoUpdateConnectionState();
    auto T2 = Now();
    Network.DoPruneConnections();
    auto T3 = Now();
    Network.DoCompactConnections();
    auto T4 = Now();
    Network.DoAddConnections();
    auto T5 = Now();
    size_t E = bench::LiveEdgeCount(Network.GetConnAlloc());
    MinE = std::min(MinE, E);
    MaxE = std::max(MaxE, E);
    if (T > 0) { // skip warmup step
      SumNs += Ns(T0, T5);
      PhFwd += Ns(T0, T1);
      PhUpd += Ns(T1, T2);
      PhPrune += Ns(T2, T3);
      PhCompact += Ns(T3, T4);
      PhAdd += Ns(T4, T5);
      ++Counted;
    }
  }
  double C = Counted ? Counted : 1;
  double MeanNs = SumNs / C;
  std::cout << "[churn] neurons=" << Neurons << " depth=" << Depth
            << " k=" << SAMPLE_K << " prune=1/" << PRUNE_DENOM
            << " steps=" << Steps << " edges_min=" << MinE
            << " edges_max=" << MaxE << " mean_step_ns=" << MeanNs
            << " throughput_sps=" << (MeanNs > 0 ? 1e9 / MeanNs : 0) << "\n";
  std::cout << "[phases] fwd=" << PhFwd / C << " upd=" << PhUpd / C
            << " prune=" << PhPrune / C << " compact=" << PhCompact / C
            << " add=" << PhAdd / C << "\n";
  return 0;
}
