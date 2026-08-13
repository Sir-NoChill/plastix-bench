// Workload 12 — per-step runtime topology CHURN, Plastix (add-cost microbench).
//
// Validates the O(N*k) sampled connection-growth path added to
// cpu::DoAddConnections (gated by GlobalState::GrowFanout). Builds a layered
// DAG (like bench 13) at level depth `--depth`, then times a single
// DoAddConnections call — the per-step growth cost a churn loop pays.
//
//   --grow-k 0  -> exhaustive all-pairs enumeration  (original, O(N^2))
//   --grow-k >0 -> sample k candidates per unit by index  (O(N*k))
//
// Two trait variants are instantiated (K=0 and K=SAMPLE_K) and selected at
// runtime, because the compression branch exposes no host-mutable global state
// (Network::Global() is private), so GrowFanout must be a compile-time default.

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
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef SAMPLE_K
#define SAMPLE_K 8
#endif

namespace {

struct ETag {};
struct IsOutputTag {};

template <uint32_t K> struct GlobalsT {
  float V = 0.0f;
  float Delta = 0.0f;
  uint32_t GrowFanout = K; // read by cpu::DoAddConnections
};

struct Forward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float Sum) {
    float Out = Sum > 0 ? Sum : 0.0f; // ReLU; cheap, unused by the add microbench
    plastix::GetActivation(U, Id) = Out;
    if (plastix::GetField<IsOutputTag>(U, Id))
      G.V = Out;
  }
};

// Grow forward edges: from a unit to a candidate exactly one level deeper.
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

template <uint32_t K> struct TraitsT : plastix::DefaultNetworkTraits<GlobalsT<K>> {
  using ForwardPass = Forward;
  using AddConn = ::AddConn;
  static constexpr bool KernelizeAdd = false;
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizePrune = false;
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
      const uint32_t Left = L - Lay + 1;
      size_t Sz = (Lay == L) ? Remaining : Remaining / Left;
      if (Sz == 0)
        Sz = 1;
      const size_t Begin = NIn + Made, End = Begin + Sz;
      const uint16_t Level = static_cast<uint16_t>(Lay);
      const size_t PrevSize = PrevEnd - PrevBegin;
      for (size_t Id = Begin; Id < End && Id < N; ++Id) {
        auto New = U.Allocate();
        GetLevel(U, New) = Level;
        GetField<IsOutputTag>(U, New) = (New == OutId) ? 1 : 0;
        const uint32_t Want =
            std::min<uint32_t>(std::min<uint32_t>(Fanin, 16u),
                               static_cast<uint32_t>(PrevSize));
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
          GetField<FromIdTag>(C, Cn) = Src;
          GetField<ToIdTag>(C, Cn) = static_cast<uint32_t>(New);
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

template <uint32_t K>
void RunOne(size_t Neurons, uint32_t NIn, uint32_t Fanin, uint32_t Depth,
            uint64_t Seed) {
  using Net = plastix::Network<TraitsT<K>>;
  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<IsOutputTag>(U, Id) = 0;
  };
  Net Network(NIn, InputInit, DepthBuilder{Neurons, NIn, Fanin, Depth, Seed});
  size_t Before = bench::LiveEdgeCount(Network.GetConnAlloc());

  auto T0 = std::chrono::steady_clock::now();
  Network.DoAddConnections();
  double AddNs =
      std::chrono::duration<double, std::nano>(
          std::chrono::steady_clock::now() - T0)
          .count();
  size_t After = bench::LiveEdgeCount(Network.GetConnAlloc());
  // Order-independent checksum of the live edge set (From,To) — lets a
  // before/after build prove the committed topology is bit-identical.
  auto &CA = Network.GetConnAlloc();
  uint64_t Sum = 0, Xor = 0;
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (plastix::GetField<plastix::DeadTag>(CA, C))
      continue;
    uint64_t Key = (static_cast<uint64_t>(plastix::GetField<plastix::FromIdTag>(CA, C)) << 32) |
                   plastix::GetField<plastix::ToIdTag>(CA, C);
    uint64_t H = Key * 0x9E3779B97F4A7C15ull;
    Sum += H;
    Xor ^= H;
  }
  std::cout << "[add] neurons=" << Neurons << " depth=" << Depth << " k=" << K
            << " edges_before=" << Before << " edges_after=" << After
            << " added=" << (After - Before) << " add_ns=" << AddNs
            << " edgesum=" << Sum << " edgexor=" << Xor << "\n";
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  size_t Neurons = static_cast<size_t>(Args.GetInt("neurons", 100000));
  uint32_t NIn = static_cast<uint32_t>(Args.GetInt("inputs", 64));
  uint32_t Fanin = static_cast<uint32_t>(Args.GetInt("fanin", 4));
  uint32_t Depth = static_cast<uint32_t>(Args.GetInt("depth", 4));
  uint32_t GrowK = static_cast<uint32_t>(Args.GetInt("grow-k", SAMPLE_K));
  uint64_t Seed = 0x1234ull + static_cast<uint64_t>(Args.Seed);
  Neurons = std::max<size_t>(Neurons, NIn + Depth + 1);

  if (GrowK == 0)
    RunOne<0>(Neurons, NIn, Fanin, Depth, Seed);
  else
    RunOne<SAMPLE_K>(Neurons, NIn, Fanin, Depth, Seed);
  return 0;
}
