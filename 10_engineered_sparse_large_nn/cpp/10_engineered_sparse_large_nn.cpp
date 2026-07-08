// Workload 10 — engineered sparse large NN (native C++).
//
// Streaming-regression on a deep (~1000-layer), sparse, irregular DAG that
// grows and shrinks while running under pipeline-propagation semantics. The
// shared `topology.bin` (produced by gen.py) defines the initial network and
// the data stream; growth/shrink is parametric and driven by a shared LCG so
// the schedule matches the other impls.
//
// The graph is a hand-rolled edge list over a flat `act[]` plus per-edge
// `weight[]`/`elig[]` arrays. The per-step pipeline is:
//   1. load inputs,
//   2. one synchronous sparse mat-vec over the *previous* activations
//      (signal advances exactly one layer per step),
//   3. predict + squared-error loss,
//   4. uniform TD(lambda) update over every edge.
// After WARMUP, every GROW_EVERY steps the network grows GROW_UNITS hidden
// units and prunes PRUNE_EDGES edges.

#include "cpp/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

// ---- spec constants --------------------------------------------------------
constexpr float LR = 0.01f;
constexpr float DECAY = 0.9f;             // = gamma*lambda
constexpr uint32_t WARMUP = 1000;
constexpr uint32_t GROW_EVERY = 500;
constexpr uint32_t GROW_UNITS = 4;
constexpr uint32_t FANIN = 4;
constexpr uint32_t PRUNE_EDGES = 8;
constexpr uint32_t MAX_UNITS = 6000;

// ---- shared LCG ------------------------------------------------------------
struct Lcg {
  uint64_t State = 0x9E3779B97F4A7C15ull;
  uint32_t Next() {
    State = State * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(State >> 33);  // 31-bit
  }
};

// ---- topology.bin loader ---------------------------------------------------
struct Topology {
  uint32_t NIn = 0;
  uint32_t NUnits = 0;
  uint32_t NEdges = 0;
  uint32_t NSteps = 0;
  uint32_t OutputId = 0;
  std::vector<uint32_t> Layer;        // [NUnits]
  std::vector<uint32_t> Src;          // [NEdges]
  std::vector<uint32_t> Dst;          // [NEdges]
  std::vector<float> X;               // [NSteps * NIn]
  std::vector<float> Y;               // [NSteps]
};

template <typename T>
static void ReadRaw(std::ifstream &In, T *Out, size_t Count) {
  In.read(reinterpret_cast<char *>(Out), static_cast<std::streamsize>(Count * sizeof(T)));
}

static bool LoadTopology(const std::filesystem::path &Path, Topology &T) {
  std::ifstream In(Path, std::ios::binary);
  if (!In)
    return false;
  char Magic[4];
  In.read(Magic, 4);
  if (std::memcmp(Magic, "ESLN", 4) != 0) {
    std::cerr << "[err] bad magic in " << Path << "\n";
    return false;
  }
  uint32_t Version = 0;
  ReadRaw(In, &Version, 1);
  ReadRaw(In, &T.NIn, 1);
  ReadRaw(In, &T.NUnits, 1);
  ReadRaw(In, &T.NEdges, 1);
  ReadRaw(In, &T.NSteps, 1);
  ReadRaw(In, &T.OutputId, 1);

  T.Layer.resize(T.NUnits);
  ReadRaw(In, T.Layer.data(), T.NUnits);

  // edges stored interleaved (src, dst)
  std::vector<uint32_t> Pairs(static_cast<size_t>(T.NEdges) * 2);
  ReadRaw(In, Pairs.data(), Pairs.size());
  T.Src.resize(T.NEdges);
  T.Dst.resize(T.NEdges);
  for (uint32_t E = 0; E < T.NEdges; ++E) {
    T.Src[E] = Pairs[2 * E];
    T.Dst[E] = Pairs[2 * E + 1];
  }

  // data: per-step (x[NIn], target)
  std::vector<float> Rec(static_cast<size_t>(T.NSteps) * (T.NIn + 1));
  ReadRaw(In, Rec.data(), Rec.size());
  T.X.resize(static_cast<size_t>(T.NSteps) * T.NIn);
  T.Y.resize(T.NSteps);
  for (uint32_t S = 0; S < T.NSteps; ++S) {
    const float *Row = Rec.data() + static_cast<size_t>(S) * (T.NIn + 1);
    std::memcpy(T.X.data() + static_cast<size_t>(S) * T.NIn, Row,
                T.NIn * sizeof(float));
    T.Y[S] = Row[T.NIn];
  }
  return static_cast<bool>(In);
}

static std::filesystem::path ResolveTopology(const bench::CliArgs &Args) {
  auto It = Args.Extras.find("topology");
  if (It != Args.Extras.end())
    return It->second;
  std::vector<std::filesystem::path> Candidates = {
      Args.DataDir / "topology.bin",
      Args.DataDir / "10_engineered_sparse_large_nn" / "topology.bin",
      "10_engineered_sparse_large_nn/topology.bin",
  };
  for (const auto &C : Candidates)
    if (std::filesystem::exists(C))
      return C;
  return {};
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  bench::MemoryProbe MP;
  MP.Start();

  size_t MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", 0));  // 0 = all
  size_t LogEvery = static_cast<size_t>(Args.GetInt("log-every", 1000));
  if (Args.Quick) {
    size_t Cap = 2000;
    MaxSteps = (MaxSteps == 0) ? Cap : std::min(MaxSteps, Cap);
    LogEvery = std::min<size_t>(LogEvery, 500);
  }

  auto TopoPath = ResolveTopology(Args);
  if (TopoPath.empty() || !std::filesystem::exists(TopoPath)) {
    std::cerr << "[err] topology.bin not found. Looked under "
              << (Args.DataDir / "topology.bin")
              << " and 10_engineered_sparse_large_nn/topology.bin\n";
    return 2;
  }

  Topology Topo;
  if (!LoadTopology(TopoPath, Topo)) {
    std::cerr << "[err] failed to load topology " << TopoPath << "\n";
    return 2;
  }
  MP.EndDataset();

  size_t N = Topo.NSteps;
  if (MaxSteps != 0)
    N = std::min<size_t>(N, MaxSteps);

  std::cout << "[info] topology=" << TopoPath.string() << " n_in=" << Topo.NIn
            << " n_units=" << Topo.NUnits << " n_edges=" << Topo.NEdges
            << " n_steps=" << Topo.NSteps << " (using " << N << ")"
            << " output_id=" << Topo.OutputId << " quick=" << (Args.Quick ? 1 : 0)
            << "\n";

  const uint32_t NIn = Topo.NIn;
  const uint32_t OutputId = Topo.OutputId;

  // ---- dynamic state ------------------------------------------------------
  // Units: act[] indexed by unit id; layer[] per unit. Growth appends new
  // units, so these can extend past the initial NUnits.
  std::vector<float> Act(Topo.NUnits, 0.0f);
  std::vector<uint32_t> Layer = Topo.Layer;
  size_t NUnits = Topo.NUnits;

  // Edges in insertion order (initial edges first, then growth edges). A
  // pruned edge is marked dead via Alive[]; live ids are kept in LiveIdx[]
  // (insertion-ordered, no holes) so prune-by-(lcg()%live_count) and the
  // structural log line up across impls.
  std::vector<uint32_t> ESrc = Topo.Src;
  std::vector<uint32_t> EDst = Topo.Dst;
  std::vector<float> Weight(Topo.NEdges, 0.0f);   // initial weights = 0
  std::vector<float> Elig(Topo.NEdges, 0.0f);
  std::vector<uint8_t> Alive(Topo.NEdges, 1u);
  std::vector<uint32_t> LiveIdx(Topo.NEdges);     // insertion-ordered live edge ids
  for (uint32_t E = 0; E < Topo.NEdges; ++E)
    LiveIdx[E] = E;

  // Max hidden layer (for new-unit layer assignment). Hidden units sit in
  // layers 1..n_layers-1; output sits one past that. We use the output's
  // layer to bound the random hidden layer like the spec's max_hidden_layer.
  uint32_t MaxHiddenLayer = Layer[OutputId];  // == n_layers
  if (MaxHiddenLayer < 2)
    MaxHiddenLayer = 2;

  Lcg Rng;
  MP.EndWeights();

  // ---- output scaffolding -------------------------------------------------
  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "engineered_sparse_large_nn");
  bench::StructuralLog Log(HistPath);

  std::vector<float> Deltas(N, 0.0f);   // per-step delta, for tail MSE

  double WindowSse = 0.0;
  size_t WindowCnt = 0;
  size_t EpochIdx = 0;

  bench::PhaseTimer PT;
  std::vector<float> Pre(NUnits, 0.0f);  // grows with units

  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 0; Step < N; ++Step) {
    PT.Tick();

    // --- step 1: inputs ---
    const float *Xt = Topo.X.data() + Step * NIn;
    for (uint32_t I = 0; I < NIn; ++I)
      Act[I] = Xt[I];

    // --- step 2: pipeline forward (one synchronous sparse mat-vec over the
    //     *previous* activations). Accumulate pre[v] for non-input units. ---
    if (Pre.size() < NUnits)
      Pre.resize(NUnits);
    std::fill(Pre.begin(), Pre.begin() + NUnits, 0.0f);
    for (uint32_t Li = 0; Li < LiveIdx.size(); ++Li) {
      uint32_t E = LiveIdx[Li];
      Pre[EDst[E]] += Weight[E] * Act[ESrc[E]];
    }
    for (size_t V = NIn; V < NUnits; ++V) {
      if (V == OutputId)
        Act[V] = std::tanh(Pre[V]); // bounded output (keeps metric finite over long horizons)
      else
        Act[V] = std::tanh(Pre[V]); // tanh hidden
    }
    PT.MarkForward();

    // --- step 3: predict / loss ---
    float Yhat = Act[OutputId];
    float Delta = Topo.Y[Step] - Yhat;
    Deltas[Step] = Delta;
    double Err = static_cast<double>(Delta);
    WindowSse += Err * Err;
    ++WindowCnt;
    PT.MarkLoss();

    // --- step 4: uniform TD(lambda) update over every (live) edge ---
    for (uint32_t Li = 0; Li < LiveIdx.size(); ++Li) {
      uint32_t E = LiveIdx[Li];
      Elig[E] = DECAY * Elig[E] + Act[ESrc[E]];
      Weight[E] += LR * Delta * Elig[E];
    }
    PT.MarkUpdate();

    // --- growth / shrink ---
    PT.Tick();
    if (Step + 1 > WARMUP && (Step + 1 - WARMUP) % GROW_EVERY == 0) {
      // Grow GROW_UNITS hidden units while under the cap.
      for (uint32_t G = 0; G < GROW_UNITS; ++G) {
        if (NUnits >= MAX_UNITS)
          break;
        uint32_t NewUnit = static_cast<uint32_t>(NUnits);
        uint32_t NewLayer = 1u + (Rng.Next() % MaxHiddenLayer);
        // extend per-unit arrays
        Layer.push_back(NewLayer);
        Act.push_back(0.0f);
        ++NUnits;

        // FANIN incoming edges from distinct earlier units.
        std::vector<uint32_t> Chosen;
        Chosen.reserve(FANIN);
        uint32_t Added = 0;
        // bounded retries to avoid pathological infinite loops while still
        // matching the draw sequence in the common case.
        uint32_t Guard = 0;
        while (Added < FANIN && Guard < 10000) {
          ++Guard;
          uint32_t S = Rng.Next() % OutputId;
          if (Layer[S] >= NewLayer)
            continue;
          if (std::find(Chosen.begin(), Chosen.end(), S) != Chosen.end())
            continue;
          Chosen.push_back(S);
          // append edge (S -> NewUnit)
          uint32_t Eid = static_cast<uint32_t>(ESrc.size());
          ESrc.push_back(S);
          EDst.push_back(NewUnit);
          Weight.push_back(0.0f);
          Elig.push_back(0.0f);
          Alive.push_back(1u);
          LiveIdx.push_back(Eid);
          ++Added;
        }
        // one edge (NewUnit -> output)
        {
          uint32_t Eid = static_cast<uint32_t>(ESrc.size());
          ESrc.push_back(NewUnit);
          EDst.push_back(OutputId);
          Weight.push_back(0.0f);
          Elig.push_back(0.0f);
          Alive.push_back(1u);
          LiveIdx.push_back(Eid);
        }
      }

      // Shrink: prune PRUNE_EDGES edges chosen by lcg() % live_edge_count,
      // skipping edges into the output (keep it connected).
      for (uint32_t P = 0; P < PRUNE_EDGES; ++P) {
        if (LiveIdx.empty())
          break;
        uint32_t Pick = Rng.Next() % static_cast<uint32_t>(LiveIdx.size());
        uint32_t E = LiveIdx[Pick];
        if (EDst[E] == OutputId)
          continue;  // skip edges into output
        Alive[E] = 0u;
        LiveIdx.erase(LiveIdx.begin() + Pick);
      }
    }
    PT.MarkPrune(); // bench only removes edges; no growth phase
    PT.StepDone();

    // --- per-epoch structural log ---
    if (WindowCnt >= LogEvery || Step + 1 == N) {
      double WindowMse = WindowCnt > 0 ? WindowSse / WindowCnt : 0.0;
      ++EpochIdx;
      // count live units: a unit that still has at least one live edge
      // incident, or is an input/output. (Cheap approximation of "dead".)
      size_t LiveEdges = LiveIdx.size();
      Log.Log(EpochIdx, NUnits, LiveEdges, nullptr, &WindowMse,
              {{"train_loss", WindowMse},
               {"epoch", static_cast<double>(EpochIdx)},
               {"step", static_cast<double>(Step + 1)},
               {"n_units", static_cast<double>(NUnits)},
               {"n_edges", static_cast<double>(LiveEdges)}});
      std::cout << "[ep " << EpochIdx << "] step=" << (Step + 1)
                << " window_mse=" << WindowMse << " n_units=" << NUnits
                << " n_edges=" << LiveEdges << "\n";
      WindowSse = 0.0;
      WindowCnt = 0;
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  // test_mse = mean delta^2 over the last 10% of steps.
  size_t TailStart = N - std::max<size_t>(1, N / 10);
  double TailSse = 0.0;
  for (size_t S = TailStart; S < N; ++S)
    TailSse += static_cast<double>(Deltas[S]) * static_cast<double>(Deltas[S]);
  double TestMse = TailSse / std::max<size_t>(1, N - TailStart);

  Log.Flush();

  size_t FinalEdges = LiveIdx.size();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"10_engineered_sparse_large_nn"});
  S.Set("max_steps", static_cast<int>(N));
  S.Set("wall_seconds", Wall);
  S.Set("test_mse", TestMse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<long long>(NUnits));
  S.Set("n_edges", static_cast<long long>(FinalEdges));
  S.Set("seed", Args.Seed);
  PT.WriteSummary(S, Wall);
  // PhaseTimer leaves backward/reset implicitly 0; make them explicit.
  S.Set("backward_ns_mean", 0.0);
  S.Set("backward_ns_std", 0.0);
  S.Set("reset_ns_mean", 0.0);
  S.Set("reset_ns_std", 0.0);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TestMse
            << "  n_units=" << NUnits << "  n_edges=" << FinalEdges << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
