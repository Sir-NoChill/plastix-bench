// Workload 9 — imprinting-learner on the audio-prediction benchmark.
//
// Streaming TD prediction task. Each step gives a 2500-dim binary observation
// (audio-spectrogram bitmap) and a {-1,0,+1} reward. The learner outputs a
// scalar prediction V_t. We score it against the discounted return
//   G_t = Σ_k γ^k · r_{t+k}
// computed offline from the reward stream, and report mean-squared error
// (over the last 10% of the run for the "test" slice, the running window for
// per-epoch history).
//
// This binary calls into the standalone imprinting-learner C++ library that
// ships under `09_imprintin_learner/cpp/{include,src,third_party}`
// (linked as `il::agent`). The bench-style scaffolding (`bench::CliArgs`,
// `bench::StructuralLog`, `bench::SummaryWriter`) is shared with every other
// `cpp/` impl in this directory.

#include "cpp/common.hpp"

#include "dataset.hpp"
#include "imprinting/imprinting_learner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>

namespace {

struct HP {
  // Default = 20k steps full / 1k steps quick. Capped to dataset size.
  size_t MaxSteps = 20000;
  size_t LogEvery = 2000;
  float Gamma = 0.99f;
  float Alpha = 3e-3f;
  float Lambda = 0.9f;
  float Eta = 0.1f;
  float EpsilonZ = 0.01f;
  uint32_t KPattern = 10;
  uint32_t KMemory = 10;
};

static std::filesystem::path
ResolveDataset(const bench::CliArgs &Args) {
  // Explicit override wins.
  auto It = Args.Extras.find("dataset");
  if (It != Args.Extras.end())
    return It->second;

  // Search order: --data-dir candidates first, then the in-tree default that
  // ships with the imprinting-learner sub-project.
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

// Discounted return G_t = Σ_k γ^k · r_{t+k}, computed backwards over the
// reward stream. Returns a vector of length N.
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
  H.Gamma = Args.GetFloat("gamma", H.Gamma);
  H.Alpha = Args.GetFloat("alpha", H.Alpha);
  H.Lambda = Args.GetFloat("lambda", H.Lambda);
  H.Eta = Args.GetFloat("eta", H.Eta);
  H.EpsilonZ = Args.GetFloat("epsilon-z", H.EpsilonZ);
  H.KPattern = static_cast<uint32_t>(Args.GetInt("k-pattern", H.KPattern));
  H.KMemory = static_cast<uint32_t>(Args.GetInt("k-memory", H.KMemory));
  if (Args.Quick) {
    H.MaxSteps = std::min<size_t>(H.MaxSteps, 1000);
    H.LogEvery = std::min<size_t>(H.LogEvery, 200);
  }

  auto DatasetPath = ResolveDataset(Args);
  if (DatasetPath.empty() || !std::filesystem::exists(DatasetPath)) {
    std::cerr << "[err] audio-prediction dataset.bin not found. Looked under\n"
              << "        " << (Args.DataDir / "audio_prediction" / "dataset.bin")
              << "\n        09_imprintin_learner/cpp/examples/"
                 "output/dataset.bin\n"
              << "      Generate via examples/prepare-cpp.py, or pass "
                 "--dataset <path>.\n";
    return 2;
  }

  audio_pred::Dataset DS(DatasetPath);
  size_t N = std::min<size_t>(H.MaxSteps, DS.Size());

  std::cout << "[info] dataset=" << DatasetPath.string() << " ("
            << DS.Size() << " steps, using " << N << ")"
            << " gamma=" << H.Gamma << " alpha=" << H.Alpha
            << " eta=" << H.Eta << " k_pattern=" << H.KPattern
            << " k_memory=" << H.KMemory << " quick=" << (Args.Quick ? 1 : 0)
            << "\n";

  // Compute the offline discounted-return target stream over the full window
  // we'll actually run; using a longer lookahead would just add float noise
  // at the tail.
  std::vector<int> Rewards(N);
  for (size_t T = 0; T < N; ++T)
    Rewards[T] = DS[T].Reward();
  auto Returns = ComputeReturns(Rewards, H.Gamma);

  il::HyperParams Hp;
  Hp.capacity = 16384;
  Hp.trace_dim = 1;
  Hp.tenure_threshold = 0.01f;
  Hp.tenure_track_threshold = 3e-4f;
  Hp.demotion_factor = 0.5f;
  Hp.gamma = H.Gamma;
  Hp.lambda = H.Lambda;
  Hp.alpha = H.Alpha;
  Hp.eta = H.Eta;
  Hp.eta_min = 1e-7f;
  Hp.decay = 0.999f;
  Hp.meta_step_size = 1e-3f;
  Hp.epsilon = 1e-5f;
  Hp.k_pattern = H.KPattern;
  Hp.k_memory = H.KMemory;
  Hp.pattern_fractions = {0.6f, 0.7f, 0.8f, 0.9f};
  Hp.pattern_min_connections = 2;
  Hp.pattern_max_connections = 8;
  Hp.memory_delay_min = 0;
  Hp.memory_delay_max = 20;
  Hp.memory_window_min = 1;
  Hp.memory_window_max = 3;
  Hp.rng_seed = static_cast<uint64_t>(Args.Seed) + 0x9E3779B97F4A7C15ull;
  Hp.epsilon_z = H.EpsilonZ;

  il::ImprintingLearner Learner(Hp);
  Learner.addObservations(audio_pred::ObservationDim);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "audio_imprinting");
  bench::StructuralLog Log(HistPath);

  std::array<uint8_t, audio_pred::ObservationDim> Obs{};
  std::vector<float> Predictions(N, 0.0f);

  // Running-window MSE inside the current "epoch" (length LogEvery). We log
  // one StructuralLog record per window for per-epoch parity with the other
  // benches in the suite.
  double WindowSse = 0.0;
  size_t WindowCnt = 0;
  size_t EpochIdx = 0;
  double GlobalSse = 0.0;

  auto T0 = std::chrono::steady_clock::now();
  for (size_t T = 0; T < N; ++T) {
    const audio_pred::StepView Step = DS[T];
    for (size_t I = 0; I < audio_pred::ObservationDim; ++I)
      Obs[I] = Step.Test(I) ? uint8_t{1} : uint8_t{0};
    float V = Learner.step(Obs, static_cast<float>(Step.Reward()));
    Predictions[T] = V;

    double Err = static_cast<double>(V) - static_cast<double>(Returns[T]);
    WindowSse += Err * Err;
    ++WindowCnt;
    GlobalSse += Err * Err;

    if (WindowCnt >= H.LogEvery || T + 1 == N) {
      double WindowMse = WindowCnt > 0 ? WindowSse / WindowCnt : 0.0;
      ++EpochIdx;
      Log.Log(EpochIdx, Learner.arena().size(), 0u, nullptr, &WindowMse,
              {{"train_loss", WindowMse},
               {"epoch", static_cast<double>(EpochIdx)},
               {"step", static_cast<double>(T + 1)},
               {"n_features", static_cast<double>(Learner.arena().size())}});
      std::cout << "[ep " << EpochIdx << "] step=" << (T + 1)
                << " window_mse=" << WindowMse
                << " features=" << Learner.arena().size() << "\n";
      WindowSse = 0.0;
      WindowCnt = 0;
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  // "Test" slice: MSE over the final 10% of predictions vs. discounted return.
  size_t TailStart = N - std::max<size_t>(1, N / 10);
  double TailSse = 0.0;
  for (size_t T = TailStart; T < N; ++T) {
    double E = static_cast<double>(Predictions[T]) -
               static_cast<double>(Returns[T]);
    TailSse += E * E;
  }
  double TestMse = TailSse / std::max<size_t>(1, N - TailStart);
  double FullMse = N > 0 ? GlobalSse / static_cast<double>(N) : 0.0;

  // Per-phase timing comes from the library's Welford accumulators, so we
  // get mean + std without storing per-step samples. The library bundles
  // "TD δ + weight update" into `backward` to mirror what PyTorch's
  // `loss.backward() + optimizer.step()` would produce; `structural`
  // covers tenure + remove + generate + activation snapshot.
  const auto &Prof = Learner.profile();
  uint64_t Steps = std::max<uint64_t>(Prof.step_count, 1);
  double FwdNsMean = Prof.forward.mean;
  double FwdNsStd = Prof.forward.stddev();
  double BwdNsMean = Prof.backward.mean;
  double BwdNsStd = Prof.backward.stddev();
  double StructNsMean = Prof.structural.mean;
  double StructNsStd = Prof.structural.stddev();
  double StepNsMean = (Wall * 1e9) / static_cast<double>(Steps);
  double OtherNsMean =
      std::max<double>(0.0, StepNsMean - FwdNsMean - BwdNsMean - StructNsMean);

  Log.Flush();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"09_imprintin_learner"});
  S.Set("dataset", std::string{"audio_prediction"});
  S.Set("max_steps", static_cast<int>(N));
  S.Set("gamma", static_cast<double>(H.Gamma));
  S.Set("alpha", static_cast<double>(H.Alpha));
  S.Set("eta", static_cast<double>(H.Eta));
  S.Set("epsilon_z", static_cast<double>(H.EpsilonZ));
  S.Set("k_pattern", static_cast<int>(H.KPattern));
  S.Set("k_memory", static_cast<int>(H.KMemory));
  S.Set("wall_seconds", Wall);
  S.Set("val_mse_final", FullMse);
  S.Set("test_mse", TestMse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<int>(Learner.arena().size()));
  S.Set("n_edges", static_cast<int>(Learner.arena().size()));
  // Per-phase ns/step. `backward` bundles "TD δ + weight update" so the
  // pytorch impl's `backward + optimizer.step` column lines up directly.
  // Loss/update/reset are 0 for this bench — the library combines them
  // inside its monolithic step() API.
  S.Set("step_count", static_cast<long long>(Steps));
  S.Set("step_ns_mean", StepNsMean);
  S.Set("forward_ns_mean", FwdNsMean);
  S.Set("forward_ns_std", FwdNsStd);
  S.Set("loss_ns_mean", 0.0);
  S.Set("loss_ns_std", 0.0);
  S.Set("backward_ns_mean", BwdNsMean);
  S.Set("backward_ns_std", BwdNsStd);
  S.Set("update_ns_mean", 0.0);
  S.Set("update_ns_std", 0.0);
  S.Set("structural_ns_mean", StructNsMean);
  S.Set("structural_ns_std", StructNsStd);
  S.Set("reset_ns_mean", 0.0);
  S.Set("reset_ns_std", 0.0);
  S.Set("other_ns_mean", OtherNsMean);
  S.Set("seed", Args.Seed);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TestMse
            << "  full_mse=" << FullMse
            << "  final_features=" << Learner.arena().size() << "\n";
  std::cout << "[phase] step=" << StepNsMean << "ns  forward=" << FwdNsMean
            << "ns  backward=" << BwdNsMean
            << "ns  structural=" << StructNsMean
            << "ns  other=" << OtherNsMean << "ns  (mean per step over "
            << Steps << " steps)\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
