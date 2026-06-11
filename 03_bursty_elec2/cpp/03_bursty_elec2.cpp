// Workload 3 / 5 — BURSTY regime, raw-C++/OpenBLAS translation.
//
// Streaming binary classification on the Elec2 dataset (auto-downloaded by
// the PyTorch reference; here we read the cached CSV). The network sits
// structurally idle for many steps, then — when a sliding window of held-out
// losses goes flat — fires a burst: grow N hidden units, then immediately
// magnitude-prune to keep edge count bounded. Mirrors
// 03_bursty_elec2.py.
//
// Per the "pre-allocate max + mask" decision, both hidden layers are
// physically (max_hidden x ...) up front; growth flips mask bits and seeds
// the new weights, no realloc.

#include "cpp/common.hpp"
#include "cpp/mlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct HP {
  size_t InitHidden = 32;
  size_t MaxHidden = 512; // pre-allocated cap on hidden width
  size_t Batch = 64;
  size_t MaxSteps = 3000;
  size_t ValEvery = 25;
  size_t ValWindow = 512;
  // mean-reduction CE: effective lr ~ PyTorch_sum_lr * batch = 1e-4 * 64.
  float Lr = 6.4e-3f;
  float BurstFrac = 0.15f;
  float BurstNoise = 0.05f;
  float PostBurstPruneFrac = 0.10f;
  size_t PlateauWindow = 8;
  float PlateauRelTol = 0.03f;
  size_t PlateauCooldown = 4;
};

struct Dataset {
  std::vector<float> X; // N x D
  std::vector<int> Y;   // N
  size_t N = 0;
  size_t D = 0;
  size_t NumClasses = 2;
};

static Dataset SynthDrift(size_t N, size_t Dim, uint32_t Seed) {
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> Norm(0.0f, 1.0f);
  std::uniform_real_distribution<float> U(0.0f, 1.0f);
  Dataset D;
  D.N = N;
  D.D = Dim;
  D.NumClasses = 2;
  D.X.resize(N * Dim);
  D.Y.resize(N);
  for (size_t I = 0; I < N; ++I) {
    float T = 4.0f * 3.14159265358979323846f * static_cast<float>(I) /
              static_cast<float>(N);
    float Dot = 0.0f;
    for (size_t J = 0; J < Dim; ++J) {
      D.X[I * Dim + J] = Norm(Rng);
      Dot += D.X[I * Dim + J] *
             std::sin(T + static_cast<float>(J));
    }
    D.Y[I] = (Dot > 0.0f) ? 1 : 0;
    (void)U;
  }
  return D;
}

static Dataset LoadElec2(const std::filesystem::path &Dir) {
  Dataset D;
  auto Path = Dir / "elec2.csv";
  if (!std::filesystem::exists(Path))
    return D;
  auto Csv = bench::ReadCsv(Path);
  if (Csv.Header.empty())
    return D;
  // Target column: 'class' if present, else last.
  size_t TargetIdx = Csv.Header.size() - 1;
  for (size_t I = 0; I < Csv.Header.size(); ++I)
    if (Csv.Header[I] == "class")
      TargetIdx = I;
  std::vector<size_t> FeatIdx;
  for (size_t I = 0; I < Csv.Header.size(); ++I)
    if (I != TargetIdx)
      FeatIdx.push_back(I);
  D.D = FeatIdx.size();
  D.N = Csv.Rows.size();
  D.X.assign(D.N * D.D, 0.0f);
  D.Y.assign(D.N, 0);
  // Detect string labels by trying to parse and remapping if needed.
  std::unordered_map<std::string, int> Remap;
  for (size_t I = 0; I < D.N; ++I) {
    const auto &R = Csv.Rows[I];
    for (size_t J = 0; J < D.D; ++J) {
      size_t Col = FeatIdx[J];
      if (Col < R.size()) {
        try {
          D.X[I * D.D + J] = std::stof(R[Col]);
        } catch (...) {
          D.X[I * D.D + J] = 0.0f;
        }
      }
    }
    int Y = 0;
    if (TargetIdx < R.size()) {
      const std::string &S = R[TargetIdx];
      try {
        Y = std::stoi(S);
      } catch (...) {
        auto It = Remap.find(S);
        if (It == Remap.end()) {
          Y = static_cast<int>(Remap.size());
          Remap[S] = Y;
        } else {
          Y = It->second;
        }
      }
    }
    D.Y[I] = Y;
  }
  // Normalise labels to {0, 1, ...} starting at 0.
  int MinY = *std::min_element(D.Y.begin(), D.Y.end());
  int MaxY = *std::max_element(D.Y.begin(), D.Y.end());
  if (MinY != 0)
    for (auto &Y : D.Y)
      Y -= MinY;
  D.NumClasses = static_cast<size_t>(MaxY - MinY + 1);
  return D;
}

// Per-feature standardisation over the first `Cut` rows.
static void StandardiseFeatures(Dataset &D, size_t Cut) {
  if (D.N == 0 || D.D == 0)
    return;
  size_t Used = std::min(Cut, D.N);
  std::vector<double> Mu(D.D, 0.0), Sd(D.D, 0.0);
  for (size_t I = 0; I < Used; ++I)
    for (size_t J = 0; J < D.D; ++J)
      Mu[J] += D.X[I * D.D + J];
  for (auto &M : Mu)
    M /= static_cast<double>(Used);
  for (size_t I = 0; I < Used; ++I)
    for (size_t J = 0; J < D.D; ++J) {
      double Diff = D.X[I * D.D + J] - Mu[J];
      Sd[J] += Diff * Diff;
    }
  for (auto &V : Sd) {
    V = std::sqrt(V / static_cast<double>(Used));
    if (V < 1e-3)
      V = 1.0;
  }
  for (size_t I = 0; I < D.N; ++I)
    for (size_t J = 0; J < D.D; ++J)
      D.X[I * D.D + J] =
          static_cast<float>((D.X[I * D.D + J] - Mu[J]) / Sd[J]);
}

// Growable MLP: in -> H -> H -> out. H grows via flipping mask bits on
// pre-allocated layers. The middle layer (H x H) is the one whose mask
// "expands" as new units come online.
struct GrowableMLP {
  raw::Linear L0; // in -> H (out_dim grows; in dim fixed)
  raw::Linear L1; // H -> H (both dims grow; sub-block (h x h))
  raw::Linear L2; // H -> out (out fixed; in dim grows)
  size_t Hidden = 0;
  size_t InDim = 0;
  size_t OutDim = 0;
  size_t MaxHidden = 0;
  size_t Bursts = 0;
  size_t Prunes = 0;

  std::vector<float> Act0; // batch x H
  std::vector<float> Act1; // batch x H
  std::vector<float> Act2; // batch x out

  void Init(size_t InDim_, size_t OutDim_, size_t InitHidden,
            size_t MaxHidden_, std::mt19937 &Rng) {
    InDim = InDim_;
    OutDim = OutDim_;
    Hidden = InitHidden;
    MaxHidden = MaxHidden_;
    L0.Init(InDim, InitHidden, InDim, MaxHidden);
    L1.Init(InitHidden, InitHidden, MaxHidden, MaxHidden);
    L2.Init(InitHidden, OutDim, MaxHidden, OutDim);
    // Xavier init for the live block of each layer.
    float Lim0 = raw::XavierLimit(InDim, InitHidden);
    float Lim1 = raw::XavierLimit(InitHidden, InitHidden);
    float Lim2 = raw::XavierLimit(InitHidden, OutDim);
    std::uniform_real_distribution<float> U0(-Lim0, Lim0);
    std::uniform_real_distribution<float> U1(-Lim1, Lim1);
    std::uniform_real_distribution<float> U2(-Lim2, Lim2);
    for (size_t I = 0; I < InitHidden; ++I)
      for (size_t J = 0; J < InDim; ++J)
        L0.Weight[I * L0.InCap + J] = U0(Rng);
    for (size_t I = 0; I < InitHidden; ++I)
      for (size_t J = 0; J < InitHidden; ++J)
        L1.Weight[I * L1.InCap + J] = U1(Rng);
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InitHidden; ++J)
        L2.Weight[I * L2.InCap + J] = U2(Rng);
    L0.EnableMask();
    L1.EnableMask();
    L2.EnableMask();
  }

  size_t UnitCount() const { return Hidden + Hidden + OutDim; }
  size_t EdgeCount() const {
    return L0.AliveCount() + L1.AliveCount() + L2.AliveCount();
  }

  bench::EdgeSet EdgeSetAlive() const {
    bench::EdgeSet Out;
    auto Pull = [&](uint32_t LI, const raw::Linear &L) {
      for (size_t I = 0; I < L.OutDim; ++I)
        for (size_t J = 0; J < L.InDim; ++J)
          if (L.Mask[I * L.InCap + J])
            Out.insert(bench::PackEdge(LI, static_cast<uint32_t>(I),
                                       static_cast<uint32_t>(J)));
    };
    Pull(0, L0);
    Pull(1, L1);
    Pull(2, L2);
    return Out;
  }

  // Forward through (in -> H -> H -> out) with ReLU between hidden layers,
  // linear at the output.
  const float *Forward(const float *X, size_t Batch) {
    Act0.assign(Batch * L0.OutDim, 0.0f);
    L0.Forward(X, Act0.data(), Batch);
    raw::ApplyReLU(Act0.data(), Batch * L0.OutDim);
    Act1.assign(Batch * L1.OutDim, 0.0f);
    L1.Forward(Act0.data(), Act1.data(), Batch);
    raw::ApplyReLU(Act1.data(), Batch * L1.OutDim);
    Act2.assign(Batch * L2.OutDim, 0.0f);
    L2.Forward(Act1.data(), Act2.data(), Batch);
    return Act2.data();
  }

  void Backward(const float *X, const float *TopGrad, size_t Batch) {
    std::vector<float> G2(Batch * L2.OutDim);
    std::memcpy(G2.data(), TopGrad, Batch * L2.OutDim * sizeof(float));
    L2.BackwardWeights(Act1.data(), G2.data(), Batch, 1.0f);
    std::vector<float> G1(Batch * L1.OutDim, 0.0f);
    L2.BackwardInput(G2.data(), G1.data(), Batch);
    raw::ReLUBackward(Act1.data(), G1.data(), Batch * L1.OutDim);
    L1.BackwardWeights(Act0.data(), G1.data(), Batch, 1.0f);
    std::vector<float> G0(Batch * L0.OutDim, 0.0f);
    L1.BackwardInput(G1.data(), G0.data(), Batch);
    raw::ReLUBackward(Act0.data(), G0.data(), Batch * L0.OutDim);
    L0.BackwardWeights(X, G0.data(), Batch, 1.0f);
  }

  void Update(float Lr) {
    L0.SGD(Lr);
    L1.SGD(Lr);
    L2.SGD(Lr);
  }

  // Grow `n_new` hidden units: extend the live block in L0 (new output rows),
  // L1 (new rows and columns), and L2 (new input columns). All new edges
  // start alive and are seeded with small Gaussian noise (matches the
  // PyTorch reference). Returns the new hidden width.
  size_t Grow(size_t NNew, float Noise, std::mt19937 &Rng) {
    if (NNew == 0)
      return Hidden;
    size_t Want = std::min(MaxHidden, Hidden + NNew);
    size_t Added = Want - Hidden;
    if (Added == 0)
      return Hidden;
    std::normal_distribution<float> N(0.0f, Noise);
    // L0: new rows [Hidden, Want) x InDim, all live, weights ~ N(0, Noise).
    for (size_t I = Hidden; I < Want; ++I) {
      for (size_t J = 0; J < InDim; ++J) {
        L0.Weight[I * L0.InCap + J] = N(Rng);
        L0.Mask[I * L0.InCap + J] = 1;
      }
      L0.Bias[I] = 0.0f;
    }
    // L1: extend logical (Hidden x Hidden) block to (Want x Want).
    //   - new rows  [Hidden, Want) x [0, Want)
    //   - new cols  [0, Hidden)     x [Hidden, Want)
    for (size_t I = Hidden; I < Want; ++I) {
      for (size_t J = 0; J < Want; ++J) {
        L1.Weight[I * L1.InCap + J] = N(Rng);
        L1.Mask[I * L1.InCap + J] = 1;
      }
      L1.Bias[I] = 0.0f;
    }
    for (size_t I = 0; I < Hidden; ++I)
      for (size_t J = Hidden; J < Want; ++J) {
        L1.Weight[I * L1.InCap + J] = N(Rng);
        L1.Mask[I * L1.InCap + J] = 1;
      }
    // L2: new cols [0, OutDim) x [Hidden, Want).
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = Hidden; J < Want; ++J) {
        L2.Weight[I * L2.InCap + J] = N(Rng);
        L2.Mask[I * L2.InCap + J] = 1;
      }
    Hidden = Want;
    L0.OutDim = Hidden;
    L1.OutDim = Hidden;
    L1.InDim = Hidden;
    L2.InDim = Hidden;
    ++Bursts;
    return Hidden;
  }

  // Global magnitude prune across L0, L1, L2's live blocks.
  size_t MagnitudePrune(float PruneFrac) {
    std::vector<float> Alive;
    auto Push = [&](const raw::Linear &L) {
      for (size_t I = 0; I < L.OutDim; ++I)
        for (size_t J = 0; J < L.InDim; ++J) {
          size_t Idx = I * L.InCap + J;
          if (L.Mask[Idx])
            Alive.push_back(std::abs(L.Weight[Idx]));
        }
    };
    Push(L0);
    Push(L1);
    Push(L2);
    if (Alive.empty())
      return 0;
    size_t K = static_cast<size_t>(PruneFrac * Alive.size());
    if (K == 0)
      return 0;
    if (K > Alive.size())
      K = Alive.size();
    std::nth_element(Alive.begin(), Alive.begin() + K - 1, Alive.end());
    float Thr = Alive[K - 1];
    size_t Killed = 0;
    auto Kill = [&](raw::Linear &L) {
      for (size_t I = 0; I < L.OutDim; ++I)
        for (size_t J = 0; J < L.InDim; ++J) {
          size_t Idx = I * L.InCap + J;
          if (L.Mask[Idx] && std::abs(L.Weight[Idx]) <= Thr) {
            L.Mask[Idx] = 0;
            L.Weight[Idx] = 0.0f;
            ++Killed;
          }
        }
    };
    Kill(L0);
    Kill(L1);
    Kill(L2);
    ++Prunes;
    return Killed;
  }
};

// Plateau detector: flat if std/mean over the rolling window < rel_tol.
struct PlateauDetector {
  std::deque<float> Win;
  size_t Window;
  float RelTol;
  size_t Cooldown;
  size_t StepsSinceBurst;
  PlateauDetector(size_t W, float Tol, size_t Cool)
      : Window(W), RelTol(Tol), Cooldown(Cool), StepsSinceBurst(Cool) {}
  bool Update(float Loss) {
    Win.push_back(Loss);
    if (Win.size() > Window)
      Win.pop_front();
    ++StepsSinceBurst;
    if (Win.size() < Window || StepsSinceBurst < Cooldown)
      return false;
    double Mu = 0.0;
    for (float V : Win)
      Mu += V;
    Mu /= Win.size();
    if (Mu <= 0.0)
      return false;
    double Var = 0.0;
    for (float V : Win)
      Var += (V - Mu) * (V - Mu);
    double Sd = std::sqrt(Var / Win.size());
    if (Sd / Mu < RelTol) {
      StepsSinceBurst = 0;
      return true;
    }
    return false;
  }
};

// Cross-entropy on a fixed slice (no gradient).
static double EvalCEOnSlice(GrowableMLP &M, const Dataset &D, size_t Start,
                            size_t Count) {
  std::vector<float> Probs(Count * D.NumClasses);
  std::vector<float> Grad(Count * D.NumClasses);
  const float *Logits = M.Forward(D.X.data() + Start * D.D, Count);
  return raw::CrossEntropyMean(Logits, D.Y.data() + Start, Count, D.NumClasses,
                               Probs.data(), Grad.data());
}

static double EvalAccOnSlice(GrowableMLP &M, const Dataset &D, size_t Start,
                             size_t Count) {
  const float *Logits = M.Forward(D.X.data() + Start * D.D, Count);
  return raw::ArgmaxAccuracy(Logits, D.Y.data() + Start, Count, D.NumClasses);
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.InitHidden = static_cast<size_t>(Args.GetInt("init-hidden", H.InitHidden));
  H.MaxHidden = static_cast<size_t>(Args.GetInt("max-hidden", H.MaxHidden));
  H.Batch = static_cast<size_t>(Args.GetInt("batch", H.Batch));
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.ValEvery = static_cast<size_t>(Args.GetInt("val-every", H.ValEvery));
  H.ValWindow = static_cast<size_t>(Args.GetInt("val-window", H.ValWindow));
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.BurstFrac = Args.GetFloat("burst-frac", H.BurstFrac);
  H.BurstNoise = Args.GetFloat("burst-noise", H.BurstNoise);
  H.PostBurstPruneFrac =
      Args.GetFloat("post-burst-prune-frac", H.PostBurstPruneFrac);
  H.PlateauWindow =
      static_cast<size_t>(Args.GetInt("plateau-window", H.PlateauWindow));
  H.PlateauRelTol = Args.GetFloat("plateau-rel-tol", H.PlateauRelTol);
  H.PlateauCooldown = static_cast<size_t>(
      Args.GetInt("plateau-cooldown", H.PlateauCooldown));
  if (Args.Quick) {
    H.MaxSteps = std::max<size_t>(50, H.MaxSteps / 5);
    H.ValEvery = std::max<size_t>(1, H.ValEvery / 2);
  }

  Dataset D;
  std::string DatasetName;
  if (Args.Synthetic) {
    D = SynthDrift(H.MaxSteps * H.Batch + 5000, 8,
                   static_cast<uint32_t>(Args.Seed));
    DatasetName = "synthetic-drift";
  } else {
    D = LoadElec2(Args.DataDir);
    if (D.N == 0) {
      std::cerr << "[warn] Elec2 not found; falling back to synthetic\n";
      D = SynthDrift(H.MaxSteps * H.Batch + 5000, 8,
                     static_cast<uint32_t>(Args.Seed));
      DatasetName = "synthetic-drift-fallback";
    } else {
      DatasetName = "elec2";
    }
  }
  size_t Cut = std::max<size_t>(static_cast<size_t>(0.1 * D.N), 256);
  StandardiseFeatures(D, Cut);

  std::cout << "[info] in_dim=" << D.D << " classes=" << D.NumClasses
            << " data=" << DatasetName << " N=" << D.N
            << " steps=" << H.MaxSteps << " init_h=" << H.InitHidden
            << " max_h=" << H.MaxHidden << "\n";

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 13u);
  GrowableMLP M;
  M.Init(D.D, D.NumClasses, H.InitHidden, H.MaxHidden, InitRng);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "bursty_elec2");
  bench::StructuralLog Log(HistPath);

  PlateauDetector Det(H.PlateauWindow, H.PlateauRelTol, H.PlateauCooldown);

  // Carve final 15% as held-out test set (mirrors pytorch/plastix). Training
  // cycles only over the first 85% so the test slice is never seen.
  size_t NTrain = static_cast<size_t>(0.85 * static_cast<double>(D.N));
  if (NTrain < H.ValWindow + 1)
    NTrain = D.N; // tiny dataset fallback
  size_t TestStart = NTrain;
  size_t TestCount = D.N - NTrain;

  // Static held-out slice for the plateau detector (within training portion).
  size_t PStart =
      std::min<size_t>(static_cast<size_t>(0.1 * NTrain),
                       NTrain > H.ValWindow + 1 ? NTrain - H.ValWindow - 1 : 0);
  size_t PCount = std::min<size_t>(H.ValWindow, NTrain - PStart);

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::vector<float> XBatch(H.Batch * D.D);
  std::vector<int> YBatch(H.Batch);
  std::vector<float> Probs(H.Batch * D.NumClasses);
  std::vector<float> Grad(H.Batch * D.NumClasses);

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    // Cyclic minibatch over the stream.
    for (size_t B = 0; B < H.Batch; ++B) {
      // Wrap within the training portion so the held-out test slice stays
      // unseen (matches pytorch reference behaviour).
      size_t Idx = ((Step - 1) * H.Batch + B) % NTrain;
      std::memcpy(XBatch.data() + B * D.D, D.X.data() + Idx * D.D,
                  D.D * sizeof(float));
      YBatch[B] = D.Y[Idx];
    }
    Timer.Tick();
    const float *Logits = M.Forward(XBatch.data(), H.Batch);
    Timer.MarkForward();
    raw::CrossEntropyMean(Logits, YBatch.data(), H.Batch, D.NumClasses,
                          Probs.data(), Grad.data());
    Timer.MarkLoss();
    M.Backward(XBatch.data(), Grad.data(), H.Batch);
    Timer.MarkBackward();
    M.Update(H.Lr);
    Timer.MarkUpdate();
    Timer.StepDone();

    if (Step % H.ValEvery == 0 || Step == H.MaxSteps) {
      double PlateauLoss = EvalCEOnSlice(M, D, PStart, PCount);
      // Streaming validation window — slides through the training portion.
      size_t VStart = (Step * H.Batch) % NTrain;
      size_t VEnd = std::min(VStart + H.ValWindow, NTrain);
      size_t VCount = VEnd - VStart;
      double VL = EvalCEOnSlice(M, D, VStart, VCount);
      double VA = EvalAccOnSlice(M, D, VStart, VCount);
      // Held-out test slice (final 15%), never seen during training.
      double TestLoss = TestCount > 0
                          ? EvalCEOnSlice(M, D, TestStart, TestCount)
                          : VL;
      double TestAcc = TestCount > 0
                          ? EvalAccOnSlice(M, D, TestStart, TestCount)
                          : VA;
      bool ShouldBurst = Det.Update(static_cast<float>(PlateauLoss));

      auto Edges = M.EdgeSetAlive();
      Log.Log(Step, M.UnitCount(), M.EdgeCount(), &Edges, &VL,
              {{"val_acc", VA},
               {"test_acc", TestAcc},
               {"test_loss", TestLoss},
               {"plateau_loss", PlateauLoss},
               {"bursts", static_cast<double>(M.Bursts)},
               {"prunes", static_cast<double>(M.Prunes)}});

      if (ShouldBurst) {
        size_t NNew =
            std::max<size_t>(1, static_cast<size_t>(H.BurstFrac * M.Hidden));
        size_t OldH = M.Hidden;
        size_t NewH = M.Grow(NNew, H.BurstNoise, Rng);
        size_t Killed = M.MagnitudePrune(H.PostBurstPruneFrac);
        std::cout << "[burst " << M.Bursts << "] step=" << Step
                  << "  hidden " << OldH << "->" << NewH
                  << "  killed=" << Killed << "  vl=" << VL << "\n";
      }
    }
  }
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
                    .count();
  Log.Flush();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"03_bursty_elec2"});
  S.Set("dataset", DatasetName);
  S.Set("in_dim", static_cast<int>(D.D));
  S.Set("n_classes", static_cast<int>(D.NumClasses));
  S.Set("init_hidden", static_cast<int>(H.InitHidden));
  S.Set("max_steps", static_cast<int>(H.MaxSteps));
  S.Set("batch", static_cast<int>(H.Batch));
  S.Set("burst_frac", static_cast<double>(H.BurstFrac));
  S.Set("post_burst_prune_frac",
        static_cast<double>(H.PostBurstPruneFrac));
  S.Set("plateau_window", static_cast<int>(H.PlateauWindow));
  S.Set("plateau_rel_tol", static_cast<double>(H.PlateauRelTol));
  S.Set("plateau_cooldown", static_cast<int>(H.PlateauCooldown));
  S.Set("wall_seconds", Wall);
  S.Set("bursts_fired", static_cast<int>(M.Bursts));
  S.Set("prunes_fired", static_cast<int>(M.Prunes));
  S.Set("hidden_final", static_cast<int>(M.Hidden));
  S.Set("edges_final", static_cast<int>(M.EdgeCount()));
  S.Set("val_loss_initial",
        Log.Records().empty() ? 0.0 : Log.Records().front().ValLoss);
  S.Set("val_loss_final",
        Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss);
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

  std::cout << "[done] wall=" << Wall << "s  bursts=" << M.Bursts
            << "  hidden_final=" << M.Hidden << "  edges=" << M.EdgeCount()
            << "  val_loss_final=" << Log.Records().back().ValLoss << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
