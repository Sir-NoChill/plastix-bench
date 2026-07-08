// Workload 2 / 5 — IDEMPOTENT SHRINKAGE, raw-C++/OpenBLAS translation.
//
// Iterative Magnitude Pruning (Frankle & Carbin 2018) of a multi-layer
// classification MLP on a UCR-style synthetic time-series task. Mirrors
// 02_idempotent_imp.py:
//
//   - alive-edge count decreases monotonically across rounds
//   - the dead-edge set is a growing union of prior rounds' kills
//   - the run terminates when a round kills zero edges (fixed point)
//
// Architecture: depth=4 = 3 ReLU hidden + 1 linear softmax readout, with
// per-layer Linear biases. Loss reduction is mean cross-entropy; the lr is
// tuned for that (effectively PyTorch_sum_lr * batch * D_logits).

#include "cpp/common.hpp"
#include "cpp/mlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct HP {
  size_t InLen = 128;
  size_t NumClasses = 5;
  size_t Hidden = 512;
  size_t Depth = 4;
  size_t NPerClass = 200;
  float Snr = 1.5f;
  size_t InitEpochs = 20;
  size_t FinetuneEpochs = 4;
  size_t MaxRounds = 20;
  float PruneFrac = 0.2f;
  size_t Batch = 128;
  // mean-reduction CE; effective lr ~ PyTorch_sum_lr * batch = 1e-4 * 128.
  float Lr = 1.28e-2f;
  float FinetuneLrScale = 0.3f;
};

struct Dataset {
  std::vector<float> X; // N x InLen
  std::vector<int> Y;   // N
  size_t N = 0;
  size_t Dim = 0;
};

static Dataset SynthUcr(size_t NPerClass, size_t NClasses, size_t Length,
                        float Snr, uint32_t Seed) {
  constexpr float Pi = 3.14159265358979323846f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> N(0.0f, 1.0f / Snr);
  Dataset D;
  D.Dim = Length;
  D.N = NPerClass * NClasses;
  D.X.resize(D.N * Length);
  D.Y.resize(D.N);
  size_t Row = 0;
  for (size_t K = 0; K < NClasses; ++K) {
    std::vector<float> Base(Length);
    for (size_t T = 0; T < Length; ++T) {
      float Tt = static_cast<float>(T) / Length;
      Base[T] = std::sin(2.0f * Pi * (K + 1) * Tt) +
                0.4f * std::sin(2.0f * Pi * (2 * K + 3) * Tt);
    }
    for (size_t I = 0; I < NPerClass; ++I) {
      for (size_t T = 0; T < Length; ++T)
        D.X[Row * Length + T] = Base[T] + N(Rng);
      D.Y[Row] = static_cast<int>(K);
      ++Row;
    }
  }
  // shuffle preserving seed-dependence
  std::vector<size_t> Idx(D.N);
  for (size_t I = 0; I < D.N; ++I)
    Idx[I] = I;
  std::shuffle(Idx.begin(), Idx.end(), Rng);
  Dataset Out;
  Out.Dim = Length;
  Out.N = D.N;
  Out.X.resize(D.N * Length);
  Out.Y.resize(D.N);
  for (size_t I = 0; I < D.N; ++I) {
    std::memcpy(Out.X.data() + I * Length, D.X.data() + Idx[I] * Length,
                Length * sizeof(float));
    Out.Y[I] = D.Y[Idx[I]];
  }
  return Out;
}

static void NormaliseInstance(Dataset &D) {
  for (size_t I = 0; I < D.N; ++I) {
    float *Row = D.X.data() + I * D.Dim;
    double M = 0.0, V = 0.0;
    for (size_t J = 0; J < D.Dim; ++J)
      M += Row[J];
    M /= D.Dim;
    for (size_t J = 0; J < D.Dim; ++J)
      V += (Row[J] - M) * (Row[J] - M);
    double Sd = std::sqrt(V / D.Dim) + 1e-6;
    for (size_t J = 0; J < D.Dim; ++J)
      Row[J] = static_cast<float>((Row[J] - M) / Sd);
  }
}

// Multi-layer ReLU MLP with optional per-layer pruning masks.
struct WideMLP {
  std::vector<raw::Linear> Layers;
  std::vector<std::vector<float>> Acts;
  std::vector<std::vector<float>> Grads;

  void Init(size_t InDim, size_t OutDim, size_t Hidden, size_t Depth,
            std::mt19937 &Rng) {
    Layers.resize(Depth);
    if (Depth == 1) {
      Layers[0].Init(InDim, OutDim, InDim, OutDim);
      Layers[0].XavierInit(Rng);
    } else {
      Layers[0].Init(InDim, Hidden, InDim, Hidden);
      Layers[0].XavierInit(Rng);
      for (size_t I = 1; I + 1 < Depth; ++I) {
        Layers[I].Init(Hidden, Hidden, Hidden, Hidden);
        Layers[I].XavierInit(Rng);
      }
      Layers[Depth - 1].Init(Hidden, OutDim, Hidden, OutDim);
      Layers[Depth - 1].XavierInit(Rng);
    }
    for (auto &L : Layers)
      L.EnableMask();
    Acts.assign(Depth, {});
    Grads.assign(Depth, {});
  }

  size_t UnitCount() const {
    size_t N = 0;
    for (const auto &L : Layers)
      N += L.OutDim;
    return N;
  }

  size_t AliveCount() const {
    size_t N = 0;
    for (const auto &L : Layers)
      N += L.AliveCount();
    return N;
  }

  size_t TotalCount() const {
    size_t N = 0;
    for (const auto &L : Layers)
      N += L.InDim * L.OutDim;
    return N;
  }

  bench::EdgeSet EdgeSetAlive() const {
    bench::EdgeSet Out;
    for (size_t LI = 0; LI < Layers.size(); ++LI) {
      const auto &L = Layers[LI];
      for (size_t I = 0; I < L.OutDim; ++I)
        for (size_t J = 0; J < L.InDim; ++J)
          if (L.Mask[I * L.InCap + J])
            Out.insert(bench::PackEdge(static_cast<uint32_t>(LI),
                                       static_cast<uint32_t>(I),
                                       static_cast<uint32_t>(J)));
    }
    return Out;
  }

  // Forward: hidden layers ReLU, last layer linear (logits).
  const float *Forward(const float *X, size_t Batch) {
    const float *Cur = X;
    for (size_t LI = 0; LI < Layers.size(); ++LI) {
      auto &L = Layers[LI];
      Acts[LI].assign(Batch * L.OutDim, 0.0f);
      L.Forward(Cur, Acts[LI].data(), Batch);
      if (LI + 1 < Layers.size())
        raw::ApplyReLU(Acts[LI].data(), Batch * L.OutDim);
      Cur = Acts[LI].data();
    }
    return Cur;
  }

  // Backward through cross-entropy. TopGrad is dL/d(logits). Computes the
  // weight gradients in each layer's GradW/GradB; call Update(Lr) to apply
  // them via SGD. The split lets the bench harness time grad vs. update.
  void Backward(const float *X, const float *TopGrad, size_t Batch) {
    size_t D = Layers.size();
    Grads.assign(D, {});
    Grads[D - 1].assign(Batch * Layers[D - 1].OutDim, 0.0f);
    std::memcpy(Grads[D - 1].data(), TopGrad,
                Batch * Layers[D - 1].OutDim * sizeof(float));
    for (ptrdiff_t LI = static_cast<ptrdiff_t>(D) - 1; LI >= 0; --LI) {
      auto &L = Layers[LI];
      const float *InData = (LI == 0) ? X : Acts[LI - 1].data();
      L.BackwardWeights(InData, Grads[LI].data(), Batch, 1.0f);
      if (LI > 0) {
        std::vector<float> GradIn(Batch * L.InDim, 0.0f);
        L.BackwardInput(Grads[LI].data(), GradIn.data(), Batch);
        // hidden layer below uses ReLU; gradient passes only where activation
        // was > 0.
        raw::ReLUBackward(Acts[LI - 1].data(), GradIn.data(),
                          Batch * L.InDim);
        Grads[LI - 1] = std::move(GradIn);
      }
    }
  }

  void Update(float Lr) {
    for (auto &L : Layers)
      L.SGD(Lr);
  }
};

// Global threshold: kth-smallest |w| over currently-alive weights.
static float ComputeThreshold(const WideMLP &M, float PruneFrac) {
  std::vector<float> Alive;
  for (const auto &L : M.Layers) {
    for (size_t I = 0; I < L.OutDim; ++I)
      for (size_t J = 0; J < L.InDim; ++J) {
        size_t Idx = I * L.InCap + J;
        if (L.Mask[Idx])
          Alive.push_back(std::abs(L.Weight[Idx]));
      }
  }
  if (Alive.empty())
    return 0.0f;
  size_t K = static_cast<size_t>(PruneFrac * Alive.size());
  if (K == 0)
    K = 1;
  if (K > Alive.size())
    K = Alive.size();
  std::nth_element(Alive.begin(), Alive.begin() + K - 1, Alive.end());
  return Alive[K - 1];
}

// Prune all alive edges with |w| <= threshold. Returns count killed.
static size_t MagnitudePrune(WideMLP &M, float Threshold) {
  size_t Killed = 0;
  for (auto &L : M.Layers) {
    for (size_t I = 0; I < L.OutDim; ++I)
      for (size_t J = 0; J < L.InDim; ++J) {
        size_t Idx = I * L.InCap + J;
        if (L.Mask[Idx] && std::abs(L.Weight[Idx]) <= Threshold) {
          L.Mask[Idx] = 0;
          L.Weight[Idx] = 0.0f;
          ++Killed;
        }
      }
  }
  return Killed;
}

static double EvalAcc(WideMLP &M, const Dataset &D, size_t Batch) {
  size_t Correct = 0;
  size_t Cnt = 0;
  for (size_t I = 0; I < D.N; I += Batch) {
    size_t B = std::min(Batch, D.N - I);
    const float *Logits = M.Forward(D.X.data() + I * D.Dim, B);
    Correct += static_cast<size_t>(
        raw::ArgmaxAccuracy(Logits, D.Y.data() + I, B,
                            M.Layers.back().OutDim) *
        static_cast<double>(B));
    Cnt += B;
  }
  return Cnt ? static_cast<double>(Correct) / Cnt : 0.0;
}

static double EvalNll(WideMLP &M, const Dataset &D, size_t Batch) {
  size_t NumClasses = M.Layers.back().OutDim;
  std::vector<float> Probs(Batch * NumClasses);
  std::vector<float> Grad(Batch * NumClasses);
  double Loss = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < D.N; I += Batch) {
    size_t B = std::min(Batch, D.N - I);
    const float *Logits = M.Forward(D.X.data() + I * D.Dim, B);
    Loss += raw::CrossEntropyMean(Logits, D.Y.data() + I, B, NumClasses,
                                  Probs.data(), Grad.data()) *
            static_cast<double>(B);
    Cnt += B;
  }
  return Cnt ? Loss / static_cast<double>(Cnt) : 0.0;
}

static void TrainEpochs(WideMLP &M, const Dataset &D, size_t Epochs,
                        size_t Batch, float Lr, std::mt19937 &Rng,
                        bench::PhaseTimer &Timer) {
  size_t NumClasses = M.Layers.back().OutDim;
  std::vector<size_t> Perm(D.N);
  for (size_t I = 0; I < D.N; ++I)
    Perm[I] = I;
  std::vector<float> XBatch(Batch * D.Dim);
  std::vector<int> YBatch(Batch);
  std::vector<float> Probs(Batch * NumClasses);
  std::vector<float> Grad(Batch * NumClasses);
  for (size_t E = 0; E < Epochs; ++E) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    for (size_t I = 0; I + Batch <= D.N; I += Batch) {
      for (size_t B = 0; B < Batch; ++B) {
        std::memcpy(XBatch.data() + B * D.Dim,
                    D.X.data() + Perm[I + B] * D.Dim,
                    D.Dim * sizeof(float));
        YBatch[B] = D.Y[Perm[I + B]];
      }
      Timer.Tick();
      const float *Logits = M.Forward(XBatch.data(), Batch);
      Timer.MarkForward();
      raw::CrossEntropyMean(Logits, YBatch.data(), Batch, NumClasses,
                            Probs.data(), Grad.data());
      Timer.MarkLoss();
      M.Backward(XBatch.data(), Grad.data(), Batch);
      Timer.MarkBackward();
      M.Update(Lr);
      Timer.MarkUpdate();
      Timer.StepDone();
    }
  }
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  bench::MemoryProbe MP;
  MP.Start();

  HP H;
  H.InLen = static_cast<size_t>(Args.GetInt("length", H.InLen));
  H.NumClasses = static_cast<size_t>(Args.GetInt("n-classes", H.NumClasses));
  H.Hidden = static_cast<size_t>(Args.GetInt("hidden", H.Hidden));
  H.Depth = static_cast<size_t>(Args.GetInt("depth", H.Depth));
  H.NPerClass = static_cast<size_t>(Args.GetInt("n-per-class", H.NPerClass));
  H.InitEpochs = static_cast<size_t>(Args.GetInt("init-epochs", H.InitEpochs));
  H.FinetuneEpochs = static_cast<size_t>(
      Args.GetInt("finetune-epochs", H.FinetuneEpochs));
  H.MaxRounds = static_cast<size_t>(Args.GetInt("max-rounds", H.MaxRounds));
  H.PruneFrac = Args.GetFloat("prune-frac", H.PruneFrac);
  H.Batch = static_cast<size_t>(Args.GetInt("batch", H.Batch));
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.FinetuneLrScale = Args.GetFloat("finetune-lr-scale", H.FinetuneLrScale);
  if (Args.Quick) {
    H.InitEpochs = std::max<size_t>(1, H.InitEpochs / 4);
    H.FinetuneEpochs = std::max<size_t>(1, H.FinetuneEpochs / 2);
    H.MaxRounds = std::max<size_t>(2, H.MaxRounds / 3);
  }

  auto Train = SynthUcr(H.NPerClass, H.NumClasses, H.InLen, H.Snr,
                        static_cast<uint32_t>(Args.Seed));
  auto Test =
      SynthUcr(std::max<size_t>(16, H.NPerClass / 4), H.NumClasses, H.InLen,
               H.Snr, static_cast<uint32_t>(Args.Seed + 1000));
  NormaliseInstance(Train);
  NormaliseInstance(Test);
  MP.EndDataset();

  std::cout << "[info] InLen=" << H.InLen << " Classes=" << H.NumClasses
            << " Hidden=" << H.Hidden << " Depth=" << H.Depth
            << " Train=" << Train.N << " Test=" << Test.N << "\n";

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 11u);
  WideMLP M;
  M.Init(H.InLen, H.NumClasses, H.Hidden, H.Depth, InitRng);
  MP.EndWeights();

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "idempotent_imp");
  bench::StructuralLog Log(HistPath);

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  bench::PhaseTimer Timer;

  std::cout << "[imp] initial dense training, " << H.InitEpochs
            << " epochs\n";
  TrainEpochs(M, Train, H.InitEpochs, H.Batch, H.Lr, Rng, Timer);
  double Acc0 = EvalAcc(M, Test, H.Batch);
  double Nll0 = EvalNll(M, Test, H.Batch);
  size_t Edges0 = M.AliveCount();
  auto Edges = M.EdgeSetAlive();
  Log.Log(0, M.UnitCount(), Edges0, &Edges, &Nll0,
          {{"val_acc", Acc0},
           {"test_acc", Acc0},
           {"round", 0.0},
           {"sparsity", 0.0},
           {"killed_this_round", 0.0}});
  std::cout << "[imp] round 0  edges=" << Edges0 << "  acc=" << Acc0
            << "  nll=" << Nll0 << "\n";

  auto T0 = std::chrono::steady_clock::now();
  size_t FixedPoint = 0;
  size_t FinalRound = 0;
  for (size_t R = 1; R <= H.MaxRounds; ++R) {
    size_t AliveBefore = M.AliveCount();
    float Thresh = ComputeThreshold(M, H.PruneFrac);
    size_t Killed = MagnitudePrune(M, Thresh);
    TrainEpochs(M, Train, H.FinetuneEpochs, H.Batch,
                H.Lr * H.FinetuneLrScale, Rng, Timer);
    size_t AliveAfter = M.AliveCount();
    double Acc = EvalAcc(M, Test, H.Batch);
    double Nll = EvalNll(M, Test, H.Batch);
    double Sparsity =
        1.0 - static_cast<double>(AliveAfter) / static_cast<double>(Edges0);
    auto Ed = M.EdgeSetAlive();
    Log.Log(R, M.UnitCount(), AliveAfter, &Ed, &Nll,
            {{"val_acc", Acc},
             {"test_acc", Acc},
             {"round", static_cast<double>(R)},
             {"sparsity", Sparsity},
             {"killed_this_round", static_cast<double>(Killed)}});
    std::cout << "[imp] round " << R << "  killed=" << Killed
              << "  alive=" << AliveAfter << "  sparsity=" << Sparsity
              << "  acc=" << Acc << "\n";
    FinalRound = R;
    (void)AliveBefore;
    if (Killed == 0) {
      FixedPoint = 1;
      std::cout << "[imp] reached fixed point at round " << R << "\n";
      break;
    }
  }
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
                    .count();

  Log.Flush();
  double FinalSparsity =
      1.0 - static_cast<double>(M.AliveCount()) / static_cast<double>(Edges0);
  double FinalAcc = EvalAcc(M, Test, H.Batch);
  bench::SummaryWriter S;
  S.Set("workload", std::string{"02_idempotent_imp"});
  S.Set("dataset", std::string{"synthetic-ucr"});
  S.Set("n_classes", static_cast<int>(H.NumClasses));
  S.Set("length", static_cast<int>(H.InLen));
  S.Set("hidden", static_cast<int>(H.Hidden));
  S.Set("depth", static_cast<int>(H.Depth));
  S.Set("init_epochs", static_cast<int>(H.InitEpochs));
  S.Set("finetune_epochs", static_cast<int>(H.FinetuneEpochs));
  S.Set("prune_frac", static_cast<double>(H.PruneFrac));
  S.Set("max_rounds", static_cast<int>(H.MaxRounds));
  S.Set("wall_seconds", Wall);
  S.Set("rounds_run", static_cast<int>(FinalRound));
  S.Set("edges_initial", static_cast<int>(Edges0));
  S.Set("edges_final", static_cast<int>(M.AliveCount()));
  S.Set("sparsity_final", FinalSparsity);
  S.Set("val_acc_initial", Acc0);
  S.Set("val_acc_final", FinalAcc);
  S.Set("fixed_point_reached", static_cast<int>(FixedPoint));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  rounds=" << FinalRound
            << "  sparsity=" << FinalSparsity << "  acc=" << FinalAcc << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
