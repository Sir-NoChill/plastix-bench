// Workload 4 / 5 — CONTINUOUS-SMALL regime, raw-C++/OpenBLAS translation.
//
// Mirrors 04_continuous_small_appliances.py: per-step
// probabilistic neuron split + per-step probabilistic edge prune on a single-
// hidden-layer regression MLP. Net structural delta per step is at most +/-1
// unit and a small number of edges, so Jaccard between consecutive live-edge
// sets sits very close to 1.0.
//
// Pre-allocate max + mask: the hidden width is physically MaxHidden up front;
// "splitting" a unit flips mask bits and seeds new weights, no realloc.

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
  size_t InitHidden = 32;
  size_t MaxHidden = 256;
  size_t Batch = 32;
  size_t MaxSteps = 3000;
  size_t ValEvery = 20;
  size_t ValWindow = 256;
  // Mean-reduction MSE; effective lr = PyTorch_sum_lr * B * OutDim
  // = 1e-5 * 32 * 1 = 3.2e-4. Bump slightly to keep the regression moving.
  float Lr = 1e-3f;
  float PSplit = 0.5f;
  float PPrune = 0.5f;
  float SplitNoise = 0.05f;
  size_t VarWindow = 64;
};

struct Dataset {
  std::vector<float> X;
  std::vector<float> Y;
  size_t N = 0;
  size_t D = 0;
};

static Dataset SynthSlowDrift(size_t N, size_t Dim, uint32_t Seed) {
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> Norm(0.0f, 1.0f);
  Dataset D;
  D.N = N;
  D.D = Dim;
  D.X.resize(N * Dim);
  D.Y.resize(N);
  for (size_t I = 0; I < N; ++I) {
    float T = 2.0f * 3.14159265358979323846f *
              static_cast<float>(I) / static_cast<float>(N);
    float Sum = 0.0f;
    for (size_t J = 0; J < Dim; ++J) {
      D.X[I * Dim + J] = Norm(Rng);
      Sum += D.X[I * Dim + J] *
             std::sin(T + 0.3f * static_cast<float>(J));
    }
    D.Y[I] = Sum + 0.1f * Norm(Rng);
  }
  return D;
}

static Dataset LoadAppliances(const std::filesystem::path &Dir) {
  Dataset D;
  auto Path = Dir / "energydata_complete.csv";
  if (!std::filesystem::exists(Path))
    return D;
  auto Csv = bench::ReadCsv(Path);
  if (Csv.Header.empty())
    return D;
  size_t TargetIdx = SIZE_MAX;
  std::vector<size_t> FeatIdx;
  for (size_t I = 0; I < Csv.Header.size(); ++I) {
    if (Csv.Header[I] == "Appliances")
      TargetIdx = I;
    else if (Csv.Header[I] != "date")
      FeatIdx.push_back(I);
  }
  if (TargetIdx == SIZE_MAX)
    return D;
  D.D = FeatIdx.size();
  D.N = Csv.Rows.size();
  D.X.assign(D.N * D.D, 0.0f);
  D.Y.assign(D.N, 0.0f);
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
    if (TargetIdx < R.size()) {
      try {
        D.Y[I] = std::stof(R[TargetIdx]);
      } catch (...) {
        D.Y[I] = 0.0f;
      }
    }
  }
  return D;
}

static void StandardiseFeatures(Dataset &D, size_t Cut) {
  if (D.N == 0 || D.D == 0)
    return;
  size_t U = std::min(Cut, D.N);
  std::vector<double> Mu(D.D, 0.0), Sd(D.D, 0.0);
  for (size_t I = 0; I < U; ++I)
    for (size_t J = 0; J < D.D; ++J)
      Mu[J] += D.X[I * D.D + J];
  for (auto &M : Mu)
    M /= static_cast<double>(U);
  for (size_t I = 0; I < U; ++I)
    for (size_t J = 0; J < D.D; ++J) {
      double Diff = D.X[I * D.D + J] - Mu[J];
      Sd[J] += Diff * Diff;
    }
  for (auto &V : Sd) {
    V = std::sqrt(V / static_cast<double>(U));
    if (V < 1e-3)
      V = 1.0;
  }
  for (size_t I = 0; I < D.N; ++I)
    for (size_t J = 0; J < D.D; ++J)
      D.X[I * D.D + J] =
          static_cast<float>((D.X[I * D.D + J] - Mu[J]) / Sd[J]);
}

static void StandardiseTarget(Dataset &D, size_t Cut) {
  if (D.N == 0)
    return;
  size_t U = std::min(Cut, D.N);
  double M = 0.0;
  for (size_t I = 0; I < U; ++I)
    M += D.Y[I];
  M /= static_cast<double>(U);
  double V = 0.0;
  for (size_t I = 0; I < U; ++I)
    V += (D.Y[I] - M) * (D.Y[I] - M);
  double Sd = std::sqrt(V / static_cast<double>(U)) + 1e-6;
  for (size_t I = 0; I < D.N; ++I)
    D.Y[I] = static_cast<float>((D.Y[I] - M) / Sd);
}

// Splittable single-hidden-layer MLP: in -> H (ReLU) -> out (linear).
// L1's mask carries the per-edge keep state (only hidden->input edges can be
// pruned). Hidden width grows via flipping unused rows live in L1 + L2.
struct SplitMLP {
  raw::Linear L1; // in -> H
  raw::Linear L2; // H -> out
  size_t Hidden = 0;
  size_t InDim = 0;
  size_t OutDim = 0;
  size_t MaxHidden = 0;
  size_t Splits = 0;
  size_t Prunes = 0;

  std::vector<float> Act1; // batch x H (post-ReLU)
  std::vector<float> Out;  // batch x out (linear)

  void Init(size_t InDim_, size_t OutDim_, size_t InitHidden,
            size_t MaxHidden_, std::mt19937 &Rng) {
    InDim = InDim_;
    OutDim = OutDim_;
    Hidden = InitHidden;
    MaxHidden = MaxHidden_;
    L1.Init(InDim, InitHidden, InDim, MaxHidden);
    L2.Init(InitHidden, OutDim, MaxHidden, OutDim);
    float Lim1 = raw::XavierLimit(InDim, InitHidden);
    float Lim2 = raw::XavierLimit(InitHidden, OutDim);
    std::uniform_real_distribution<float> U1(-Lim1, Lim1);
    std::uniform_real_distribution<float> U2(-Lim2, Lim2);
    for (size_t I = 0; I < InitHidden; ++I)
      for (size_t J = 0; J < InDim; ++J)
        L1.Weight[I * L1.InCap + J] = U1(Rng);
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InitHidden; ++J)
        L2.Weight[I * L2.InCap + J] = U2(Rng);
    L1.EnableMask();
    // L2 has no per-edge pruning; mask stays empty (dense).
  }

  size_t UnitCount() const { return Hidden + OutDim; }
  size_t EdgeCount() const {
    // L1 alive edges + L2 dense edges (hidden->out is all alive).
    return L1.AliveCount() + OutDim * Hidden;
  }

  bench::EdgeSet EdgeSetAlive() const {
    bench::EdgeSet S;
    for (size_t I = 0; I < L1.OutDim; ++I)
      for (size_t J = 0; J < L1.InDim; ++J)
        if (L1.Mask[I * L1.InCap + J])
          S.insert(bench::PackEdge(0u, static_cast<uint32_t>(I),
                                   static_cast<uint32_t>(J)));
    for (size_t I = 0; I < L2.OutDim; ++I)
      for (size_t J = 0; J < L2.InDim; ++J)
        S.insert(bench::PackEdge(1u, static_cast<uint32_t>(I),
                                 static_cast<uint32_t>(J)));
    return S;
  }

  const float *Forward(const float *X, size_t Batch) {
    Act1.assign(Batch * L1.OutDim, 0.0f);
    L1.Forward(X, Act1.data(), Batch);
    raw::ApplyReLU(Act1.data(), Batch * L1.OutDim);
    Out.assign(Batch * L2.OutDim, 0.0f);
    L2.Forward(Act1.data(), Out.data(), Batch);
    return Out.data();
  }

  void Backward(const float *X, const float *TopGrad, size_t Batch) {
    std::vector<float> G2(Batch * L2.OutDim);
    std::memcpy(G2.data(), TopGrad, Batch * L2.OutDim * sizeof(float));
    L2.BackwardWeights(Act1.data(), G2.data(), Batch, 1.0f);
    std::vector<float> G1(Batch * L1.OutDim, 0.0f);
    L2.BackwardInput(G2.data(), G1.data(), Batch);
    raw::ReLUBackward(Act1.data(), G1.data(), Batch * L1.OutDim);
    L1.BackwardWeights(X, G1.data(), Batch, 1.0f);
  }

  void Update(float Lr) {
    L1.SGD(Lr);
    L2.SGD(Lr);
  }

  // Duplicate hidden unit `idx`: copy its incoming row in L1 with small
  // noise; halve l2[:, idx] and write the halved copy to the new column.
  // PyTorch reference resets the optimizer to clear momentum — pure SGD has
  // none, so this is a no-op on our side.
  bool Split(size_t Idx, float Noise, std::mt19937 &Rng) {
    if (Hidden >= MaxHidden)
      return false;
    if (Idx >= Hidden)
      return false;
    size_t New = Hidden;
    std::normal_distribution<float> N(0.0f, Noise);
    for (size_t J = 0; J < InDim; ++J) {
      L1.Weight[New * L1.InCap + J] =
          L1.Weight[Idx * L1.InCap + J] + N(Rng);
      L1.Mask[New * L1.InCap + J] = 1;
    }
    L1.Bias[New] = L1.Bias[Idx];
    for (size_t I = 0; I < OutDim; ++I) {
      float Half = 0.5f * L2.Weight[I * L2.InCap + Idx];
      L2.Weight[I * L2.InCap + Idx] = Half;
      L2.Weight[I * L2.InCap + New] = Half;
    }
    ++Hidden;
    L1.OutDim = Hidden;
    L2.InDim = Hidden;
    ++Splits;
    return true;
  }

  // Kill the smallest |w| alive edge in L1. Returns true on success.
  bool KillSmallestEdge() {
    float Best = std::numeric_limits<float>::infinity();
    size_t BestI = 0, BestJ = 0;
    bool Found = false;
    for (size_t I = 0; I < L1.OutDim; ++I)
      for (size_t J = 0; J < L1.InDim; ++J) {
        size_t Idx = I * L1.InCap + J;
        if (!L1.Mask[Idx])
          continue;
        float V = std::abs(L1.Weight[Idx]);
        if (V < Best) {
          Best = V;
          BestI = I;
          BestJ = J;
          Found = true;
        }
      }
    if (!Found)
      return false;
    L1.Mask[BestI * L1.InCap + BestJ] = 0;
    L1.Weight[BestI * L1.InCap + BestJ] = 0.0f;
    ++Prunes;
    return true;
  }
};

// Per-unit activation-variance tracker over a sliding window. The "hottest"
// unit (argmax variance) is the one to split.
struct ActStats {
  std::vector<std::vector<float>> Buf; // window x hidden_capacity
  size_t Window;
  size_t Idx = 0;
  size_t Filled = 0;
  size_t HiddenCap;
  ActStats(size_t W, size_t HCap) : Window(W), HiddenCap(HCap) {
    Buf.assign(W, std::vector<float>(HCap, 0.0f));
  }
  // Push mean-over-batch of activations (only the live hidden block; rest
  // stays at whatever value, ignored by Hottest).
  void Push(const float *H, size_t Batch, size_t HiddenLive) {
    auto &Row = Buf[Idx];
    for (size_t J = 0; J < HiddenLive; ++J) {
      float Sum = 0.0f;
      for (size_t B = 0; B < Batch; ++B)
        Sum += H[B * HiddenLive + J];
      Row[J] = Sum / static_cast<float>(Batch);
    }
    Idx = (Idx + 1) % Window;
    if (Filled < Window)
      ++Filled;
  }
  size_t Hottest(size_t HiddenLive, std::mt19937 &Rng) const {
    if (Filled < 2) {
      std::uniform_int_distribution<size_t> U(0, HiddenLive - 1);
      return U(Rng);
    }
    size_t Best = 0;
    float BestVar = -1.0f;
    for (size_t J = 0; J < HiddenLive; ++J) {
      double M = 0.0;
      for (size_t I = 0; I < Filled; ++I)
        M += Buf[I][J];
      M /= static_cast<double>(Filled);
      double V = 0.0;
      for (size_t I = 0; I < Filled; ++I)
        V += (Buf[I][J] - M) * (Buf[I][J] - M);
      V /= static_cast<double>(Filled);
      if (V > BestVar) {
        BestVar = static_cast<float>(V);
        Best = J;
      }
    }
    return Best;
  }
};

static double EvalMseOnSlice(SplitMLP &M, const Dataset &D, size_t Start,
                             size_t Count) {
  const float *Pred = M.Forward(D.X.data() + Start * D.D, Count);
  return raw::MSEEvalMean(Pred, D.Y.data() + Start, Count, M.OutDim);
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
  H.PSplit = Args.GetFloat("p-split", H.PSplit);
  H.PPrune = Args.GetFloat("p-prune", H.PPrune);
  H.SplitNoise = Args.GetFloat("split-noise", H.SplitNoise);
  H.VarWindow = static_cast<size_t>(Args.GetInt("var-window", H.VarWindow));
  if (Args.Quick) {
    H.MaxSteps = std::max<size_t>(50, H.MaxSteps / 5);
    H.ValEvery = std::max<size_t>(1, H.ValEvery / 2);
  }

  Dataset D;
  std::string DatasetName;
  if (Args.Synthetic) {
    D = SynthSlowDrift(H.MaxSteps * H.Batch + 5000, 25,
                       static_cast<uint32_t>(Args.Seed));
    DatasetName = "synthetic-slow-drift";
  } else {
    D = LoadAppliances(Args.DataDir);
    if (D.N == 0) {
      std::cerr << "[warn] Appliances not found; using synthetic\n";
      D = SynthSlowDrift(H.MaxSteps * H.Batch + 5000, 25,
                         static_cast<uint32_t>(Args.Seed));
      DatasetName = "synthetic-slow-drift-fallback";
    } else {
      DatasetName = "uci-appliances";
    }
  }
  size_t Cut = std::max<size_t>(static_cast<size_t>(0.1 * D.N), 512);
  StandardiseFeatures(D, Cut);
  StandardiseTarget(D, Cut);

  std::cout << "[info] in_dim=" << D.D << " data=" << DatasetName
            << " N=" << D.N << " steps=" << H.MaxSteps
            << " init_h=" << H.InitHidden << "\n";

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 17u);
  SplitMLP M;
  M.Init(D.D, 1, H.InitHidden, H.MaxHidden, InitRng);
  ActStats Stats(H.VarWindow, H.MaxHidden);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "continuous_small_appliances");
  bench::StructuralLog Log(HistPath);

  // Carve final 15% as held-out test set (mirrors pytorch). Training cycles
  // only over the first 85% so the test slice is never seen.
  size_t NTrain = static_cast<size_t>(0.85 * static_cast<double>(D.N));
  if (NTrain < H.ValWindow + 1)
    NTrain = D.N;
  size_t TestStart = NTrain;
  size_t TestCount = D.N - NTrain;

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::uniform_real_distribution<float> Prob(0.0f, 1.0f);

  std::vector<float> XBatch(H.Batch * D.D);
  std::vector<float> YBatch(H.Batch * M.OutDim);
  std::vector<float> Grad(H.Batch * M.OutDim);

  std::vector<int> DeltaUnits;
  std::vector<int> DeltaEdges;
  DeltaUnits.reserve(H.MaxSteps);
  DeltaEdges.reserve(H.MaxSteps);

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    for (size_t B = 0; B < H.Batch; ++B) {
      // Wrap within training portion so held-out test slice stays unseen.
      size_t Idx = ((Step - 1) * H.Batch + B) % NTrain;
      std::memcpy(XBatch.data() + B * D.D, D.X.data() + Idx * D.D,
                  D.D * sizeof(float));
      YBatch[B] = D.Y[Idx];
    }
    Timer.Tick();
    const float *Pred = M.Forward(XBatch.data(), H.Batch);
    Timer.MarkForward();
    raw::MSELossMean(Pred, YBatch.data(), H.Batch, M.OutDim, Grad.data());
    Timer.MarkLoss();
    M.Backward(XBatch.data(), Grad.data(), H.Batch);
    Timer.MarkBackward();
    M.Update(H.Lr);
    Timer.MarkUpdate();
    Timer.StepDone();
    Stats.Push(M.Act1.data(), H.Batch, M.Hidden);

    size_t UnitsBefore = M.UnitCount();
    size_t EdgesBefore = M.EdgeCount();
    bool DidChange = false;

    if (Prob(Rng) < H.PSplit && M.Hidden < H.MaxHidden) {
      size_t Hot = Stats.Hottest(M.Hidden, Rng);
      if (M.Split(Hot, H.SplitNoise, Rng))
        DidChange = true;
    }
    if (Prob(Rng) < H.PPrune && M.L1.AliveCount() > M.InDim) {
      if (M.KillSmallestEdge())
        DidChange = true;
    }

    DeltaUnits.push_back(static_cast<int>(M.UnitCount()) -
                          static_cast<int>(UnitsBefore));
    DeltaEdges.push_back(static_cast<int>(M.EdgeCount()) -
                          static_cast<int>(EdgesBefore));

    if (Step % H.ValEvery == 0 || Step == H.MaxSteps || DidChange) {
      size_t VStart = (Step * H.Batch) % NTrain;
      size_t VEnd = std::min(VStart + H.ValWindow, NTrain);
      size_t VCount = VEnd - VStart;
      double VL = EvalMseOnSlice(M, D, VStart, VCount);
      double TestMSE = TestCount > 0
                         ? EvalMseOnSlice(M, D, TestStart, TestCount)
                         : VL;
      auto Edges = M.EdgeSetAlive();
      Log.Log(Step, M.UnitCount(), M.EdgeCount(), &Edges, &VL,
              {{"hidden", static_cast<double>(M.Hidden)},
               {"test_mse", TestMSE},
               {"splits", static_cast<double>(M.Splits)},
               {"prunes", static_cast<double>(M.Prunes)},
               {"delta_units", static_cast<double>(DeltaUnits.back())},
               {"delta_edges", static_cast<double>(DeltaEdges.back())}});
    }
  }
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
                    .count();
  Log.Flush();

  auto Pctile = [](std::vector<int> V, double P) -> int {
    if (V.empty())
      return 0;
    for (auto &X : V)
      X = std::abs(X);
    size_t K = static_cast<size_t>(P * static_cast<double>(V.size()) / 100.0);
    if (K >= V.size())
      K = V.size() - 1;
    std::nth_element(V.begin(), V.begin() + K, V.end());
    return V[K];
  };
  int DuMax = 0, DeMax = 0;
  for (auto V : DeltaUnits)
    DuMax = std::max(DuMax, std::abs(V));
  for (auto V : DeltaEdges)
    DeMax = std::max(DeMax, std::abs(V));

  float JMin = 1.0f;
  double JMean = 0.0;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JMean += R.Jaccard;
  }
  if (!Log.Records().empty())
    JMean /= static_cast<double>(Log.Records().size());

  bench::SummaryWriter S;
  S.Set("workload", std::string{"04_continuous_small_appliances"});
  S.Set("dataset", DatasetName);
  S.Set("in_dim", static_cast<int>(D.D));
  S.Set("init_hidden", static_cast<int>(H.InitHidden));
  S.Set("max_steps", static_cast<int>(H.MaxSteps));
  S.Set("batch", static_cast<int>(H.Batch));
  S.Set("p_split", static_cast<double>(H.PSplit));
  S.Set("p_prune", static_cast<double>(H.PPrune));
  S.Set("wall_seconds", Wall);
  S.Set("splits_fired", static_cast<int>(M.Splits));
  S.Set("prunes_fired", static_cast<int>(M.Prunes));
  S.Set("hidden_final", static_cast<int>(M.Hidden));
  S.Set("edges_final", static_cast<int>(M.EdgeCount()));
  S.Set("val_loss_initial",
        Log.Records().empty() ? 0.0 : Log.Records().front().ValLoss);
  S.Set("val_loss_final",
        Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss);
  S.Set("delta_units_p99_abs", Pctile(DeltaUnits, 99.0));
  S.Set("delta_units_max_abs", DuMax);
  S.Set("delta_edges_p99_abs", Pctile(DeltaEdges, 99.0));
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_mean", JMean);
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  splits=" << M.Splits
            << "  prunes=" << M.Prunes << "  hidden_final=" << M.Hidden
            << "  |du|_max=" << DuMax << "  jaccard_mean=" << JMean << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  (void)DeMax;
  return 0;
}
