// Workload 1 / 5 — STATIC regime, raw-C++/OpenBLAS translation.
//
// Mirrors 01_static_etth1.py: a fixed feedforward MLP trained
// against the ETTh1 long-horizon forecasting target with mean-squared error.
// Topology never changes; the topology hash and edge Jaccard sit at constants
// for the entire run.
//
// Architecture matches the PyTorch reference: GeLU-hidden / linear-output,
// depth=3 (= 2 GeLU hidden + 1 linear readout), Linear layers with bias.
// Loss reduction is *mean* (not PyTorch's sum) so the lr is tuned for the
// mean-reduction setting; see HP::Lr below.

#include "cpp/common.hpp"
#include "cpp/mlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct HP {
  size_t InLen = 96;
  size_t OutLen = 24;
  size_t Hidden = 256;
  size_t Depth = 3;           // PyTorch convention: total Linear layers.
  size_t Epochs = 20;
  size_t Batch = 64;
  // Mean-reduction MSE: effective lr = PyTorch sum-lr * B * out_dim, so
  // 1e-5 * 64 * 168 ~ 0.1 reproduces the PyTorch reference's step magnitude.
  float Lr = 0.1f;
  size_t MaxTrainRows = 0;
};

struct Series {
  std::vector<float> Flat;
  size_t T = 0;
  size_t C = 0;
};

static Series SynthesiseEtth1(size_t T, size_t C, uint32_t Seed) {
  constexpr float Pi = 3.14159265358979323846f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> N(0.0f, 0.1f);
  Series S;
  S.T = T;
  S.C = C;
  S.Flat.assign(T * C, 0.0f);
  for (size_t I = 0; I < T; ++I) {
    float Tt = static_cast<float>(I);
    float Base = std::sin(2.0f * Pi * Tt / 24.0f) +
                 0.4f * std::sin(2.0f * Pi * Tt / (24.0f * 7.0f));
    for (size_t Ch = 0; Ch < C; ++Ch)
      S.Flat[I * C + Ch] = Base + N(Rng);
  }
  return S;
}

static Series LoadEtth1(const std::filesystem::path &Dir, bool ForceSynth,
                        uint32_t Seed) {
  if (!ForceSynth) {
    auto Path = Dir / "ETTh1.csv";
    if (std::filesystem::exists(Path)) {
      auto Csv = bench::ReadCsv(Path);
      if (!Csv.Header.empty()) {
        std::vector<size_t> ColIdx;
        for (size_t I = 0; I < Csv.Header.size(); ++I) {
          std::string H = Csv.Header[I];
          for (auto &Ch : H)
            Ch = static_cast<char>(std::tolower(Ch));
          if (H != "date")
            ColIdx.push_back(I);
        }
        Series S;
        S.T = Csv.Rows.size();
        S.C = ColIdx.size();
        S.Flat.reserve(S.T * S.C);
        for (size_t I = 0; I < S.T; ++I) {
          const auto &R = Csv.Rows[I];
          for (size_t J : ColIdx) {
            float V = 0.0f;
            if (J < R.size()) {
              try {
                V = std::stof(R[J]);
              } catch (...) {
                V = 0.0f;
              }
            }
            S.Flat.push_back(V);
          }
        }
        std::cerr << "[data] loaded ETTh1 (" << S.T << " rows x " << S.C
                  << " channels)\n";
        return S;
      }
    }
  }
  std::cerr << "[data] using synthetic ETTh1 stand-in\n";
  return SynthesiseEtth1(17'420, 7, Seed);
}

// Per-channel z-score over the first `Cut` rows. Matches the PyTorch
// reference's standardisation policy.
static void StandardisePerChannel(Series &S, size_t Cut) {
  if (S.T == 0 || S.C == 0)
    return;
  size_t N = std::min(Cut, S.T);
  std::vector<double> Mu(S.C, 0.0), Sd(S.C, 0.0);
  for (size_t I = 0; I < N; ++I)
    for (size_t Ch = 0; Ch < S.C; ++Ch)
      Mu[Ch] += S.Flat[I * S.C + Ch];
  for (auto &M : Mu)
    M /= static_cast<double>(N);
  for (size_t I = 0; I < N; ++I)
    for (size_t Ch = 0; Ch < S.C; ++Ch) {
      double D = S.Flat[I * S.C + Ch] - Mu[Ch];
      Sd[Ch] += D * D;
    }
  for (auto &V : Sd)
    V = std::sqrt(V / static_cast<double>(N)) + 1e-6;
  for (size_t I = 0; I < S.T; ++I)
    for (size_t Ch = 0; Ch < S.C; ++Ch) {
      double X = S.Flat[I * S.C + Ch];
      S.Flat[I * S.C + Ch] = static_cast<float>((X - Mu[Ch]) / Sd[Ch]);
    }
}

struct Dataset {
  std::vector<float> X; // flat row-major, NRows x InDim
  std::vector<float> Y; // flat row-major, NRows x OutDim
  size_t NRows = 0;
  size_t InDim = 0;
  size_t OutDim = 0;
};

static Dataset Window(const Series &S, size_t InLen, size_t OutLen) {
  Dataset D;
  if (S.T < InLen + OutLen)
    return D;
  D.InDim = InLen * S.C;
  D.OutDim = OutLen * S.C;
  D.NRows = S.T - InLen - OutLen + 1;
  D.X.resize(D.NRows * D.InDim);
  D.Y.resize(D.NRows * D.OutDim);
  for (size_t I = 0; I < D.NRows; ++I) {
    float *XRow = D.X.data() + I * D.InDim;
    float *YRow = D.Y.data() + I * D.OutDim;
    for (size_t T = 0; T < InLen; ++T)
      for (size_t Ch = 0; Ch < S.C; ++Ch)
        XRow[T * S.C + Ch] = S.Flat[(I + T) * S.C + Ch];
    for (size_t T = 0; T < OutLen; ++T)
      for (size_t Ch = 0; Ch < S.C; ++Ch)
        YRow[T * S.C + Ch] = S.Flat[(I + InLen + T) * S.C + Ch];
  }
  return D;
}

// Static MLP: in -> H -> ... -> out. Hidden layers use GeLU, output linear.
struct StaticMLP {
  std::vector<raw::Linear> Layers;
  std::vector<std::vector<float>> Acts;    // per-layer output activations (post-act)
  std::vector<std::vector<float>> Preacts; // per-layer pre-activations (z)
  std::vector<std::vector<float>> Grads;   // per-layer dL/da

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
    Acts.assign(Depth, {});
    Preacts.assign(Depth, {});
    Grads.assign(Depth, {});
  }

  size_t UnitCount() const {
    size_t N = 0;
    for (const auto &L : Layers)
      N += L.OutDim;
    return N;
  }

  size_t EdgeCount() const {
    size_t N = 0;
    for (const auto &L : Layers)
      N += L.InDim * L.OutDim;
    return N;
  }

  bench::EdgeSet EdgeSet() const {
    bench::EdgeSet Out;
    for (size_t LI = 0; LI < Layers.size(); ++LI) {
      const auto &L = Layers[LI];
      for (size_t I = 0; I < L.OutDim; ++I)
        for (size_t J = 0; J < L.InDim; ++J)
          Out.insert(bench::PackEdge(static_cast<uint32_t>(LI),
                                     static_cast<uint32_t>(I),
                                     static_cast<uint32_t>(J)));
    }
    return Out;
  }

  // Forward over a batch of Batch x InDim, returns final-layer output buffer.
  const float *Forward(const float *X, size_t Batch) {
    const float *Cur = X;
    for (size_t LI = 0; LI < Layers.size(); ++LI) {
      auto &L = Layers[LI];
      Preacts[LI].assign(Batch * L.OutDim, 0.0f);
      Acts[LI].assign(Batch * L.OutDim, 0.0f);
      L.Forward(Cur, Preacts[LI].data(), Batch);
      if (LI + 1 < Layers.size()) {
        raw::ApplyGeLU(Preacts[LI].data(), Acts[LI].data(),
                       Batch * L.OutDim);
      } else {
        std::memcpy(Acts[LI].data(), Preacts[LI].data(),
                    Batch * L.OutDim * sizeof(float));
      }
      Cur = Acts[LI].data();
    }
    return Cur;
  }

  // Backward: TopGrad is dL/d(output) shape Batch x OutDim. Computes the
  // weight gradients (in each layer's GradW / GradB) but does NOT apply them
  // — call Update(Lr) for the SGD step. Splitting the phases lets the bench
  // harness time them separately (cf. PyTorch's `loss.backward()` vs
  // `optimizer.step()`).
  void Backward(const float *X, const float *TopGrad, size_t Batch) {
    size_t D = Layers.size();
    Grads.assign(D, {});
    // Top-down: dL/dz at the readout = dL/da * 1 (linear), so just copy.
    Grads[D - 1].assign(Batch * Layers[D - 1].OutDim, 0.0f);
    std::memcpy(Grads[D - 1].data(), TopGrad,
                Batch * Layers[D - 1].OutDim * sizeof(float));
    // For hidden layers: dL/dz = dL/da * GeLU'(z).
    for (ptrdiff_t LI = static_cast<ptrdiff_t>(D) - 1; LI >= 0; --LI) {
      auto &L = Layers[LI];
      const float *InData = (LI == 0) ? X : Acts[LI - 1].data();
      // Weight gradient against current dL/dz, mean-reduced over batch.
      L.BackwardWeights(InData, Grads[LI].data(), Batch, 1.0f);
      // If there's a layer below, propagate dL/d(input).
      if (LI > 0) {
        std::vector<float> GradIn(Batch * L.InDim, 0.0f);
        L.BackwardInput(Grads[LI].data(), GradIn.data(), Batch);
        // dL/dz at the layer below = dL/da * GeLU'(z_below).
        raw::GeLUBackwardFromPreact(Preacts[LI - 1].data(), GradIn.data(),
                                    Batch * L.InDim);
        Grads[LI - 1] = std::move(GradIn);
      }
    }
  }

  // Apply the SGD step using gradients computed by the preceding Backward().
  void Update(float Lr) {
    for (auto &L : Layers)
      L.SGD(Lr);
  }
};

static double EvalMse(StaticMLP &M, const std::vector<float> &X,
                      const std::vector<float> &Y, size_t InDim, size_t OutDim,
                      size_t NRows, size_t Batch) {
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < NRows; I += Batch) {
    size_t B = std::min(Batch, NRows - I);
    const float *Out = M.Forward(X.data() + I * InDim, B);
    Sum += raw::MSEEvalMean(Out, Y.data() + I * OutDim, B, OutDim) *
           static_cast<double>(B * OutDim);
    Cnt += B * OutDim;
  }
  return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.InLen = static_cast<size_t>(Args.GetInt("in-len", H.InLen));
  H.OutLen = static_cast<size_t>(Args.GetInt("out-len", H.OutLen));
  H.Hidden = static_cast<size_t>(Args.GetInt("hidden", H.Hidden));
  H.Depth = static_cast<size_t>(Args.GetInt("depth", H.Depth));
  H.Epochs = static_cast<size_t>(Args.GetInt("epochs", H.Epochs));
  H.Batch = static_cast<size_t>(Args.GetInt("batch", H.Batch));
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.MaxTrainRows = static_cast<size_t>(
      Args.GetInt("max-train-rows", static_cast<int>(H.MaxTrainRows)));
  if (Args.Quick)
    H.Epochs = std::max<size_t>(1, H.Epochs / 4);

  auto Raw = LoadEtth1(Args.DataDir, Args.Synthetic,
                       static_cast<uint32_t>(Args.Seed));
  StandardisePerChannel(Raw, std::max<size_t>(64, (Raw.T * 7) / 10));
  auto D = Window(Raw, H.InLen, H.OutLen);
  size_t InDim = D.InDim;
  size_t OutDim = D.OutDim;
  size_t NTotal = D.NRows;
  size_t NTr = static_cast<size_t>(0.7f * NTotal);
  size_t NVa = static_cast<size_t>(0.15f * NTotal);
  size_t NTe = NTotal - NTr - NVa;
  if (H.MaxTrainRows > 0)
    NTr = std::min(NTr, H.MaxTrainRows);

  std::cout << "[info] InLen=" << H.InLen << " OutLen=" << H.OutLen
            << " Channels=" << Raw.C << " InDim=" << InDim
            << " OutDim=" << OutDim << " Hidden=" << H.Hidden
            << " Depth=" << H.Depth << " Epochs=" << H.Epochs
            << " Batch=" << H.Batch << " Train=" << NTr << " Val=" << NVa
            << " Test=" << NTe << "\n";

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 7u);
  StaticMLP M;
  M.Init(InDim, OutDim, H.Hidden, H.Depth, InitRng);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "static_etth1");
  bench::StructuralLog Log(HistPath);

  auto Edges = M.EdgeSet();
  double InitVal = EvalMse(M, D.X, D.Y, InDim, OutDim, 0 /* placeholder */,
                           H.Batch);
  // The placeholder is awkward; recompute against the actual val slice.
  // Validation slice = rows [NTr, NTr+NVa); reuse the same flat X/Y.
  auto EvalSlice = [&](size_t Start, size_t Count) {
    double Sum = 0.0;
    size_t Cnt = 0;
    for (size_t I = 0; I < Count; I += H.Batch) {
      size_t B = std::min(H.Batch, Count - I);
      const float *Out = M.Forward(D.X.data() + (Start + I) * InDim, B);
      Sum += raw::MSEEvalMean(Out, D.Y.data() + (Start + I) * OutDim, B,
                              OutDim) *
             static_cast<double>(B * OutDim);
      Cnt += B * OutDim;
    }
    return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
  };
  (void)InitVal;
  InitVal = EvalSlice(NTr, NVa);
  Log.Log(0, M.UnitCount(), M.EdgeCount(), &Edges, &InitVal,
          {{"train_loss", 0.0}, {"epoch", 0.0}});

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::vector<size_t> Perm(NTr);
  for (size_t I = 0; I < NTr; ++I)
    Perm[I] = I;

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  std::vector<float> XBatch(H.Batch * InDim);
  std::vector<float> YBatch(H.Batch * OutDim);
  std::vector<float> Grad(H.Batch * OutDim);
  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    double TrainLoss = 0.0;
    size_t TrainCnt = 0;
    for (size_t I = 0; I + H.Batch <= NTr; I += H.Batch) {
      for (size_t B = 0; B < H.Batch; ++B) {
        size_t Idx = Perm[I + B];
        std::memcpy(XBatch.data() + B * InDim, D.X.data() + Idx * InDim,
                    InDim * sizeof(float));
        std::memcpy(YBatch.data() + B * OutDim, D.Y.data() + Idx * OutDim,
                    OutDim * sizeof(float));
      }
      Timer.Tick();
      const float *Pred = M.Forward(XBatch.data(), H.Batch);
      Timer.MarkForward();
      double Loss =
          raw::MSELossMean(Pred, YBatch.data(), H.Batch, OutDim, Grad.data());
      Timer.MarkLoss();
      M.Backward(XBatch.data(), Grad.data(), H.Batch);
      Timer.MarkBackward();
      M.Update(H.Lr);
      Timer.MarkUpdate();
      Timer.StepDone();
      TrainLoss += Loss * static_cast<double>(H.Batch * OutDim);
      TrainCnt += H.Batch * OutDim;
    }
    TrainLoss /= std::max<size_t>(TrainCnt, 1);
    double VaMse = EvalSlice(NTr, NVa);
    auto Ed = M.EdgeSet();
    Log.Log(Ep, M.UnitCount(), M.EdgeCount(), &Ed, &VaMse,
            {{"train_loss", TrainLoss},
             {"epoch", static_cast<double>(Ep)}});
    std::cout << "[ep " << Ep << "] train_mse=" << TrainLoss
              << "  val_mse=" << VaMse << "\n";
  }
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
                    .count();

  double TeMse = EvalSlice(NTr + NVa, NTe);
  // MAE for parity with the PyTorch summary CSV.
  auto EvalMaeSlice = [&](size_t Start, size_t Count) {
    double Sum = 0.0;
    size_t Cnt = 0;
    for (size_t I = 0; I < Count; I += H.Batch) {
      size_t B = std::min(H.Batch, Count - I);
      const float *Out = M.Forward(D.X.data() + (Start + I) * InDim, B);
      const float *Yp = D.Y.data() + (Start + I) * OutDim;
      for (size_t K = 0; K < B * OutDim; ++K)
        Sum += std::abs(Out[K] - Yp[K]);
      Cnt += B * OutDim;
    }
    return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
  };
  double TeMae = EvalMaeSlice(NTr + NVa, NTe);
  Log.Flush();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"01_static_etth1"});
  S.Set("dataset", Args.Synthetic ? std::string{"synthetic-etth1"}
                                  : std::string{"ETTh1"});
  S.Set("in_len", static_cast<int>(H.InLen));
  S.Set("out_len", static_cast<int>(H.OutLen));
  S.Set("hidden", static_cast<int>(H.Hidden));
  S.Set("depth", static_cast<int>(H.Depth));
  S.Set("epochs", static_cast<int>(H.Epochs));
  S.Set("batch", static_cast<int>(H.Batch));
  S.Set("lr", static_cast<double>(H.Lr));
  S.Set("wall_seconds", Wall);
  S.Set("val_mse_final", Log.Records().back().ValLoss);
  S.Set("test_mse", TeMse);
  S.Set("test_mae", TeMae);
  S.Set("n_units", static_cast<int>(M.UnitCount()));
  S.Set("n_edges", static_cast<int>(M.EdgeCount()));
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

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TeMse
            << "  test_mae=" << TeMae << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
