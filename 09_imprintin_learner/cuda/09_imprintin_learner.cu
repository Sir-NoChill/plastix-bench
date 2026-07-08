// Workload 9 — imprinting-learner, pure-CUDA implementation.
//
// A from-scratch GPU port of the linear TD(λ) core (the same algorithm the
// pytorch/ and cpp/ impls run): streaming value prediction over the APBD
// audio-prediction dataset. Unlike the plastix/ impl — whose policies fall
// back to the host (struct forward accumulator, host-RNG structural ops) — this
// keeps all per-step state resident on the device and uses cuBLAS BLAS-1 ops
// (dot / scal / axpy) for the linear algebra, with two tiny custom kernels for
// the binary-observation expand and the replacing-trace set.
//
// Per step t (matching pytorch/run_benchmark.py::td_lambda_run):
//   V_t       = w · x_t                         (cublasSdot)
//   δ_t       = r_t + γ·V_t − V_{t−1}
//   e_t       = γλ·e_{t−1}; e_t[i]=1 for active i  (scal + replace kernel)
//   w        += (α / nnz)·δ_t · e_t             (cublasSaxpy)
//
// Build (standalone): nvcc -O3 -std=c++17 -I common 09.../cuda/09.cu -lcublas

#include "cpp/common.hpp"
#include "../cpp/examples/dataset.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      std::cerr << "[cuda] " << cudaGetErrorString(e_) << " at " << __LINE__   \
                << "\n";                                                       \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "[cublas] error " << s_ << " at " << __LINE__ << "\n";      \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

namespace {

// Expand one packed-bit observation (LSB-first) into a dense float vector.
__global__ void ExpandObs(const std::uint8_t *Packed, float *X, size_t D) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= D)
    return;
  X[I] = (Packed[I >> 3] >> (I & 7)) & 1u ? 1.0f : 0.0f;
}

// Replacing trace: e[i] = 1 wherever the current observation is active.
__global__ void SetTrace(const float *X, float *E, size_t D) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= D)
    return;
  if (X[I] != 0.0f)
    E[I] = 1.0f;
}

// --- sparse path: gather over the (few) active inputs instead of dense ops ---
// V = Σ w[active]  via a single-block shared-memory reduction over the step's
// active indices (≈ n_freq_bins of them, ≪ D).
__global__ void SparseDot(const int *Act, int Cnt, const float *W, float *V) {
  __shared__ float S[256];
  int T = threadIdx.x;
  float A = 0.0f;
  for (int K = T; K < Cnt; K += blockDim.x)
    A += W[Act[K]];
  S[T] = A;
  __syncthreads();
  for (int O = blockDim.x / 2; O > 0; O >>= 1) {
    if (T < O)
      S[T] += S[T + O];
    __syncthreads();
  }
  if (T == 0)
    *V = S[0];
}

// Replacing trace over active indices only: e[active] = 1.
__global__ void SparseTrace(const int *Act, int Cnt, float *E) {
  int K = blockIdx.x * blockDim.x + threadIdx.x;
  if (K < Cnt)
    E[Act[K]] = 1.0f;
}

// --- truncated dynamic eligibility trace -----------------------------------
// Keep a dynamic "live" set of features whose trace is non-negligible. Every
// op (decay / update) touches only the live set — O(|live|) instead of O(D).
// |live| is bounded by active-per-step / (1-γλ), ≪ D for large networks.
__global__ void DecayLive(const int *Live, int Cnt, float *E, float Decay) {
  int K = blockIdx.x * blockDim.x + threadIdx.x;
  if (K < Cnt)
    E[Live[K]] *= Decay;
}
// Compact survivors (decayed trace ≥ eps) into NewLive; evict the rest (e←0).
__global__ void CompactLive(const int *Old, int Cnt, float *E, float Eps,
                            int *New, int *NewCnt, int *InList) {
  int K = blockIdx.x * blockDim.x + threadIdx.x;
  if (K >= Cnt)
    return;
  int Idx = Old[K];
  if (E[Idx] >= Eps) {
    New[atomicAdd(NewCnt, 1)] = Idx;
  } else {
    E[Idx] = 0.0f;
    InList[Idx] = 0;
  }
}
// Replacing trace: set e[active]=1 and append newly-live actives to the set.
__global__ void AddActiveLive(const int *Act, int Cnt, float *E, int *New,
                              int *NewCnt, int *InList) {
  int K = blockIdx.x * blockDim.x + threadIdx.x;
  if (K >= Cnt)
    return;
  int Idx = Act[K];
  E[Idx] = 1.0f;
  if (atomicExch(&InList[Idx], 1) == 0)
    New[atomicAdd(NewCnt, 1)] = Idx;
}
// w[live] += coef · e[live].
__global__ void UpdateLive(const int *Live, int Cnt, float *W, const float *E,
                           float Coef) {
  int K = blockIdx.x * blockDim.x + threadIdx.x;
  if (K < Cnt)
    W[Live[K]] += Coef * E[Live[K]];
}

std::filesystem::path ResolveDataset(const bench::CliArgs &Args) {
  auto It = Args.Extras.find("dataset");
  if (It != Args.Extras.end())
    return It->second;
  std::vector<std::filesystem::path> C = {
      Args.DataDir / "audio_prediction" / "dataset.bin",
      Args.DataDir / "audio" / "dataset.bin", Args.DataDir / "dataset.bin",
      "09_imprintin_learner/cpp/examples/output/dataset.bin"};
  for (const auto &P : C)
    if (std::filesystem::exists(P))
      return P;
  return {};
}

std::vector<float> ComputeReturns(const std::vector<int> &R, float Gamma) {
  std::vector<float> G(R.size(), 0.0f);
  if (R.empty())
    return G;
  G.back() = static_cast<float>(R.back());
  for (std::ptrdiff_t I = static_cast<std::ptrdiff_t>(R.size()) - 2; I >= 0; --I)
    G[I] = static_cast<float>(R[I]) + Gamma * G[I + 1];
  return G;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  size_t MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", 20000));
  size_t LogEvery = static_cast<size_t>(Args.GetInt("log-every", 500));
  const float Gamma = Args.GetFloat("gamma", 0.99f);
  const float Lambda = Args.GetFloat("lambda", 0.9f);
  const float Alpha = Args.GetFloat("alpha", 0.1f);
  // sparse: 0=dense, 1=sparse forward + dense trace, 2=+ truncated sparse trace.
  const int SparseMode = Args.GetInt("sparse", 0);
  const bool Sparse = SparseMode > 0; // gather over active bits
  const float TraceEps = Args.GetFloat("trace-eps", 1e-3f);
  if (Args.Quick) {
    MaxSteps = std::min<size_t>(MaxSteps, 400);
    LogEvery = std::min<size_t>(LogEvery, 100);
  }

  bench::MemoryProbe MP;
  MP.Start();

  auto Path = ResolveDataset(Args);
  if (Path.empty()) {
    std::cerr << "[err] audio-prediction dataset.bin not found.\n";
    return 2;
  }
  audio_pred::Dataset DS(Path);
  const size_t D = DS.ObservationDim();
  const size_t N = std::min<size_t>(MaxSteps, DS.Size());
  const size_t PackedBytes = DS.PackedBytes();
  std::cout << "[info] cuda IL  dataset=" << Path.string() << " obs_dim=" << D
            << " steps=" << N << "\n";

  // Rewards, returns, and per-step nnz (host); packed obs bytes for upload.
  std::vector<int> Rewards(N);
  std::vector<float> HostNnz(N, 0.0f);
  std::vector<std::uint8_t> HostPacked(N * PackedBytes);
  for (size_t T = 0; T < N; ++T) {
    audio_pred::StepView S = DS[T];
    Rewards[T] = S.Reward();
    HostNnz[T] = static_cast<float>(S.Popcount());
    std::memcpy(&HostPacked[T * PackedBytes], S.RawBytes(), PackedBytes);
  }
  auto Returns = ComputeReturns(Rewards, Gamma);

  // Sparse path: flat active-index list per step (≈ n_freq_bins entries each).
  std::vector<int> Active;
  std::vector<int> ActiveOff(N + 1, 0);
  if (Sparse) {
    for (size_t T = 0; T < N; ++T) {
      audio_pred::StepView S = DS[T];
      for (size_t I = 0; I < D; ++I)
        if (S.Test(I))
          Active.push_back(static_cast<int>(I));
      ActiveOff[T + 1] = static_cast<int>(Active.size());
    }
  }
  MP.EndDataset();

  // Device state — everything resident on the GPU for the whole run.
  float *dW = nullptr, *dE = nullptr, *dX = nullptr, *dV = nullptr;
  std::uint8_t *dPacked = nullptr;
  int *dAct = nullptr;
  int *dLive = nullptr, *dLive2 = nullptr, *dInList = nullptr, *dLiveCnt = nullptr;
  CUDA_CHECK(cudaMalloc(&dW, D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dE, D * sizeof(float)));
  CUDA_CHECK(cudaMemset(dW, 0, D * sizeof(float)));
  CUDA_CHECK(cudaMemset(dE, 0, D * sizeof(float)));
  if (Sparse) {
    CUDA_CHECK(cudaMalloc(&dV, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dAct, std::max<size_t>(Active.size(), 1) * sizeof(int)));
    if (!Active.empty())
      CUDA_CHECK(cudaMemcpy(dAct, Active.data(), Active.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
    if (SparseMode == 2) {
      CUDA_CHECK(cudaMalloc(&dLive, D * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&dLive2, D * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&dInList, D * sizeof(int)));
      CUDA_CHECK(cudaMemset(dInList, 0, D * sizeof(int)));
      CUDA_CHECK(cudaMallocManaged(&dLiveCnt, sizeof(int))); // host reads count
    }
  } else {
    CUDA_CHECK(cudaMalloc(&dX, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPacked, HostPacked.size()));
    CUDA_CHECK(cudaMemcpy(dPacked, HostPacked.data(), HostPacked.size(),
                          cudaMemcpyHostToDevice));
  }
  MP.EndWeights();

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  const unsigned Block = 256;
  const unsigned Grid = static_cast<unsigned>((D + Block - 1) / Block);
  const float Decay = Gamma * Lambda;

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "audio_imprinting");
  bench::StructuralLog Log(HistPath);
  bench::PhaseTimer Timer;

  std::vector<float> Predictions(N, 0.0f);
  float VOld = 0.0f; // w starts at 0 ⇒ V_old = 0
  int LiveCnt = 0;   // size of the truncated-trace live set (SparseMode==2)
  double WindowSse = 0.0, GlobalSse = 0.0;
  size_t WindowCnt = 0, EpochIdx = 0;

  auto T0 = std::chrono::steady_clock::now();
  for (size_t T = 0; T < N; ++T) {
    float V = 0.0f;
    float Coef;
    if (Sparse) {
      const int Off = ActiveOff[T];
      const int Cnt = ActiveOff[T + 1] - Off;
      Timer.Tick();
      SparseDot<<<1, 256>>>(dAct + Off, Cnt, dW, dV); // V = Σ w[active]
      CUDA_CHECK(cudaMemcpy(&V, dV, sizeof(float), cudaMemcpyDeviceToHost));
      Timer.MarkForward();
      float Delta = static_cast<float>(Rewards[T]) + Gamma * V - VOld;
      Coef = (Alpha / std::max(HostNnz[T], 1.0f)) * Delta;
      if (SparseMode == 2) {
        // Truncated dynamic trace: every op touches only the live set.
        if (LiveCnt > 0)
          DecayLive<<<(LiveCnt + 255) / 256, 256>>>(dLive, LiveCnt, dE, Decay);
        *dLiveCnt = 0;
        if (LiveCnt > 0)
          CompactLive<<<(LiveCnt + 255) / 256, 256>>>(dLive, LiveCnt, dE,
                                                      TraceEps, dLive2,
                                                      dLiveCnt, dInList);
        if (Cnt > 0)
          AddActiveLive<<<(Cnt + 255) / 256, 256>>>(dAct + Off, Cnt, dE, dLive2,
                                                    dLiveCnt, dInList);
        CUDA_CHECK(cudaDeviceSynchronize()); // read back the new live count
        LiveCnt = *dLiveCnt;
        std::swap(dLive, dLive2);
        if (LiveCnt > 0)
          UpdateLive<<<(LiveCnt + 255) / 256, 256>>>(dLive, LiveCnt, dW, dE,
                                                     Coef);
        CUDA_CHECK(cudaDeviceSynchronize());
      } else {
        // mode 1: sparse forward, dense trace decay + replace + update.
        CUBLAS_CHECK(cublasSscal(Bl, static_cast<int>(D), &Decay, dE, 1));
        if (Cnt > 0)
          SparseTrace<<<(Cnt + 255) / 256, 256>>>(dAct + Off, Cnt, dE);
        CUBLAS_CHECK(cublasSaxpy(Bl, static_cast<int>(D), &Coef, dE, 1, dW, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      Timer.MarkBackward();
    } else {
      const std::uint8_t *StepPacked = dPacked + T * PackedBytes;
      ExpandObs<<<Grid, Block>>>(StepPacked, dX, D);
      Timer.Tick();
      CUBLAS_CHECK(cublasSdot(Bl, static_cast<int>(D), dW, 1, dX, 1, &V)); // syncs
      Timer.MarkForward();
      float Delta = static_cast<float>(Rewards[T]) + Gamma * V - VOld;
      // e ← γλ·e ; e[active] ← 1 ; w ← w + (α/nnz)·δ·e  (TD update bundle)
      CUBLAS_CHECK(cublasSscal(Bl, static_cast<int>(D), &Decay, dE, 1));
      SetTrace<<<Grid, Block>>>(dX, dE, D);
      Coef = (Alpha / std::max(HostNnz[T], 1.0f)) * Delta;
      CUBLAS_CHECK(cublasSaxpy(Bl, static_cast<int>(D), &Coef, dE, 1, dW, 1));
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkBackward();
    }
    Timer.StepDone();

    Predictions[T] = V;
    VOld = V;
    double Err = static_cast<double>(V) - static_cast<double>(Returns[T]);
    WindowSse += Err * Err;
    GlobalSse += Err * Err;
    ++WindowCnt;
    if (WindowCnt >= LogEvery || T + 1 == N) {
      double WindowMse = WindowSse / WindowCnt;
      ++EpochIdx;
      Log.Log(EpochIdx, D, D, nullptr, &WindowMse,
              {{"train_loss", WindowMse},
               {"epoch", static_cast<double>(EpochIdx)},
               {"step", static_cast<double>(T + 1)},
               {"test_mse", WindowMse}});
      std::cout << "[ep " << EpochIdx << "] step=" << (T + 1)
                << " window_mse=" << WindowMse << "\n";
      WindowSse = 0.0;
      WindowCnt = 0;
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  size_t TailStart = N - std::max<size_t>(1, N / 10);
  double TailSse = 0.0;
  for (size_t T = TailStart; T < N; ++T) {
    double E = static_cast<double>(Predictions[T]) -
               static_cast<double>(Returns[T]);
    TailSse += E * E;
  }
  double TestMse = TailSse / std::max<size_t>(1, N - TailStart);
  double FullMse = N > 0 ? GlobalSse / static_cast<double>(N) : 0.0;

  Log.Flush();

  bench::SummaryWriter S;
  S.Set("workload", std::string{"09_imprintin_learner"});
  S.Set("dataset", std::string{"audio_prediction"});
  S.Set("max_steps", static_cast<int>(N));
  S.Set("wall_seconds", Wall);
  S.Set("val_mse_final", FullMse);
  S.Set("test_mse", TestMse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<int>(D));
  S.Set("n_edges", static_cast<int>(D));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TestMse
            << "  full_mse=" << FullMse << "  obs_dim=" << D << "\n";

  cublasDestroy(Bl);
  cudaFree(dW);
  cudaFree(dE);
  if (dX) cudaFree(dX);
  if (dPacked) cudaFree(dPacked);
  if (dV) cudaFree(dV);
  if (dAct) cudaFree(dAct);
  if (dLive) cudaFree(dLive);
  if (dLive2) cudaFree(dLive2);
  if (dInList) cudaFree(dInList);
  if (dLiveCnt) cudaFree(dLiveCnt);
  (void)LogPath;
  return 0;
}
