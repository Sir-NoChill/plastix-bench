// Workload 4 / 5 — CONTINUOUS-SMALL regime, pure-CUDA / cuBLAS implementation.
//
// A GPU port of 04_continuous_small_appliances.cpp (the OpenBLAS impl): a
// single-hidden-layer regression MLP (in -> H ReLU -> out linear) trained on
// the UCI Appliances energy dataset, with a per-step probabilistic neuron
// split + per-step probabilistic edge prune. Net structural delta per step is
// at most +/-1 unit and a small number of edges, so the Jaccard between
// consecutive live-edge sets sits very close to 1.0.
//
// The structural machinery (mask flips, split / prune, host RNG, the
// activation-variance tracker) is identical to the cpp impl and runs on the
// host; the dense linear algebra of the forward / backward / update path runs
// on the GPU via cuBLAS (cublasSgemm / Sgemv), with tiny custom kernels for
// ReLU, its backward, the elementwise MSE-gradient and the SGD step.
//
// Following the cpp impl's "pre-allocate max + mask" decision, the hidden
// width is physically MaxHidden up front; "splitting" a unit flips L1 mask bits
// and seeds new weights, no realloc. Each layer's weights are canonical on the
// host (so the host structural logic from the cpp impl is reused verbatim);
// they are uploaded to the device before the forward pass and the SGD-updated
// weights are downloaded after the update. cuBLAS uses column-major; we exploit
// that the host (OutCap x InCap) row-major matrix IS a column-major
// (InCap x OutCap) matrix with leading dim InCap, so W^T (the math we want) is a
// plain column-major read with no transpose flag.
//
// Build (standalone):
//   nvcc -O3 -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -arch=sm_89 -I common -I/usr/local/cuda/include \
//     04_continuous_small_appliances/cuda/04_continuous_small_appliances.cu \
//     -L/usr/local/cuda/lib64 -lcublas -o run_benchmark

#include "cpp/common.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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

// --- tiny elementwise kernels --------------------------------------------

// In-place ReLU: Y = max(Y, 0).
__global__ void ReluInPlace(float *Y, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N && Y[I] < 0.0f)
    Y[I] = 0.0f;
}

// ReLU backward gate: G[i] = 0 where Y[i] <= 0 (Y is the post-ReLU activation).
__global__ void ReluGate(const float *Y, float *G, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N && Y[I] <= 0.0f)
    G[I] = 0.0f;
}

// Add bias broadcast over the batch: Y[b*Out + j] += Bias[j].
__global__ void AddBias(float *Y, const float *Bias, size_t Batch, size_t Out) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < Batch * Out)
    Y[I] += Bias[I % Out];
}

// MSE mean gradient: G = 2 (pred - target) / N, N == B*D.
__global__ void MseGrad(const float *Pred, const float *Target, float *Grad,
                        size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    Grad[I] = 2.0f * (Pred[I] - Target[I]) / static_cast<float>(N);
}

// SGD step over the full physical buffer; dead-edge masking is enforced
// host-side, so this is a plain elementwise update (weights and biases alike).
__global__ void SgdStep(float *W, const float *G, float Lr, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    W[I] -= Lr * G[I];
}

inline unsigned Grid(size_t N, unsigned Block) {
  return static_cast<unsigned>((N + Block - 1) / Block);
}

constexpr unsigned kBlock = 256;

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

// ===========================================================================
// Host-side data loading — copied verbatim from
// 04_continuous_small_appliances.cpp.
// ===========================================================================

struct Dataset {
  std::vector<float> X;
  std::vector<float> Y;
  size_t N = 0;
  size_t D = 0;
};

Dataset SynthSlowDrift(size_t N, size_t Dim, uint32_t Seed) {
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

Dataset LoadAppliances(const std::filesystem::path &Dir) {
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

void StandardiseFeatures(Dataset &D, size_t Cut) {
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

void StandardiseTarget(Dataset &D, size_t Cut) {
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

// --- Linear layer: host-canonical weights, GPU matmul / SGD ---------------
//
// Mirrors raw::Linear's storage (row-major OutCap x InCap, stride InCap). The
// host buffers stay the source of truth for the structural logic (mask flips,
// split, prune). For each step we upload Weight/Bias, run the gemms and the SGD
// on device, then download the updated Weight/Bias back to host.
struct Linear {
  size_t InDim = 0, OutDim = 0, InCap = 0, OutCap = 0;
  std::vector<float> Weight; // OutCap x InCap row-major
  std::vector<float> Bias;   // OutCap
  std::vector<uint8_t> Mask; // OutCap x InCap, empty == no mask

  float *dW = nullptr, *dB = nullptr, *dGW = nullptr, *dGB = nullptr;

  void Init(size_t InDim_, size_t OutDim_, size_t InCap_, size_t OutCap_) {
    InDim = InDim_;
    OutDim = OutDim_;
    InCap = InCap_;
    OutCap = OutCap_;
    Weight.assign(OutCap * InCap, 0.0f);
    Bias.assign(OutCap, 0.0f);
    CUDA_CHECK(cudaMalloc(&dW, OutCap * InCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, OutCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGW, OutCap * InCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGB, OutCap * sizeof(float)));
  }

  void EnableMask() { Mask.assign(OutCap * InCap, 1); }

  void Upload() {
    CUDA_CHECK(cudaMemcpy(dW, Weight.data(), OutCap * InCap * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, Bias.data(), OutCap * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  void Download() {
    CUDA_CHECK(cudaMemcpy(Weight.data(), dW, OutCap * InCap * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Bias.data(), dB, OutCap * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Y[Batch x OutDim] = X[Batch x InDim] @ W[OutDim x InDim]^T + Bias.
  // Col-major: Y_cm(Out x B) = W_blk(Out x In) * X_cm(In x B). W host row-major
  // (Out x In, stride InCap) == col-major (In x Out, ld=InCap); pass op_T to get
  // (Out x In). X host (B x In) == col-major (In x B, ld=InDim), op_N.
  void Forward(cublasHandle_t Bl, const float *dX, float *dY,
               size_t Batch) const {
    if (Batch == 0 || OutDim == 0)
      return;
    const float Alpha = 1.0f, Beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        Bl, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(OutDim),
        static_cast<int>(Batch), static_cast<int>(InDim), &Alpha, dW,
        static_cast<int>(InCap), dX, static_cast<int>(InDim), &Beta, dY,
        static_cast<int>(OutDim)));
    AddBias<<<Grid(Batch * OutDim, kBlock), kBlock>>>(dY, dB, Batch, OutDim);
  }

  // GradX[Batch x InDim] = GradY[Batch x OutDim] @ W[OutDim x InDim].
  // Col-major: GX_cm(In x B) = W_blk(In x Out) * GY_cm(Out x B); op_N for W
  // (host (Out x In) as col-major (In x Out), ld=InCap), op_N for GY.
  void BackwardInput(cublasHandle_t Bl, const float *dGY, float *dGX,
                     size_t Batch) const {
    if (Batch == 0 || InDim == 0)
      return;
    const float Alpha = 1.0f, Beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        Bl, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(InDim),
        static_cast<int>(Batch), static_cast<int>(OutDim), &Alpha, dW,
        static_cast<int>(InCap), dGY, static_cast<int>(OutDim), &Beta, dGX,
        static_cast<int>(InDim)));
  }

  // GradW[OutDim x InDim] = Scale * GradY^T @ X (row-major, stride InCap).
  // Col-major GW(In x Out, ld=InCap) = Xcm(In x B) * GYcm(Out x B)^T:
  // op_N for X, op_T for GY, m=In, n=Out, k=B. Also bias-grad = col-sum of GY.
  void BackwardWeights(cublasHandle_t Bl, const float *dX, const float *dGY,
                       size_t Batch, float Scale) {
    if (Batch == 0)
      return;
    const float Beta = 0.0f;
    // Zero the full GradW buffer first so capacity beyond OutDim/InDim stays 0.
    CUDA_CHECK(cudaMemset(dGW, 0, OutCap * InCap * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(
        Bl, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(InDim),
        static_cast<int>(OutDim), static_cast<int>(Batch), &Scale, dX,
        static_cast<int>(InDim), dGY, static_cast<int>(OutDim), &Beta, dGW,
        static_cast<int>(InCap)));
    if (dOnes_)
      CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_N, static_cast<int>(OutDim),
                               static_cast<int>(Batch), &Scale, dGY,
                               static_cast<int>(OutDim), dOnes_, 1, &Beta, dGB,
                               1));
  }

  void SGD(float Lr) {
    size_t Nw = OutCap * InCap;
    SgdStep<<<Grid(Nw, kBlock), kBlock>>>(dW, dGW, Lr, Nw);
    SgdStep<<<Grid(OutCap, kBlock), kBlock>>>(dB, dGB, Lr, OutCap);
  }

  size_t AliveCount() const {
    if (Mask.empty())
      return InDim * OutDim;
    size_t N = 0;
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InDim; ++J)
        if (Mask[I * InCap + J])
          ++N;
    return N;
  }

  void EnforceMask() {
    if (Mask.empty())
      return;
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InDim; ++J) {
        size_t Idx = I * InCap + J;
        if (!Mask[Idx])
          Weight[Idx] = 0.0f;
      }
  }

  const float *dOnes_ = nullptr;
};

float XavierLimit(size_t In, size_t Out) {
  return std::sqrt(6.0f / static_cast<float>(In + Out));
}

// Splittable single-hidden-layer MLP: in -> H (ReLU) -> out (linear).
// L1's mask carries the per-edge keep state (only hidden->input edges can be
// pruned). Hidden width grows via flipping unused rows live in L1 + L2.
//
// Structural logic mirrors the cpp SplitMLP; the dense math runs on the GPU.
struct SplitMLP {
  Linear L1; // in -> H
  Linear L2; // H -> out
  size_t Hidden = 0;
  size_t InDim = 0;
  size_t OutDim = 0;
  size_t MaxHidden = 0;
  size_t Splits = 0;
  size_t Prunes = 0;

  cublasHandle_t Bl = nullptr;
  size_t BatchCap = 0;
  // Device scratch (sized to BatchCap x MaxHidden / OutDim capacity).
  float *dXB = nullptr;  // Batch x InDim
  float *dAct1 = nullptr;// Batch x MaxHidden (post-ReLU hidden)
  float *dOut = nullptr; // Batch x OutDim
  float *dG2 = nullptr;  // Batch x OutDim (top gradient)
  float *dG1 = nullptr;  // Batch x MaxHidden (hidden gradient)
  float *dOnes = nullptr;// BatchCap (ones)

  std::vector<float> Act1; // host mirror of post-ReLU hidden (Batch x Hidden)
  std::vector<float> Out;  // host mirror of output (Batch x OutDim)

  void Init(size_t InDim_, size_t OutDim_, size_t InitHidden,
            size_t MaxHidden_, size_t BatchCap_, std::mt19937 &Rng,
            cublasHandle_t Handle) {
    InDim = InDim_;
    OutDim = OutDim_;
    Hidden = InitHidden;
    MaxHidden = MaxHidden_;
    BatchCap = BatchCap_;
    Bl = Handle;
    L1.Init(InDim, InitHidden, InDim, MaxHidden);
    L2.Init(InitHidden, OutDim, MaxHidden, OutDim);
    float Lim1 = XavierLimit(InDim, InitHidden);
    float Lim2 = XavierLimit(InitHidden, OutDim);
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

    CUDA_CHECK(cudaMalloc(&dXB, BatchCap * InDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dAct1, BatchCap * MaxHidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, BatchCap * OutDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dG2, BatchCap * OutDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dG1, BatchCap * MaxHidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOnes, BatchCap * sizeof(float)));
    std::vector<float> Ones(BatchCap, 1.0f);
    CUDA_CHECK(cudaMemcpy(dOnes, Ones.data(), BatchCap * sizeof(float),
                          cudaMemcpyHostToDevice));
    L1.dOnes_ = dOnes;
    L2.dOnes_ = dOnes;
    Act1.assign(BatchCap * MaxHidden, 0.0f);
    Out.assign(BatchCap * OutDim, 0.0f);
    UploadAll();
  }

  void UploadAll() {
    L1.Upload();
    L2.Upload();
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

  // Forward over a device batch (dX, row-major Batch x InDim). Leaves dAct1 and
  // dOut populated on device. Caller may DownloadAct1 / DownloadOut as needed.
  void ForwardDev(const float *dX, size_t Batch) {
    // h = relu(L1(x))
    L1.Forward(Bl, dX, dAct1, Batch);
    ReluInPlace<<<Grid(Batch * L1.OutDim, kBlock), kBlock>>>(
        dAct1, Batch * L1.OutDim);
    // out = L2(h)  (linear)
    L2.Forward(Bl, dAct1, dOut, Batch);
  }

  void DownloadAct1(size_t Batch) {
    CUDA_CHECK(cudaMemcpy(Act1.data(), dAct1, Batch * Hidden * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  void DownloadOut(size_t Batch) {
    CUDA_CHECK(cudaMemcpy(Out.data(), dOut, Batch * OutDim * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Backward given the top gradient already on device (dGrad, Batch x OutDim).
  void BackwardDev(const float *dX, const float *dGrad, size_t Batch) {
    // d_out: linear. L2 weight grad := dGrad^T @ act1; d/dact1 = L2.W^T @ dGrad.
    L2.BackwardWeights(Bl, dAct1, dGrad, Batch, 1.0f);
    L2.BackwardInput(Bl, dGrad, dG1, Batch);
    // through h = relu(L1(x)): zero grad where act1 <= 0.
    ReluGate<<<Grid(Batch * L1.OutDim, kBlock), kBlock>>>(
        dAct1, dG1, Batch * L1.OutDim);
    L1.BackwardWeights(Bl, dX, dG1, Batch, 1.0f);
  }

  void UpdateDev(float Lr) {
    L1.SGD(Lr);
    L2.SGD(Lr);
  }

  // Pull SGD-updated weights to host, re-enforce the L1 mask, re-upload so the
  // device sees the masked weights for the next forward.
  void DownloadAndEnforce() {
    L1.Download();
    L2.Download();
    L1.EnforceMask();
    L1.Upload();
  }

  // --- structural ops: identical to the cpp impl (host weight buffers) -----

  // Duplicate hidden unit `idx`: copy its incoming L1 row with small noise;
  // halve l2[:, idx] and write the halved copy to the new column.
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
// unit (argmax variance) is the one to split. Verbatim from the cpp impl.
struct ActStats {
  std::vector<std::vector<float>> Buf; // window x hidden_capacity
  size_t Window;
  size_t Idx = 0;
  size_t Filled = 0;
  size_t HiddenCap;
  ActStats(size_t W, size_t HCap) : Window(W), HiddenCap(HCap) {
    Buf.assign(W, std::vector<float>(HCap, 0.0f));
  }
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

// Evaluate MSE over a contiguous slice [Start, Start+Count) of the host
// dataset. Uploads each minibatch's inputs, runs the forward, reduces on host
// in double (matching raw::MSEEvalMean accumulation).
double EvalMseOnSlice(SplitMLP &M, const Dataset &D, size_t Start,
                      size_t Count) {
  if (Count == 0)
    return 0.0;
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < Count; I += M.BatchCap) {
    size_t B = std::min(M.BatchCap, Count - I);
    CUDA_CHECK(cudaMemcpy(M.dXB, D.X.data() + (Start + I) * D.D,
                          B * D.D * sizeof(float), cudaMemcpyHostToDevice));
    M.ForwardDev(M.dXB, B);
    M.DownloadOut(B);
    double S = 0.0;
    for (size_t K = 0; K < B * M.OutDim; ++K) {
      double E = static_cast<double>(M.Out[K]) -
                 static_cast<double>(D.Y[Start + I + K]);
      S += E * E;
    }
    Sum += S;
    Cnt += B * M.OutDim;
  }
  return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
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

  std::cout << "[info] cuda continuous-small  in_dim=" << D.D
            << " data=" << DatasetName << " N=" << D.N
            << " steps=" << H.MaxSteps << " init_h=" << H.InitHidden << "\n";

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 17u);
  SplitMLP M;
  M.Init(D.D, 1, H.InitHidden, H.MaxHidden, H.Batch, InitRng, Bl);
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

  // Device buffer for the per-step gradient (Batch x OutDim).
  float *dGrad = nullptr;
  CUDA_CHECK(cudaMalloc(&dGrad, H.Batch * M.OutDim * sizeof(float)));
  float *dYB = nullptr;
  CUDA_CHECK(cudaMalloc(&dYB, H.Batch * M.OutDim * sizeof(float)));

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
    // Upload inputs + targets for this step.
    CUDA_CHECK(cudaMemcpy(M.dXB, XBatch.data(), H.Batch * D.D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dYB, YBatch.data(),
                          H.Batch * M.OutDim * sizeof(float),
                          cudaMemcpyHostToDevice));

    Timer.Tick();
    M.ForwardDev(M.dXB, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkForward();

    size_t Nout = H.Batch * M.OutDim;
    MseGrad<<<Grid(Nout, kBlock), kBlock>>>(M.dOut, dYB, dGrad, Nout);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkLoss();

    M.BackwardDev(M.dXB, dGrad, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkBackward();

    M.UpdateDev(H.Lr);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Pull updated weights, re-enforce the L1 mask, re-upload.
    M.DownloadAndEnforce();
    Timer.MarkUpdate();
    Timer.StepDone();

    // Pull the post-ReLU hidden activations for the variance tracker.
    M.DownloadAct1(H.Batch);
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

    // If the host weights/masks changed structurally, re-sync the device.
    if (DidChange)
      M.UploadAll();

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
  CUDA_CHECK(cudaDeviceSynchronize());
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
  S.Set("metric_kind", std::string{"mse"});
  S.Set("n_units", static_cast<int>(M.UnitCount()));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  splits=" << M.Splits
            << "  prunes=" << M.Prunes << "  hidden_final=" << M.Hidden
            << "  |du|_max=" << DuMax << "  jaccard_mean=" << JMean << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  cudaFree(dGrad);
  cudaFree(dYB);
  cublasDestroy(Bl);
  (void)LogPath;
  (void)DeMax;
  return 0;
}
