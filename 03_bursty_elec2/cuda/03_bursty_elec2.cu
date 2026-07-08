// Workload 3 / 5 — BURSTY regime, pure-CUDA / cuBLAS implementation.
//
// A GPU port of 03_bursty_elec2.cpp (the OpenBLAS impl): streaming binary
// classification on the Elec2 dataset. The network sits structurally idle for
// many steps, then — when a sliding window of held-out losses goes flat —
// fires a burst: grow N hidden units, then immediately magnitude-prune to keep
// edge count bounded.
//
// The structural machinery (mask flips, grow / magnitude-prune, host RNG, the
// plateau detector) is identical to the cpp impl and runs on the host; the
// dense linear algebra of the forward / backward / update path runs on the GPU
// via cuBLAS (cublasSgemm / cublasSgemv / cublasSaxpy is folded into the SGD
// kernel), with tiny custom kernels for the ReLU activation, its backward, the
// bias add and the elementwise SGD step.
//
// Architecture: in -> H (ReLU) -> H (ReLU) -> out (linear), with a softmax /
// cross-entropy loss. Per the "pre-allocate max + mask" decision, all three
// layers live at MaxHidden physical capacity from the start; "grow" flips mask
// bits on new hidden rows/cols and magnitude-prune flips them off.
//
// Each layer's weights are canonical on the host (so the host structural logic
// from the cpp impl is reused verbatim); they are uploaded to the device, the
// gemms / SGD run on the GPU, and the SGD-updated weights are downloaded after
// each step so the host structural ops see current weights. cuBLAS is
// column-major; the host (OutCap x InCap) row-major matrix IS a column-major
// (InCap x OutCap) matrix with leading dim InCap, so the matmuls below use that
// row-major-host-as-column-major trick (identical to the 01/05 CUDA impls).
//
// Build (standalone):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++20 --extended-lambda \
//     --expt-relaxed-constexpr -arch=sm_89 -I common -I/usr/local/cuda/include \
//     03_bursty_elec2/cuda/03_bursty_elec2.cu \
//     -L/usr/local/cuda/lib64 -lcublas -o run_benchmark

#include "cpp/common.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
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

__global__ void ReluInPlace(float *Y, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N && Y[I] < 0.0f)
    Y[I] = 0.0f;
}

// G zeroed where Y <= 0 (ReLU backward gate; matches raw::ReLUBackward).
__global__ void ReluGate(float *G, const float *Y, size_t N) {
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

// SGD step over the full physical (OutCap x InCap) weight buffer; dead-edge
// masking is handled host-side, so this is a plain elementwise update. Used for
// weights and biases alike.
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

// ===========================================================================
// Host-side data loading — copied verbatim from 03_bursty_elec2.cpp.
// ===========================================================================

struct Dataset {
  std::vector<float> X; // N x D
  std::vector<int> Y;   // N
  size_t N = 0;
  size_t D = 0;
  size_t NumClasses = 2;
};

Dataset SynthDrift(size_t N, size_t Dim, uint32_t Seed) {
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
      Dot += D.X[I * Dim + J] * std::sin(T + static_cast<float>(J));
    }
    D.Y[I] = (Dot > 0.0f) ? 1 : 0;
    (void)U;
  }
  return D;
}

Dataset LoadElec2(const std::filesystem::path &Dir) {
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
void StandardiseFeatures(Dataset &D, size_t Cut) {
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

float XavierLimit(size_t In, size_t Out) {
  return std::sqrt(6.0f / static_cast<float>(In + Out));
}

// ===========================================================================
// Loss helpers (host-side; match raw:: exactly, run on downloaded logits).
// ===========================================================================

void SoftmaxRowwise(float *Logits, size_t B, size_t Dd) {
  for (size_t I = 0; I < B; ++I) {
    float *Row = Logits + I * Dd;
    float Max = Row[0];
    for (size_t J = 1; J < Dd; ++J)
      if (Row[J] > Max)
        Max = Row[J];
    float Sum = 0.0f;
    for (size_t J = 0; J < Dd; ++J) {
      Row[J] = std::exp(Row[J] - Max);
      Sum += Row[J];
    }
    float InvSum = 1.0f / Sum;
    for (size_t J = 0; J < Dd; ++J)
      Row[J] *= InvSum;
  }
}

// Cross-entropy mean reduction; writes grad w.r.t logits (p - onehot)/B into
// Grad. Mirrors raw::CrossEntropyMean.
double CrossEntropyMean(const float *Logits, const int *Targets, size_t B,
                        size_t Dd, float *Probs, float *Grad) {
  std::memcpy(Probs, Logits, sizeof(float) * B * Dd);
  SoftmaxRowwise(Probs, B, Dd);
  double Loss = 0.0;
  for (size_t I = 0; I < B; ++I) {
    int T = Targets[I];
    Loss -= std::log(std::max(Probs[I * Dd + T], 1e-12f));
  }
  Loss /= static_cast<double>(B);
  for (size_t I = 0; I < B; ++I)
    for (size_t J = 0; J < Dd; ++J) {
      float P = Probs[I * Dd + J];
      float Y = (static_cast<int>(J) == Targets[I]) ? 1.0f : 0.0f;
      Grad[I * Dd + J] = (P - Y) / static_cast<float>(B);
    }
  return Loss;
}

double ArgmaxAccuracy(const float *Logits, const int *Targets, size_t B,
                      size_t Dd) {
  size_t Correct = 0;
  for (size_t I = 0; I < B; ++I) {
    const float *Row = Logits + I * Dd;
    int Arg = 0;
    for (size_t J = 1; J < Dd; ++J)
      if (Row[J] > Row[Arg])
        Arg = static_cast<int>(J);
    if (Arg == Targets[I])
      ++Correct;
  }
  return B ? static_cast<double>(Correct) / B : 0.0;
}

// ===========================================================================
// Linear layer: host-canonical weights, GPU matmul / SGD. Mirrors raw::Linear
// storage (row-major OutCap x InCap, stride InCap). The host buffers stay the
// source of truth for the structural logic; for each step the weights are
// uploaded, the gemms / SGD run on device, then weights are downloaded.
// ===========================================================================

struct Linear {
  size_t InDim = 0, OutDim = 0, InCap = 0, OutCap = 0;
  std::vector<float> Weight; // OutCap x InCap row-major
  std::vector<float> Bias;   // OutCap
  std::vector<uint8_t> Mask; // OutCap x InCap, empty == no mask

  // Device mirrors (sized to physical capacity).
  float *dW = nullptr, *dB = nullptr, *dGW = nullptr, *dGB = nullptr;
  const float *dOnes_ = nullptr; // shared ones vector (bias-grad reduction)

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
  // Col-major: Y_cm(Out x B) = W^T(Out x In) * X(In x B).
  //   dW host (Out x In, ld=InCap) as col-major is (In x Out); op_T -> (Out x In)
  //   dX host (B x In, ld=InDim) as col-major is (In x B); op_N
  // => sgemm(op_T, op_N, m=Out, n=B, k=In, A=dW ld=InCap, B=dX ld=InDim,
  //          C=dY ld=Out).
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
  // Col-major: GX_cm(In x B) = W(In x Out) * GY(Out x B).
  // => sgemm(op_N, op_N, m=In, n=B, k=Out, A=dW ld=InCap, B=dGY ld=OutDim,
  //          C=dGX ld=In).
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

  // GradW[OutDim x InDim] = Scale * GradY^T @ X, stored row-major (ld=InCap).
  // Col-major GW(In x Out) = X(In x B) * GY^T(B x Out).
  // => sgemm(op_N, op_T, m=In, n=Out, k=B, A=dX ld=InDim, B=dGY ld=OutDim,
  //          C=dGW ld=InCap). Also bias grad = column-sum(GY)*Scale via sgemv.
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
    // Bias gradient: column-sum of GY, scaled. GY host (B x Out) col-major is
    // (Out x B, ld=OutDim); (Out x B) * ones(B) with op_N gives the sum.
    if (dOnes_)
      CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_N, static_cast<int>(OutDim),
                               static_cast<int>(Batch), &Scale, dGY,
                               static_cast<int>(OutDim), dOnes_, 1, &Beta, dGB,
                               1));
  }

  // SGD: W -= Lr*GW; B -= Lr*GB over the full physical buffers. Dead edges are
  // re-zeroed host-side after Download (the cpp impl's mask enforcement).
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

  // Force pruned weights (and grads, via SGD pre-step) to zero on the host
  // buffer. Matches the cpp impl's SGD mask enforcement.
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
};

// ===========================================================================
// Growable MLP: in -> H -> H -> out. Structural logic on host, math on GPU.
// Mirrors GrowableMLP in the cpp impl.
// ===========================================================================
struct GrowableMLP {
  Linear L0; // in -> H
  Linear L1; // H -> H
  Linear L2; // H -> out
  size_t Hidden = 0, InDim = 0, OutDim = 0, MaxHidden = 0;
  size_t Bursts = 0, Prunes = 0;

  cublasHandle_t Bl = nullptr;
  size_t BatchCap = 0;
  // Device scratch (sized to BatchCap x capacity).
  float *dA0 = nullptr;  // Batch x MaxHidden (post-ReLU act0)
  float *dA1 = nullptr;  // Batch x MaxHidden (post-ReLU act1)
  float *dA2 = nullptr;  // Batch x OutDim    (logits)
  float *dG2 = nullptr;  // Batch x OutDim    (top grad)
  float *dG1 = nullptr;  // Batch x MaxHidden
  float *dG0 = nullptr;  // Batch x MaxHidden
  float *dXB = nullptr;  // Batch x InDim     (inputs)
  float *dOnes = nullptr;
  std::vector<float> HostLogits; // Batch x OutDim, for loss/eval reads

  void Init(size_t InDim_, size_t OutDim_, size_t InitHidden, size_t MaxHidden_,
            size_t BatchCap_, std::mt19937 &Rng, cublasHandle_t Handle) {
    InDim = InDim_;
    OutDim = OutDim_;
    Hidden = InitHidden;
    MaxHidden = MaxHidden_;
    BatchCap = BatchCap_;
    Bl = Handle;
    L0.Init(InDim, InitHidden, InDim, MaxHidden);
    L1.Init(InitHidden, InitHidden, MaxHidden, MaxHidden);
    L2.Init(InitHidden, OutDim, MaxHidden, OutDim);
    float Lim0 = XavierLimit(InDim, InitHidden);
    float Lim1 = XavierLimit(InitHidden, InitHidden);
    float Lim2 = XavierLimit(InitHidden, OutDim);
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

    size_t MH = MaxHidden;
    CUDA_CHECK(cudaMalloc(&dA0, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA1, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA2, BatchCap * OutDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dG2, BatchCap * OutDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dG1, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dG0, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXB, BatchCap * InDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOnes, BatchCap * sizeof(float)));
    std::vector<float> Ones(BatchCap, 1.0f);
    CUDA_CHECK(cudaMemcpy(dOnes, Ones.data(), BatchCap * sizeof(float),
                          cudaMemcpyHostToDevice));
    L0.dOnes_ = dOnes;
    L1.dOnes_ = dOnes;
    L2.dOnes_ = dOnes;
    HostLogits.assign(BatchCap * OutDim, 0.0f);
    UploadAll();
  }

  void UploadAll() {
    L0.Upload();
    L1.Upload();
    L2.Upload();
  }

  size_t UnitCount() const { return Hidden + Hidden + OutDim; }
  size_t EdgeCount() const {
    return L0.AliveCount() + L1.AliveCount() + L2.AliveCount();
  }

  bench::EdgeSet EdgeSetAlive() const {
    bench::EdgeSet Out;
    auto Pull = [&](uint32_t LI, const Linear &L) {
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

  // Forward over device inputs dX (row-major Batch x InDim). Leaves dA2
  // populated and HostLogits mirrored. Returns the host logits pointer.
  const float *Forward(const float *dX, size_t Batch) {
    L0.Forward(Bl, dX, dA0, Batch);
    ReluInPlace<<<Grid(Batch * L0.OutDim, kBlock), kBlock>>>(
        dA0, Batch * L0.OutDim);
    L1.Forward(Bl, dA0, dA1, Batch);
    ReluInPlace<<<Grid(Batch * L1.OutDim, kBlock), kBlock>>>(
        dA1, Batch * L1.OutDim);
    L2.Forward(Bl, dA1, dA2, Batch);
    CUDA_CHECK(cudaMemcpy(HostLogits.data(), dA2,
                          Batch * OutDim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return HostLogits.data();
  }

  // Backward given the top gradient already on device (dGrad, Batch x OutDim).
  void Backward(const float *dX, const float *dGrad, size_t Batch) {
    CUDA_CHECK(cudaMemcpyAsync(dG2, dGrad, Batch * OutDim * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    L2.BackwardWeights(Bl, dA1, dG2, Batch, 1.0f);
    L2.BackwardInput(Bl, dG2, dG1, Batch);
    ReluGate<<<Grid(Batch * L1.OutDim, kBlock), kBlock>>>(dG1, dA1,
                                                          Batch * L1.OutDim);
    L1.BackwardWeights(Bl, dA0, dG1, Batch, 1.0f);
    L1.BackwardInput(Bl, dG1, dG0, Batch);
    ReluGate<<<Grid(Batch * L0.OutDim, kBlock), kBlock>>>(dG0, dA0,
                                                          Batch * L0.OutDim);
    L0.BackwardWeights(Bl, dX, dG0, Batch, 1.0f);
  }

  void Update(float Lr) {
    L0.SGD(Lr);
    L1.SGD(Lr);
    L2.SGD(Lr);
  }

  // Pull SGD-updated weights to host, re-enforce masks, re-upload so the device
  // sees the masked weights for the next forward.
  void DownloadAndEnforce() {
    L0.Download();
    L1.Download();
    L2.Download();
    L0.EnforceMask();
    L1.EnforceMask();
    L2.EnforceMask();
    UploadAll();
  }

  // --- structural ops: identical to the cpp impl (host weight buffers) -----

  // Grow `NNew` hidden units. Returns new hidden width. Host RNG, verbatim.
  size_t Grow(size_t NNew, float Noise, std::mt19937 &Rng) {
    if (NNew == 0)
      return Hidden;
    size_t Want = std::min(MaxHidden, Hidden + NNew);
    size_t Added = Want - Hidden;
    if (Added == 0)
      return Hidden;
    std::normal_distribution<float> N(0.0f, Noise);
    for (size_t I = Hidden; I < Want; ++I) {
      for (size_t J = 0; J < InDim; ++J) {
        L0.Weight[I * L0.InCap + J] = N(Rng);
        L0.Mask[I * L0.InCap + J] = 1;
      }
      L0.Bias[I] = 0.0f;
    }
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

  // Global magnitude prune across L0, L1, L2 live blocks. Host, verbatim.
  size_t MagnitudePrune(float PruneFrac) {
    std::vector<float> Alive;
    auto Push = [&](const Linear &L) {
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
    auto Kill = [&](Linear &L) {
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
// Verbatim from the cpp impl.
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

// Cross-entropy on a fixed slice (no gradient). Uploads the slice to the device
// in minibatches, forwards, reduces CE on host. dGradScratch unused.
double EvalCEOnSlice(GrowableMLP &M, const Dataset &D, size_t Start,
                     size_t Count, size_t Batch, std::vector<float> &Probs,
                     std::vector<float> &Grad) {
  if (Count == 0)
    return 0.0;
  // Match the cpp impl: it forwards the whole slice in one call. Here we cap to
  // the batch capacity and accumulate per-minibatch CE * count, then divide.
  double SumLoss = 0.0;
  size_t Total = 0;
  for (size_t I = 0; I < Count; I += Batch) {
    size_t B = std::min(Batch, Count - I);
    CUDA_CHECK(cudaMemcpy(M.dXB, D.X.data() + (Start + I) * D.D,
                          B * D.D * sizeof(float), cudaMemcpyHostToDevice));
    const float *Logits = M.Forward(M.dXB, B);
    double L = CrossEntropyMean(Logits, D.Y.data() + Start + I, B, D.NumClasses,
                                Probs.data(), Grad.data());
    SumLoss += L * static_cast<double>(B);
    Total += B;
  }
  return Total ? SumLoss / static_cast<double>(Total) : 0.0;
}

double EvalAccOnSlice(GrowableMLP &M, const Dataset &D, size_t Start,
                      size_t Count, size_t Batch) {
  if (Count == 0)
    return 0.0;
  size_t Correct = 0, Total = 0;
  for (size_t I = 0; I < Count; I += Batch) {
    size_t B = std::min(Batch, Count - I);
    CUDA_CHECK(cudaMemcpy(M.dXB, D.X.data() + (Start + I) * D.D,
                          B * D.D * sizeof(float), cudaMemcpyHostToDevice));
    const float *Logits = M.Forward(M.dXB, B);
    double Acc =
        ArgmaxAccuracy(Logits, D.Y.data() + Start + I, B, D.NumClasses);
    Correct += static_cast<size_t>(std::llround(Acc * static_cast<double>(B)));
    Total += B;
  }
  return Total ? static_cast<double>(Correct) / static_cast<double>(Total) : 0.0;
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
  H.PlateauCooldown =
      static_cast<size_t>(Args.GetInt("plateau-cooldown", H.PlateauCooldown));
  if (Args.Quick) {
    H.MaxSteps = std::max<size_t>(50, H.MaxSteps / 5);
    H.ValEvery = std::max<size_t>(1, H.ValEvery / 2);
  }

  bench::MemoryProbe MP;
  MP.Start();

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
  MP.EndDataset();

  std::cout << "[info] cuda bursty MLP  in_dim=" << D.D
            << " classes=" << D.NumClasses << " data=" << DatasetName
            << " N=" << D.N << " steps=" << H.MaxSteps
            << " init_h=" << H.InitHidden << " max_h=" << H.MaxHidden << "\n";

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 13u);
  GrowableMLP M;
  // Batch capacity must cover the eval-slice minibatches too; cpp evals the
  // whole slice at once, but here we cap at the larger of Batch / ValWindow so
  // the eval minibatch granularity stays loss-neutral against the cpp impl.
  size_t BatchCap = std::max(H.Batch, H.ValWindow);
  M.Init(D.D, D.NumClasses, H.InitHidden, H.MaxHidden, BatchCap, InitRng, Bl);
  MP.EndWeights();

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "bursty_elec2");
  bench::StructuralLog Log(HistPath);

  PlateauDetector Det(H.PlateauWindow, H.PlateauRelTol, H.PlateauCooldown);

  // Carve final 15% as held-out test set (mirrors pytorch/plastix).
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
  std::vector<float> Probs(BatchCap * D.NumClasses);
  std::vector<float> Grad(BatchCap * D.NumClasses);

  // Device buffer for the per-step gradient (Batch x NumClasses).
  float *dGrad = nullptr;
  CUDA_CHECK(cudaMalloc(&dGrad, H.Batch * D.NumClasses * sizeof(float)));

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    // Cyclic minibatch over the training portion (held-out test stays unseen).
    for (size_t B = 0; B < H.Batch; ++B) {
      size_t Idx = ((Step - 1) * H.Batch + B) % NTrain;
      std::memcpy(XBatch.data() + B * D.D, D.X.data() + Idx * D.D,
                  D.D * sizeof(float));
      YBatch[B] = D.Y[Idx];
    }
    CUDA_CHECK(cudaMemcpy(M.dXB, XBatch.data(), H.Batch * D.D * sizeof(float),
                          cudaMemcpyHostToDevice));

    Timer.Tick();
    const float *Logits = M.Forward(M.dXB, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkForward();

    CrossEntropyMean(Logits, YBatch.data(), H.Batch, D.NumClasses, Probs.data(),
                     Grad.data());
    CUDA_CHECK(cudaMemcpy(dGrad, Grad.data(),
                          H.Batch * D.NumClasses * sizeof(float),
                          cudaMemcpyHostToDevice));
    Timer.MarkLoss();

    M.Backward(M.dXB, dGrad, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkBackward();

    M.Update(H.Lr);
    CUDA_CHECK(cudaDeviceSynchronize());
    M.DownloadAndEnforce();
    Timer.MarkUpdate();
    Timer.StepDone();

    if (Step % H.ValEvery == 0 || Step == H.MaxSteps) {
      double PlateauLoss =
          EvalCEOnSlice(M, D, PStart, PCount, BatchCap, Probs, Grad);
      size_t VStart = (Step * H.Batch) % NTrain;
      size_t VEnd = std::min(VStart + H.ValWindow, NTrain);
      size_t VCount = VEnd - VStart;
      double VL = EvalCEOnSlice(M, D, VStart, VCount, BatchCap, Probs, Grad);
      double VA = EvalAccOnSlice(M, D, VStart, VCount, BatchCap);
      double TestLoss =
          TestCount > 0
              ? EvalCEOnSlice(M, D, TestStart, TestCount, BatchCap, Probs, Grad)
              : VL;
      double TestAcc = TestCount > 0
                           ? EvalAccOnSlice(M, D, TestStart, TestCount, BatchCap)
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
        // Structural change to host weights/masks: re-sync the device.
        M.UploadAll();
        std::cout << "[burst " << M.Bursts << "] step=" << Step << "  hidden "
                  << OldH << "->" << NewH << "  killed=" << Killed
                  << "  vl=" << VL << "\n";
      }
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
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
  S.Set("post_burst_prune_frac", static_cast<double>(H.PostBurstPruneFrac));
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
  S.Set("metric_kind", std::string{"cross_entropy"});
  S.Set("n_units", static_cast<int>(M.UnitCount()));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  bursts=" << M.Bursts
            << "  hidden_final=" << M.Hidden << "  edges=" << M.EdgeCount()
            << "  val_loss_final="
            << (Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss)
            << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  cudaFree(dGrad);
  cublasDestroy(Bl);
  (void)LogPath;
  return 0;
}
