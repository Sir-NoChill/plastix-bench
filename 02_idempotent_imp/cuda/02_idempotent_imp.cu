// Workload 2 / 5 — IDEMPOTENT SHRINKAGE, pure-CUDA / cuBLAS implementation.
//
// A GPU port of 02_idempotent_imp.cpp (the OpenBLAS impl): Iterative Magnitude
// Pruning (Frankle & Carbin 2018) of a multi-layer classification MLP on a
// UCR-style synthetic time-series task. Mirrors 02_idempotent_imp.py /.cpp:
//
//   - alive-edge count decreases monotonically across rounds
//   - the dead-edge set is a growing union of prior rounds' kills
//   - the run terminates when a round kills zero edges (fixed point)
//
// Architecture: depth=4 = 3 ReLU hidden + 1 linear softmax readout, per-layer
// Linear biases. Loss reduction is mean cross-entropy.
//
// The structural machinery (mask flips, magnitude threshold, prune, host RNG)
// is identical to the cpp impl and runs on the host; the dense linear algebra
// of the forward / backward / update path runs on the GPU via cuBLAS
// (cublasSgemm / cublasSgemv), with tiny custom kernels for ReLU, its backward,
// bias-add, and the SGD step. Cross-entropy / softmax / argmax are computed on
// host from the downloaded logits (matches the cpp accumulation exactly).
//
// Each layer's weights are canonical on the host (so the host structural logic
// from the cpp impl is reused verbatim); they are uploaded to the device before
// the forward pass and the SGD-updated weights are downloaded after the update.
// cuBLAS uses column-major; the host (OutCap x InCap) row-major matrix IS a
// column-major (InCap x OutCap) matrix with leading dim InCap.
//
// Build (standalone):
//   nvcc -O3 -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -arch=sm_89 -I common -I/usr/local/cuda/include \
//     02_idempotent_imp/cuda/02_idempotent_imp.cu \
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

// ReLU in place over the live (Batch x OutDim) block (row-major, contiguous).
__global__ void ReluInPlace(float *Y, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N && Y[I] < 0.0f)
    Y[I] = 0.0f;
}

// ReLU backward gate: where the post-activation Y <= 0, zero the gradient.
__global__ void ReluGate(const float *Y, float *GradY, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N && Y[I] <= 0.0f)
    GradY[I] = 0.0f;
}

// Add bias broadcast over the batch: Y[b*Out + j] += Bias[j].
__global__ void AddBias(float *Y, const float *Bias, size_t Batch, size_t Out) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < Batch * Out)
    Y[I] += Bias[I % Out];
}

// SGD step over the full physical (OutCap x InCap) weight buffer; dead-edge
// masking is handled host-side, so this is a plain elementwise update. We use
// it for weights and biases alike.
__global__ void SgdStep(float *W, const float *G, float Lr, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    W[I] -= Lr * G[I];
}

inline unsigned Grid(size_t N, unsigned Block) {
  return static_cast<unsigned>((N + Block - 1) / Block);
}

constexpr unsigned kBlock = 256;

// ===========================================================================
// Host-side hyperparameters / dataset — copied verbatim from the cpp impl.
// ===========================================================================

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

Dataset SynthUcr(size_t NPerClass, size_t NClasses, size_t Length, float Snr,
                 uint32_t Seed) {
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

void NormaliseInstance(Dataset &D) {
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

// --- Host CE / softmax / argmax helpers (copied from raw::, mlp.hpp) -------

void SoftmaxRowwise(float *Logits, size_t B, size_t D) {
  for (size_t I = 0; I < B; ++I) {
    float *Row = Logits + I * D;
    float Max = Row[0];
    for (size_t J = 1; J < D; ++J)
      if (Row[J] > Max)
        Max = Row[J];
    float Sum = 0.0f;
    for (size_t J = 0; J < D; ++J) {
      Row[J] = std::exp(Row[J] - Max);
      Sum += Row[J];
    }
    float InvSum = 1.0f / Sum;
    for (size_t J = 0; J < D; ++J)
      Row[J] *= InvSum;
  }
}

double CrossEntropyMean(const float *Logits, const int *Targets, size_t B,
                        size_t D, float *Probs, float *Grad) {
  std::memcpy(Probs, Logits, sizeof(float) * B * D);
  SoftmaxRowwise(Probs, B, D);
  double Loss = 0.0;
  for (size_t I = 0; I < B; ++I) {
    int T = Targets[I];
    Loss -= std::log(std::max(Probs[I * D + T], 1e-12f));
  }
  Loss /= static_cast<double>(B);
  for (size_t I = 0; I < B; ++I)
    for (size_t J = 0; J < D; ++J) {
      float P = Probs[I * D + J];
      float Y = (static_cast<int>(J) == Targets[I]) ? 1.0f : 0.0f;
      Grad[I * D + J] = (P - Y) / static_cast<float>(B);
    }
  return Loss;
}

double ArgmaxAccuracy(const float *Logits, const int *Targets, size_t B,
                      size_t D) {
  size_t Correct = 0;
  for (size_t I = 0; I < B; ++I) {
    const float *Row = Logits + I * D;
    int Arg = 0;
    for (size_t J = 1; J < D; ++J)
      if (Row[J] > Row[Arg])
        Arg = static_cast<int>(J);
    if (Arg == Targets[I])
      ++Correct;
  }
  return B ? static_cast<double>(Correct) / B : 0.0;
}

float XavierLimit(size_t In, size_t Out) {
  return std::sqrt(6.0f / static_cast<float>(In + Out));
}

// --- Linear layer: host-canonical masked weights, GPU matmul / SGD --------
//
// Mirrors raw::Linear's storage (row-major OutCap x InCap, stride InCap). The
// host buffers stay the source of truth for the structural logic (mask flips,
// magnitude threshold, prune). For each step we keep the device weights synced
// with the host (Upload), run the gemms and the SGD on device, then download
// the SGD-updated weights back to host (Download).
struct Linear {
  size_t InDim = 0, OutDim = 0, InCap = 0, OutCap = 0;
  std::vector<float> Weight; // OutCap x InCap row-major
  std::vector<float> Bias;   // OutCap
  std::vector<uint8_t> Mask; // OutCap x InCap, empty == no mask

  // Device mirrors (sized to physical capacity).
  float *dW = nullptr, *dB = nullptr, *dGW = nullptr, *dGB = nullptr;
  const float *dOnes_ = nullptr; // shared ones-vector for bias-grad reduction

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

  void XavierInit(std::mt19937 &Rng) {
    float Lim = XavierLimit(InDim, OutDim);
    std::uniform_real_distribution<float> U(-Lim, Lim);
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InDim; ++J)
        Weight[I * InCap + J] = U(Rng);
    for (size_t I = 0; I < OutDim; ++I)
      Bias[I] = 0.0f;
  }

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
  // col-major: Y_cm(Out x B) = W^T_blk * X_cm(In x B); W host row-major
  // (Out x In, ld=InCap) is col-major (In x Out) -> op_T gives (Out x In).
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
  // col-major: GX_cm(In x B) = W_blk(In x Out) * GY_cm(Out x B), op_N for both.
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
  // col-major: GW_cm(In x Out) = X_cm(In x B) * GY_cm(Out x B)^T,
  // op_N for X, op_T for GY. Bias grad = Scale * column-sum of GY.
  void BackwardWeights(cublasHandle_t Bl, const float *dX, const float *dGY,
                       size_t Batch, float Scale) {
    if (Batch == 0)
      return;
    const float Beta = 0.0f;
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

  // SGD: W -= Lr*GW; B -= Lr*GB on the full physical buffers. The cpp impl
  // zeroes the dead-edge gradients before the step; since GradW capacity past
  // OutDim/InDim is zero (memset above) and the gemm only fills the live block,
  // we instead re-enforce the mask host-side after Download (EnforceMask),
  // which matches the cpp's post-update re-zeroing.
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

  // Zero the gradient on dead edges (host buffer). Used implicitly by re-zeroing
  // weights below; kept for clarity / parity with the cpp impl.
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

// Multi-layer ReLU MLP with per-layer pruning masks. Host weights canonical.
struct WideMLP {
  std::vector<Linear> Layers;
  cublasHandle_t Bl = nullptr;
  size_t Cap = 0; // batch capacity for scratch buffers

  // Device scratch, capacity Cap x (max OutDim) each, indexed per layer.
  std::vector<float *> Preacts; // pre-activation z (= AddBias result)
  std::vector<float *> Acts;    // post-activation a (ReLU); last == preact
  std::vector<float *> Grads;   // dL/dz per layer (Batch x OutDim)
  std::vector<float *> GradIn;  // scratch dL/d(input) (Batch x InDim)
  float *dOnes = nullptr;       // ones (Cap) for bias-grad reductions

  void Init(size_t InDim, size_t OutDim, size_t Hidden, size_t Depth,
            size_t BatchCap, std::mt19937 &Rng, cublasHandle_t Handle) {
    Bl = Handle;
    Cap = BatchCap;
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

    CUDA_CHECK(cudaMalloc(&dOnes, Cap * sizeof(float)));
    std::vector<float> Ones(Cap, 1.0f);
    CUDA_CHECK(cudaMemcpy(dOnes, Ones.data(), Cap * sizeof(float),
                          cudaMemcpyHostToDevice));

    Preacts.assign(Depth, nullptr);
    Acts.assign(Depth, nullptr);
    Grads.assign(Depth, nullptr);
    GradIn.assign(Depth, nullptr);
    for (size_t LI = 0; LI < Depth; ++LI) {
      Layers[LI].dOnes_ = dOnes;
      size_t Outp = Layers[LI].OutDim;
      size_t In = Layers[LI].InDim;
      CUDA_CHECK(cudaMalloc(&Preacts[LI], Cap * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&Acts[LI], Cap * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&Grads[LI], Cap * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&GradIn[LI], Cap * In * sizeof(float)));
    }
    UploadAll();
  }

  void UploadAll() {
    for (auto &L : Layers)
      L.Upload();
  }
  void DownloadAll() {
    for (auto &L : Layers)
      L.Download();
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

  // Forward over device batch dX (row-major Batch x InDim). Hidden layers ReLU,
  // last layer linear (logits). Returns final-layer output device buffer.
  const float *Forward(const float *dX, size_t Batch) {
    const float *Cur = dX;
    for (size_t LI = 0; LI < Layers.size(); ++LI) {
      Linear &L = Layers[LI];
      L.Forward(Bl, Cur, Preacts[LI], Batch);
      size_t Nout = Batch * L.OutDim;
      if (LI + 1 < Layers.size()) {
        CUDA_CHECK(cudaMemcpyAsync(Acts[LI], Preacts[LI], Nout * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
        ReluInPlace<<<Grid(Nout, kBlock), kBlock>>>(Acts[LI], Nout);
      } else {
        CUDA_CHECK(cudaMemcpyAsync(Acts[LI], Preacts[LI], Nout * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
      }
      Cur = Acts[LI];
    }
    return Cur;
  }

  // Backward through CE. dTopGrad is dL/d(logits) on device (Batch x OutDim).
  // Computes per-layer weight/bias gradients but does NOT apply them.
  void Backward(const float *dX, const float *dTopGrad, size_t Batch) {
    size_t D = Layers.size();
    CUDA_CHECK(cudaMemcpyAsync(Grads[D - 1], dTopGrad,
                               Batch * Layers[D - 1].OutDim * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    for (ptrdiff_t LI = static_cast<ptrdiff_t>(D) - 1; LI >= 0; --LI) {
      Linear &L = Layers[LI];
      const float *InData = (LI == 0) ? dX : Acts[LI - 1];
      L.BackwardWeights(Bl, InData, Grads[LI], Batch, 1.0f);
      if (LI > 0) {
        L.BackwardInput(Bl, Grads[LI], GradIn[LI], Batch);
        // hidden layer below uses ReLU; gradient passes only where act > 0.
        size_t Nin = Batch * L.InDim;
        ReluGate<<<Grid(Nin, kBlock), kBlock>>>(Acts[LI - 1], GradIn[LI], Nin);
        CUDA_CHECK(cudaMemcpyAsync(Grads[LI - 1], GradIn[LI],
                                   Nin * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
      }
    }
  }

  void Update(float Lr) {
    for (auto &L : Layers)
      L.SGD(Lr);
  }

  void Free() {
    for (auto &L : Layers) {
      cudaFree(L.dW);
      cudaFree(L.dB);
      cudaFree(L.dGW);
      cudaFree(L.dGB);
    }
    for (auto *P : Preacts)
      cudaFree(P);
    for (auto *P : Acts)
      cudaFree(P);
    for (auto *P : Grads)
      cudaFree(P);
    for (auto *P : GradIn)
      cudaFree(P);
    cudaFree(dOnes);
  }
};

// Global threshold: kth-smallest |w| over currently-alive weights (host).
float ComputeThreshold(const WideMLP &M, float PruneFrac) {
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

// Prune all alive edges with |w| <= threshold. Returns count killed (host).
size_t MagnitudePrune(WideMLP &M, float Threshold) {
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

// --- device-batch helpers --------------------------------------------------

struct DevState {
  float *dX = nullptr;    // full dataset N x Dim (uploaded once)
  float *dXB = nullptr;   // per-batch staged inputs (Cap x Dim)
  float *dGrad = nullptr; // per-batch top gradient (Cap x NumClasses)
  size_t Dim = 0, NumClasses = 0, Cap = 0;
  std::vector<float> HostLogits; // Cap x NumClasses for CE/eval reads
};

double EvalAcc(WideMLP &M, DevState &DS, const Dataset &D, size_t Batch) {
  size_t Correct = 0, Cnt = 0;
  size_t NC = M.Layers.back().OutDim;
  for (size_t I = 0; I < D.N; I += Batch) {
    size_t B = std::min(Batch, D.N - I);
    CUDA_CHECK(cudaMemcpy(DS.dXB, D.X.data() + I * D.Dim,
                          B * D.Dim * sizeof(float), cudaMemcpyHostToDevice));
    const float *Logits = M.Forward(DS.dXB, B);
    CUDA_CHECK(cudaMemcpy(DS.HostLogits.data(), Logits,
                          B * NC * sizeof(float), cudaMemcpyDeviceToHost));
    Correct += static_cast<size_t>(
        ArgmaxAccuracy(DS.HostLogits.data(), D.Y.data() + I, B, NC) *
        static_cast<double>(B));
    Cnt += B;
  }
  return Cnt ? static_cast<double>(Correct) / Cnt : 0.0;
}

double EvalNll(WideMLP &M, DevState &DS, const Dataset &D, size_t Batch) {
  size_t NC = M.Layers.back().OutDim;
  std::vector<float> Probs(Batch * NC), Grad(Batch * NC);
  double Loss = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < D.N; I += Batch) {
    size_t B = std::min(Batch, D.N - I);
    CUDA_CHECK(cudaMemcpy(DS.dXB, D.X.data() + I * D.Dim,
                          B * D.Dim * sizeof(float), cudaMemcpyHostToDevice));
    const float *Logits = M.Forward(DS.dXB, B);
    CUDA_CHECK(cudaMemcpy(DS.HostLogits.data(), Logits,
                          B * NC * sizeof(float), cudaMemcpyDeviceToHost));
    Loss += CrossEntropyMean(DS.HostLogits.data(), D.Y.data() + I, B, NC,
                             Probs.data(), Grad.data()) *
            static_cast<double>(B);
    Cnt += B;
  }
  return Cnt ? Loss / static_cast<double>(Cnt) : 0.0;
}

void TrainEpochs(WideMLP &M, DevState &DS, const Dataset &D, size_t Epochs,
                 size_t Batch, float Lr, std::mt19937 &Rng,
                 bench::PhaseTimer &Timer, size_t MaxSteps, size_t &StepsDone) {
  size_t NC = M.Layers.back().OutDim;
  std::vector<size_t> Perm(D.N);
  for (size_t I = 0; I < D.N; ++I)
    Perm[I] = I;
  std::vector<float> XBatch(Batch * D.Dim);
  std::vector<int> YBatch(Batch);
  std::vector<float> Probs(Batch * NC), Grad(Batch * NC);
  for (size_t E = 0; E < Epochs; ++E) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    for (size_t I = 0; I + Batch <= D.N; I += Batch) {
      for (size_t B = 0; B < Batch; ++B) {
        std::memcpy(XBatch.data() + B * D.Dim, D.X.data() + Perm[I + B] * D.Dim,
                    D.Dim * sizeof(float));
        YBatch[B] = D.Y[Perm[I + B]];
      }
      CUDA_CHECK(cudaMemcpy(DS.dXB, XBatch.data(), Batch * D.Dim * sizeof(float),
                            cudaMemcpyHostToDevice));
      Timer.Tick();
      const float *Logits = M.Forward(DS.dXB, Batch);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkForward();

      // CE / softmax on host (matches the cpp accumulation); upload grad.
      CUDA_CHECK(cudaMemcpy(DS.HostLogits.data(), Logits,
                            Batch * NC * sizeof(float), cudaMemcpyDeviceToHost));
      CrossEntropyMean(DS.HostLogits.data(), YBatch.data(), Batch, NC,
                       Probs.data(), Grad.data());
      CUDA_CHECK(cudaMemcpy(DS.dGrad, Grad.data(), Batch * NC * sizeof(float),
                            cudaMemcpyHostToDevice));
      Timer.MarkLoss();

      M.Backward(DS.dXB, DS.dGrad, Batch);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkBackward();

      M.Update(Lr);
      CUDA_CHECK(cudaDeviceSynchronize());
      // Pull SGD-updated weights to host, re-enforce masks (re-zero pruned
      // weights in case the update pushed them off zero), then re-upload so the
      // device sees the masked weights for the next forward — mirrors the cpp
      // SGD()'s mask enforcement.
      M.DownloadAll();
      for (auto &L : M.Layers)
        L.EnforceMask();
      M.UploadAll();
      Timer.MarkUpdate();
      Timer.StepDone();

      ++StepsDone;
      if (MaxSteps > 0 && StepsDone >= MaxSteps)
        return;
    }
  }
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.InLen = static_cast<size_t>(Args.GetInt("length", H.InLen));
  H.NumClasses = static_cast<size_t>(Args.GetInt("n-classes", H.NumClasses));
  H.Hidden = static_cast<size_t>(Args.GetInt("hidden", H.Hidden));
  H.Depth = static_cast<size_t>(Args.GetInt("depth", H.Depth));
  H.NPerClass = static_cast<size_t>(Args.GetInt("n-per-class", H.NPerClass));
  H.InitEpochs = static_cast<size_t>(Args.GetInt("init-epochs", H.InitEpochs));
  H.FinetuneEpochs =
      static_cast<size_t>(Args.GetInt("finetune-epochs", H.FinetuneEpochs));
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

  // --max-steps caps the number of optimisation steps across the whole run
  // (for quick smoke runs); 0 = unlimited. Mirrors the 01/05 CUDA impls.
  size_t MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", 0));

  bench::MemoryProbe MP;
  MP.Start();

  auto Train = SynthUcr(H.NPerClass, H.NumClasses, H.InLen, H.Snr,
                        static_cast<uint32_t>(Args.Seed));
  auto Test =
      SynthUcr(std::max<size_t>(16, H.NPerClass / 4), H.NumClasses, H.InLen,
               H.Snr, static_cast<uint32_t>(Args.Seed + 1000));
  NormaliseInstance(Train);
  NormaliseInstance(Test);
  MP.EndDataset();

  std::cout << "[info] cuda IMP  InLen=" << H.InLen
            << " Classes=" << H.NumClasses << " Hidden=" << H.Hidden
            << " Depth=" << H.Depth << " Train=" << Train.N
            << " Test=" << Test.N << "\n";

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 11u);
  WideMLP M;
  M.Init(H.InLen, H.NumClasses, H.Hidden, H.Depth, H.Batch, InitRng, Bl);
  MP.EndWeights();

  // Per-batch staging buffers on the device + host logit mirror.
  DevState DS;
  DS.Dim = Train.Dim;
  DS.NumClasses = H.NumClasses;
  DS.Cap = H.Batch;
  CUDA_CHECK(cudaMalloc(&DS.dXB, H.Batch * Train.Dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&DS.dGrad, H.Batch * H.NumClasses * sizeof(float)));
  DS.HostLogits.assign(H.Batch * H.NumClasses, 0.0f);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "idempotent_imp");
  bench::StructuralLog Log(HistPath);

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  bench::PhaseTimer Timer;
  size_t StepsDone = 0;

  std::cout << "[imp] initial dense training, " << H.InitEpochs << " epochs\n";
  TrainEpochs(M, DS, Train, H.InitEpochs, H.Batch, H.Lr, Rng, Timer, MaxSteps,
              StepsDone);
  double Acc0 = EvalAcc(M, DS, Test, H.Batch);
  double Nll0 = EvalNll(M, DS, Test, H.Batch);
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
    bool Capped = (MaxSteps > 0 && StepsDone >= MaxSteps);
    if (Capped)
      break;
    size_t AliveBefore = M.AliveCount();
    float Thresh = ComputeThreshold(M, H.PruneFrac);
    size_t Killed = MagnitudePrune(M, Thresh);
    // Prune mutated host weights/masks; re-sync the device.
    M.UploadAll();
    TrainEpochs(M, DS, Train, H.FinetuneEpochs, H.Batch,
                H.Lr * H.FinetuneLrScale, Rng, Timer, MaxSteps, StepsDone);
    size_t AliveAfter = M.AliveCount();
    double Acc = EvalAcc(M, DS, Test, H.Batch);
    double Nll = EvalNll(M, DS, Test, H.Batch);
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
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  Log.Flush();
  double FinalSparsity =
      1.0 - static_cast<double>(M.AliveCount()) / static_cast<double>(Edges0);
  double FinalAcc = EvalAcc(M, DS, Test, H.Batch);

  float JMin = 1.0f, JMax = 1.0f;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JMax = std::max(JMax, R.Jaccard);
  }

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
  S.Set("metric_kind", std::string{"accuracy"});
  S.Set("n_units", static_cast<int>(M.UnitCount()));
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_max", static_cast<double>(JMax));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  rounds=" << FinalRound
            << "  sparsity=" << FinalSparsity << "  acc=" << FinalAcc << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  M.Free();
  cudaFree(DS.dXB);
  cudaFree(DS.dGrad);
  cublasDestroy(Bl);
  (void)LogPath;
  return 0;
}
