// Workload 5 / 5 — CONTINUOUS-LARGE regime, pure-CUDA / cuBLAS implementation.
//
// A GPU port of 05_continuous_large_mackey_glass.cpp: a sparse-recurrent-style
// network on the Mackey-Glass chaotic time series, with heavy-tailed per-step
// unit deltas (Pareto), periodic Watts-Strogatz edge rewiring, and growth-
// momentum-triggered bursts. The structural machinery (mask flips, grow /
// shrink / rewire, host RNG) is identical to the cpp impl and runs on the host;
// the dense linear algebra of the forward / backward / update path runs on the
// GPU via cuBLAS (cublasSgemm), with tiny custom kernels for the tanh
// activation, its backward, and the elementwise MSE-gradient / SGD steps.
//
// Architecture: in -> H (tanh) + recurrent H -> H (sparse) -> tanh -> out
// (linear). Per the "pre-allocate max + mask" decision, all three layers live
// at MaxHidden physical capacity from the start; "grow" / "shrink" flips mask
// bits on whole hidden rows/cols.
//
// Each layer's weights are canonical on the host (so the host structural logic
// from the cpp impl is reused verbatim); they are uploaded to the device before
// the forward pass and the SGD-updated weights are downloaded after the update.
// cuBLAS uses column-major; we exploit that the host (OutCap x InCap) row-major
// matrix IS a column-major (InCap x OutCap) matrix with leading dim InCap, so
// W^T (the math we want) is a plain column-major read with no transpose flag.
//
// Build (standalone):
//   nvcc -O3 -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -arch=sm_89 -I common -I/usr/local/cuda/include \
//     05_continuous_large_mackey_glass/cuda/05_continuous_large_mackey_glass.cu \
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

__global__ void TanhInPlace(float *Y, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    Y[I] = tanhf(Y[I]);
}

// Z = tanh(A + B), elementwise (used for h2 = tanh(h1 + rec)).
__global__ void TanhSum(const float *A, const float *B, float *Z, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    Z[I] = tanhf(A[I] + B[I]);
}

// G *= (1 - Y^2), the tanh backward gate.
__global__ void TanhGate(float *G, const float *Y, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    G[I] *= (1.0f - Y[I] * Y[I]);
}

// C = A + B, elementwise.
__global__ void AddVec(const float *A, const float *B, float *C, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    C[I] = A[I] + B[I];
}

// Add bias broadcast over the batch: Y[b*Out + j] += Bias[j].
__global__ void AddBias(float *Y, const float *Bias, size_t Batch, size_t Out) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < Batch * Out)
    Y[I] += Bias[I % Out];
}

// MSE mean gradient: G = 2 (pred - target) / (B*D). D == 1 here.
__global__ void MseGrad(const float *Pred, const float *Target, float *Grad,
                        size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    Grad[I] = 2.0f * (Pred[I] - Target[I]) / static_cast<float>(N);
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

struct HP {
  size_t SeriesLen = 5000;
  size_t Tau = 17;
  size_t InLen = 32;
  size_t Horizon = 1;
  size_t InitHidden = 128;
  size_t MinHidden = 16;
  size_t MaxHidden = 512;
  float RecurDensity = 0.05f;
  size_t Batch = 64;
  size_t MaxSteps = 1000;
  size_t ValEvery = 25;
  float Lr = 5e-2f;
  float ParetoAlpha = 1.5f;
  size_t MaxDeltaPerStep = 20;
  size_t RewireEvery = 3;
  float RewireFrac = 0.25f;
  float MomentumThreshold = 200.0f;
  size_t MomentumBurst = 10;
};

// --- Mackey-Glass series + windowing (verbatim from the cpp impl) ---------

std::vector<float> MackeyGlass(size_t N, size_t Tau, uint32_t Seed) {
  float Beta = 0.2f, Gamma = 0.1f, X0 = 1.2f, H = 0.1f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> Norm(0.0f, 0.01f);
  size_t Sub = static_cast<size_t>(1.0f / H);
  if (Sub < 1)
    Sub = 1;
  size_t History = Tau + 1;
  std::vector<double> Buf;
  Buf.reserve(Tau * Sub * 4 + N * Sub + History);
  for (size_t I = 0; I < History; ++I)
    Buf.push_back(X0 + Norm(Rng));
  double Cur = Buf.back();
  size_t Burn = Tau * 20 * Sub;
  auto Step = [&]() {
    double Delayed =
        Buf.size() >= Tau * Sub ? Buf[Buf.size() - Tau * Sub] : Buf[0];
    double Dx = Beta * Delayed / (1.0 + std::pow(Delayed, 10.0)) - Gamma * Cur;
    Cur += H * Dx;
    Buf.push_back(Cur);
    if (Buf.size() > Tau * Sub * 2) {
      std::vector<double> Tail(Buf.end() - Tau * Sub * 2, Buf.end());
      Buf = std::move(Tail);
    }
  };
  for (size_t I = 0; I < Burn; ++I)
    Step();
  std::vector<float> Out(N, 0.0f);
  size_t Written = 0;
  for (size_t I = 0; I < N * Sub && Written < N; ++I) {
    Step();
    if (I % Sub == 0)
      Out[Written++] = static_cast<float>(Cur);
  }
  return Out;
}

struct Windowed {
  std::vector<float> X; // N x InLen
  std::vector<float> Y; // N x 1
  size_t N = 0;
  size_t InLen = 0;
};

Windowed WindowSeries(const std::vector<float> &S, size_t InLen,
                      size_t Horizon) {
  Windowed W;
  if (S.size() < InLen + Horizon)
    return W;
  W.N = S.size() - InLen - Horizon + 1;
  W.InLen = InLen;
  W.X.resize(W.N * InLen);
  W.Y.resize(W.N);
  for (size_t I = 0; I < W.N; ++I) {
    for (size_t J = 0; J < InLen; ++J)
      W.X[I * InLen + J] = S[I + J];
    W.Y[I] = S[I + InLen + Horizon - 1];
  }
  return W;
}

// --- Linear layer: host-canonical weights, GPU matmul / SGD ---------------
//
// Mirrors raw::Linear's storage (row-major OutCap x InCap, stride InCap). The
// host buffers stay the source of truth for the structural logic (mask flips,
// grow / shrink / rewire). For each step we upload Weight/Bias, run the gemms
// and the SGD on device, then download the updated Weight/Bias back to host.
struct Linear {
  size_t InDim = 0, OutDim = 0, InCap = 0, OutCap = 0;
  std::vector<float> Weight; // OutCap x InCap row-major
  std::vector<float> Bias;   // OutCap
  std::vector<uint8_t> Mask; // OutCap x InCap, empty == no mask
  std::vector<float> GradW;  // OutCap x InCap (host mirror, for grad-norm)
  std::vector<float> GradB;  // OutCap

  // Device mirrors (sized to physical capacity).
  float *dW = nullptr, *dB = nullptr, *dGW = nullptr, *dGB = nullptr;

  void Init(size_t InDim_, size_t OutDim_, size_t InCap_, size_t OutCap_) {
    InDim = InDim_;
    OutDim = OutDim_;
    InCap = InCap_;
    OutCap = OutCap_;
    Weight.assign(OutCap * InCap, 0.0f);
    Bias.assign(OutCap, 0.0f);
    GradW.assign(OutCap * InCap, 0.0f);
    GradB.assign(OutCap, 0.0f);
    CUDA_CHECK(cudaMalloc(&dW, OutCap * InCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, OutCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGW, OutCap * InCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGB, OutCap * sizeof(float)));
  }

  void EnableMask() { Mask.assign(OutCap * InCap, 1); }

  // Push host weights/biases to the device (call before forward/backward).
  void Upload() {
    CUDA_CHECK(cudaMemcpy(dW, Weight.data(), OutCap * InCap * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, Bias.data(), OutCap * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  // Pull device weights/biases back (call after the SGD step).
  void Download() {
    CUDA_CHECK(cudaMemcpy(Weight.data(), dW, OutCap * InCap * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Bias.data(), dB, OutCap * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  void DownloadGrads() {
    CUDA_CHECK(cudaMemcpy(GradW.data(), dGW, OutCap * InCap * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(GradB.data(), dGB, OutCap * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Y[Batch x OutDim] = X[Batch x InDim] @ W[OutDim x InDim]^T + Bias.
  // Device buffers are row-major; we reinterpret them as column-major to call
  // cuBLAS. Row-major Y (B x Out) == col-major (Out x B). We compute, in
  // col-major: Y_cm(Out x B) = W_blk(Out x In) * X_cm(In x B).
  //   W host row-major (OutDim x InDim, stride InCap) == col-major (In x Out,
  //   ld=InCap); we want it as (Out x In) so we pass it transposed (op_T).
  //   X host row-major (B x In) == col-major (In x B, ld=InDim); already (In x
  //   B), pass non-transposed (op_N).
  // => cublasSgemm(op_T for W, op_N for X, m=Out, n=B, k=In,
  //                A=dW ld=InCap, B=dX ld=InDim, C=dY ld=Out).
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
  // Col-major: GX_cm(In x B) = W^T_blk(In x Out) * GY_cm(Out x B).
  //   W host (Out x In, ld=InCap) as col-major (In x Out) is op_N => gives
  //   (In x Out). GY host (B x Out) col-major (Out x B, ld=OutDim) op_N.
  // => op_N for W, op_N for GY, m=In, n=B, k=Out, A=dW ld=InCap,
  //    B=dGY ld=OutDim, C=dGX ld=In.
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

  // GradW[OutDim x InDim] = Scale * GradY^T @ X  (row-major result, stride
  // InCap). Col-major we want GW_cm reinterpreted as host row-major (Out x In,
  // ld=InCap), i.e. col-major (In x Out, ld=InCap): GW_cm(In x Out) =
  // X^T(In x B) ... -> = sum_b X[b,:]^T GY[b,:].  In col-major:
  //   GW(In x Out) = Xcm(In x B) * GYcm(Out x B)^T
  //   Xcm = host X (B x In) col-major (In x B, ld=InDim) op_N.
  //   GYcm^T: host GY (B x Out) col-major (Out x B, ld=OutDim), transposed
  //   gives (B x Out) -> we need (B x Out) as middle... use op_T on GY.
  // => op_N for X, op_T for GY, m=In, n=Out, k=B, A=dX ld=InDim,
  //    B=dGY ld=OutDim, C=dGW ld=InCap.
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
    // Bias gradient: column-sum of GY, scaled. cuBLAS sgemv: dGB = Scale *
    // GY^T(col-major B x Out, ld=OutDim) * ones(B). GY host (B x Out) col-major
    // is (Out x B, ld=OutDim); to sum over batch we do (Out x B) * ones(B)
    // with op_N.
    if (dOnes_)
      CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_N, static_cast<int>(OutDim),
                               static_cast<int>(Batch), &Scale, dGY,
                               static_cast<int>(OutDim), dOnes_, 1, &Beta, dGB,
                               1));
  }

  // SGD: W -= Lr*GW; B -= Lr*GB on the full physical buffers. Dead edges are
  // re-zeroed host-side after Download (the cpp impl's mask enforcement).
  void SGD(cublasHandle_t Bl, float Lr) {
    (void)Bl;
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

  // Shared ones-vector for the bias-gradient reduction (set by the model).
  const float *dOnes_ = nullptr;
};

float XavierLimit(size_t In, size_t Out) {
  return std::sqrt(6.0f / static_cast<float>(In + Out));
}

// --- Sparse recurrent network: structural logic on host, math on GPU ------
struct SparseRecurrent {
  Linear LIn, LRec, LOut;
  size_t Hidden = 0, MaxHidden = 0, InDim = 0, OutDim = 0;
  size_t Rewires = 0, Grows = 0, Shrinks = 0, Bursts = 0;

  cublasHandle_t Bl = nullptr;
  // Device scratch (sized to Batch x MaxHidden capacity).
  float *dX = nullptr;     // Batch x InDim
  float *dH1 = nullptr;    // Batch x MaxHidden
  float *dRec = nullptr;   // Batch x MaxHidden
  float *dH2 = nullptr;    // Batch x MaxHidden
  float *dOut = nullptr;   // Batch x OutDim
  float *dY = nullptr;     // Batch x OutDim (targets)
  float *dGh2 = nullptr;   // Batch x MaxHidden
  float *dGhRec = nullptr; // Batch x MaxHidden
  float *dGh1 = nullptr;   // Batch x MaxHidden
  float *dOnes = nullptr;  // Batch (ones)
  size_t BatchCap = 0;
  std::vector<float> HostOut; // Batch x OutDim, for loss/eval reads

  void Init(size_t InDim_, size_t OutDim_, size_t InitHidden, size_t MaxHidden_,
            size_t BatchCap_, float RecurDensity, std::mt19937 &Rng,
            cublasHandle_t Handle) {
    InDim = InDim_;
    OutDim = OutDim_;
    Hidden = InitHidden;
    MaxHidden = MaxHidden_;
    BatchCap = BatchCap_;
    Bl = Handle;
    LIn.Init(InDim, InitHidden, InDim, MaxHidden);
    LRec.Init(InitHidden, InitHidden, MaxHidden, MaxHidden);
    LOut.Init(InitHidden, OutDim, MaxHidden, OutDim);

    float Lim0 = XavierLimit(InDim, InitHidden);
    float Lim1 = XavierLimit(InitHidden, InitHidden);
    float Lim2 = XavierLimit(InitHidden, OutDim);
    std::uniform_real_distribution<float> U0(-Lim0, Lim0);
    std::uniform_real_distribution<float> U1(-Lim1, Lim1);
    std::uniform_real_distribution<float> U2(-Lim2, Lim2);
    for (size_t I = 0; I < InitHidden; ++I)
      for (size_t J = 0; J < InDim; ++J)
        LIn.Weight[I * LIn.InCap + J] = U0(Rng);
    for (size_t I = 0; I < InitHidden; ++I)
      for (size_t J = 0; J < InitHidden; ++J)
        LRec.Weight[I * LRec.InCap + J] = U1(Rng) * 0.3f;
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InitHidden; ++J)
        LOut.Weight[I * LOut.InCap + J] = U2(Rng);
    LRec.EnableMask();
    std::bernoulli_distribution Bern(RecurDensity);
    for (size_t I = 0; I < InitHidden; ++I)
      for (size_t J = 0; J < InitHidden; ++J) {
        size_t Idx = I * LRec.InCap + J;
        if (I != J && Bern(Rng))
          LRec.Mask[Idx] = 1;
        else {
          LRec.Mask[Idx] = 0;
          LRec.Weight[Idx] = 0.0f;
        }
      }

    // Device scratch.
    size_t MH = MaxHidden;
    CUDA_CHECK(cudaMalloc(&dX, BatchCap * InDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dH1, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRec, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dH2, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, BatchCap * OutDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, BatchCap * OutDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGh2, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGhRec, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGh1, BatchCap * MH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOnes, BatchCap * sizeof(float)));
    std::vector<float> Ones(BatchCap, 1.0f);
    CUDA_CHECK(cudaMemcpy(dOnes, Ones.data(), BatchCap * sizeof(float),
                          cudaMemcpyHostToDevice));
    LIn.dOnes_ = dOnes;
    LRec.dOnes_ = dOnes;
    LOut.dOnes_ = dOnes;
    HostOut.assign(BatchCap * OutDim, 0.0f);
    UploadAll();
  }

  void UploadAll() {
    LIn.Upload();
    LRec.Upload();
    LOut.Upload();
  }

  size_t UnitCount() const { return Hidden + OutDim; }
  size_t EdgeCount() const {
    return LIn.OutDim * LIn.InDim + LRec.AliveCount() +
           LOut.OutDim * LOut.InDim;
  }

  bench::EdgeSet EdgeSetAlive() const {
    bench::EdgeSet S;
    for (size_t I = 0; I < LIn.OutDim; ++I)
      for (size_t J = 0; J < LIn.InDim; ++J)
        S.insert(bench::PackEdge(0u, static_cast<uint32_t>(I),
                                 static_cast<uint32_t>(J)));
    for (size_t I = 0; I < LRec.OutDim; ++I)
      for (size_t J = 0; J < LRec.InDim; ++J)
        if (LRec.Mask[I * LRec.InCap + J])
          S.insert(bench::PackEdge(1u, static_cast<uint32_t>(I),
                                   static_cast<uint32_t>(J)));
    for (size_t I = 0; I < LOut.OutDim; ++I)
      for (size_t J = 0; J < LOut.InDim; ++J)
        S.insert(bench::PackEdge(2u, static_cast<uint32_t>(I),
                                 static_cast<uint32_t>(J)));
    return S;
  }

  // Forward, leaving dOut populated on device and HostOut mirrored on host.
  // dXSrc points to a device buffer holding the Batch x InDim inputs (row
  // major), already uploaded by the caller.
  void ForwardDev(const float *dXSrc, size_t Batch) {
    // h1 = tanh(L_in(x))
    LIn.Forward(Bl, dXSrc, dH1, Batch);
    TanhInPlace<<<Grid(Batch * LIn.OutDim, kBlock), kBlock>>>(
        dH1, Batch * LIn.OutDim);
    // rec = L_rec(h1)  (sparse via masked host weights)
    LRec.Forward(Bl, dH1, dRec, Batch);
    // h2 = tanh(h1 + rec)
    TanhSum<<<Grid(Batch * LRec.OutDim, kBlock), kBlock>>>(
        dH1, dRec, dH2, Batch * LRec.OutDim);
    // out = L_out(h2)
    LOut.Forward(Bl, dH2, dOut, Batch);
  }

  void DownloadOut(size_t Batch) {
    CUDA_CHECK(cudaMemcpy(HostOut.data(), dOut, Batch * OutDim * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Backward given the top gradient already on device (dGrad, Batch x OutDim).
  void BackwardDev(const float *dXSrc, const float *dGrad, size_t Batch) {
    // d_out: linear. weight grad := dGrad^T @ h2 ; d/dh2 = L_out.W^T @ dGrad.
    LOut.BackwardWeights(Bl, dH2, dGrad, Batch, 1.0f);
    LOut.BackwardInput(Bl, dGrad, dGh2, Batch);
    // through h2 = tanh(h1 + rec): d/dArg = (1 - h2^2) * Gh2
    TanhGate<<<Grid(Batch * LRec.OutDim, kBlock), kBlock>>>(
        dGh2, dH2, Batch * LRec.OutDim);
    // L_rec weight grad: Gh2^T @ h1 ; d/dh1_via_rec: Gh2 @ L_rec.W
    LRec.BackwardWeights(Bl, dH1, dGh2, Batch, 1.0f);
    LRec.BackwardInput(Bl, dGh2, dGhRec, Batch);
    // d/dh1 = Gh2 + GhFromRec
    AddVec<<<Grid(Batch * LIn.OutDim, kBlock), kBlock>>>(
        dGh2, dGhRec, dGh1, Batch * LIn.OutDim);
    // through h1 = tanh(L_in(x)): * (1 - h1^2)
    TanhGate<<<Grid(Batch * LIn.OutDim, kBlock), kBlock>>>(
        dGh1, dH1, Batch * LIn.OutDim);
    LIn.BackwardWeights(Bl, dXSrc, dGh1, Batch, 1.0f);
  }

  void UpdateDev(float Lr) {
    LIn.SGD(Bl, Lr);
    LRec.SGD(Bl, Lr);
    LOut.SGD(Bl, Lr);
  }

  // Pull SGD-updated weights to host and re-enforce the recurrent mask, then
  // re-upload (so the device sees the masked weights for the next forward).
  void DownloadAndEnforce() {
    LIn.Download();
    LRec.Download();
    LOut.Download();
    LRec.EnforceMask();
    LRec.Upload(); // re-sync the masked recurrent weights
  }

  void DownloadGradsAll() {
    LIn.DownloadGrads();
    LRec.DownloadGrads();
    LOut.DownloadGrads();
  }

  // --- structural ops: identical to the cpp impl (host weight buffers) -----

  void Grow(size_t N, float InitScale, std::mt19937 &Rng) {
    if (N == 0)
      return;
    size_t Want = std::min(MaxHidden, Hidden + N);
    if (Want == Hidden)
      return;
    std::normal_distribution<float> Norm(0.0f, InitScale);
    std::bernoulli_distribution Bern(0.05);
    for (size_t I = Hidden; I < Want; ++I) {
      for (size_t J = 0; J < InDim; ++J)
        LIn.Weight[I * LIn.InCap + J] = Norm(Rng);
      LIn.Bias[I] = 0.0f;
    }
    for (size_t I = Hidden; I < Want; ++I)
      for (size_t J = 0; J < Want; ++J) {
        size_t Idx = I * LRec.InCap + J;
        LRec.Weight[Idx] = Norm(Rng);
        LRec.Mask[Idx] = (I != J && Bern(Rng)) ? 1 : 0;
        if (!LRec.Mask[Idx])
          LRec.Weight[Idx] = 0.0f;
      }
    for (size_t I = 0; I < Hidden; ++I)
      for (size_t J = Hidden; J < Want; ++J) {
        size_t Idx = I * LRec.InCap + J;
        LRec.Weight[Idx] = Norm(Rng);
        LRec.Mask[Idx] = (I != J && Bern(Rng)) ? 1 : 0;
        if (!LRec.Mask[Idx])
          LRec.Weight[Idx] = 0.0f;
      }
    for (size_t I = Hidden; I < Want; ++I)
      LRec.Bias[I] = 0.0f;
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = Hidden; J < Want; ++J)
        LOut.Weight[I * LOut.InCap + J] = Norm(Rng);
    Hidden = Want;
    LIn.OutDim = Hidden;
    LRec.InDim = Hidden;
    LRec.OutDim = Hidden;
    LOut.InDim = Hidden;
    ++Grows;
  }

  void Shrink(size_t N, size_t MinHidden) {
    if (N == 0)
      return;
    size_t Target = (Hidden > N) ? Hidden - N : MinHidden;
    if (Target < MinHidden)
      Target = MinHidden;
    if (Target >= Hidden)
      return;
    for (size_t I = Target; I < Hidden; ++I) {
      for (size_t J = 0; J < InDim; ++J)
        LIn.Weight[I * LIn.InCap + J] = 0.0f;
      LIn.Bias[I] = 0.0f;
    }
    for (size_t I = Target; I < Hidden; ++I)
      for (size_t J = 0; J < Hidden; ++J) {
        size_t Idx = I * LRec.InCap + J;
        LRec.Mask[Idx] = 0;
        LRec.Weight[Idx] = 0.0f;
      }
    for (size_t I = 0; I < Hidden; ++I)
      for (size_t J = Target; J < Hidden; ++J) {
        size_t Idx = I * LRec.InCap + J;
        LRec.Mask[Idx] = 0;
        LRec.Weight[Idx] = 0.0f;
      }
    for (size_t I = Target; I < Hidden; ++I)
      LRec.Bias[I] = 0.0f;
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = Target; J < Hidden; ++J)
        LOut.Weight[I * LOut.InCap + J] = 0.0f;
    Hidden = Target;
    LIn.OutDim = Hidden;
    LRec.InDim = Hidden;
    LRec.OutDim = Hidden;
    LOut.InDim = Hidden;
    ++Shrinks;
  }

  size_t Rewire(float Frac, std::mt19937 &Rng) {
    std::vector<std::pair<size_t, size_t>> Alive;
    for (size_t I = 0; I < LRec.OutDim; ++I)
      for (size_t J = 0; J < LRec.InDim; ++J)
        if (LRec.Mask[I * LRec.InCap + J])
          Alive.emplace_back(I, J);
    if (Alive.empty())
      return 0;
    size_t NRewire = std::max<size_t>(
        1, static_cast<size_t>(Frac * static_cast<float>(Alive.size())));
    std::shuffle(Alive.begin(), Alive.end(), Rng);
    if (NRewire > Alive.size())
      NRewire = Alive.size();
    std::uniform_int_distribution<size_t> Pick(0, Hidden - 1);
    size_t Commit = 0;
    for (size_t I = 0; I < NRewire; ++I) {
      auto [Row, Col] = Alive[I];
      float W = LRec.Weight[Row * LRec.InCap + Col];
      LRec.Mask[Row * LRec.InCap + Col] = 0;
      LRec.Weight[Row * LRec.InCap + Col] = 0.0f;
      size_t S = Pick(Rng), D = Pick(Rng);
      if (S == D)
        continue;
      size_t Idx = S * LRec.InCap + D;
      if (LRec.Mask[Idx])
        continue;
      LRec.Mask[Idx] = 1;
      LRec.Weight[Idx] = W;
      ++Commit;
    }
    ++Rewires;
    return Commit;
  }

  double ComputeGradNorm() const {
    double Sum = 0.0;
    auto AccLayer = [&](const Linear &L) {
      for (size_t I = 0; I < L.OutDim; ++I)
        for (size_t J = 0; J < L.InDim; ++J) {
          float G = L.GradW[I * L.InCap + J];
          Sum += static_cast<double>(G) * G;
        }
      for (size_t I = 0; I < L.OutDim; ++I)
        Sum += static_cast<double>(L.GradB[I]) * L.GradB[I];
    };
    AccLayer(LIn);
    AccLayer(LRec);
    AccLayer(LOut);
    return std::sqrt(Sum);
  }
};

// Evaluate MSE on a slice [Start, Start+Count) of the windowed data. Uploads
// each minibatch's inputs to the device, runs the forward, and reduces on host.
double EvalMseSlice(SparseRecurrent &M, const Windowed &W, size_t Start,
                    size_t Count, size_t Batch) {
  if (Count == 0)
    return 0.0;
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < Count; I += Batch) {
    size_t B = std::min(Batch, Count - I);
    CUDA_CHECK(cudaMemcpy(M.dX, W.X.data() + (Start + I) * W.InLen,
                          B * W.InLen * sizeof(float), cudaMemcpyHostToDevice));
    M.ForwardDev(M.dX, B);
    M.DownloadOut(B);
    double S = 0.0;
    for (size_t K = 0; K < B; ++K) {
      double E = static_cast<double>(M.HostOut[K]) -
                 static_cast<double>(W.Y[Start + I + K]);
      S += E * E;
    }
    Sum += S; // B*D with D==1, so the mean over the minibatch * B == sum
    Cnt += B;
  }
  return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.SeriesLen = static_cast<size_t>(Args.GetInt("series-len", H.SeriesLen));
  H.Tau = static_cast<size_t>(Args.GetInt("tau", H.Tau));
  H.InLen = static_cast<size_t>(Args.GetInt("in-len", H.InLen));
  H.Horizon = static_cast<size_t>(Args.GetInt("horizon", H.Horizon));
  H.InitHidden = static_cast<size_t>(Args.GetInt("init-hidden", H.InitHidden));
  H.MinHidden = static_cast<size_t>(Args.GetInt("min-hidden", H.MinHidden));
  H.MaxHidden = static_cast<size_t>(Args.GetInt("max-hidden", H.MaxHidden));
  H.RecurDensity = Args.GetFloat("recur-density", H.RecurDensity);
  H.Batch = static_cast<size_t>(Args.GetInt("batch", H.Batch));
  H.MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", H.MaxSteps));
  H.ValEvery = static_cast<size_t>(Args.GetInt("val-every", H.ValEvery));
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.ParetoAlpha = Args.GetFloat("pareto-alpha", H.ParetoAlpha);
  H.MaxDeltaPerStep =
      static_cast<size_t>(Args.GetInt("max-delta-per-step", H.MaxDeltaPerStep));
  H.RewireEvery =
      static_cast<size_t>(Args.GetInt("rewire-every", H.RewireEvery));
  H.RewireFrac = Args.GetFloat("rewire-frac", H.RewireFrac);
  H.MomentumThreshold =
      Args.GetFloat("momentum-threshold", H.MomentumThreshold);
  H.MomentumBurst =
      static_cast<size_t>(Args.GetInt("momentum-burst", H.MomentumBurst));
  if (Args.Quick) {
    H.MaxSteps = std::max<size_t>(20, H.MaxSteps / 5);
    H.ValEvery = std::max<size_t>(1, H.ValEvery / 2);
  }

  bench::MemoryProbe MP;
  MP.Start();

  auto Series =
      MackeyGlass(H.SeriesLen, H.Tau, static_cast<uint32_t>(Args.Seed));
  double Mu = 0.0;
  for (float V : Series)
    Mu += V;
  Mu /= Series.size();
  double Var = 0.0;
  for (float V : Series)
    Var += (V - Mu) * (V - Mu);
  double Sd = std::sqrt(Var / Series.size()) + 1e-6;
  for (auto &V : Series)
    V = static_cast<float>((V - Mu) / Sd);

  auto W = WindowSeries(Series, H.InLen, H.Horizon);
  MP.EndDataset();
  size_t NTr = static_cast<size_t>(0.7f * W.N);
  size_t NVa = static_cast<size_t>(0.15f * W.N);
  size_t NTe = W.N - NTr - NVa;

  std::cout << "[info] cuda MG  N=" << W.N << " train=" << NTr << " val=" << NVa
            << " test=" << NTe << " steps=" << H.MaxSteps
            << " init_h=" << H.InitHidden
            << " recur_density=" << H.RecurDensity << "\n";

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 19u);
  SparseRecurrent M;
  M.Init(H.InLen, 1, H.InitHidden, H.MaxHidden, H.Batch, H.RecurDensity,
         InitRng, Bl);
  MP.EndWeights();

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "continuous_large_mg");
  bench::StructuralLog Log(HistPath);

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::uniform_real_distribution<float> Prob(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> PickTrain(0, NTr - 1);
  auto Pareto = [&]() {
    float U = std::max(1e-9f, Prob(Rng));
    return std::pow(1.0f - U, -1.0f / H.ParetoAlpha) - 1.0f;
  };

  std::vector<float> XBatch(H.Batch * H.InLen);
  std::vector<float> YBatch(H.Batch);
  std::vector<int> DeltaUnits;
  DeltaUnits.reserve(H.MaxSteps);

  // Device buffer for the per-step gradient (Batch x OutDim, OutDim==1).
  float *dGrad = nullptr;
  CUDA_CHECK(cudaMalloc(&dGrad, H.Batch * sizeof(float)));

  float GrowthMomentum = 0.0f;
  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    for (size_t B = 0; B < H.Batch; ++B) {
      size_t Idx = PickTrain(Rng);
      std::memcpy(XBatch.data() + B * H.InLen, W.X.data() + Idx * H.InLen,
                  H.InLen * sizeof(float));
      YBatch[B] = W.Y[Idx];
    }
    // Upload inputs + targets for this step.
    CUDA_CHECK(cudaMemcpy(M.dX, XBatch.data(), H.Batch * H.InLen * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M.dY, YBatch.data(), H.Batch * sizeof(float),
                          cudaMemcpyHostToDevice));

    Timer.Tick();
    M.ForwardDev(M.dX, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkForward();

    // MSE mean gradient (D == 1): G = 2(pred - y)/(B).
    MseGrad<<<Grid(H.Batch, kBlock), kBlock>>>(M.dOut, M.dY, dGrad, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkLoss();

    M.BackwardDev(M.dX, dGrad, H.Batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Timer.MarkBackward();

    M.UpdateDev(H.Lr);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Pull grads for the grad-norm; pull + mask-enforce + re-upload weights.
    M.DownloadGradsAll();
    M.DownloadAndEnforce();
    // Re-upload LIn/LOut weights too (SGD modified them on-device; the host now
    // matches the device, but the next forward reads device buffers — which are
    // already current. Only LRec needed the mask re-upload, done above.)
    Timer.MarkUpdate();
    Timer.StepDone();
    float GNorm = static_cast<float>(M.ComputeGradNorm());

    size_t UnitsBefore = M.UnitCount();
    int Sign = (Prob(Rng) > 0.5f) ? +1 : -1;
    size_t Mag = std::max<size_t>(1, static_cast<size_t>(std::ceil(Pareto())));
    if (Mag > H.MaxDeltaPerStep)
      Mag = H.MaxDeltaPerStep;
    bool Structural = false;
    if (Sign > 0 && M.Hidden + Mag <= H.MaxHidden) {
      M.Grow(Mag, 0.05f, Rng);
      Structural = true;
    } else if (Sign < 0 && M.Hidden > H.MinHidden + Mag - 1) {
      M.Shrink(Mag, H.MinHidden);
      Structural = true;
    }
    if (Step % H.RewireEvery == 0 && M.Hidden >= 8) {
      M.Rewire(H.RewireFrac, Rng);
      Structural = true;
    }

    GrowthMomentum += GNorm;
    if (GrowthMomentum > H.MomentumThreshold) {
      size_t Burst = std::max<size_t>(2, H.MomentumBurst);
      if (M.Hidden + Burst <= H.MaxHidden) {
        M.Grow(Burst, 0.05f, Rng);
        ++M.Bursts;
        Structural = true;
      }
      GrowthMomentum = 0.0f;
    }

    // If the host weights/masks changed structurally, re-sync the device.
    if (Structural)
      M.UploadAll();

    DeltaUnits.push_back(static_cast<int>(M.UnitCount()) -
                         static_cast<int>(UnitsBefore));

    if (Step % H.ValEvery == 0 || Step == H.MaxSteps) {
      double VL = EvalMseSlice(M, W, NTr, NVa, H.Batch);
      double TestMSE = EvalMseSlice(M, W, NTr + NVa, NTe, H.Batch);
      auto Edges = M.EdgeSetAlive();
      Log.Log(Step, M.UnitCount(), M.EdgeCount(), &Edges, &VL,
              {{"hidden", static_cast<double>(M.Hidden)},
               {"test_mse", TestMSE},
               {"grows", static_cast<double>(M.Grows)},
               {"shrinks", static_cast<double>(M.Shrinks)},
               {"rewires", static_cast<double>(M.Rewires)},
               {"bursts", static_cast<double>(M.Bursts)},
               {"delta_units", static_cast<double>(DeltaUnits.back())}});
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();
  Log.Flush();

  double TestMse = EvalMseSlice(M, W, NTr + NVa, NTe, H.Batch);

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
  int DuMax = 0;
  double DuNzMean = 0.0;
  size_t DuNzCnt = 0;
  for (auto V : DeltaUnits) {
    int A = std::abs(V);
    if (A > DuMax)
      DuMax = A;
    if (A > 0) {
      DuNzMean += A;
      ++DuNzCnt;
    }
  }
  if (DuNzCnt)
    DuNzMean /= static_cast<double>(DuNzCnt);

  float JMin = 1.0f;
  double JMean = 0.0;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JMean += R.Jaccard;
  }
  if (!Log.Records().empty())
    JMean /= static_cast<double>(Log.Records().size());

  bench::SummaryWriter S;
  S.Set("workload", std::string{"05_continuous_large_mg"});
  S.Set("dataset", std::string{"mackey-glass"});
  S.Set("tau", static_cast<int>(H.Tau));
  S.Set("in_len", static_cast<int>(H.InLen));
  S.Set("horizon", static_cast<int>(H.Horizon));
  S.Set("init_hidden", static_cast<int>(H.InitHidden));
  S.Set("recur_density", static_cast<double>(H.RecurDensity));
  S.Set("pareto_alpha", static_cast<double>(H.ParetoAlpha));
  S.Set("rewire_every", static_cast<int>(H.RewireEvery));
  S.Set("rewire_frac", static_cast<double>(H.RewireFrac));
  S.Set("max_steps", static_cast<int>(H.MaxSteps));
  S.Set("batch", static_cast<int>(H.Batch));
  S.Set("wall_seconds", Wall);
  S.Set("grows", static_cast<int>(M.Grows));
  S.Set("shrinks", static_cast<int>(M.Shrinks));
  S.Set("rewires", static_cast<int>(M.Rewires));
  S.Set("bursts", static_cast<int>(M.Bursts));
  S.Set("hidden_final", static_cast<int>(M.Hidden));
  S.Set("edges_final", static_cast<int>(M.EdgeCount()));
  S.Set("val_loss_initial",
        Log.Records().empty() ? 0.0 : Log.Records().front().ValLoss);
  S.Set("val_loss_final",
        Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss);
  S.Set("test_mse", TestMse);
  S.Set("metric_kind", std::string{"mse"});
  S.Set("delta_units_p50_abs", Pctile(DeltaUnits, 50.0));
  S.Set("delta_units_p95_abs", Pctile(DeltaUnits, 95.0));
  S.Set("delta_units_max_abs", DuMax);
  S.Set("delta_units_mean_nonzero", DuNzMean);
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_mean", JMean);
  S.Set("n_units", static_cast<int>(M.UnitCount()));
  S.Set("n_params", static_cast<int>(M.EdgeCount()));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  grows=" << M.Grows
            << "  shrinks=" << M.Shrinks << "  rewires=" << M.Rewires
            << "  hidden=" << M.Hidden << "  test_mse=" << TestMse
            << "  jaccard_mean=" << JMean
            << "  |du|_p95=" << Pctile(DeltaUnits, 95.0)
            << "  |du|_max=" << DuMax << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  cudaFree(dGrad);
  cublasDestroy(Bl);
  (void)LogPath;
  return 0;
}
