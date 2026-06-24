// Workload 1 / 5 — STATIC regime, pure-CUDA / cuBLAS translation.
//
// A from-scratch GPU port of 01_static_etth1.cpp (the OpenBLAS impl): a fixed
// feedforward MLP trained against the ETTh1 long-horizon forecasting target
// with mean-squared error. Topology never changes. Architecture matches the
// PyTorch / cpp reference: GeLU-hidden / linear-output, depth=3 (= 2 GeLU
// hidden + 1 linear readout), Linear layers with bias, mean-reduction MSE.
//
// Unlike the cpp impl, every per-step/per-epoch tensor (weights, biases,
// activations, pre-activations, gradients) lives on the device for the whole
// run. The MLP matmuls (forward, backward-input, backward-weights) go through
// cublasSgemm; activations / bias / loss-gradient are tiny custom kernels.
// Only the CSV/synthetic data loading stays on the host (copied verbatim from
// the cpp impl).
//
// Build (standalone):
//   nvcc -O3 -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -arch=sm_89 -I common -I/usr/local/cuda/include \
//     01_static_etth1/cuda/01_static_etth1.cu \
//     -L/usr/local/cuda/lib64 -lcublas -o run_benchmark

#include "cpp/common.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

// ===========================================================================
// Host-side data loading — copied verbatim from 01_static_etth1.cpp.
// ===========================================================================

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

// ===========================================================================
// Device kernels (GeLU, bias add/grad, MSE gradient).
// ===========================================================================

constexpr float kInv2Sqrt = 0.70710678118654752440f;   // 1/sqrt(2)
constexpr float kInvSqrt2Pi = 0.39894228040143267794f; // 1/sqrt(2 pi)

// Y = exact-GeLU(Z). Matches raw::GeLUExact (PyTorch F.gelu default = erf form).
__global__ void GeluForward(const float *Z, float *Y, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  float z = Z[I];
  Y[I] = 0.5f * z * (1.0f + erff(z * kInv2Sqrt));
}

// GradY *= GeLU'(Z). Matches raw::GeLUBackwardFromPreact.
__global__ void GeluBackward(const float *Z, float *GradY, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  float z = Z[I];
  float cdf = 0.5f * (1.0f + erff(z * kInv2Sqrt));
  float pdf = kInvSqrt2Pi * expf(-0.5f * z * z);
  GradY[I] *= (cdf + z * pdf);
}

// Add per-output bias to a row-major (Batch x OutDim) activation buffer.
__global__ void AddBias(float *Y, const float *Bias, size_t Batch,
                        size_t OutDim) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  size_t N = Batch * OutDim;
  if (I >= N)
    return;
  Y[I] += Bias[I % OutDim];
}

// MSE gradient w.r.t. predictions: Grad = 2 * (Pred - Target) / N.
__global__ void MseGrad(const float *Pred, const float *Target, float *Grad,
                        size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  Grad[I] = 2.0f * (Pred[I] - Target[I]) / static_cast<float>(N);
}

// SGD step: W -= Lr * GradW (length N).
__global__ void SgdStep(float *W, const float *GradW, float Lr, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  W[I] -= Lr * GradW[I];
}

// Column-sum of a row-major (Batch x OutDim) buffer into Bias-grad (length
// OutDim), scaled. One thread per output unit.
__global__ void ColSum(const float *GradY, float *GradB, size_t Batch,
                       size_t OutDim, float Scale) {
  size_t J = blockIdx.x * blockDim.x + threadIdx.x;
  if (J >= OutDim)
    return;
  float s = 0.0f;
  for (size_t B = 0; B < Batch; ++B)
    s += GradY[B * OutDim + J];
  GradB[J] = s * Scale;
}

inline unsigned Grid(size_t N, unsigned Block) {
  return static_cast<unsigned>((N + Block - 1) / Block);
}

// ===========================================================================
// Device-resident MLP. All matmuls via cuBLAS (column-major), all element-wise
// ops via the kernels above.
//
// cuBLAS is column-major. A row-major (R x C) buffer is bit-identical to a
// column-major (C x R) buffer. So a row-major (Batch x Dim) activation reads
// as column-major (Dim x Batch).
//
//   Forward         Y[B x O] = X[B x I] @ W[O x I]^T
//     col-major:    Y_cm(O x B) = (W as I x O col-major)^T * (X as I x B)
//                   => sgemm(T, N, m=O, n=B, k=I, A=W lda=I, B=X ldb=I, ldc=O)
//   BackwardInput   GradX[B x I] = GradY[B x O] @ W[O x I]
//     col-major:    GradX_cm(I x B) = (W as I x O col-major) * (GradY as O x B)
//                   => sgemm(N, N, m=I, n=B, k=O, A=W lda=I, B=GradY ldb=O, ldc=I)
//   BackwardWeights GradW[O x I] = GradY[B x O]^T @ X[B x I] * Scale
//     col-major:    GradW as I x O col-major = (X as I x B) * (GradY as O x B)^T
//                   => sgemm(N, T, m=I, n=O, k=B, A=X lda=I, B=GradY ldb=O, ldc=I)
// ===========================================================================

struct DevLinear {
  size_t InDim = 0;
  size_t OutDim = 0;
  float *W = nullptr;     // row-major OutDim x InDim (col-major InDim x OutDim)
  float *Bias = nullptr;  // OutDim
  float *GradW = nullptr; // same shape as W
  float *GradB = nullptr; // OutDim
};

struct StaticMLP {
  std::vector<DevLinear> Layers;
  // Per-layer device buffers, capacity Batch x OutDim each.
  std::vector<float *> Preacts; // pre-activation (z)
  std::vector<float *> Acts;    // post-activation (a); last layer == preact
  std::vector<float *> Grads;   // dL/dz per layer
  std::vector<float *> GradIn;  // scratch dL/d(input) per layer (Batch x InDim)
  cublasHandle_t Bl = nullptr;
  size_t Cap = 0; // batch capacity buffers are sized for

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

  void Init(size_t InDim, size_t OutDim, size_t Hidden, size_t Depth,
            size_t BatchCap, std::mt19937 &Rng, cublasHandle_t Handle) {
    Bl = Handle;
    Cap = BatchCap;
    Layers.resize(Depth);
    auto Dims = [&](size_t LI) -> std::pair<size_t, size_t> {
      if (Depth == 1)
        return {InDim, OutDim};
      if (LI == 0)
        return {InDim, Hidden};
      if (LI + 1 == Depth)
        return {Hidden, OutDim};
      return {Hidden, Hidden};
    };
    for (size_t LI = 0; LI < Depth; ++LI) {
      auto [In, Outp] = Dims(LI);
      DevLinear &L = Layers[LI];
      L.InDim = In;
      L.OutDim = Outp;
      // Xavier-uniform on host, matching raw::Linear::XavierInit; bias = 0.
      // Host buffer is row-major OutDim x InDim (identical layout to device).
      std::vector<float> HW(In * Outp, 0.0f);
      float Lim = std::sqrt(6.0f / static_cast<float>(In + Outp));
      std::uniform_real_distribution<float> U(-Lim, Lim);
      for (size_t I = 0; I < Outp; ++I)
        for (size_t J = 0; J < In; ++J)
          HW[I * In + J] = U(Rng);
      CUDA_CHECK(cudaMalloc(&L.W, In * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&L.GradW, In * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&L.Bias, Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&L.GradB, Outp * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(L.W, HW.data(), In * Outp * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(L.Bias, 0, Outp * sizeof(float)));
    }
    Preacts.assign(Depth, nullptr);
    Acts.assign(Depth, nullptr);
    Grads.assign(Depth, nullptr);
    GradIn.assign(Depth, nullptr);
    for (size_t LI = 0; LI < Depth; ++LI) {
      size_t Outp = Layers[LI].OutDim;
      size_t In = Layers[LI].InDim;
      CUDA_CHECK(cudaMalloc(&Preacts[LI], Cap * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&Acts[LI], Cap * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&Grads[LI], Cap * Outp * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&GradIn[LI], Cap * In * sizeof(float)));
    }
  }

  // Forward over a device batch X (row-major Batch x InDim). Returns the
  // final-layer output buffer (row-major Batch x OutDim, on device).
  const float *Forward(const float *X, size_t Batch) {
    const float *Cur = X;
    const unsigned BlockSz = 256;
    const float One = 1.0f, Zero = 0.0f;
    for (size_t LI = 0; LI < Layers.size(); ++LI) {
      DevLinear &L = Layers[LI];
      int m = static_cast<int>(L.OutDim);
      int n = static_cast<int>(Batch);
      int k = static_cast<int>(L.InDim);
      // Y_cm(O x B) = W^T(O x I) * X(I x B); A=W(I x O col-major), opA=T.
      CUBLAS_CHECK(cublasSgemm(Bl, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &One, L.W,
                               k, Cur, k, &Zero, Preacts[LI], m));
      size_t Nout = Batch * L.OutDim;
      AddBias<<<Grid(Nout, BlockSz), BlockSz>>>(Preacts[LI], L.Bias, Batch,
                                                L.OutDim);
      if (LI + 1 < Layers.size()) {
        GeluForward<<<Grid(Nout, BlockSz), BlockSz>>>(Preacts[LI], Acts[LI],
                                                      Nout);
      } else {
        CUDA_CHECK(cudaMemcpyAsync(Acts[LI], Preacts[LI],
                                   Nout * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
      }
      Cur = Acts[LI];
    }
    return Cur;
  }

  // Backward: TopGrad is dL/d(output) (device, Batch x OutDim). Computes
  // per-layer weight/bias gradients but does NOT apply them.
  void Backward(const float *X, const float *TopGrad, size_t Batch) {
    size_t D = Layers.size();
    const unsigned BlockSz = 256;
    const float One = 1.0f, Zero = 0.0f;
    // dL/dz at readout = dL/da (linear output): copy TopGrad into Grads[D-1].
    CUDA_CHECK(cudaMemcpyAsync(Grads[D - 1], TopGrad,
                               Batch * Layers[D - 1].OutDim * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    for (ptrdiff_t LI = static_cast<ptrdiff_t>(D) - 1; LI >= 0; --LI) {
      DevLinear &L = Layers[LI];
      const float *InData = (LI == 0) ? X : Acts[LI - 1];
      int m = static_cast<int>(L.InDim);
      int n = static_cast<int>(L.OutDim);
      int k = static_cast<int>(Batch);
      // GradW as (I x O col-major) = X(I x B) * GradY^T(B x O). The mean
      // reduction over the batch is already folded into the loss gradient
      // (MseGrad divides by B*OutDim), so the GEMM scale is 1.0 — matching the
      // cpp impl which calls raw::Linear::BackwardWeights with Scale = 1.0f.
      CUBLAS_CHECK(cublasSgemm(Bl, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &One,
                               InData, m, Grads[LI], n, &Zero, L.GradW, m));
      ColSum<<<Grid(L.OutDim, BlockSz), BlockSz>>>(Grads[LI], L.GradB, Batch,
                                                   L.OutDim, 1.0f);
      if (LI > 0) {
        // GradX_cm(I x B) = W(I x O col-major) * GradY(O x B); opA=N.
        int mi = static_cast<int>(L.InDim);
        int ni = static_cast<int>(Batch);
        int ki = static_cast<int>(L.OutDim);
        CUBLAS_CHECK(cublasSgemm(Bl, CUBLAS_OP_N, CUBLAS_OP_N, mi, ni, ki, &One,
                                 L.W, mi, Grads[LI], ki, &Zero, GradIn[LI],
                                 mi));
        // dL/dz at layer below = GradIn * GeLU'(z_below).
        size_t Nin = Batch * L.InDim;
        GeluBackward<<<Grid(Nin, BlockSz), BlockSz>>>(Preacts[LI - 1],
                                                      GradIn[LI], Nin);
        CUDA_CHECK(cudaMemcpyAsync(Grads[LI - 1], GradIn[LI],
                                   Nin * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
      }
    }
  }

  void Update(float Lr) {
    const unsigned BlockSz = 256;
    for (auto &L : Layers) {
      size_t Nw = L.InDim * L.OutDim;
      SgdStep<<<Grid(Nw, BlockSz), BlockSz>>>(L.W, L.GradW, Lr, Nw);
      SgdStep<<<Grid(L.OutDim, BlockSz), BlockSz>>>(L.Bias, L.GradB, Lr,
                                                    L.OutDim);
    }
  }

  void Free() {
    for (auto &L : Layers) {
      cudaFree(L.W);
      cudaFree(L.GradW);
      cudaFree(L.Bias);
      cudaFree(L.GradB);
    }
    for (auto *P : Preacts)
      cudaFree(P);
    for (auto *P : Acts)
      cudaFree(P);
    for (auto *P : Grads)
      cudaFree(P);
    for (auto *P : GradIn)
      cudaFree(P);
  }
};

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

  // --max-steps caps the number of optimisation steps across all epochs (for
  // quick smoke runs); 0 = unlimited.
  size_t MaxSteps = static_cast<size_t>(Args.GetInt("max-steps", 0));

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

  std::cout << "[info] cuda static MLP  InLen=" << H.InLen
            << " OutLen=" << H.OutLen << " Channels=" << Raw.C
            << " InDim=" << InDim << " OutDim=" << OutDim
            << " Hidden=" << H.Hidden << " Depth=" << H.Depth
            << " Epochs=" << H.Epochs << " Batch=" << H.Batch
            << " Train=" << NTr << " Val=" << NVa << " Test=" << NTe << "\n";

  if (NTotal == 0 || InDim == 0) {
    std::cerr << "[err] empty dataset (not enough rows for windowing)\n";
    return 2;
  }

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 7u);
  StaticMLP M;
  M.Init(InDim, OutDim, H.Hidden, H.Depth, H.Batch, InitRng, Bl);

  // Upload the full windowed dataset to the device once.
  float *dX = nullptr, *dY = nullptr;
  CUDA_CHECK(cudaMalloc(&dX, D.X.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, D.Y.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX, D.X.data(), D.X.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dY, D.Y.data(), D.Y.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Per-batch staging buffers on the device (gathered rows + grad/pred scratch).
  float *dXB = nullptr, *dYB = nullptr, *dGrad = nullptr;
  CUDA_CHECK(cudaMalloc(&dXB, H.Batch * InDim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dYB, H.Batch * OutDim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dGrad, H.Batch * OutDim * sizeof(float)));

  const unsigned BlockSz = 256;

  // Evaluate MSE over a contiguous slice [Start, Start+Count) of the device
  // dataset. Copies prediction & target back to host for the (double) sum to
  // match the cpp impl's accumulation exactly.
  std::vector<float> HPred(H.Batch * OutDim), HTgt(H.Batch * OutDim);
  auto EvalSliceMse = [&](size_t Start, size_t Count) -> double {
    double Sum = 0.0;
    size_t Cnt = 0;
    for (size_t I = 0; I < Count; I += H.Batch) {
      size_t B = std::min(H.Batch, Count - I);
      const float *Out = M.Forward(dX + (Start + I) * InDim, B);
      CUDA_CHECK(cudaMemcpy(HPred.data(), Out, B * OutDim * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(HTgt.data(), dY + (Start + I) * OutDim,
                            B * OutDim * sizeof(float), cudaMemcpyDeviceToHost));
      double Local = 0.0;
      for (size_t K = 0; K < B * OutDim; ++K) {
        double E = static_cast<double>(HPred[K]) - HTgt[K];
        Local += E * E;
      }
      Sum += Local; // sum of squared errors; normalise below
      Cnt += B * OutDim;
    }
    return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
  };
  auto EvalSliceMae = [&](size_t Start, size_t Count) -> double {
    double Sum = 0.0;
    size_t Cnt = 0;
    for (size_t I = 0; I < Count; I += H.Batch) {
      size_t B = std::min(H.Batch, Count - I);
      const float *Out = M.Forward(dX + (Start + I) * InDim, B);
      CUDA_CHECK(cudaMemcpy(HPred.data(), Out, B * OutDim * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(HTgt.data(), dY + (Start + I) * OutDim,
                            B * OutDim * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t K = 0; K < B * OutDim; ++K)
        Sum += std::abs(static_cast<double>(HPred[K]) - HTgt[K]);
      Cnt += B * OutDim;
    }
    return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
  };

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "static_etth1");
  bench::StructuralLog Log(HistPath);

  auto Edges = M.EdgeSet();
  double InitVal = EvalSliceMse(NTr, NVa);
  Log.Log(0, M.UnitCount(), M.EdgeCount(), &Edges, &InitVal,
          {{"train_loss", 0.0}, {"epoch", 0.0}});

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::vector<size_t> Perm(NTr);
  for (size_t I = 0; I < NTr; ++I)
    Perm[I] = I;

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  size_t TotalSteps = 0;
  bool StepCapped = false;
  for (size_t Ep = 1; Ep <= H.Epochs && !StepCapped; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    double TrainLoss = 0.0;
    size_t TrainCnt = 0;
    for (size_t I = 0; I + H.Batch <= NTr; I += H.Batch) {
      // Gather the shuffled batch rows on the device with row-wise copies
      // (cheap; avoids re-uploading from host).
      for (size_t B = 0; B < H.Batch; ++B) {
        size_t Idx = Perm[I + B];
        CUDA_CHECK(cudaMemcpyAsync(dXB + B * InDim, dX + Idx * InDim,
                                   InDim * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(dYB + B * OutDim, dY + Idx * OutDim,
                                   OutDim * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
      }
      Timer.Tick();
      const float *Pred = M.Forward(dXB, H.Batch);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkForward();

      size_t Nout = H.Batch * OutDim;
      MseGrad<<<Grid(Nout, BlockSz), BlockSz>>>(Pred, dYB, dGrad, Nout);
      // Accumulate the train loss (sum of squared errors) on host for parity.
      CUDA_CHECK(cudaMemcpy(HPred.data(), Pred, Nout * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(HTgt.data(), dYB, Nout * sizeof(float),
                            cudaMemcpyDeviceToHost));
      double Loss = 0.0;
      for (size_t K = 0; K < Nout; ++K) {
        double E = static_cast<double>(HPred[K]) - HTgt[K];
        Loss += E * E;
      }
      Loss /= static_cast<double>(Nout);
      Timer.MarkLoss();

      M.Backward(dXB, dGrad, H.Batch);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkBackward();

      M.Update(H.Lr);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkUpdate();
      Timer.StepDone();

      TrainLoss += Loss * static_cast<double>(Nout);
      TrainCnt += Nout;
      ++TotalSteps;
      if (MaxSteps > 0 && TotalSteps >= MaxSteps) {
        StepCapped = true;
        break;
      }
    }
    TrainLoss /= std::max<size_t>(TrainCnt, 1);
    double VaMse = EvalSliceMse(NTr, NVa);
    auto Ed = M.EdgeSet();
    Log.Log(Ep, M.UnitCount(), M.EdgeCount(), &Ed, &VaMse,
            {{"train_loss", TrainLoss}, {"epoch", static_cast<double>(Ep)}});
    std::cout << "[ep " << Ep << "] train_mse=" << TrainLoss
              << "  val_mse=" << VaMse << "\n";
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
                    .count();

  double TeMse = EvalSliceMse(NTr + NVa, NTe);
  double TeMae = EvalSliceMae(NTr + NVa, NTe);
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
  S.Set("metric_kind", std::string{"mse"});
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

  M.Free();
  cudaFree(dX);
  cudaFree(dY);
  cudaFree(dXB);
  cudaFree(dYB);
  cudaFree(dGrad);
  cublasDestroy(Bl);
  (void)LogPath;
  return 0;
}
