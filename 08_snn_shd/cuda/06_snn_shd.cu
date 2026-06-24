// Workload 8 (06_snn_shd) — sparse spiking neural network on SHD, pure-CUDA /
// cuBLAS translation.
//
// GPU port of 08_snn_shd/cpp/06_snn_shd.cpp (the OpenBLAS impl): a surrogate-
// gradient spiking network on Spiking-Heidelberg-Digits. 700 input channels →
// K-sparse fan-in into a layer of N_hid LIF neurons → dense readout into N_out
// non-spiking integrators, trained with the same e-prop learning rule +
// random-feedback alignment for the hidden learning signal.
//
// The e-prop rule used by the cpp impl is a *forward-mode* eligibility trace
// (not full BPTT): per timestep it maintains rank-1 eligibility traces and, at
// the terminal step, multiplies them by the learning signal for the weight
// step. That structure ports directly to the GPU — there is no unrolled BPTT
// graph to reconstruct, so the backward is faithful, not approximated.
//
// Every weight / mask / eligibility-trace / per-unit-state tensor lives on the
// device for the whole run. The per-timestep matmuls go through cuBLAS:
//   forward  : 2 × cublasSgemv (W_in · u, W_out · spk)
//   backward : 1 × cublasSgemv (B_out^T · L_out)
//   elig     : cublasSscal (decay) + 2 × cublasSger (rank-1 outer products)
// LIF membrane update, surrogate gradient, softmax loss signal, and the masked
// clipped weight steps are tiny custom kernels. Only host-side .plxbin loading
// is copied verbatim from the cpp impl. float32 throughout.
//
// Build (standalone):
//   nvcc -O3 -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -arch=sm_89 -I common -I/usr/local/cuda/include \
//     08_snn_shd/cuda/06_snn_shd.cu \
//     -L/usr/local/cuda/lib64 -lcublas -o run_benchmark

#include "cpp/common.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

// ---------------------------------------------------------------------------
// Hyperparameters (verbatim from the cpp impl).
// ---------------------------------------------------------------------------

struct HP {
  size_t NBins = 50;
  size_t NHid = 256;
  size_t FanIn = 32;
  size_t NumClasses = 20;
  size_t Epochs = 80;
  float Lr = 1e-3f;
  float Beta = 0.9f;           // membrane decay
  float BetaTrace = 0.9f;      // eligibility-trace decay
  float Threshold = 1.0f;
  float SurrogateSlope = 25.0f;
  float WeightScale = 1.0f;
  float FeedbackScale = 1.0f;
  float ClipDelta = 0.1f;
  float WMax = 5.0f;
  size_t MaxTrainRows = 0;     // 0 = no cap
  size_t MaxEvalRows = 0;
  size_t EvalEvery = 1;
};

// ---------------------------------------------------------------------------
// Dataset loader (.plxbin) — copied verbatim from the cpp impl.
// ---------------------------------------------------------------------------

struct Dataset {
  std::vector<float> X;       // n_samples * n_bins * n_channels, float32
  std::vector<int64_t> Y;
  size_t NSamples = 0;
  size_t NBins = 0;
  size_t NChannels = 0;
  const float *Sample(size_t I, size_t T) const {
    return X.data() + (I * NBins + T) * NChannels;
  }
};

static Dataset LoadPlxbin(const std::filesystem::path &Path) {
  std::ifstream In(Path, std::ios::binary);
  if (!In) {
    std::cerr << "failed to open " << Path << "\n";
    std::exit(2);
  }
  uint32_t Header[6];
  In.read(reinterpret_cast<char *>(Header), sizeof(Header));
  if (Header[0] != 0x53484430u) {
    std::cerr << "bad magic in " << Path << "\n";
    std::exit(2);
  }
  Dataset D;
  D.NSamples = Header[1];
  D.NBins = Header[2];
  D.NChannels = Header[3];
  D.X.resize(D.NSamples * D.NBins * D.NChannels);
  In.read(reinterpret_cast<char *>(D.X.data()),
          D.X.size() * sizeof(float));
  D.Y.resize(D.NSamples);
  In.read(reinterpret_cast<char *>(D.Y.data()),
          D.Y.size() * sizeof(int64_t));
  return D;
}

// ---------------------------------------------------------------------------
// Device kernels.
// ---------------------------------------------------------------------------

inline unsigned Grid(size_t N, unsigned Block) {
  return static_cast<unsigned>((N + Block - 1) / Block);
}

// Hidden LIF subtract-reset, fused.
//   Mem = beta*Mem + Drive;  Z = Mem - thr;  Spk = (Z>=0);  Mem -= thr*Spk
__global__ void LifHidden(const float *Drive, float *Mem, float *Spk, float *Z,
                          float beta, float thr, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  float m = beta * Mem[I] + Drive[I];
  float z = m - thr;
  float spk = (z >= 0.0f) ? 1.0f : 0.0f;
  m -= thr * spk;
  Mem[I] = m;
  Spk[I] = spk;
  Z[I] = z;
}

// Output integrator + scaled logit accumulation.
//   MemOut = beta*MemOut + Drive;  LogitAcc += MemOut * logit_scale
__global__ void OutIntegrate(const float *Drive, float *MemOut, float *LogitAcc,
                             float beta, float logit_scale, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  float m = beta * MemOut[I] + Drive[I];
  MemOut[I] = m;
  LogitAcc[I] += m * logit_scale;
}

// Surrogate derivative psi[h] = 1 / (1 + slope*|z|)^2.
__global__ void Surrogate(const float *Z, float *Psi, float slope, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  float den = 1.0f + slope * fabsf(Z[I]);
  Psi[I] = 1.0f / (den * den);
}

// L_hid *= psi (per-unit gate after the B_out^T·L_out gemv).
__global__ void GateByPsi(float *LHid, const float *Psi, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    LHid[I] *= Psi[I];
}

// Masked clipped W_in step. dW[h,i] = lr * L_hid[h] * E_in[h,i], clipped, then
// W -= dW; W clipped to [-wmax, wmax]; dead positions (mask 0) skipped.
__global__ void WInStep(float *WIn, const float *EIn, const uint8_t *MIn,
                        const float *LHid, float lr, float clip, float wmax,
                        size_t NHid, size_t NIn) {
  size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (Idx >= NHid * NIn)
    return;
  if (!MIn[Idx])
    return;
  size_t h = Idx / NIn;
  float lh = LHid[h];
  if (lh == 0.0f)
    return;
  float delta = lr * lh * EIn[Idx];
  if (delta > clip) delta = clip;
  if (delta < -clip) delta = -clip;
  float w = WIn[Idx] - delta;
  if (w > wmax) w = wmax;
  if (w < -wmax) w = -wmax;
  WIn[Idx] = w;
}

// Dense clipped W_out step. dW[o,h] = lr * L_out[o] * E_out[o,h].
__global__ void WOutStep(float *WOut, const float *EOut, const float *LOut,
                         float lr, float clip, float wmax, size_t NOut,
                         size_t NHid) {
  size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (Idx >= NOut * NHid)
    return;
  size_t o = Idx / NHid;
  float lo = LOut[o];
  if (lo == 0.0f)
    return;
  float delta = lr * lo * EOut[Idx];
  if (delta > clip) delta = clip;
  if (delta < -clip) delta = -clip;
  float w = WOut[Idx] - delta;
  if (w > wmax) w = wmax;
  if (w < -wmax) w = -wmax;
  WOut[Idx] = w;
}

// Softmax + dL/dlogit = softmax - one_hot, single-block reduction over NOut
// (NOut is tiny — 20). Launch with one block, blockDim >= NOut rounded.
__global__ void SoftmaxLossSignal(const float *LogitAcc, float *LOut,
                                  int target, size_t NOut) {
  // single-threaded scan is fine for NOut=20; keep it simple & exact.
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  float maxl = -1e30f;
  for (size_t o = 0; o < NOut; ++o)
    if (LogitAcc[o] > maxl) maxl = LogitAcc[o];
  double Z = 0.0;
  for (size_t o = 0; o < NOut; ++o)
    Z += exp((double)(LogitAcc[o] - maxl));
  for (size_t o = 0; o < NOut; ++o) {
    float p = (float)(exp((double)(LogitAcc[o] - maxl)) / Z);
    LOut[o] = p - ((int)o == target ? 1.0f : 0.0f);
  }
}

// ---------------------------------------------------------------------------
// Device-resident model. cuBLAS is column-major; a row-major (R x C) buffer is
// bit-identical to a column-major (C x R) buffer. The cpp impl uses
// cblas_sgemv RowMajor; we map those to the column-major equivalents:
//   RowMajor NoTrans (M x N) * x  ==  col-major (N x M) with OP_T
//   RowMajor Trans   (M x N) * x  ==  col-major (N x M) with OP_N
// and cblas_sger RowMajor (M x N) A += alpha*x*y^T (x len M, y len N) maps to
// col-major (N x M): cublasSger(N, M, alpha, y, 1, x, 1, A, N).
// ---------------------------------------------------------------------------

struct Model {
  size_t NIn = 0, NHid = 0, NOut = 0;
  size_t NLiveIn = 0;

  // Device weights / masks / traces / state.
  float *WIn = nullptr;     // NHid x NIn (row-major)
  float *WOut = nullptr;    // NOut x NHid
  float *BOut = nullptr;    // NOut x NHid (random feedback)
  uint8_t *MIn = nullptr;   // NHid x NIn
  float *EIn = nullptr;     // NHid x NIn
  float *EOut = nullptr;    // NOut x NHid

  float *MemHid = nullptr;  // NHid
  float *MemOut = nullptr;  // NOut
  float *SpkHid = nullptr;  // NHid
  float *ZHid = nullptr;    // NHid
  float *LogitAcc = nullptr;// NOut
  float *LHid = nullptr;    // NHid
  float *LOut = nullptr;    // NOut
  float *Psi = nullptr;     // NHid
  float *Drive = nullptr;   // max(NHid, NOut)
  float *Ones = nullptr;    // NOut (for E_out rank-1)

  float *dU = nullptr;      // NIn — current timestep input (uploaded per step)

  cublasHandle_t Bl = nullptr;

  void Build(size_t n_in, size_t n_hid, size_t n_out, size_t fan_in,
             float w_scale, float fb_scale, uint64_t seed, cublasHandle_t bl) {
    NIn = n_in; NHid = n_hid; NOut = n_out; Bl = bl;

    // Build weights / mask on host (mirrors the cpp impl exactly).
    std::vector<float> hWIn(NHid * NIn, 0.0f);
    std::vector<float> hWOut(NOut * NHid, 0.0f);
    std::vector<float> hBOut(NOut * NHid, 0.0f);
    std::vector<uint8_t> hMIn(NHid * NIn, 0);

    std::mt19937_64 Rng(seed);
    float BoundIn =
        w_scale * std::sqrt(6.0f / static_cast<float>(fan_in + NHid));
    float BoundOut = std::sqrt(6.0f / static_cast<float>(NHid + NOut));
    float BoundFb =
        fb_scale * std::sqrt(6.0f / static_cast<float>(NHid + NOut));
    std::uniform_real_distribution<float> Win(-BoundIn, BoundIn);
    std::uniform_real_distribution<float> Wout(-BoundOut, BoundOut);
    std::uniform_real_distribution<float> Bout(-BoundFb, BoundFb);

    std::vector<size_t> Pool(NIn);
    std::iota(Pool.begin(), Pool.end(), 0);
    for (size_t H = 0; H < NHid; ++H) {
      size_t K = std::min(fan_in, NIn);
      for (size_t I = 0; I < K; ++I) {
        std::uniform_int_distribution<size_t> Pick(I, NIn - 1);
        std::swap(Pool[I], Pool[Pick(Rng)]);
        size_t Src = Pool[I];
        hMIn[H * NIn + Src] = 1;
        hWIn[H * NIn + Src] = Win(Rng);
      }
    }
    NLiveIn = NHid * fan_in;
    for (auto &v : hWOut) v = Wout(Rng);
    for (auto &v : hBOut) v = Bout(Rng);

    auto AllocF = [](float **p, size_t n) {
      CUDA_CHECK(cudaMalloc(p, n * sizeof(float)));
    };
    AllocF(&WIn, NHid * NIn);
    AllocF(&WOut, NOut * NHid);
    AllocF(&BOut, NOut * NHid);
    CUDA_CHECK(cudaMalloc(&MIn, NHid * NIn * sizeof(uint8_t)));
    AllocF(&EIn, NHid * NIn);
    AllocF(&EOut, NOut * NHid);
    AllocF(&MemHid, NHid);
    AllocF(&MemOut, NOut);
    AllocF(&SpkHid, NHid);
    AllocF(&ZHid, NHid);
    AllocF(&LogitAcc, NOut);
    AllocF(&LHid, NHid);
    AllocF(&LOut, NOut);
    AllocF(&Psi, NHid);
    AllocF(&Drive, std::max(NHid, NOut));
    AllocF(&Ones, NOut);
    AllocF(&dU, NIn);

    CUDA_CHECK(cudaMemcpy(WIn, hWIn.data(), hWIn.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(WOut, hWOut.data(), hWOut.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(BOut, hBOut.data(), hBOut.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(MIn, hMIn.data(), hMIn.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    std::vector<float> ones(NOut, 1.0f);
    CUDA_CHECK(cudaMemcpy(Ones, ones.data(), NOut * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  void ResetSequence() {
    CUDA_CHECK(cudaMemset(MemHid, 0, NHid * sizeof(float)));
    CUDA_CHECK(cudaMemset(MemOut, 0, NOut * sizeof(float)));
    CUDA_CHECK(cudaMemset(SpkHid, 0, NHid * sizeof(float)));
    CUDA_CHECK(cudaMemset(ZHid, 0, NHid * sizeof(float)));
    CUDA_CHECK(cudaMemset(LogitAcc, 0, NOut * sizeof(float)));
    CUDA_CHECK(cudaMemset(LHid, 0, NHid * sizeof(float)));
    CUDA_CHECK(cudaMemset(LOut, 0, NOut * sizeof(float)));
    CUDA_CHECK(cudaMemset(EIn, 0, NHid * NIn * sizeof(float)));
    CUDA_CHECK(cudaMemset(EOut, 0, NOut * NHid * sizeof(float)));
  }

  // Upload one timestep's input vector (NIn) into dU.
  void SetInput(const float *u_host) {
    CUDA_CHECK(cudaMemcpy(dU, u_host, NIn * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // Forward step: assumes dU holds the current input.
  void ForwardStep(float beta, float thr, float logit_scale) {
    const float One = 1.0f, Zero = 0.0f;
    const unsigned B = 256;
    // Drive_hid = W_in · u  (RowMajor NoTrans NHid x NIn => col-major OP_T)
    CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_T, (int)NIn, (int)NHid, &One, WIn,
                             (int)NIn, dU, 1, &Zero, Drive, 1));
    LifHidden<<<Grid(NHid, B), B>>>(Drive, MemHid, SpkHid, ZHid, beta, thr,
                                    NHid);
    // Drive_out = W_out · spk  (RowMajor NoTrans NOut x NHid => col-major OP_T)
    CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_T, (int)NHid, (int)NOut, &One, WOut,
                             (int)NHid, SpkHid, 1, &Zero, Drive, 1));
    OutIntegrate<<<Grid(NOut, B), B>>>(Drive, MemOut, LogitAcc, beta,
                                       logit_scale, NOut);
  }

  // Backward + eligibility + weight step for this timestep. `target` >= 0 only
  // on the terminal step (else no loss signal / no weight update).
  void EpropStep(const HP &H, float beta_trace, float lr, int target) {
    const float One = 1.0f, Zero = 0.0f;
    const unsigned B = 256;

    // 1. Loss signal at terminal step; else L_out = 0.
    if (target >= 0) {
      SoftmaxLossSignal<<<1, 32>>>(LogitAcc, LOut, target, NOut);
    } else {
      CUDA_CHECK(cudaMemset(LOut, 0, NOut * sizeof(float)));
    }

    // 2. Surrogate psi (reused for elig + gate).
    Surrogate<<<Grid(NHid, B), B>>>(ZHid, Psi, H.SurrogateSlope, NHid);

    // L_hid = (B_out^T · L_out) ⊙ psi.
    //   RowMajor Trans (NOut x NHid) * L_out  =>  col-major OP_N (NHid x NOut)
    CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_N, (int)NHid, (int)NOut, &One, BOut,
                             (int)NHid, LOut, 1, &Zero, LHid, 1));
    GateByPsi<<<Grid(NHid, B), B>>>(LHid, Psi, NHid);

    // 3. Eligibility traces (rank-1 updates).
    //   E_in[h,i]  = beta_trace·E_in  + psi[h]·u[i]
    //   E_out[o,h] = beta_trace·E_out + 1·spk[h]
    CUBLAS_CHECK(cublasSscal(Bl, (int)(NHid * NIn), &beta_trace, EIn, 1));
    // cblas_sger RowMajor (NHid x NIn) A += psi(len NHid) * u(len NIn)^T
    //   => col-major (NIn x NHid): cublasSger(NIn, NHid, 1, u, 1, psi, 1, A, NIn)
    CUBLAS_CHECK(cublasSger(Bl, (int)NIn, (int)NHid, &One, dU, 1, Psi, 1, EIn,
                            (int)NIn));
    CUBLAS_CHECK(cublasSscal(Bl, (int)(NOut * NHid), &beta_trace, EOut, 1));
    // cblas_sger RowMajor (NOut x NHid) A += ones(len NOut) * spk(len NHid)^T
    //   => col-major (NHid x NOut): cublasSger(NHid, NOut, 1, spk, 1, ones, 1, A, NHid)
    CUBLAS_CHECK(cublasSger(Bl, (int)NHid, (int)NOut, &One, SpkHid, 1, Ones, 1,
                            EOut, (int)NHid));

    // 4. Weight step — only at terminal step (target >= 0).
    if (target < 0)
      return;
    WInStep<<<Grid(NHid * NIn, B), B>>>(WIn, EIn, MIn, LHid, lr, H.ClipDelta,
                                        H.WMax, NHid, NIn);
    WOutStep<<<Grid(NOut * NHid, B), B>>>(WOut, EOut, LOut, lr, H.ClipDelta,
                                          H.WMax, NOut, NHid);
  }

  // Argmax over LogitAcc — done on host (NOut tiny).
  int Argmax(std::vector<float> &buf) const {
    CUDA_CHECK(cudaMemcpy(buf.data(), LogitAcc, NOut * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int best = 0;
    float bestv = buf[0];
    for (size_t o = 1; o < NOut; ++o)
      if (buf[o] > bestv) { bestv = buf[o]; best = (int)o; }
    return best;
  }

  void GetLogits(std::vector<float> &buf) const {
    CUDA_CHECK(cudaMemcpy(buf.data(), LogitAcc, NOut * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  double SumSpikes(std::vector<float> &buf) const {
    CUDA_CHECK(cudaMemcpy(buf.data(), SpkHid, NHid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (size_t h = 0; h < NHid; ++h) s += buf[h];
    return s;
  }

  void Free() {
    cudaFree(WIn); cudaFree(WOut); cudaFree(BOut); cudaFree(MIn);
    cudaFree(EIn); cudaFree(EOut); cudaFree(MemHid); cudaFree(MemOut);
    cudaFree(SpkHid); cudaFree(ZHid); cudaFree(LogitAcc); cudaFree(LHid);
    cudaFree(LOut); cudaFree(Psi); cudaFree(Drive); cudaFree(Ones);
    cudaFree(dU);
  }
};

// ---------------------------------------------------------------------------
// Eval helpers (pure forward).
// ---------------------------------------------------------------------------

static double EvalAccuracy(Model &M, const Dataset &D, const HP &H,
                           std::vector<float> &logitbuf,
                           size_t cap = 0,
                           bool time_shuffle = false,
                           uint32_t shuffle_seed = 0) {
  size_t N = D.NSamples;
  if (cap > 0 && cap < N) N = cap;
  size_t correct = 0;
  float logit_scale = 1.0f / static_cast<float>(D.NBins);
  std::mt19937 sr(shuffle_seed);
  std::vector<size_t> perm(D.NBins);
  std::iota(perm.begin(), perm.end(), 0);
  for (size_t i = 0; i < N; ++i) {
    M.ResetSequence();
    if (time_shuffle) std::shuffle(perm.begin(), perm.end(), sr);
    for (size_t T = 0; T < D.NBins; ++T) {
      const float *src = time_shuffle ? D.Sample(i, perm[T]) : D.Sample(i, T);
      M.SetInput(src);
      M.ForwardStep(H.Beta, H.Threshold, logit_scale);
    }
    if (M.Argmax(logitbuf) == static_cast<int>(D.Y[i])) ++correct;
  }
  return N ? static_cast<double>(correct) / N : 0.0;
}

static double MeanFiringRate(Model &M, const Dataset &D, const HP &H,
                             std::vector<float> &spkbuf, size_t cap) {
  size_t N = std::min(cap, D.NSamples);
  if (N == 0) return 0.0;
  double total = 0.0;
  size_t slots = 0;
  float logit_scale = 1.0f / static_cast<float>(D.NBins);
  for (size_t i = 0; i < N; ++i) {
    M.ResetSequence();
    for (size_t T = 0; T < D.NBins; ++T) {
      M.SetInput(D.Sample(i, T));
      M.ForwardStep(H.Beta, H.Threshold, logit_scale);
      total += M.SumSpikes(spkbuf);
      slots += M.NHid;
    }
  }
  return slots ? total / static_cast<double>(slots) : 0.0;
}

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.NBins         = static_cast<size_t>(Args.GetInt("n-bins",   H.NBins));
  H.NHid          = static_cast<size_t>(Args.GetInt("n-hid",    H.NHid));
  H.FanIn         = static_cast<size_t>(Args.GetInt("fan-in",   H.FanIn));
  H.NumClasses    = static_cast<size_t>(Args.GetInt("n-classes",H.NumClasses));
  H.Epochs        = static_cast<size_t>(Args.GetInt("epochs",   H.Epochs));
  H.Lr            = Args.GetFloat("lr",            H.Lr);
  H.Beta          = Args.GetFloat("beta",          H.Beta);
  H.BetaTrace     = Args.GetFloat("beta-trace",    H.BetaTrace);
  H.Threshold     = Args.GetFloat("threshold",     H.Threshold);
  H.SurrogateSlope = Args.GetFloat("surrogate-slope", H.SurrogateSlope);
  H.WeightScale   = Args.GetFloat("weight-scale",  H.WeightScale);
  H.FeedbackScale = Args.GetFloat("feedback-scale",H.FeedbackScale);
  H.ClipDelta     = Args.GetFloat("clip-delta",    H.ClipDelta);
  H.WMax          = Args.GetFloat("w-max",         H.WMax);
  H.MaxTrainRows  = static_cast<size_t>(Args.GetInt(
      "max-train-rows", static_cast<int>(H.MaxTrainRows)));
  H.MaxEvalRows   = static_cast<size_t>(Args.GetInt(
      "max-eval-rows",  static_cast<int>(H.MaxEvalRows)));
  H.EvalEvery     = static_cast<size_t>(Args.GetInt(
      "eval-every",     static_cast<int>(H.EvalEvery)));
  if (Args.Quick) {
    H.Epochs = std::max<size_t>(1, H.Epochs / 4);
    if (H.MaxTrainRows == 0) H.MaxTrainRows = 1024;
    if (H.MaxEvalRows == 0)  H.MaxEvalRows  = 512;
  }

  auto TrainPath = Args.DataDir / "SHD_cache" /
                   ("train_n" + std::to_string(H.NBins) + ".plxbin");
  auto TestPath  = Args.DataDir / "SHD_cache" /
                   ("test_n"  + std::to_string(H.NBins) + ".plxbin");
  if (!std::filesystem::exists(TrainPath)) {
    std::cerr << "missing " << TrainPath << "\n"
              << "Run:\n  uv run python snn_shd/pytorch/data.py "
                 "--export --n-bins " << H.NBins << "\n";
    return 2;
  }

  std::cout << "[data] loading " << TrainPath << "\n";
  auto Train = LoadPlxbin(TrainPath);
  std::cout << "[data] loading " << TestPath << "\n";
  auto Test = LoadPlxbin(TestPath);
  size_t NIn = Train.NChannels;
  std::cout << "[info] cuda snn_shd  train=" << Train.NSamples
            << " test=" << Test.NSamples << " n_in=" << NIn
            << " n_hid=" << H.NHid << " n_out=" << H.NumClasses
            << " n_bins=" << H.NBins << " fan_in=" << H.FanIn
            << " epochs=" << H.Epochs << " lr=" << H.Lr << "\n";

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  Model M;
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 41ull;
  M.Build(NIn, H.NHid, H.NumClasses, H.FanIn,
          H.WeightScale, H.FeedbackScale, SeedBase + 1, Bl);
  size_t NConns = M.NLiveIn + M.NOut * M.NHid;
  std::cout << "[info] live conns=" << NConns
            << "  (vs " << (NIn * H.NHid + H.NHid * H.NumClasses)
            << " dense)\n";

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "snn_shd");
  bench::StructuralLog Log(HistPath);

  std::vector<float> LogitBuf(H.NumClasses, 0.0f);
  std::vector<float> SpkBuf(H.NHid, 0.0f);

  size_t EvalCap = H.MaxEvalRows;
  double InitTest = EvalAccuracy(M, Test, H, LogitBuf, EvalCap);
  double InitVal  = 1.0 - InitTest;
  Log.Log(0, NIn + H.NHid + H.NumClasses, NConns, nullptr, &InitVal,
          {{"test_acc", InitTest},
           {"firing_rate", 0.0},
           {"train_loss", 0.0},
           {"epoch", 0.0}});
  std::cout << "[ep   0] test_acc=" << InitTest << "\n";

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  size_t TrainCap = H.MaxTrainRows > 0
                       ? std::min(H.MaxTrainRows, Train.NSamples)
                       : Train.NSamples;
  std::vector<size_t> Perm(TrainCap);
  std::iota(Perm.begin(), Perm.end(), 0);

  double BestTest = InitTest;
  float logit_scale = 1.0f / static_cast<float>(Train.NBins);
  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();

  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    double LossSum = 0.0;
    size_t LossCnt = 0;
    size_t Correct = 0;

    for (size_t Idx : Perm) {
      M.ResetSequence();
      int tgt = static_cast<int>(Train.Y[Idx]);

      for (size_t T = 0; T < Train.NBins; ++T) {
        bool final = (T + 1 == Train.NBins);
        M.SetInput(Train.Sample(Idx, T));
        Timer.Tick();
        M.ForwardStep(H.Beta, H.Threshold, logit_scale);
        CUDA_CHECK(cudaDeviceSynchronize());
        Timer.MarkForward();
        // EpropStep bundles loss + surrogate-gradient backward + eligibility
        // + weight update — reported under backward to line up with PyTorch's
        // loss.backward()+optimizer.step() bundle.
        M.EpropStep(H, H.BetaTrace, H.Lr, final ? tgt : -1);
        CUDA_CHECK(cudaDeviceSynchronize());
        Timer.MarkBackward();
        Timer.StepDone();
      }

      // Cross-entropy loss for reporting (final-step LogitAcc).
      M.GetLogits(LogitBuf);
      float maxl = *std::max_element(LogitBuf.begin(), LogitBuf.end());
      double Z = 0.0;
      for (float v : LogitBuf) Z += std::exp(v - maxl);
      float P = static_cast<float>(std::exp(LogitBuf[tgt] - maxl) / Z);
      LossSum += -std::log(std::max(P, 1e-30f));
      ++LossCnt;
      int best = 0;
      float bestv = LogitBuf[0];
      for (size_t o = 1; o < H.NumClasses; ++o)
        if (LogitBuf[o] > bestv) { bestv = LogitBuf[o]; best = (int)o; }
      if (best == tgt) ++Correct;
    }

    double TrLoss = LossCnt ? LossSum / LossCnt : 0.0;
    double TrAcc = TrainCap ? static_cast<double>(Correct) / TrainCap : 0.0;
    double TestAcc = 0.0, Rate = 0.0;
    if (Ep == H.Epochs || (Ep % H.EvalEvery) == 0) {
      TestAcc = EvalAccuracy(M, Test, H, LogitBuf, EvalCap);
      Rate = MeanFiringRate(M, Train, H, SpkBuf, std::min<size_t>(128, TrainCap));
    }
    if (TestAcc > BestTest) BestTest = TestAcc;

    double NegMetric = 1.0 - TestAcc;
    Log.Log(Ep, NIn + H.NHid + H.NumClasses, NConns, nullptr, &NegMetric,
            {{"test_acc", TestAcc},
             {"firing_rate", Rate},
             {"train_loss", TrLoss},
             {"train_acc", TrAcc},
             {"epoch", static_cast<double>(Ep)}});
    std::cout << "[ep " << std::setw(3) << Ep
              << "] train_loss=" << TrLoss
              << " train_acc=" << TrAcc
              << " test_acc=" << TestAcc
              << " rate=" << Rate << "\n" << std::flush;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  double FinalTest = EvalAccuracy(M, Test, H, LogitBuf, EvalCap);
  double ShuffledTest = EvalAccuracy(M, Test, H, LogitBuf, EvalCap,
                                     /*time_shuffle=*/true,
                                     /*shuffle_seed=*/Args.Seed + 9999);

  Log.Flush();
  bench::SummaryWriter S;
  S.Set("workload", std::string{"06_snn_shd"});
  S.Set("dataset", std::string{"SHD"});
  S.Set("backend", std::string{"cuda"});
  S.Set("n_in", static_cast<int>(NIn));
  S.Set("n_hid", static_cast<int>(H.NHid));
  S.Set("n_out", static_cast<int>(H.NumClasses));
  S.Set("fan_in", static_cast<int>(H.FanIn));
  S.Set("n_bins", static_cast<int>(H.NBins));
  S.Set("n_conns", static_cast<int>(NConns));
  S.Set("epochs", static_cast<int>(H.Epochs));
  S.Set("lr", static_cast<double>(H.Lr));
  S.Set("beta", static_cast<double>(H.Beta));
  S.Set("beta_trace", static_cast<double>(H.BetaTrace));
  S.Set("threshold", static_cast<double>(H.Threshold));
  S.Set("surrogate_slope", static_cast<double>(H.SurrogateSlope));
  S.Set("wall_seconds", Wall);
  S.Set("val_acc_best", BestTest);
  S.Set("test_acc", FinalTest);
  S.Set("test_acc_shuffled", ShuffledTest);
  S.Set("ablation_drop", FinalTest - ShuffledTest);
  S.Set("metric_kind", std::string{"test_acc"});
  S.Set("n_units", static_cast<int>(NIn + H.NHid + H.NumClasses));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_acc=" << FinalTest
            << "  shuffled=" << ShuffledTest
            << "  drop=" << (FinalTest - ShuffledTest) << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  M.Free();
  cublasDestroy(Bl);
  (void)LogPath;
  return 0;
}
