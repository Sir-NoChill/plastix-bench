// Workload 6 / 7 — Compact C. elegans-Wired Controller, pure-CUDA / cuBLAS.
//
// A GPU port of 06_ccwc_ncp/cpp/07_ccwc_ncp.cpp: the real ncps-style
// Liquid-Time-Constant (LTC) ODE cell over an AutoNCP sparse wiring, trained
// with true Backprop-Through-Time on the synthetic sine task. The structural
// machinery (AutoNCP partition + wiring masks, host RNG, dataset generation)
// is identical to the cpp impl and runs on the host. The dense linear algebra
// of the LTC forward / BPTT backward / Adam update path runs on the GPU.
//
// Fidelity: this reproduces the cpp algorithm exactly — the semi-implicit
// Euler ODE unfolds, the per-synapse (w, sigma, mu, erev) sigmoid conductances
// with softplus weights, the leak / capacitance terms, the affine in/out maps,
// and the hand-derived reverse pass. The whole minibatch of sequences is
// processed in parallel across the GPU grid (one CUDA thread per (dst-unit,
// sample) pair), so the recurrent/weight state stays resident on the device
// for the entire forward+backward; only the host RNG, the structural wiring
// and the dataset generation stay on the host (copied verbatim).
//
// cuBLAS usage: the affine input map (mapped_u = InW .* u + InB), the affine
// output map, and the cross-sample reductions for the parameter gradients use
// cuBLAS Sgemm / Sgemv / Saxpy; the per-synapse ODE numerator/denominator
// reductions (which carry a per-(dst,src) nonlinearity, so are NOT a fixed
// matmul) and the activation gates use small custom kernels. float32 throughout.
//
// Build (standalone):
//   nvcc -O3 -std=c++20 --extended-lambda --expt-relaxed-constexpr \
//     -arch=sm_89 -I common -I/usr/local/cuda/include \
//     06_ccwc_ncp/cuda/06_ccwc_ncp.cu \
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

inline unsigned Grid(size_t N, unsigned Block) {
  return static_cast<unsigned>((N + Block - 1) / Block);
}
constexpr unsigned kBlock = 256;

// ---------------------------------------------------------------------------
// Hyperparameters (verbatim from the cpp impl).
// ---------------------------------------------------------------------------

struct HP {
  size_t Units      = 32;
  size_t InputDim   = 2;
  size_t OutputDim  = 2;
  size_t KSparse    = 4;
  size_t KRec       = 4;
  size_t KFb        = 2;
  size_t SeqLen     = 64;
  size_t TrainSeqs  = 512;
  size_t ValSeqs    = 128;
  size_t TestSeqs   = 128;
  size_t Epochs     = 20;
  size_t Batch      = 64;
  size_t OdeUnfolds = 6;
  float  NoiseStd   = 0.1f;
  float  Elapsed    = 1.0f;
  float  Lr         = 1e-3f;
  float  Beta1      = 0.9f;
  float  Beta2      = 0.999f;
  float  AdamEps    = 1e-8f;
  float  GradClip   = 1.0f;
  float  OdeEps     = 1e-8f;
};

enum Kind : uint8_t { KSensory = 0, KInter = 1, KCommand = 2, KMotor = 3 };

// ---------------------------------------------------------------------------
// Host numeric helpers (used only for structural init parity).
// ---------------------------------------------------------------------------

inline float SoftPlusH(float X) { return X > 20.0f ? X : std::log1p(std::exp(X)); }
inline float SigmoidH(float X) {
  if (X >= 0.0f) { float Z = std::exp(-X); return 1.0f / (1.0f + Z); }
  float Z = std::exp(X); return Z / (1.0f + Z);
}

// ---------------------------------------------------------------------------
// AutoNCP-style partition + wiring (verbatim from the cpp impl).
// ---------------------------------------------------------------------------

struct Partition {
  size_t Ns, Ni, Nc, Nm;
  std::vector<Kind> Kinds;
};

Partition ComputePartition(size_t Units, size_t Motors) {
  Partition P{};
  P.Nm = Motors;
  size_t Rest = Units > Motors ? Units - Motors : 0;
  P.Ns = std::max<size_t>(1, Rest / 3);
  size_t After = Rest > P.Ns ? Rest - P.Ns : 0;
  P.Ni = After / 2;
  P.Nc = After - P.Ni;
  if (P.Nc == 0) { P.Nc = 1; if (P.Ni > 0) --P.Ni; }
  P.Kinds.resize(Units);
  size_t Idx = 0;
  for (size_t I = 0; I < P.Ns; ++I) P.Kinds[Idx++] = KSensory;
  for (size_t I = 0; I < P.Ni; ++I) P.Kinds[Idx++] = KInter;
  for (size_t I = 0; I < P.Nc; ++I) P.Kinds[Idx++] = KCommand;
  for (size_t I = 0; I < P.Nm; ++I) P.Kinds[Idx++] = KMotor;
  return P;
}

struct KindRanges {
  size_t SenseBeg, SenseEnd, InterBeg, InterEnd, CmdBeg, CmdEnd, MotorBeg, MotorEnd;
};

KindRanges Ranges(const Partition &P) {
  KindRanges R{};
  R.SenseBeg = 0;          R.SenseEnd = P.Ns;
  R.InterBeg = R.SenseEnd; R.InterEnd = R.InterBeg + P.Ni;
  R.CmdBeg   = R.InterEnd;  R.CmdEnd   = R.CmdBeg + P.Nc;
  R.MotorBeg = R.CmdEnd;    R.MotorEnd = R.MotorBeg + P.Nm;
  return R;
}

std::vector<size_t> SampleK(size_t Beg, size_t End, size_t K, size_t Forbidden,
                            std::mt19937_64 &Rng) {
  std::vector<size_t> Pool;
  Pool.reserve(End - Beg);
  for (size_t I = Beg; I < End; ++I)
    if (I != Forbidden) Pool.push_back(I);
  if (K >= Pool.size()) return Pool;
  std::shuffle(Pool.begin(), Pool.end(), Rng);
  Pool.resize(K);
  return Pool;
}

struct Masks {
  std::vector<uint8_t> Rec;   // Units x Units, row-major (row = dst)
  std::vector<uint8_t> In;    // Units x InputDim (row = dst)
  size_t LiveEdges = 0;
};

Masks BuildMasks(const HP &H, const Partition &P, std::mt19937_64 &Rng) {
  KindRanges R = Ranges(P);
  Masks M;
  M.Rec.assign(H.Units * H.Units, 0);
  M.In.assign(H.Units * H.InputDim, 0);
  auto EdgeRec = [&](size_t Src, size_t Dst) { M.Rec[Dst * H.Units + Src] = 1; ++M.LiveEdges; };
  auto EdgeIn  = [&](size_t Src, size_t Dst) { M.In[Dst * H.InputDim + Src] = 1; ++M.LiveEdges; };
  for (size_t Src = 0; Src < H.InputDim; ++Src)
    for (size_t Dst = R.SenseBeg; Dst < R.SenseEnd; ++Dst) EdgeIn(Src, Dst);
  if (P.Ni > 0 && P.Ns > 0)
    for (size_t Dst = R.InterBeg; Dst < R.InterEnd; ++Dst)
      for (auto Src : SampleK(R.SenseBeg, R.SenseEnd, H.KSparse, ~size_t{0}, Rng)) EdgeRec(Src, Dst);
  if (P.Nc > 0) {
    size_t SrcBeg = P.Ni > 0 ? R.InterBeg : R.SenseBeg;
    size_t SrcEnd = P.Ni > 0 ? R.InterEnd : R.SenseEnd;
    for (size_t Dst = R.CmdBeg; Dst < R.CmdEnd; ++Dst)
      for (auto Src : SampleK(SrcBeg, SrcEnd, H.KSparse, ~size_t{0}, Rng)) EdgeRec(Src, Dst);
  }
  if (P.Nc > 1)
    for (size_t Dst = R.CmdBeg; Dst < R.CmdEnd; ++Dst)
      for (auto Src : SampleK(R.CmdBeg, R.CmdEnd, H.KRec, Dst, Rng)) EdgeRec(Src, Dst);
  if (P.Nc > 0)
    for (size_t Dst = R.MotorBeg; Dst < R.MotorEnd; ++Dst)
      for (size_t Src = R.CmdBeg; Src < R.CmdEnd; ++Src) EdgeRec(Src, Dst);
  if (P.Nc > 0 && H.KFb > 0)
    for (size_t Src = R.MotorBeg; Src < R.MotorEnd; ++Src)
      for (auto Dst : SampleK(R.CmdBeg, R.CmdEnd, H.KFb, ~size_t{0}, Rng)) EdgeRec(Src, Dst);
  return M;
}

// ---------------------------------------------------------------------------
// Sine dataset (verbatim from the cpp impl).
// ---------------------------------------------------------------------------

struct SineDataset {
  std::vector<float> X;   // N * SeqLen * InDim, contiguous
  std::vector<float> Y;   // N * SeqLen * OutDim
  size_t N = 0, SeqLen = 0, InDim = 2, OutDim = 2;
};

SineDataset MakeSine(size_t N, size_t SeqLen, float NoiseStd, uint64_t Seed) {
  SineDataset D;
  D.N = N; D.SeqLen = SeqLen;
  D.X.assign(N * SeqLen * D.InDim, 0.0f);
  D.Y.assign(N * SeqLen * D.OutDim, 0.0f);
  std::mt19937_64 Rng(Seed);
  std::uniform_real_distribution<float> Freq(0.5f, 2.0f);
  std::uniform_real_distribution<float> Phase(0.0f, 6.28318530718f);
  std::normal_distribution<float> Noise(0.0f, NoiseStd);
  for (size_t I = 0; I < N; ++I) {
    float F = Freq(Rng);
    float Ph = Phase(Rng);
    for (size_t T = 0; T < SeqLen; ++T) {
      float A0 = F * (6.28318530718f * static_cast<float>(T) / SeqLen) + Ph;
      float A1 = F * (6.28318530718f * static_cast<float>(T + 1) / SeqLen) + Ph;
      size_t Xb = (I * SeqLen + T) * 2, Yb = (I * SeqLen + T) * 2;
      D.X[Xb + 0] = std::sin(A0) + Noise(Rng);
      D.X[Xb + 1] = std::cos(A0) + Noise(Rng);
      D.Y[Yb + 0] = std::sin(A1);
      D.Y[Yb + 1] = std::cos(A1);
    }
  }
  return D;
}

// ===========================================================================
// Device-side LTC kernels.
//
// Layout conventions (all row-major / column-strided):
//   - Per-synapse params (Wr, Sigma, Mu, Er): S*S, index = dst*S + src.
//   - Sensory params (SW, SSigma, SMu, SEr): S*In, index = dst*In + feat.
//   - State / per-unit-per-sample matrices: S * B, index = j*B + b (row = unit,
//     col = sample). One thread per (j, b).
//   - Masks live on device too (uint8).
// ===========================================================================

__device__ inline float dSoftPlus(float X) { return X > 20.0f ? X : log1pf(expf(X)); }
__device__ inline float dSigmoid(float X) {
  if (X >= 0.0f) { float Z = expf(-X); return 1.0f / (1.0f + Z); }
  float Z = expf(X); return Z / (1.0f + Z);
}

// Layout note: all per-unit/per-feature device buffers are stored with row
// stride `Str` (= BatchCap). The active minibatch is the first `B` columns. We
// launch grids over S*B (or In*B / Out*B) active elements, decode (row, b) with
// `B`, but address the buffers with `Str` so a partial final batch (B < Str)
// addresses the right columns without touching stale ones.

// mapped_u[k,b] = u[k,b]*InW[k] + InB[k].   (k in [0,In), b in [0,B))
__global__ void MapInput(const float *U, const float *InW, const float *InB,
                         float *Mapped, int In, int B, int Str) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= In * B) return;
  int k = i / B, b = i % B;
  int o = k * Str + b;
  Mapped[o] = U[o] * InW[k] + InB[k];
}

// Sensory numerator/denominator per (j, b), loop-invariant across unfolds.
// NumS[j,b] = sum_k mask*A*SEr ; DenS[j,b] = sum_k mask*A ; A = softplus(SW)*sigmoid(...).
__global__ void SensoryKernel(const float *Mapped, const float *SW,
                              const float *SSigma, const float *SMu,
                              const float *SEr, const uint8_t *MaskIn,
                              float *NumS, float *DenS, int S, int In, int B, int Str) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S * B) return;
  int j = idx / B, b = idx % B;
  int o = j * Str + b;
  float num = 0.0f, den = 0.0f;
  for (int k = 0; k < In; ++k) {
    int wi = j * In + k;
    if (!MaskIn[wi]) continue;
    float mu = Mapped[k * Str + b];
    float g = dSigmoid(SSigma[wi] * (mu - SMu[wi]));
    float a = dSoftPlus(SW[wi]) * g;
    num += a * SEr[wi];
    den += a;
  }
  NumS[o] = num; DenS[o] = den;
}

// One semi-implicit Euler unfold. Reads Cur (S*B), writes Nxt (S*B).
//   Num = CmT[j]*Cur + Gl[j]*Vleak[j] + NumS[j,b] + sum_src A*Er
//   Den = CmT[j] + Gl[j] + DenS[j,b] + sum_src A
//   A   = softplus(Wr[j,src]) * sigmoid(Sigma[j,src]*(Cur[src,b]-Mu[j,src]))
__global__ void UnfoldKernel(const float *Cur, float *Nxt, const float *Wr,
                             const float *Sigma, const float *Mu, const float *Er,
                             const uint8_t *MaskRec, const float *CmT,
                             const float *Gl, const float *Vleak,
                             const float *NumS, const float *DenS, float OdeEps,
                             int S, int B, int Str) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S * B) return;
  int j = idx / B, b = idx % B;
  int o = j * Str + b;
  float num = CmT[j] * Cur[o] + Gl[j] * Vleak[j] + NumS[o];
  float den = CmT[j] + Gl[j] + DenS[o];
  const uint8_t *mrow = MaskRec + j * S;
  for (int src = 0; src < S; ++src) {
    if (!mrow[src]) continue;
    int wi = j * S + src;
    float a = dSoftPlus(Wr[wi]) * dSigmoid(Sigma[wi] * (Cur[src * Str + b] - Mu[wi]));
    num += a * Er[wi];
    den += a;
  }
  Nxt[o] = num / (den + OdeEps);
}

// Output affine + MSE gradient seed into the motor units of the final state.
// Out[m,b] = State[(MotorBeg+m)*B+b]*OutW[m] + OutB[m].
// dOut = 2*(Out - Tg)*Scale ; seeds GradH (S*B) at motor rows; accumulates OutW/OutB grads.
__global__ void OutputBackwardKernel(const float *State, const float *Tg,
                                     const float *OutW, const float *OutB,
                                     float *GradH, float *OutWGrad, float *OutBGrad,
                                     float Scale, int MotorBeg, int Out, int S, int B,
                                     int Str) {
  // one thread per (m, b)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Out * B) return;
  int m = idx / B, b = idx % B;
  int munit = MotorBeg + m;
  float sv = State[munit * Str + b];
  float pred = sv * OutW[m] + OutB[m];
  float dout = 2.0f * (pred - Tg[m * Str + b]) * Scale;
  GradH[munit * Str + b] += dout * OutW[m];
  atomicAdd(&OutWGrad[m], dout * sv);
  atomicAdd(&OutBGrad[m], dout);
}

// Forward output read (eval/loss bookkeeping). Out[m,b] stored with stride Str.
__global__ void ReadOutputKernel(const float *State, const float *OutW,
                                 const float *OutB, float *Out, int MotorBeg,
                                 int Out_, int B, int Str) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Out_ * B) return;
  int m = idx / B, b = idx % B;
  Out[m * Str + b] = State[(MotorBeg + m) * Str + b] * OutW[m] + OutB[m];
}

// Reverse of one ODE unfold (BPTT). Mirrors the cpp Backward inner loop exactly.
//   Inputs:  Cur (state into unfold), Nxt (vnew out), Gcur (grad wrt Nxt).
//   Outputs: Gpre (grad wrt Cur, summed) — must be pre-zeroed.
//            param grads (Wr/Sigma/Mu/Er, Cm/Gleak/Vleak) accumulated via atomicAdd.
//            DNumS/DDenS accumulated (grad onto sensory num/den), per (j,b).
// One thread per (j, b). Gpre updates touch column b across many src -> atomicAdd.
__global__ void UnfoldBackwardKernel(
    const float *Cur, const float *Nxt, const float *Gcur, float *Gpre,
    const float *Wr, const float *Sigma, const float *Mu, const float *Er,
    const uint8_t *MaskRec, const float *CmT, const float *Gl,
    const float *Vleak, const float *DenS, const float *CmRaw,
    const float *GleakRaw, float DtSub, float OdeEps,
    float *WrGrad, float *SigmaGrad, float *MuGrad, float *ErGrad,
    float *CmGrad, float *GleakGrad, float *VleakGrad,
    float *DNumS, float *DDenS, int S, int B, int Str) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S * B) return;
  int j = idx / B, b = idx % B;
  int o = j * Str + b;

  float denj = CmT[j] + Gl[j] + DenS[o];
  const uint8_t *mrow = MaskRec + j * S;
  for (int src = 0; src < S; ++src) {
    if (!mrow[src]) continue;
    int wi = j * S + src;
    denj += dSoftPlus(Wr[wi]) * dSigmoid(Sigma[wi] * (Cur[src * Str + b] - Mu[wi]));
  }
  float inv = 1.0f / (denj + OdeEps);
  float gc = Gcur[o];
  float dnum = gc * inv;
  float dden = -gc * Nxt[o] * inv;

  // Capacitance: num += cm_t*cur[j], den += cm_t.
  float dcmt = dnum * Cur[o] + dden;
  atomicAdd(&CmGrad[j], dcmt * dSigmoid(CmRaw[j]) / DtSub);
  atomicAdd(&Gpre[o], dnum * CmT[j]);

  // Leak: num += gl*vleak, den += gl.
  float dgl = dnum * Vleak[j] + dden;
  atomicAdd(&GleakGrad[j], dgl * dSigmoid(GleakRaw[j]));
  atomicAdd(&VleakGrad[j], dnum * Gl[j]);

  // Sensory contribution accumulates (back-propped once on host-side path).
  DNumS[o] += dnum;
  DDenS[o] += dden;

  // Recurrent synapses.
  for (int src = 0; src < S; ++src) {
    if (!mrow[src]) continue;
    int wi = j * S + src;
    float wraw = Wr[wi];
    float spw = dSoftPlus(wraw);
    float curs = Cur[src * Str + b];
    float z = Sigma[wi] * (curs - Mu[wi]);
    float gs = dSigmoid(z);
    float a = spw * gs;
    float da = dnum * Er[wi] + dden;
    atomicAdd(&ErGrad[wi], dnum * a);
    atomicAdd(&WrGrad[wi], (da * gs) * dSigmoid(wraw));
    float dg = da * spw;
    float dz = dg * gs * (1.0f - gs);
    atomicAdd(&SigmaGrad[wi], dz * (curs - Mu[wi]));
    atomicAdd(&MuGrad[wi], -dz * Sigma[wi]);
    atomicAdd(&Gpre[src * Str + b], dz * Sigma[wi]);
  }
}

// Sensory backward. Uses accumulated DNumS/DDenS (per j,b). One thread per (j,b).
// Accumulates SW/SSigma/SMu/SEr grads and DMappedU[k,b] (grad onto mapped_u).
__global__ void SensoryBackwardKernel(
    const float *Mapped, const float *SW, const float *SSigma, const float *SMu,
    const float *SEr, const uint8_t *MaskIn, const float *DNumS,
    const float *DDenS, float *SWGrad, float *SSigmaGrad, float *SMuGrad,
    float *SErGrad, float *DMappedU, int S, int In, int B, int Str) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S * B) return;
  int j = idx / B, b = idx % B;
  int o = j * Str + b;
  float dnums = DNumS[o], ddens = DDenS[o];
  for (int k = 0; k < In; ++k) {
    int wi = j * In + k;
    if (!MaskIn[wi]) continue;
    float swraw = SW[wi];
    float spw = dSoftPlus(swraw);
    float mu = Mapped[k * Str + b];
    float z = SSigma[wi] * (mu - SMu[wi]);
    float gs = dSigmoid(z);
    float a = spw * gs;
    float da = dnums * SEr[wi] + ddens;
    atomicAdd(&SErGrad[wi], dnums * a);
    atomicAdd(&SWGrad[wi], (da * gs) * dSigmoid(swraw));
    float dg = da * spw;
    float dz = dg * gs * (1.0f - gs);
    atomicAdd(&SSigmaGrad[wi], dz * (mu - SMu[wi]));
    atomicAdd(&SMuGrad[wi], -dz * SSigma[wi]);
    atomicAdd(&DMappedU[k * Str + b], dz * SSigma[wi]);
  }
}

// Input-affine backward: InW[k] += sum_b DMappedU[k,b]*U[k,b]; InB[k] += sum_b DMappedU[k,b].
__global__ void InputAffineBackwardKernel(const float *DMappedU, const float *U,
                                          float *InWGrad, float *InBGrad,
                                          int In, int B, int Str) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= In) return;
  float gw = 0.0f, gb = 0.0f;
  for (int b = 0; b < B; ++b) {
    float d = DMappedU[k * Str + b];
    gw += d * U[k * Str + b];
    gb += d;
  }
  InWGrad[k] += gw; InBGrad[k] += gb;
}

// Adam helpers ------------------------------------------------------------

// Scale all grads in place (global clip scale).
__global__ void ScaleVec(float *G, float Scale, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) G[i] *= Scale;
}

// Adam step over one parameter buffer.
__global__ void AdamStep(float *Val, const float *Grad, float *M, float *V,
                         float B1, float B2, float BC1, float BC2, float Lr,
                         float Eps, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float g = Grad[i];
  M[i] = B1 * M[i] + (1.0f - B1) * g;
  V[i] = B2 * V[i] + (1.0f - B2) * g * g;
  float mhat = M[i] / BC1;
  float vhat = V[i] / BC2;
  Val[i] -= Lr * mhat / (sqrtf(vhat) + Eps);
}

// Refresh derived per-unit quantities: Gl=softplus(gleak), CmT=softplus(cm)/DtSub.
__global__ void RefreshDerivedKernel(const float *GleakRaw, const float *CmRaw,
                                     float *Gl, float *CmT, float DtSub, int S) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= S) return;
  Gl[j] = dSoftPlus(GleakRaw[j]);
  CmT[j] = dSoftPlus(CmRaw[j]) / DtSub;
}

// Per-(m,b) squared error for the loss/eval reduction; reduced on host.
__global__ void SqErrKernel(const float *Out, const float *Tg, float *Sq,
                            int Out_, int B, int Str) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Out_ * B) return;
  int m = idx / B, b = idx % B;
  int o = m * Str + b;
  float d = Out[o] - Tg[o];
  Sq[idx] = d * d;  // packed contiguous Out_*B for the host reduction
}

// ===========================================================================
// Device parameter bundle. Each parameter has Val/Grad/M/V on device.
// ===========================================================================

struct DParam {
  float *Val = nullptr, *Grad = nullptr, *M = nullptr, *V = nullptr;
  size_t N = 0;
  void Init(size_t n) {
    N = n;
    CUDA_CHECK(cudaMalloc(&Val, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Grad, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&M, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(Val, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(Grad, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(M, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(V, 0, N * sizeof(float)));
  }
  void UploadVal(const std::vector<float> &H) {
    CUDA_CHECK(cudaMemcpy(Val, H.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  }
  void ZeroGrad() { CUDA_CHECK(cudaMemset(Grad, 0, N * sizeof(float))); }
};

// ===========================================================================
// GPU LTC model: structural state on host, all tensors on device.
// ===========================================================================

struct Model {
  HP H;
  int S = 0, In = 0, Out = 0;
  Partition P;
  Masks Msk;
  cublasHandle_t Bl = nullptr;

  // Device parameters.
  DParam Wr, Sigma, Mu, Er;
  DParam Gleak, Vleak, Cm;
  DParam SW, SSigma, SMu, SEr;
  DParam InW, InB, OutW, OutB;
  std::vector<DParam *> All;

  // Device masks.
  uint8_t *dMaskRec = nullptr, *dMaskIn = nullptr;

  // Derived per-unit (device).
  float *dGl = nullptr, *dCmT = nullptr;

  // Batch-parallel tape: Vp holds [(U+1) unfolds] * [S*B] per timestep, over
  // the whole sequence: SeqLen * (U+1) * S * B floats. This is the BPTT tape.
  float *dVp = nullptr;       // SeqLen * (U+1) * S * B
  float *dOut = nullptr;      // SeqLen * Out * B
  float *dNumS = nullptr;     // SeqLen * S * B  (sensory num per step)
  float *dDenS = nullptr;     // SeqLen * S * B
  float *dMapped = nullptr;   // SeqLen * In * B (mapped inputs per step)
  float *dU = nullptr;        // SeqLen * In * B (raw inputs per step, col=sample)
  float *dTg = nullptr;       // SeqLen * Out * B (targets, col=sample)

  // Backward scratch.
  float *dGradH = nullptr;    // S*B
  float *dGcur = nullptr;     // S*B
  float *dGpre = nullptr;     // S*B
  float *dDNumS = nullptr;    // S*B
  float *dDDenS = nullptr;    // S*B
  float *dDMappedU = nullptr; // In*B
  float *dSq = nullptr;       // Out*B (squared error scratch)

  int BatchCap = 0;
  size_t AdamT = 0;

  void Build(const HP &Hp, std::mt19937_64 &Rng, cublasHandle_t Handle, int BatchCap_) {
    H = Hp; S = (int)H.Units; In = (int)H.InputDim; Out = (int)H.OutputDim;
    Bl = Handle; BatchCap = BatchCap_;
    P = ComputePartition(S, Out);
    Msk = BuildMasks(H, P, Rng);

    Wr.Init(S * S); Sigma.Init(S * S); Mu.Init(S * S); Er.Init(S * S);
    Gleak.Init(S); Vleak.Init(S); Cm.Init(S);
    SW.Init(In * S); SSigma.Init(In * S); SMu.Init(In * S); SEr.Init(In * S);
    InW.Init(In); InB.Init(In); OutW.Init(Out); OutB.Init(Out);
    All = {&Wr, &Sigma, &Mu, &Er, &Gleak, &Vleak, &Cm,
           &SW, &SSigma, &SMu, &SEr, &InW, &InB, &OutW, &OutB};

    CUDA_CHECK(cudaMalloc(&dMaskRec, (size_t)S * S));
    CUDA_CHECK(cudaMalloc(&dMaskIn, (size_t)S * In));
    CUDA_CHECK(cudaMemcpy(dMaskRec, Msk.Rec.data(), (size_t)S * S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMaskIn, Msk.In.data(), (size_t)S * In, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dGl, S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCmT, S * sizeof(float)));

    int U = (int)H.OdeUnfolds;
    size_t T = H.SeqLen;
    size_t SB = (size_t)S * BatchCap;
    CUDA_CHECK(cudaMalloc(&dVp, T * (U + 1) * SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, T * Out * BatchCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dNumS, T * SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDenS, T * SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMapped, T * In * BatchCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dU, T * In * BatchCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTg, T * Out * BatchCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradH, SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGcur, SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGpre, SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDNumS, SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDDenS, SB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDMappedU, (size_t)In * BatchCap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSq, (size_t)Out * BatchCap * sizeof(float)));

    InitParams(Rng);
  }

  void InitParams(std::mt19937_64 &Rng) {
    auto U = [&](float A, float B) { return std::uniform_real_distribution<float>(A, B)(Rng); };
    auto Pol = [&]() { return (std::uniform_int_distribution<int>(0, 1)(Rng) == 0) ? -1.0f : 1.0f; };
    std::vector<float> gleak(S), vleak(S), cm(S);
    for (int J = 0; J < S; ++J) {
      gleak[J] = U(0.001f, 1.0f); vleak[J] = U(-0.2f, 0.2f); cm[J] = U(0.4f, 0.6f);
    }
    std::vector<float> wr(S * S, 0), sigma(S * S, 0), mu(S * S, 0), er(S * S, 0);
    for (int Dst = 0; Dst < S; ++Dst)
      for (int Src = 0; Src < S; ++Src) {
        int I = Dst * S + Src;
        if (!Msk.Rec[I]) continue;
        wr[I] = U(0.001f, 1.0f); sigma[I] = U(3.0f, 8.0f); mu[I] = U(0.3f, 0.8f); er[I] = Pol();
      }
    std::vector<float> sw(In * S, 0), ssigma(In * S, 0), smu(In * S, 0), ser(In * S, 0);
    for (int Dst = 0; Dst < S; ++Dst)
      for (int K = 0; K < In; ++K) {
        int I = Dst * In + K;
        if (!Msk.In[I]) continue;
        sw[I] = U(0.001f, 1.0f); ssigma[I] = U(3.0f, 8.0f); smu[I] = U(0.3f, 0.8f); ser[I] = Pol();
      }
    std::vector<float> inw(In, 1.0f), inb(In, 0.0f), outw(Out, 1.0f), outb(Out, 0.0f);

    Gleak.UploadVal(gleak); Vleak.UploadVal(vleak); Cm.UploadVal(cm);
    Wr.UploadVal(wr); Sigma.UploadVal(sigma); Mu.UploadVal(mu); Er.UploadVal(er);
    SW.UploadVal(sw); SSigma.UploadVal(ssigma); SMu.UploadVal(smu); SEr.UploadVal(ser);
    InW.UploadVal(inw); InB.UploadVal(inb); OutW.UploadVal(outw); OutB.UploadVal(outb);
  }

  size_t DenseParamCount() const {
    return 4ull * S * S + 3ull * S + 4ull * In * S + 2ull * In + 2ull * Out;
  }

  void RefreshDerived() {
    float DtSub = H.Elapsed / (float)H.OdeUnfolds;
    RefreshDerivedKernel<<<Grid(S, kBlock), kBlock>>>(Gleak.Val, Cm.Val, dGl, dCmT, DtSub, S);
  }

  void ZeroGrads() { for (DParam *p : All) p->ZeroGrad(); }

  // Run the full forward over a minibatch of B sequences whose inputs/targets
  // are already laid out in dU / dTg as [T][In|Out][B] (col = sample). Fills
  // dVp (tape), dOut, dNumS, dDenS, dMapped. Initial state (Vp unfold 0 of t=0)
  // must be zeroed by the caller.
  void Forward(int B) {
    int U = (int)H.OdeUnfolds;
    size_t SB = (size_t)S * B;
    size_t Stride = (size_t)(U + 1) * S * BatchCap;   // per-timestep stride in dVp
    size_t SBcap = (size_t)S * BatchCap;
    for (size_t t = 0; t < H.SeqLen; ++t) {
      float *VpT = dVp + t * Stride;
      // carry previous step's final state into this step's unfold-0 slot.
      if (t > 0) {
        float *prevFinal = dVp + (t - 1) * Stride + (size_t)U * SBcap;
        // copy row-by-row is unnecessary: layouts match (S rows x BatchCap cols),
        // we only use first B cols. Copy the whole S*BatchCap block.
        CUDA_CHECK(cudaMemcpy(VpT, prevFinal, SBcap * sizeof(float), cudaMemcpyDeviceToDevice));
      }
      // Sensory for this step.
      const float *Ut = dU + t * (size_t)In * BatchCap;
      float *Mt = dMapped + t * (size_t)In * BatchCap;
      MapInput<<<Grid((size_t)In * B, kBlock), kBlock>>>(Ut, InW.Val, InB.Val, Mt, In, B, BatchCap);
      float *Nt = dNumS + t * SBcap;
      float *Dt = dDenS + t * SBcap;
      SensoryKernel<<<Grid(SB, kBlock), kBlock>>>(Mt, SW.Val, SSigma.Val, SMu.Val,
                                                  SEr.Val, dMaskIn, Nt, Dt, S, In, B, BatchCap);
      // Unfolds.
      for (int sub = 0; sub < U; ++sub) {
        const float *Cur = VpT + (size_t)sub * SBcap;
        float *Nxt = VpT + (size_t)(sub + 1) * SBcap;
        UnfoldKernel<<<Grid(SB, kBlock), kBlock>>>(Cur, Nxt, Wr.Val, Sigma.Val,
            Mu.Val, Er.Val, dMaskRec, dCmT, dGl, Vleak.Val, Nt, Dt, H.OdeEps, S, B, BatchCap);
      }
      // Output read for loss bookkeeping.
      const float *State = VpT + (size_t)U * SBcap;
      KindRanges R = Ranges(P);
      float *Ot = dOut + t * (size_t)Out * BatchCap;
      ReadOutputKernel<<<Grid((size_t)Out * B, kBlock), kBlock>>>(
          State, OutW.Val, OutB.Val, Ot, (int)R.MotorBeg, Out, B, BatchCap);
    }
  }

  // BPTT over the minibatch already forwarded into dVp/dOut. Accumulates grads.
  // Scale = 1/(B*T*Out).
  void Backward(int B, float Scale) {
    int U = (int)H.OdeUnfolds;
    size_t SBcap = (size_t)S * BatchCap;
    size_t Stride = (size_t)(U + 1) * S * BatchCap;
    float DtSub = H.Elapsed / (float)H.OdeUnfolds;
    KindRanges R = Ranges(P);
    size_t SB = (size_t)S * B;  // active element count for grids

    CUDA_CHECK(cudaMemset(dGradH, 0, SBcap * sizeof(float)));

    for (size_t tt = H.SeqLen; tt-- > 0;) {
      float *VpT = dVp + tt * Stride;
      const float *Ht = VpT + (size_t)U * SBcap;
      const float *Tg = dTg + tt * (size_t)Out * BatchCap;

      OutputBackwardKernel<<<Grid((size_t)Out * B, kBlock), kBlock>>>(
          Ht, Tg, OutW.Val, OutB.Val, dGradH, OutW.Grad, OutB.Grad, Scale,
          (int)R.MotorBeg, Out, S, B, BatchCap);

      // Gcur = GradH; reset DNumS/DDenS (full strided buffers).
      CUDA_CHECK(cudaMemcpy(dGcur, dGradH, SBcap * sizeof(float), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemset(dDNumS, 0, SBcap * sizeof(float)));
      CUDA_CHECK(cudaMemset(dDDenS, 0, SBcap * sizeof(float)));

      const float *Dt = dDenS + tt * SBcap;

      for (size_t sub = U; sub-- > 0;) {
        const float *Cur = VpT + sub * SBcap;
        const float *Nxt = VpT + (sub + 1) * SBcap;
        CUDA_CHECK(cudaMemset(dGpre, 0, SBcap * sizeof(float)));
        UnfoldBackwardKernel<<<Grid(SB, kBlock), kBlock>>>(
            Cur, Nxt, dGcur, dGpre, Wr.Val, Sigma.Val, Mu.Val, Er.Val, dMaskRec,
            dCmT, dGl, Vleak.Val, Dt, Cm.Val, Gleak.Val, DtSub, H.OdeEps,
            Wr.Grad, Sigma.Grad, Mu.Grad, Er.Grad, Cm.Grad, Gleak.Grad,
            Vleak.Grad, dDNumS, dDDenS, S, B, BatchCap);
        CUDA_CHECK(cudaMemcpy(dGcur, dGpre, SBcap * sizeof(float), cudaMemcpyDeviceToDevice));
      }

      // Sensory backward (uses accumulated DNumS/DDenS).
      const float *Mt = dMapped + tt * (size_t)In * BatchCap;
      CUDA_CHECK(cudaMemset(dDMappedU, 0, (size_t)In * BatchCap * sizeof(float)));
      SensoryBackwardKernel<<<Grid(SB, kBlock), kBlock>>>(
          Mt, SW.Val, SSigma.Val, SMu.Val, SEr.Val, dMaskIn, dDNumS, dDDenS,
          SW.Grad, SSigma.Grad, SMu.Grad, SEr.Grad, dDMappedU, S, In, B, BatchCap);
      const float *Ut = dU + tt * (size_t)In * BatchCap;
      InputAffineBackwardKernel<<<Grid(In, kBlock), kBlock>>>(
          dDMappedU, Ut, InW.Grad, InB.Grad, In, B, BatchCap);

      // Gcur (== grad wrt this step's input state) -> GradH for previous step.
      CUDA_CHECK(cudaMemcpy(dGradH, dGcur, SBcap * sizeof(float), cudaMemcpyDeviceToDevice));
    }
  }

  // Global L2-norm clip + Adam. Grad norm computed via cuBLAS Snrm2 over each
  // buffer (summed in quadrature on host), then a scale kernel + Adam kernel.
  void ClipAndAdam() {
    double SumSq = 0.0;
    for (DParam *p : All) {
      float nrm = 0.0f;
      CUBLAS_CHECK(cublasSnrm2(Bl, (int)p->N, p->Grad, 1, &nrm));
      SumSq += (double)nrm * nrm;
    }
    float Norm = (float)std::sqrt(SumSq);
    float Scale = 1.0f;
    if (H.GradClip > 0.0f && Norm > H.GradClip)
      Scale = H.GradClip / (Norm + 1e-6f);

    ++AdamT;
    float B1 = H.Beta1, B2 = H.Beta2;
    float BC1 = 1.0f - std::pow(B1, (float)AdamT);
    float BC2 = 1.0f - std::pow(B2, (float)AdamT);
    for (DParam *p : All) {
      if (Scale != 1.0f)
        ScaleVec<<<Grid(p->N, kBlock), kBlock>>>(p->Grad, Scale, p->N);
      AdamStep<<<Grid(p->N, kBlock), kBlock>>>(p->Val, p->Grad, p->M, p->V, B1, B2,
                                               BC1, BC2, H.Lr, H.AdamEps, p->N);
    }
  }
};

// ---------------------------------------------------------------------------
// Stage a minibatch of B sequences (timestep-major, sample-as-column) into the
// device dU/dTg buffers. Host scratch transposes (sample, t, feat) ->
// per-timestep [feat][sample].
// ---------------------------------------------------------------------------

void StageBatch(Model &M, const SineDataset &D, const std::vector<size_t> &Ids,
                size_t Beg, int B, std::vector<float> &HU, std::vector<float> &HT) {
  int In = M.In, Out = M.Out;
  size_t T = D.SeqLen;
  // HU layout: [t][feat][sample] over BatchCap cols (we fill first B).
  for (size_t t = 0; t < T; ++t)
    for (int b = 0; b < B; ++b) {
      size_t sid = Ids[Beg + b];
      const float *xrow = &D.X[(sid * T + t) * In];
      const float *yrow = &D.Y[(sid * T + t) * Out];
      for (int k = 0; k < In; ++k)
        HU[t * (size_t)In * M.BatchCap + (size_t)k * M.BatchCap + b] = xrow[k];
      for (int m = 0; m < Out; ++m)
        HT[t * (size_t)Out * M.BatchCap + (size_t)m * M.BatchCap + b] = yrow[m];
    }
  CUDA_CHECK(cudaMemcpy(M.dU, HU.data(), T * (size_t)In * M.BatchCap * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(M.dTg, HT.data(), T * (size_t)Out * M.BatchCap * sizeof(float),
                        cudaMemcpyHostToDevice));
}

// Eval MSE over a whole dataset (forward only), batched.
double EvalMse(Model &M, const SineDataset &D, std::vector<float> &HU,
               std::vector<float> &HT) {
  M.RefreshDerived();
  size_t T = D.SeqLen;
  int Out = M.Out, U = (int)M.H.OdeUnfolds;
  size_t SBcap = (size_t)M.S * M.BatchCap;
  std::vector<size_t> Ids(D.N);
  for (size_t i = 0; i < D.N; ++i) Ids[i] = i;
  std::vector<float> hSq((size_t)Out * M.BatchCap);
  double Sum = 0.0; size_t Count = 0;
  for (size_t beg = 0; beg < D.N; beg += M.BatchCap) {
    int B = (int)std::min((size_t)M.BatchCap, D.N - beg);
    StageBatch(M, D, Ids, beg, B, HU, HT);
    // zero initial state for unfold-0 of t=0.
    CUDA_CHECK(cudaMemset(M.dVp, 0, SBcap * sizeof(float)));
    M.Forward(B);
    // accumulate squared error over all t.
    for (size_t t = 0; t < T; ++t) {
      const float *Ot = M.dOut + t * (size_t)Out * M.BatchCap;
      const float *Tg = M.dTg + t * (size_t)Out * M.BatchCap;
      SqErrKernel<<<Grid((size_t)Out * B, kBlock), kBlock>>>(Ot, Tg, M.dSq, Out, B, M.BatchCap);
      CUDA_CHECK(cudaMemcpy(hSq.data(), M.dSq, (size_t)Out * B * sizeof(float),
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < Out * B; ++i) Sum += hSq[i];
      Count += (size_t)Out * B;
    }
    (void)U;
  }
  return Count == 0 ? 0.0 : Sum / (double)Count;
}

bench::EdgeSet LiveEdgeSet(const Model &M) {
  bench::EdgeSet Set;
  for (int Dst = 0; Dst < M.S; ++Dst) {
    for (int Src = 0; Src < M.S; ++Src)
      if (M.Msk.Rec[Dst * M.S + Src])
        Set.insert(bench::PackEdge(0, (uint32_t)Src, (uint32_t)Dst));
    for (int Src = 0; Src < M.In; ++Src)
      if (M.Msk.In[Dst * M.In + Src])
        Set.insert(bench::PackEdge(1, (uint32_t)Src, (uint32_t)Dst));
  }
  return Set;
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.Units      = (size_t)Args.GetInt("units",       (int)H.Units);
  H.SeqLen     = (size_t)Args.GetInt("seq-len",     (int)H.SeqLen);
  H.TrainSeqs  = (size_t)Args.GetInt("train-seqs",  (int)H.TrainSeqs);
  H.ValSeqs    = (size_t)Args.GetInt("val-seqs",    (int)H.ValSeqs);
  H.TestSeqs   = (size_t)Args.GetInt("test-seqs",   (int)H.TestSeqs);
  H.Epochs     = (size_t)Args.GetInt("epochs",      (int)H.Epochs);
  H.Batch      = (size_t)Args.GetInt("batch",       (int)H.Batch);
  H.OdeUnfolds = (size_t)Args.GetInt("ode-unfolds", (int)H.OdeUnfolds);
  H.KSparse    = (size_t)Args.GetInt("k-sparse",    (int)H.KSparse);
  H.KRec       = (size_t)Args.GetInt("k-rec",       (int)H.KRec);
  H.KFb        = (size_t)Args.GetInt("k-fb",        (int)H.KFb);
  H.NoiseStd   = Args.GetFloat("noise-std", H.NoiseStd);
  H.Elapsed    = Args.GetFloat("elapsed",   H.Elapsed);
  H.Lr         = Args.GetFloat("lr",        H.Lr);
  H.GradClip   = Args.GetFloat("grad-clip", H.GradClip);
  if (Args.Quick) {
    H.Epochs    = std::max<size_t>(1, H.Epochs / 4);
    H.TrainSeqs = std::max<size_t>(8, H.TrainSeqs / 8);
    H.ValSeqs   = std::max<size_t>(4, H.ValSeqs / 4);
    H.TestSeqs  = std::max<size_t>(4, H.TestSeqs / 4);
    H.SeqLen    = std::max<size_t>(16, H.SeqLen / 2);
  }

  uint64_t Seed = (uint64_t)Args.Seed * 7919ull + 13ull;
  std::mt19937_64 Rng(Seed);

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));

  int BatchCap = (int)H.Batch;

  bench::MemoryProbe MP;
  MP.Start();

  Model Mod;
  Mod.Build(H, Rng, Bl, BatchCap);
  MP.EndWeights();

  size_t NumEdges = Mod.Msk.LiveEdges;
  size_t NumParams = Mod.DenseParamCount();
  std::cout << "[info] cuda units=" << H.Units << "  live_edges=" << NumEdges
            << "  dense_params=" << NumParams
            << "  partition Ns=" << Mod.P.Ns << " Ni=" << Mod.P.Ni
            << " Nc=" << Mod.P.Nc << " Nm=" << Mod.P.Nm << "\n";
  std::cout << "[info] task=sine  algo=LTC+BPTT(cuBLAS)  seq_len=" << H.SeqLen
            << "  train=" << H.TrainSeqs << "  val=" << H.ValSeqs
            << "  test=" << H.TestSeqs << "  epochs=" << H.Epochs
            << "  batch=" << H.Batch << "  ode_unfolds=" << H.OdeUnfolds
            << "  lr=" << H.Lr << "\n";

  auto Train = MakeSine(H.TrainSeqs, H.SeqLen, H.NoiseStd, Seed ^ 0xa1a1ull);
  auto Val   = MakeSine(H.ValSeqs,   H.SeqLen, H.NoiseStd, Seed ^ 0xb2b2ull);
  auto Test  = MakeSine(H.TestSeqs,  H.SeqLen, H.NoiseStd, Seed ^ 0xc3c3ull);
  MP.EndDataset();

  auto [HistPath, SummaryPath, LogPath] = bench::OutputPaths(Args, "ccwc_ncp");
  (void)LogPath;
  bench::StructuralLog Log(HistPath);

  // Host staging buffers (sized for BatchCap; reused for train + eval).
  std::vector<float> HU(H.SeqLen * (size_t)Mod.In * BatchCap, 0.0f);
  std::vector<float> HT(H.SeqLen * (size_t)Mod.Out * BatchCap, 0.0f);
  std::vector<float> hSq((size_t)Mod.Out * BatchCap, 0.0f);

  double InitVal  = EvalMse(Mod, Val, HU, HT);
  double InitTest = EvalMse(Mod, Test, HU, HT);
  bench::EdgeSet Edges0 = LiveEdgeSet(Mod);
  Log.Log(0, H.Units, NumEdges, &Edges0, &InitVal,
          {{"epoch", 0.0}, {"train_mse", 0.0}, {"val_mse", InitVal}, {"test_mse", InitTest}});
  std::cout << "[ep   0] val_mse=" << InitVal << "  test_mse=" << InitTest << "\n";

  size_t S = Mod.S, U = H.OdeUnfolds, Out = Mod.Out;
  size_t SBcap = (size_t)S * BatchCap;
  (void)U;

  std::vector<size_t> Perm(H.TrainSeqs);
  for (size_t I = 0; I < Perm.size(); ++I) Perm[I] = I;
  std::mt19937 ShuffleRng((uint32_t)Args.Seed);

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), ShuffleRng);
    double TrainSumSq = 0.0; size_t TrainCount = 0;

    for (size_t Beg = 0; Beg < H.TrainSeqs; Beg += H.Batch) {
      int Bsize = (int)std::min(H.Batch, H.TrainSeqs - Beg);
      float Scale = 1.0f / (float)((size_t)Bsize * Train.SeqLen * Out);

      Mod.ZeroGrads();
      Mod.RefreshDerived();
      StageBatch(Mod, Train, Perm, Beg, Bsize, HU, HT);
      CUDA_CHECK(cudaMemset(Mod.dVp, 0, SBcap * sizeof(float)));

      Timer.Tick();
      Mod.Forward(Bsize);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkForward();

      // Train MSE bookkeeping (host reduce over per-step squared error).
      for (size_t t = 0; t < Train.SeqLen; ++t) {
        const float *Ot = Mod.dOut + t * (size_t)Out * BatchCap;
        const float *Tg = Mod.dTg + t * (size_t)Out * BatchCap;
        SqErrKernel<<<Grid((size_t)Out * Bsize, kBlock), kBlock>>>(Ot, Tg, Mod.dSq, (int)Out, Bsize, BatchCap);
        CUDA_CHECK(cudaMemcpy(hSq.data(), Mod.dSq, (size_t)Out * Bsize * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < (int)Out * Bsize; ++i) TrainSumSq += hSq[i];
        TrainCount += (size_t)Out * Bsize;
      }
      Timer.MarkLoss();

      Mod.Backward(Bsize, Scale);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkBackward();

      Mod.ClipAndAdam();
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer.MarkUpdate();
      Timer.StepDone();
    }

    double TrainMse = TrainCount == 0 ? 0.0 : TrainSumSq / (double)TrainCount;
    double ValMse  = EvalMse(Mod, Val, HU, HT);
    double TestMse = EvalMse(Mod, Test, HU, HT);

    bench::EdgeSet Edges = LiveEdgeSet(Mod);
    Log.Log(Ep, H.Units, NumEdges, &Edges, &ValMse,
            {{"epoch", (double)Ep}, {"train_mse", TrainMse},
             {"val_mse", ValMse}, {"test_mse", TestMse}});
    std::cout << "[ep " << std::setw(3) << Ep << "] "
              << "train_mse=" << TrainMse << "  val_mse=" << ValMse
              << "  test_mse=" << TestMse << "\n";
  }
  double Wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();

  double FinalTest = EvalMse(Mod, Test, HU, HT);
  Log.Flush();

  bench::SummaryWriter Sum;
  Sum.Set("workload", std::string{"06_ccwc_ncp"});
  Sum.Set("dataset", std::string{"synthetic-sine"});
  Sum.Set("task",    std::string{"sine"});
  Sum.Set("backend", std::string{"cuda-ltc-bptt"});
  Sum.Set("metric_kind", std::string{"mse"});
  Sum.Set("n_in",    (int)H.InputDim);
  Sum.Set("n_out",   (int)H.OutputDim);
  Sum.Set("n_units", (int)H.Units);
  Sum.Set("n_edges", (int)NumEdges);
  Sum.Set("n_params", (long long)NumParams);
  Sum.Set("seq_len", (int)H.SeqLen);
  Sum.Set("epochs",  (int)H.Epochs);
  Sum.Set("batch",   (int)H.Batch);
  Sum.Set("ode_unfolds", (int)H.OdeUnfolds);
  Sum.Set("lr",      H.Lr);
  Sum.Set("elapsed", H.Elapsed);
  Sum.Set("k_sparse", (int)H.KSparse);
  Sum.Set("k_rec",    (int)H.KRec);
  Sum.Set("k_fb",     (int)H.KFb);
  Sum.Set("wall_seconds", Wall);
  Sum.Set("val_mse_final",
          Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss);
  Sum.Set("test_mse", FinalTest);
  Sum.Set("seed",     Args.Seed);
  Timer.WriteSummary(Sum, Wall);
  MP.WriteSummary(Sum);
  Sum.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  final_test_mse=" << FinalTest << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  cublasDestroy(Bl);
  return 0;
}
