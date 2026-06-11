// Workload 7 — Compact C. elegans-Wired Controller, raw C++ (LTC + BPTT).
//
// *** This is the "same algorithm as PyTorch model A" rendering. ***
//
// The other two ports in this benchmark (06_ccwc_ncp/plastix/ and
// the historical e-prop C++ port) deliberately run a *different* algorithm: a
// leaky-tanh "LTC-lite" RNN trained with one-step e-prop (O(1) state, no
// temporal tape). That made them fast but was not what the PyTorch reference
// computes.
//
// This file instead reimplements, in raw C++, the actual ncps "model A":
//
//   * the real Liquid-Time-Constant (LTC) ODE cell — sigmoid synaptic
//     conductances with per-synapse (w, sigma, mu, erev), a leak conductance
//     (gleak, vleak), membrane capacitance (cm), and `ode_unfolds`
//     semi-implicit Euler sub-steps per timestep, with affine input/output
//     maps — over the same AutoNCP-style sparse wiring (sensory -> inter ->
//     command -> motor, command recurrence, motor -> command feedback);
//
//   * trained with true Backprop-Through-Time: the full sequence is unrolled,
//     a per-sample reverse-mode gradient is computed by hand through every
//     unfold of the ODE solver, gradients are accumulated across a minibatch,
//     clipped to a global L2 norm, and applied with Adam.
//
// The parameterisation is identical in *shape* to ncps `LTC(AutoNCP(units,
// out))`: at units=32, in=2, out=2 the trainable tensor count is
//   4*S^2 (w,sigma,mu,erev) + 3*S (gleak,vleak,cm) + 4*in*S (sensory x4)
//   + 2*in + 2*out (affine)  =  4096 + 96 + 256 + 4 + 4  =  4456,
// matching PyTorch's `count_params(model_A)` exactly. We do NOT claim
// bit-identical weights (random init streams differ, and ncps' AutoNCP wiring
// algorithm differs in detail from the sampler here) — the comparison is at
// the algorithm / model-shape level, the same standard the rest of the suite
// uses across pytorch/plastix/cpp.
//
// Why keep this around: it isolates the *framework* axis from the *algorithm*
// axis. The e-prop ports answered "is a one-step local rule cheap?"; this port
// answers "if C++ runs the very same BPTT-through-an-ODE that PyTorch runs, how
// much of PyTorch's cost was the framework (Python dispatch, autograd tape,
// dense masked tensors) versus the algorithm itself?"
//
// Output schema (.history.jsonl + .summary.csv) mirrors the other ports so the
// orchestrator can drive all three uniformly.

#include "cpp/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Hyperparameters (sine-task defaults match 06_ccwc_ncp/pytorch).
// ---------------------------------------------------------------------------

struct HP {
  size_t Units      = 32;     // total NCP neurons (sensory+inter+command+motor)
  size_t InputDim   = 2;
  size_t OutputDim  = 2;
  size_t KSparse    = 4;      // fan-in for inter and command layers
  size_t KRec       = 4;      // fan-in for command -> command recurrence
  size_t KFb        = 2;      // fan-in for motor -> command feedback
  size_t SeqLen     = 64;
  size_t TrainSeqs  = 512;
  size_t ValSeqs    = 128;
  size_t TestSeqs   = 128;
  size_t Epochs     = 20;
  size_t Batch      = 64;
  size_t OdeUnfolds = 6;      // semi-implicit Euler sub-steps per timestep
  float  NoiseStd   = 0.1f;
  float  Elapsed    = 1.0f;   // ODE elapsed_time per step (ncps default ts=1)
  float  Lr         = 1e-3f;  // Adam
  float  Beta1      = 0.9f;
  float  Beta2      = 0.999f;
  float  AdamEps    = 1e-8f;
  float  GradClip   = 1.0f;   // global grad-norm clip (matches PyTorch)
  float  OdeEps     = 1e-8f;  // divide-by-zero guard in the ODE solver
};

enum Kind : uint8_t { KSensory = 0, KInter = 1, KCommand = 2, KMotor = 3 };

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

inline float SoftPlus(float X) {
  // log(1 + e^x), numerically guarded. d/dx softplus(x) = sigmoid(x).
  return X > 20.0f ? X : std::log1p(std::exp(X));
}
inline float Sigmoid(float X) {
  if (X >= 0.0f) {
    float Z = std::exp(-X);
    return 1.0f / (1.0f + Z);
  }
  float Z = std::exp(X);
  return Z / (1.0f + Z);
}

// ---------------------------------------------------------------------------
// AutoNCP-style partition + wiring masks (shape-parity with the other ports).
// ---------------------------------------------------------------------------

struct Partition {
  size_t Ns, Ni, Nc, Nm;
  std::vector<Kind> Kinds;            // length Units; per-unit role
};

Partition ComputePartition(size_t Units, size_t Motors) {
  Partition P{};
  P.Nm = Motors;
  size_t Rest = Units > Motors ? Units - Motors : 0;
  P.Ns = std::max<size_t>(1, Rest / 3);
  size_t After = Rest > P.Ns ? Rest - P.Ns : 0;
  P.Ni = After / 2;
  P.Nc = After - P.Ni;
  if (P.Nc == 0) {
    P.Nc = 1;
    if (P.Ni > 0) --P.Ni;
  }
  P.Kinds.resize(Units);
  size_t Idx = 0;
  for (size_t I = 0; I < P.Ns; ++I) P.Kinds[Idx++] = KSensory;
  for (size_t I = 0; I < P.Ni; ++I) P.Kinds[Idx++] = KInter;
  for (size_t I = 0; I < P.Nc; ++I) P.Kinds[Idx++] = KCommand;
  for (size_t I = 0; I < P.Nm; ++I) P.Kinds[Idx++] = KMotor;
  return P;
}

struct KindRanges {
  size_t SenseBeg, SenseEnd;
  size_t InterBeg, InterEnd;
  size_t CmdBeg,   CmdEnd;
  size_t MotorBeg, MotorEnd;
};

KindRanges Ranges(const Partition &P) {
  KindRanges R{};
  R.SenseBeg = 0;             R.SenseEnd = P.Ns;
  R.InterBeg = R.SenseEnd;    R.InterEnd = R.InterBeg + P.Ni;
  R.CmdBeg   = R.InterEnd;    R.CmdEnd   = R.CmdBeg + P.Nc;
  R.MotorBeg = R.CmdEnd;      R.MotorEnd = R.MotorBeg + P.Nm;
  return R;
}

std::vector<size_t> SampleK(size_t Beg, size_t End, size_t K,
                            size_t Forbidden, std::mt19937_64 &Rng) {
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
  std::vector<uint8_t> Rec;       // Units x Units, row-major (row = dst)
  std::vector<uint8_t> In;        // Units x InputDim (row = dst)
  size_t LiveEdges = 0;
};

Masks BuildMasks(const HP &H, const Partition &P, std::mt19937_64 &Rng) {
  KindRanges R = Ranges(P);
  Masks M;
  M.Rec.assign(H.Units * H.Units, 0);
  M.In.assign(H.Units * H.InputDim, 0);

  auto EdgeRec = [&](size_t Src, size_t Dst) {
    M.Rec[Dst * H.Units + Src] = 1; ++M.LiveEdges;
  };
  auto EdgeIn = [&](size_t Src, size_t Dst) {
    M.In[Dst * H.InputDim + Src] = 1; ++M.LiveEdges;
  };

  // (1) Inputs -> Sensory: dense.
  for (size_t Src = 0; Src < H.InputDim; ++Src)
    for (size_t Dst = R.SenseBeg; Dst < R.SenseEnd; ++Dst)
      EdgeIn(Src, Dst);

  // (2) Sensory -> Inter: KSparse fan-in.
  if (P.Ni > 0 && P.Ns > 0)
    for (size_t Dst = R.InterBeg; Dst < R.InterEnd; ++Dst)
      for (auto Src : SampleK(R.SenseBeg, R.SenseEnd, H.KSparse, ~size_t{0}, Rng))
        EdgeRec(Src, Dst);

  // (3) Inter -> Command (or Sensory -> Command if Ni == 0).
  if (P.Nc > 0) {
    size_t SrcBeg = P.Ni > 0 ? R.InterBeg : R.SenseBeg;
    size_t SrcEnd = P.Ni > 0 ? R.InterEnd : R.SenseEnd;
    for (size_t Dst = R.CmdBeg; Dst < R.CmdEnd; ++Dst)
      for (auto Src : SampleK(SrcBeg, SrcEnd, H.KSparse, ~size_t{0}, Rng))
        EdgeRec(Src, Dst);
  }

  // (4) Command -> Command: recurrent. Self-loops excluded.
  if (P.Nc > 1)
    for (size_t Dst = R.CmdBeg; Dst < R.CmdEnd; ++Dst)
      for (auto Src : SampleK(R.CmdBeg, R.CmdEnd, H.KRec, Dst, Rng))
        EdgeRec(Src, Dst);

  // (5) Command -> Motor: dense.
  if (P.Nc > 0)
    for (size_t Dst = R.MotorBeg; Dst < R.MotorEnd; ++Dst)
      for (size_t Src = R.CmdBeg; Src < R.CmdEnd; ++Src)
        EdgeRec(Src, Dst);

  // (6) Motor -> Command: feedback.
  if (P.Nc > 0 && H.KFb > 0)
    for (size_t Src = R.MotorBeg; Src < R.MotorEnd; ++Src)
      for (auto Dst : SampleK(R.CmdBeg, R.CmdEnd, H.KFb, ~size_t{0}, Rng))
        EdgeRec(Src, Dst);

  return M;
}

// ---------------------------------------------------------------------------
// Sine dataset (noisy input, clean next-step target). Shape-parity with the
// PyTorch make_sine_dataset; seeded independently.
// ---------------------------------------------------------------------------

struct SineDataset {
  std::vector<std::vector<float>> X;   // N seqs, each SeqLen x InDim row-major
  std::vector<std::vector<float>> Y;   // N seqs, each SeqLen x OutDim
  size_t SeqLen = 0;
  size_t InDim = 2;
  size_t OutDim = 2;
};

SineDataset MakeSine(size_t N, size_t SeqLen, float NoiseStd, uint64_t Seed) {
  SineDataset D;
  D.SeqLen = SeqLen;
  D.X.resize(N); D.Y.resize(N);
  std::mt19937_64 Rng(Seed);
  std::uniform_real_distribution<float> Freq(0.5f, 2.0f);
  std::uniform_real_distribution<float> Phase(0.0f, 6.28318530718f);
  std::normal_distribution<float> Noise(0.0f, NoiseStd);
  for (size_t I = 0; I < N; ++I) {
    float F = Freq(Rng);
    float Ph = Phase(Rng);
    D.X[I].assign(SeqLen * D.InDim, 0.0f);
    D.Y[I].assign(SeqLen * D.OutDim, 0.0f);
    for (size_t T = 0; T < SeqLen; ++T) {
      float A0 = F * (6.28318530718f * static_cast<float>(T) / SeqLen) + Ph;
      float A1 = F * (6.28318530718f * static_cast<float>(T + 1) / SeqLen) + Ph;
      D.X[I][T * 2 + 0] = std::sin(A0) + Noise(Rng);
      D.X[I][T * 2 + 1] = std::cos(A0) + Noise(Rng);
      D.Y[I][T * 2 + 0] = std::sin(A1);
      D.Y[I][T * 2 + 1] = std::cos(A1);
    }
  }
  return D;
}

// ---------------------------------------------------------------------------
// A trainable parameter array: value + gradient + Adam moments.
// ---------------------------------------------------------------------------

struct Param {
  std::vector<float> Val, Grad, M, V;
  void Init(size_t N) {
    Val.assign(N, 0.0f); Grad.assign(N, 0.0f);
    M.assign(N, 0.0f);   V.assign(N, 0.0f);
  }
};

// ---------------------------------------------------------------------------
// LTC model — full ncps-style ODE cell, dense-with-mask, trained by BPTT.
// ---------------------------------------------------------------------------

struct Model {
  HP H;
  size_t S = 0;          // = Units
  size_t In = 0;
  size_t Out = 0;
  Partition P;
  Masks Msk;

  // Recurrent synapse params, indexed idx = dst*S + src (row = dst).
  Param Wr, Sigma, Mu, Er;
  // Per-neuron leak / capacitance.
  Param Gleak, Vleak, Cm;
  // Sensory synapse params, indexed idx = dst*In + feature.
  Param SW, SSigma, SMu, SEr;
  // Affine input/output maps.
  Param InW, InB, OutW, OutB;

  std::vector<Param *> All;     // registry for generic Adam / clip / zero

  // Per-minibatch derived (param-only) scratch.
  std::vector<float> Gl, CmT;   // softplus(gleak), softplus(cm)/(elapsed/unfolds)

  // Reusable backward scratch (single-threaded, sequence-serial).
  std::vector<float> Den, DenS, NumS;
  std::vector<float> Gcur, Gpre, GradH, DNumS, DDenS;
  std::vector<float> MappedU, DMappedU;

  size_t AdamT = 0;

  // -------- construction --------
  void Build(const HP &Hp, std::mt19937_64 &Rng) {
    H = Hp; S = H.Units; In = H.InputDim; Out = H.OutputDim;
    P = ComputePartition(S, Out);
    Msk = BuildMasks(H, P, Rng);

    Wr.Init(S * S); Sigma.Init(S * S); Mu.Init(S * S); Er.Init(S * S);
    Gleak.Init(S); Vleak.Init(S); Cm.Init(S);
    SW.Init(In * S); SSigma.Init(In * S); SMu.Init(In * S); SEr.Init(In * S);
    InW.Init(In); InB.Init(In); OutW.Init(Out); OutB.Init(Out);

    All = {&Wr, &Sigma, &Mu, &Er, &Gleak, &Vleak, &Cm,
           &SW, &SSigma, &SMu, &SEr, &InW, &InB, &OutW, &OutB};

    Gl.assign(S, 0.0f); CmT.assign(S, 0.0f);
    Den.assign(S, 0.0f); DenS.assign(S, 0.0f); NumS.assign(S, 0.0f);
    Gcur.assign(S, 0.0f); Gpre.assign(S, 0.0f); GradH.assign(S, 0.0f);
    DNumS.assign(S, 0.0f); DDenS.assign(S, 0.0f);
    MappedU.assign(In, 0.0f); DMappedU.assign(In, 0.0f);

    InitParams(Rng);
  }

  // ncps default init ranges (raw values; positive ones pass through softplus).
  void InitParams(std::mt19937_64 &Rng) {
    auto U = [&](float A, float B) {
      return std::uniform_real_distribution<float>(A, B)(Rng);
    };
    auto Pol = [&]() {
      return (std::uniform_int_distribution<int>(0, 1)(Rng) == 0) ? -1.0f : 1.0f;
    };
    for (size_t J = 0; J < S; ++J) {
      Gleak.Val[J] = U(0.001f, 1.0f);
      Vleak.Val[J] = U(-0.2f, 0.2f);
      Cm.Val[J]    = U(0.4f, 0.6f);
    }
    for (size_t Dst = 0; Dst < S; ++Dst)
      for (size_t Src = 0; Src < S; ++Src) {
        size_t I = Dst * S + Src;
        if (!Msk.Rec[I]) continue;
        Wr.Val[I]    = U(0.001f, 1.0f);
        Sigma.Val[I] = U(3.0f, 8.0f);
        Mu.Val[I]    = U(0.3f, 0.8f);
        Er.Val[I]    = Pol();
      }
    for (size_t Dst = 0; Dst < S; ++Dst)
      for (size_t K = 0; K < In; ++K) {
        size_t I = Dst * In + K;
        if (!Msk.In[I]) continue;
        SW.Val[I]     = U(0.001f, 1.0f);
        SSigma.Val[I] = U(3.0f, 8.0f);
        SMu.Val[I]    = U(0.3f, 0.8f);
        SEr.Val[I]    = Pol();
      }
    for (size_t K = 0; K < In; ++K)  { InW.Val[K] = 1.0f; InB.Val[K] = 0.0f; }
    for (size_t M = 0; M < Out; ++M) { OutW.Val[M] = 1.0f; OutB.Val[M] = 0.0f; }
  }

  size_t DenseParamCount() const {
    return 4 * S * S + 3 * S + 4 * In * S + 2 * In + 2 * Out;
  }

  // Param-only derived quantities; recompute once per minibatch / eval pass.
  void RefreshDerived() {
    float DtSub = H.Elapsed / static_cast<float>(H.OdeUnfolds);
    for (size_t J = 0; J < S; ++J) {
      Gl[J]  = SoftPlus(Gleak.Val[J]);
      CmT[J] = SoftPlus(Cm.Val[J]) / DtSub;
    }
  }

  // Sensory pathway is loop-invariant across the ODE unfolds, so precompute its
  // numerator/denominator contribution once per timestep. Fills NumS_, DenS_.
  void Sensory(const float *Uin) {
    for (size_t K = 0; K < In; ++K)
      MappedU[K] = Uin[K] * InW.Val[K] + InB.Val[K];
    for (size_t J = 0; J < S; ++J) {
      float NumAcc = 0.0f, DenAcc = 0.0f;
      for (size_t K = 0; K < In; ++K) {
        size_t I = J * In + K;
        if (!Msk.In[I]) continue;
        float G = Sigmoid(SSigma.Val[I] * (MappedU[K] - SMu.Val[I]));
        float A = SoftPlus(SW.Val[I]) * G;
        NumAcc += A * SEr.Val[I];
        DenAcc += A;
      }
      NumS[J] = NumAcc; DenS[J] = DenAcc;
    }
  }

  // One LTC cell step. Reads VpTape[0..S) as the previous state, writes the
  // per-unfold state into VpTape[(s+1)*S .. ]; the final state is the last
  // block. Sensory() must have run for this timestep's input first.
  void CellForward(float *VpTape) {
    size_t U = H.OdeUnfolds;
    for (size_t Sub = 0; Sub < U; ++Sub) {
      const float *Cur = VpTape + Sub * S;
      float *Nxt = VpTape + (Sub + 1) * S;
      for (size_t J = 0; J < S; ++J) {
        float Num = CmT[J] * Cur[J] + Gl[J] * Vleak.Val[J] + NumS[J];
        float Den_ = CmT[J] + Gl[J] + DenS[J];
        for (size_t Src = 0; Src < S; ++Src) {
          size_t I = J * S + Src;
          if (!Msk.Rec[I]) continue;
          float A = SoftPlus(Wr.Val[I]) *
                    Sigmoid(Sigma.Val[I] * (Cur[Src] - Mu.Val[I]));
          Num += A * Er.Val[I];
          Den_ += A;
        }
        Nxt[J] = Num / (Den_ + H.OdeEps);
      }
    }
  }

  void ReadOutput(const float *State, float *OutBuf) const {
    KindRanges R = Ranges(P);
    for (size_t M = 0; M < Out; ++M)
      OutBuf[M] = State[R.MotorBeg + M] * OutW.Val[M] + OutB.Val[M];
  }

  // -------- gradient bookkeeping --------
  void ZeroGrads() {
    for (Param *Pb : All)
      std::fill(Pb->Grad.begin(), Pb->Grad.end(), 0.0f);
  }

  void ClipAndAdam() {
    // Global L2-norm clip (matches torch.nn.utils.clip_grad_norm_).
    double SumSq = 0.0;
    for (Param *Pb : All)
      for (float G : Pb->Grad) SumSq += static_cast<double>(G) * G;
    float Norm = static_cast<float>(std::sqrt(SumSq));
    float Scale = 1.0f;
    if (H.GradClip > 0.0f && Norm > H.GradClip)
      Scale = H.GradClip / (Norm + 1e-6f);

    ++AdamT;
    float B1 = H.Beta1, B2 = H.Beta2;
    float BC1 = 1.0f - std::pow(B1, static_cast<float>(AdamT));
    float BC2 = 1.0f - std::pow(B2, static_cast<float>(AdamT));
    for (Param *Pb : All) {
      for (size_t I = 0; I < Pb->Val.size(); ++I) {
        float G = Pb->Grad[I] * Scale;
        Pb->M[I] = B1 * Pb->M[I] + (1.0f - B1) * G;
        Pb->V[I] = B2 * Pb->V[I] + (1.0f - B2) * G * G;
        float MHat = Pb->M[I] / BC1;
        float VHat = Pb->V[I] / BC2;
        Pb->Val[I] -= H.Lr * MHat / (std::sqrt(VHat) + H.AdamEps);
      }
    }
  }

  // BPTT over one sample. `VpAll` is the forward tape for this sample
  // (length SeqLen*(U+1)*S), `OutAll` its per-step outputs (SeqLen*Out),
  // `Xseq`/`Yseq` the input/target streams, `Scale` = 1/(B*T*Out) so that
  // accumulating across the minibatch yields the mean-MSE gradient.
  void Backward(const float *VpAll, const float *OutAll, const float *Xseq,
                const float *Yseq, size_t T, float Scale) {
    size_t U = H.OdeUnfolds;
    size_t Stride = (U + 1) * S;
    float DtSub = H.Elapsed / static_cast<float>(H.OdeUnfolds);
    KindRanges R = Ranges(P);

    std::fill(GradH.begin(), GradH.end(), 0.0f);

    for (size_t Tt = T; Tt-- > 0;) {
      const float *Vp = VpAll + Tt * Stride;       // per-unfold states
      const float *Ht = Vp + U * S;                // final state this step
      const float *Out_ = OutAll + Tt * Out;
      const float *Tg = Yseq + Tt * Out;
      const float *Uin = Xseq + Tt * In;

      // dL/d(output affine) and seed grad into the motor units' final state.
      for (size_t M = 0; M < Out; ++M) {
        float DOut = 2.0f * (Out_[M] - Tg[M]) * Scale;
        size_t MUnit = R.MotorBeg + M;
        GradH[MUnit] += DOut * OutW.Val[M];
        OutW.Grad[M] += DOut * Ht[MUnit];
        OutB.Grad[M] += DOut;
      }

      // Recompute the sensory contribution for this step (loop-invariant
      // across unfolds); also gives DenS used in the den recomputation.
      Sensory(Uin);

      std::copy(GradH.begin(), GradH.end(), Gcur.begin());  // grad wrt Ht
      std::fill(DNumS.begin(), DNumS.end(), 0.0f);
      std::fill(DDenS.begin(), DDenS.end(), 0.0f);

      // Reverse over the ODE unfolds.
      for (size_t Sub = U; Sub-- > 0;) {
        const float *Cur = Vp + Sub * S;            // input state of this unfold
        const float *Nxt = Vp + (Sub + 1) * S;      // output state (= vnew)
        std::fill(Gpre.begin(), Gpre.end(), 0.0f);

        for (size_t J = 0; J < S; ++J) {
          // Recompute the denominator for this unfold/dst.
          float DenJ = CmT[J] + Gl[J] + DenS[J];
          for (size_t Src = 0; Src < S; ++Src) {
            size_t I = J * S + Src;
            if (!Msk.Rec[I]) continue;
            DenJ += SoftPlus(Wr.Val[I]) *
                    Sigmoid(Sigma.Val[I] * (Cur[Src] - Mu.Val[I]));
          }
          float Inv = 1.0f / (DenJ + H.OdeEps);
          float DNum = Gcur[J] * Inv;
          float DDen = -Gcur[J] * Nxt[J] * Inv;

          // Membrane capacitance term: num += cm_t*cur[j], den += cm_t.
          float DCmT = DNum * Cur[J] + DDen;
          Cm.Grad[J] += DCmT * Sigmoid(Cm.Val[J]) / DtSub;
          Gpre[J] += DNum * CmT[J];

          // Leak term: num += gl*vleak, den += gl.
          float DGl = DNum * Vleak.Val[J] + DDen;
          Gleak.Grad[J] += DGl * Sigmoid(Gleak.Val[J]);
          Vleak.Grad[J] += DNum * Gl[J];

          // Sensory contribution accumulates (back-propped once, below).
          DNumS[J] += DNum;
          DDenS[J] += DDen;

          // Recurrent synapses.
          for (size_t Src = 0; Src < S; ++Src) {
            size_t I = J * S + Src;
            if (!Msk.Rec[I]) continue;
            float Wraw = Wr.Val[I];
            float Spw = SoftPlus(Wraw);
            float Z = Sigma.Val[I] * (Cur[Src] - Mu.Val[I]);
            float Gs = Sigmoid(Z);
            float A = Spw * Gs;

            float DA = DNum * Er.Val[I] + DDen;
            Er.Grad[I] += DNum * A;
            Wr.Grad[I] += (DA * Gs) * Sigmoid(Wraw);     // through softplus
            float DG = DA * Spw;
            float DZ = DG * Gs * (1.0f - Gs);            // through sigmoid
            Sigma.Grad[I] += DZ * (Cur[Src] - Mu.Val[I]);
            Mu.Grad[I] += -DZ * Sigma.Val[I];
            Gpre[Src] += DZ * Sigma.Val[I];
          }
        }
        std::copy(Gpre.begin(), Gpre.end(), Gcur.begin());
      }
      // After the unfold loop, Gcur = grad wrt this step's input state =
      // grad wrt the previous timestep's final state.

      // Sensory backward (uses accumulated DNumS / DDenS for this step).
      for (size_t K = 0; K < In; ++K) MappedU[K] = Uin[K] * InW.Val[K] + InB.Val[K];
      std::fill(DMappedU.begin(), DMappedU.end(), 0.0f);
      for (size_t J = 0; J < S; ++J) {
        for (size_t K = 0; K < In; ++K) {
          size_t I = J * In + K;
          if (!Msk.In[I]) continue;
          float Swraw = SW.Val[I];
          float Spw = SoftPlus(Swraw);
          float Z = SSigma.Val[I] * (MappedU[K] - SMu.Val[I]);
          float Gs = Sigmoid(Z);
          float A = Spw * Gs;

          float DA = DNumS[J] * SEr.Val[I] + DDenS[J];
          SEr.Grad[I] += DNumS[J] * A;
          SW.Grad[I] += (DA * Gs) * Sigmoid(Swraw);
          float DG = DA * Spw;
          float DZ = DG * Gs * (1.0f - Gs);
          SSigma.Grad[I] += DZ * (MappedU[K] - SMu.Val[I]);
          SMu.Grad[I] += -DZ * SSigma.Val[I];
          DMappedU[K] += DZ * SSigma.Val[I];
        }
      }
      for (size_t K = 0; K < In; ++K) {
        InW.Grad[K] += DMappedU[K] * Uin[K];
        InB.Grad[K] += DMappedU[K];
      }

      std::copy(Gcur.begin(), Gcur.end(), GradH.begin());  // -> previous step
    }
  }
};

// ---------------------------------------------------------------------------
// Eval — forward only, mean MSE over all timesteps and motor outputs.
// ---------------------------------------------------------------------------

double EvalMse(Model &Mod, const SineDataset &D) {
  Mod.RefreshDerived();
  size_t S = Mod.S, In = Mod.In, Out = Mod.Out, U = Mod.H.OdeUnfolds;
  std::vector<float> Tape((U + 1) * S, 0.0f);
  std::vector<float> OutBuf(Out, 0.0f);
  double SumSq = 0.0;
  size_t Count = 0;
  for (size_t Seq = 0; Seq < D.X.size(); ++Seq) {
    std::fill(Tape.begin(), Tape.begin() + S, 0.0f);   // state <- 0
    const float *Xseq = D.X[Seq].data();
    const float *Yseq = D.Y[Seq].data();
    for (size_t T = 0; T < D.SeqLen; ++T) {
      Mod.Sensory(Xseq + T * In);
      Mod.CellForward(Tape.data());
      const float *State = Tape.data() + U * S;
      Mod.ReadOutput(State, OutBuf.data());
      const float *Tg = Yseq + T * Out;
      for (size_t M = 0; M < Out; ++M) {
        float Diff = OutBuf[M] - Tg[M];
        SumSq += static_cast<double>(Diff) * Diff;
      }
      Count += Out;
      std::copy(State, State + S, Tape.begin());        // carry state forward
    }
  }
  return Count == 0 ? 0.0 : SumSq / static_cast<double>(Count);
}

// Edge set for the structural log (matches the other ports' convention).
bench::EdgeSet LiveEdgeSet(const Model &Mod) {
  bench::EdgeSet Set;
  for (size_t Dst = 0; Dst < Mod.S; ++Dst) {
    for (size_t Src = 0; Src < Mod.S; ++Src)
      if (Mod.Msk.Rec[Dst * Mod.S + Src])
        Set.insert(bench::PackEdge(0, static_cast<uint32_t>(Src),
                                   static_cast<uint32_t>(Dst)));
    for (size_t Src = 0; Src < Mod.In; ++Src)
      if (Mod.Msk.In[Dst * Mod.In + Src])
        Set.insert(bench::PackEdge(1, static_cast<uint32_t>(Src),
                                   static_cast<uint32_t>(Dst)));
  }
  return Set;
}

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.Units      = static_cast<size_t>(Args.GetInt("units",       static_cast<int>(H.Units)));
  H.SeqLen     = static_cast<size_t>(Args.GetInt("seq-len",     static_cast<int>(H.SeqLen)));
  H.TrainSeqs  = static_cast<size_t>(Args.GetInt("train-seqs",  static_cast<int>(H.TrainSeqs)));
  H.ValSeqs    = static_cast<size_t>(Args.GetInt("val-seqs",    static_cast<int>(H.ValSeqs)));
  H.TestSeqs   = static_cast<size_t>(Args.GetInt("test-seqs",   static_cast<int>(H.TestSeqs)));
  H.Epochs     = static_cast<size_t>(Args.GetInt("epochs",      static_cast<int>(H.Epochs)));
  H.Batch      = static_cast<size_t>(Args.GetInt("batch",       static_cast<int>(H.Batch)));
  H.OdeUnfolds = static_cast<size_t>(Args.GetInt("ode-unfolds", static_cast<int>(H.OdeUnfolds)));
  H.KSparse    = static_cast<size_t>(Args.GetInt("k-sparse",    static_cast<int>(H.KSparse)));
  H.KRec       = static_cast<size_t>(Args.GetInt("k-rec",       static_cast<int>(H.KRec)));
  H.KFb        = static_cast<size_t>(Args.GetInt("k-fb",        static_cast<int>(H.KFb)));
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

  uint64_t Seed = static_cast<uint64_t>(Args.Seed) * 7919ull + 13ull;
  std::mt19937_64 Rng(Seed);

  Model Mod;
  Mod.Build(H, Rng);

  size_t NumEdges = Mod.Msk.LiveEdges;
  size_t NumParams = Mod.DenseParamCount();
  std::cout << "[info] units=" << H.Units << "  live_edges=" << NumEdges
            << "  dense_params=" << NumParams
            << "  partition Ns=" << Mod.P.Ns << " Ni=" << Mod.P.Ni
            << " Nc=" << Mod.P.Nc << " Nm=" << Mod.P.Nm << "\n";
  std::cout << "[info] task=sine  algo=LTC+BPTT  seq_len=" << H.SeqLen
            << "  train=" << H.TrainSeqs << "  val=" << H.ValSeqs
            << "  test=" << H.TestSeqs << "  epochs=" << H.Epochs
            << "  batch=" << H.Batch << "  ode_unfolds=" << H.OdeUnfolds
            << "  lr=" << H.Lr << "\n";

  auto Train = MakeSine(H.TrainSeqs, H.SeqLen, H.NoiseStd, Seed ^ 0xa1a1ull);
  auto Val   = MakeSine(H.ValSeqs,   H.SeqLen, H.NoiseStd, Seed ^ 0xb2b2ull);
  auto Test  = MakeSine(H.TestSeqs,  H.SeqLen, H.NoiseStd, Seed ^ 0xc3c3ull);

  auto [HistPath, SummaryPath, LogPath] = bench::OutputPaths(Args, "ccwc_ncp");
  (void)LogPath;
  bench::StructuralLog Log(HistPath);

  double InitVal  = EvalMse(Mod, Val);
  double InitTest = EvalMse(Mod, Test);
  bench::EdgeSet Edges0 = LiveEdgeSet(Mod);
  Log.Log(0, H.Units, NumEdges, &Edges0, &InitVal,
          {{"epoch", 0.0}, {"train_mse", 0.0},
           {"val_mse", InitVal}, {"test_mse", InitTest}});
  std::cout << "[ep   0] val_mse=" << InitVal << "  test_mse=" << InitTest << "\n";

  // Per-minibatch BPTT tape buffers (reused across minibatches).
  size_t S = Mod.S, U = H.OdeUnfolds, In = Mod.In, Out = Mod.Out;
  size_t Stride = (U + 1) * S;
  std::vector<float> BatchVp(H.Batch * H.SeqLen * Stride, 0.0f);
  std::vector<float> BatchOut(H.Batch * H.SeqLen * Out, 0.0f);

  std::vector<size_t> Perm(H.TrainSeqs);
  for (size_t I = 0; I < Perm.size(); ++I) Perm[I] = I;
  std::mt19937 ShuffleRng(static_cast<uint32_t>(Args.Seed));

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), ShuffleRng);
    double TrainSumSq = 0.0;
    size_t TrainCount = 0;

    for (size_t Beg = 0; Beg < H.TrainSeqs; Beg += H.Batch) {
      size_t Bsize = std::min(H.Batch, H.TrainSeqs - Beg);
      float Scale = 1.0f /
          static_cast<float>(Bsize * Train.SeqLen * Out);

      Mod.ZeroGrads();
      Mod.RefreshDerived();

      // ---- forward (whole minibatch), storing per-sample tapes ----
      Timer.Tick();
      for (size_t B = 0; B < Bsize; ++B) {
        size_t Sid = Perm[Beg + B];
        const float *Xseq = Train.X[Sid].data();
        float *Vp = BatchVp.data() + B * Train.SeqLen * Stride;
        float *Ob = BatchOut.data() + B * Train.SeqLen * Out;
        std::fill(Vp, Vp + S, 0.0f);                 // initial state 0
        for (size_t T = 0; T < Train.SeqLen; ++T) {
          float *VpT = Vp + T * Stride;
          if (T > 0)  // carry previous step's final state into this step's input
            std::copy(VpT - Stride + U * S, VpT - Stride + U * S + S, VpT);
          Mod.Sensory(Xseq + T * In);
          Mod.CellForward(VpT);
          Mod.ReadOutput(VpT + U * S, Ob + T * Out);
        }
      }
      Timer.MarkForward();

      // ---- loss (train MSE bookkeeping; gradient seed happens in backward) ----
      for (size_t B = 0; B < Bsize; ++B) {
        size_t Sid = Perm[Beg + B];
        const float *Yseq = Train.Y[Sid].data();
        const float *Ob = BatchOut.data() + B * Train.SeqLen * Out;
        for (size_t T = 0; T < Train.SeqLen; ++T)
          for (size_t M = 0; M < Out; ++M) {
            float Diff = Ob[T * Out + M] - Yseq[T * Out + M];
            TrainSumSq += static_cast<double>(Diff) * Diff;
            ++TrainCount;
          }
      }
      Timer.MarkLoss();

      // ---- backward (BPTT), accumulating grads across the minibatch ----
      for (size_t B = 0; B < Bsize; ++B) {
        size_t Sid = Perm[Beg + B];
        const float *Vp = BatchVp.data() + B * Train.SeqLen * Stride;
        const float *Ob = BatchOut.data() + B * Train.SeqLen * Out;
        Mod.Backward(Vp, Ob, Train.X[Sid].data(), Train.Y[Sid].data(),
                     Train.SeqLen, Scale);
      }
      Timer.MarkBackward();

      // ---- optimiser (clip + Adam) ----
      Mod.ClipAndAdam();
      Timer.MarkUpdate();
      Timer.StepDone();
    }

    double TrainMse = TrainCount == 0 ? 0.0
                                      : TrainSumSq / static_cast<double>(TrainCount);
    double ValMse  = EvalMse(Mod, Val);
    double TestMse = EvalMse(Mod, Test);

    bench::EdgeSet Edges = LiveEdgeSet(Mod);
    Log.Log(Ep, H.Units, NumEdges, &Edges, &ValMse,
            {{"epoch", static_cast<double>(Ep)},
             {"train_mse", TrainMse},
             {"val_mse", ValMse},
             {"test_mse", TestMse}});
    std::cout << "[ep " << std::setw(3) << Ep << "] "
              << "train_mse=" << TrainMse << "  val_mse=" << ValMse
              << "  test_mse=" << TestMse << "\n";
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();

  double FinalTest = EvalMse(Mod, Test);
  Log.Flush();

  bench::SummaryWriter Sum;
  Sum.Set("workload", std::string{"07_ccwc_ncp"});
  Sum.Set("dataset", std::string{"synthetic-sine"});
  Sum.Set("task",    std::string{"sine"});
  Sum.Set("backend", std::string{"cpp-ltc-bptt"});
  Sum.Set("metric_kind", std::string{"mse"});
  Sum.Set("n_in",    static_cast<int>(H.InputDim));
  Sum.Set("n_out",   static_cast<int>(H.OutputDim));
  Sum.Set("n_units", static_cast<int>(H.Units));
  Sum.Set("n_edges", static_cast<int>(NumEdges));
  Sum.Set("n_params", static_cast<long long>(NumParams));
  Sum.Set("seq_len", static_cast<int>(H.SeqLen));
  Sum.Set("epochs",  static_cast<int>(H.Epochs));
  Sum.Set("batch",   static_cast<int>(H.Batch));
  Sum.Set("ode_unfolds", static_cast<int>(H.OdeUnfolds));
  Sum.Set("lr",      H.Lr);
  Sum.Set("elapsed", H.Elapsed);
  Sum.Set("k_sparse", static_cast<int>(H.KSparse));
  Sum.Set("k_rec",    static_cast<int>(H.KRec));
  Sum.Set("k_fb",     static_cast<int>(H.KFb));
  Sum.Set("wall_seconds", Wall);
  Sum.Set("val_mse_final",
          Log.Records().empty() ? 0.0 : Log.Records().back().ValLoss);
  Sum.Set("test_mse", FinalTest);
  Sum.Set("seed",     Args.Seed);
  Timer.WriteSummary(Sum, Wall);
  Sum.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  final_test_mse=" << FinalTest << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";
  return 0;
}
