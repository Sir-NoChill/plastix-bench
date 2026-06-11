#ifndef TRADITIONAL_RAW_MLP_HPP
#define TRADITIONAL_RAW_MLP_HPP

// Small MLP building blocks for the traditional-raw suite.
//
// Linear layer: weight matrix is (out_dim x in_dim), row-major, optionally
// masked. Forward / backward route through OpenBLAS sgemm. Bias included.
//
// The masked-Linear pattern is shared across workloads 02 (IMP), 03 (bursty),
// and 04 (continuous-small); the unmasked version is what 01 (static) and 05
// (continuous-large readout) use.

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace raw {

// Xavier-uniform initialiser, matching PyTorch's nn.Linear default
// (sqrt(6/(in+out)) range).
inline float XavierLimit(size_t In, size_t Out) {
  return std::sqrt(6.0f / static_cast<float>(In + Out));
}

inline void FillUniform(std::vector<float> &W, float Limit, std::mt19937 &Rng) {
  std::uniform_real_distribution<float> U(-Limit, Limit);
  for (auto &V : W)
    V = U(Rng);
}

inline void FillNormal(std::vector<float> &W, float Sigma, std::mt19937 &Rng) {
  std::normal_distribution<float> N(0.0f, Sigma);
  for (auto &V : W)
    V = N(Rng);
}

// ---------------------------------------------------------------------------
// Linear layer
// ---------------------------------------------------------------------------

struct Linear {
  size_t InDim = 0;
  size_t OutDim = 0;
  // Logical (active) shape can be <= physical (capacity) shape when the
  // workload pre-allocates max width and grows in-place.
  size_t InCap = 0;
  size_t OutCap = 0;
  std::vector<float> Weight; // row-major, OutCap x InCap
  std::vector<float> Bias;   // OutCap
  std::vector<uint8_t> Mask; // OutCap x InCap, 0/1. Empty == no masking.
  std::vector<float> GradW;  // OutCap x InCap
  std::vector<float> GradB;  // OutCap

  void Init(size_t InDim_, size_t OutDim_, size_t InCap_, size_t OutCap_) {
    InDim = InDim_;
    OutDim = OutDim_;
    InCap = InCap_;
    OutCap = OutCap_;
    Weight.assign(OutCap * InCap, 0.0f);
    Bias.assign(OutCap, 0.0f);
    GradW.assign(OutCap * InCap, 0.0f);
    GradB.assign(OutCap, 0.0f);
  }

  void EnableMask() { Mask.assign(OutCap * InCap, 1); }

  void XavierInit(std::mt19937 &Rng) {
    float Lim = XavierLimit(InDim, OutDim);
    std::uniform_real_distribution<float> U(-Lim, Lim);
    // Only initialise the live (InDim x OutDim) block; rows/cols beyond the
    // current logical size stay zero until growth promotes them.
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InDim; ++J)
        Weight[I * InCap + J] = U(Rng);
    for (size_t I = 0; I < OutDim; ++I)
      Bias[I] = 0.0f;
  }

  // Y = X @ W^T + b  (Y: BxOutDim, X: BxInDim). Uses the (OutDim x InDim)
  // logical block; ignores rows/cols beyond it.
  void Forward(const float *X, float *Y, size_t Batch) const {
    if (Batch == 0 || OutDim == 0)
      return;
    // sgemm with row-major OutDim cols on the right. W has stride InCap
    // (physical), but we only sweep InDim cols. Output Y has stride OutDim.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(Batch), static_cast<int>(OutDim),
                static_cast<int>(InDim), 1.0f, X, static_cast<int>(InDim),
                Weight.data(), static_cast<int>(InCap), 0.0f, Y,
                static_cast<int>(OutDim));
    // Add bias broadcast.
    for (size_t B = 0; B < Batch; ++B) {
      float *Row = Y + B * OutDim;
      for (size_t J = 0; J < OutDim; ++J)
        Row[J] += Bias[J];
    }
  }

  // GradX[BxInDim] = GradY[BxOutDim] @ W[OutDim x InDim]
  void BackwardInput(const float *GradY, float *GradX, size_t Batch) const {
    if (Batch == 0 || InDim == 0)
      return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(Batch), static_cast<int>(InDim),
                static_cast<int>(OutDim), 1.0f, GradY,
                static_cast<int>(OutDim), Weight.data(),
                static_cast<int>(InCap), 0.0f, GradX,
                static_cast<int>(InDim));
  }

  // GradW[OutDim x InDim] = GradY^T @ X / Batch  (mean reduction); also
  // accumulates GradB. Stride-aware so the unused capacity past OutDim/InDim
  // stays zero.
  void BackwardWeights(const float *X, const float *GradY, size_t Batch,
                       float Scale) {
    if (Batch == 0)
      return;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(OutDim), static_cast<int>(InDim),
                static_cast<int>(Batch), Scale, GradY,
                static_cast<int>(OutDim), X, static_cast<int>(InDim), 0.0f,
                GradW.data(), static_cast<int>(InCap));
    // Bias gradient: column-sum of GradY.
    for (size_t J = 0; J < OutDim; ++J)
      GradB[J] = 0.0f;
    for (size_t B = 0; B < Batch; ++B) {
      const float *Row = GradY + B * OutDim;
      for (size_t J = 0; J < OutDim; ++J)
        GradB[J] += Row[J];
    }
    for (size_t J = 0; J < OutDim; ++J)
      GradB[J] *= Scale;
  }

  // SGD step: W -= lr * GradW; B -= lr * GradB. If a mask is active, both the
  // weight and its incoming gradient are forced to zero on dead edges before
  // applying the update.
  void SGD(float Lr) {
    if (!Mask.empty()) {
      // Zero gradients on dead edges so we don't gradually drift them off zero
      // through some other arithmetic path (shouldn't happen, but cheap to
      // enforce).
      for (size_t I = 0; I < OutDim; ++I)
        for (size_t J = 0; J < InDim; ++J) {
          size_t Idx = I * InCap + J;
          if (!Mask[Idx])
            GradW[Idx] = 0.0f;
        }
    }
    for (size_t I = 0; I < OutDim; ++I)
      for (size_t J = 0; J < InDim; ++J) {
        size_t Idx = I * InCap + J;
        Weight[Idx] -= Lr * GradW[Idx];
      }
    for (size_t J = 0; J < OutDim; ++J)
      Bias[J] -= Lr * GradB[J];
    if (!Mask.empty()) {
      // Re-zero pruned weights in case the update pushed them off zero.
      for (size_t I = 0; I < OutDim; ++I)
        for (size_t J = 0; J < InDim; ++J) {
          size_t Idx = I * InCap + J;
          if (!Mask[Idx])
            Weight[Idx] = 0.0f;
        }
    }
  }

  // Convenience for forward when the mask must be respected explicitly
  // (Forward() reads Weight directly, so the caller is responsible for
  // keeping pruned-weight entries at zero; this helper does the zeroing).
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
};

// ---------------------------------------------------------------------------
// Activations and their gradients-from-output / gradients-from-preact.
// ---------------------------------------------------------------------------

inline void ApplyReLU(float *Y, size_t N) {
  for (size_t I = 0; I < N; ++I)
    if (Y[I] < 0.0f)
      Y[I] = 0.0f;
}

inline void ReLUBackward(const float *Y, float *GradY, size_t N) {
  for (size_t I = 0; I < N; ++I)
    if (Y[I] <= 0.0f)
      GradY[I] = 0.0f;
}

inline void ApplyTanh(float *Y, size_t N) {
  for (size_t I = 0; I < N; ++I)
    Y[I] = std::tanh(Y[I]);
}

inline void TanhBackwardFromOut(const float *Y, float *GradY, size_t N) {
  for (size_t I = 0; I < N; ++I)
    GradY[I] *= (1.0f - Y[I] * Y[I]);
}

// PyTorch's F.gelu default is the exact (erf) form.
inline float GeLUExact(float Z) {
  constexpr float Inv2Sqrt = 0.70710678118654752440f; // 1/sqrt(2)
  return 0.5f * Z * (1.0f + std::erf(Z * Inv2Sqrt));
}

inline float GeLUExactGrad(float Z) {
  constexpr float Inv2Sqrt = 0.70710678118654752440f;
  constexpr float InvSqrt2Pi = 0.39894228040143267794f; // 1/sqrt(2 pi)
  float CdfPart = 0.5f * (1.0f + std::erf(Z * Inv2Sqrt));
  float PdfPart = InvSqrt2Pi * std::exp(-0.5f * Z * Z);
  return CdfPart + Z * PdfPart;
}

inline void ApplyGeLU(float *Z, float *Y, size_t N) {
  for (size_t I = 0; I < N; ++I)
    Y[I] = GeLUExact(Z[I]);
}

inline void GeLUBackwardFromPreact(const float *Z, float *GradY, size_t N) {
  for (size_t I = 0; I < N; ++I)
    GradY[I] *= GeLUExactGrad(Z[I]);
}

// ---------------------------------------------------------------------------
// Loss helpers (mean reduction over batch).
// ---------------------------------------------------------------------------

// Per-element MSE: loss = mean over (B, D) of (pred - target)^2.
// Grad w.r.t pred = 2 * (pred - target) / (B*D).
inline double MSELossMean(const float *Pred, const float *Target, size_t B,
                          size_t D, float *Grad) {
  double Sum = 0.0;
  size_t N = B * D;
  for (size_t I = 0; I < N; ++I) {
    float E = Pred[I] - Target[I];
    Sum += static_cast<double>(E) * E;
    Grad[I] = 2.0f * E / static_cast<float>(N);
  }
  return Sum / static_cast<double>(N);
}

// Per-element MSE, no gradient — for validation reads.
inline double MSEEvalMean(const float *Pred, const float *Target, size_t B,
                          size_t D) {
  double Sum = 0.0;
  size_t N = B * D;
  for (size_t I = 0; I < N; ++I) {
    float E = Pred[I] - Target[I];
    Sum += static_cast<double>(E) * E;
  }
  return N ? Sum / static_cast<double>(N) : 0.0;
}

// In-place softmax over the trailing dimension D, length B*D.
inline void SoftmaxRowwise(float *Logits, size_t B, size_t D) {
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

// Cross-entropy loss with mean reduction; expects integer targets in [0,D).
// Writes the gradient w.r.t the logits (after softmax: (p - onehot) / B) into
// Grad. Probs is a scratch buffer of size B*D.
inline double CrossEntropyMean(const float *Logits, const int *Targets,
                               size_t B, size_t D, float *Probs, float *Grad) {
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

// Argmax-accuracy over a batch of logits.
inline double ArgmaxAccuracy(const float *Logits, const int *Targets, size_t B,
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

} // namespace raw

#endif // TRADITIONAL_RAW_MLP_HPP
