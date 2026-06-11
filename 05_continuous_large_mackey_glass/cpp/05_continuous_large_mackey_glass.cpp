// Workload 5 / 5 — CONTINUOUS-LARGE regime, raw-C++/OpenBLAS translation.
//
// Mirrors 05_continuous_large_mackey_glass.py: a sparse-recurrent-
// style network on the Mackey-Glass chaotic time series, with heavy-tailed
// per-step unit deltas (Pareto), periodic Watts-Strogatz edge rewiring, and
// growth-momentum-triggered bursts. Per the "pre-allocate max + mask"
// decision, all three layers live at MaxHidden physical capacity from the
// start; "grow" / "shrink" flips mask bits on whole hidden rows/cols.
//
// Architecture: in -> H (tanh) + recurrent H -> H (sparse) -> tanh -> out
// (linear). Recurrent block is one ESN-style update per forward pass.

#include "common.hpp"
#include "mlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

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
  // mean-reduction MSE; effective ~= PyTorch_sum_lr * B * OutDim
  // = 1e-3 * 64 * 1 = 0.064; PyTorch uses Adam so step magnitude differs
  // anyway. Plain SGD value tuned by inspection.
  float Lr = 5e-2f;
  float ParetoAlpha = 1.5f;
  size_t MaxDeltaPerStep = 20;
  size_t RewireEvery = 3;
  float RewireFrac = 0.25f;
  float MomentumThreshold = 200.0f;
  size_t MomentumBurst = 10;
};

// Euler integration of Mackey-Glass DDE (matches the Python reference's
// shortcut). Returns `n` samples after a long warm-up.
static std::vector<float> MackeyGlass(size_t N, size_t Tau, uint32_t Seed) {
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
    double Dx =
        Beta * Delayed / (1.0 + std::pow(Delayed, 10.0)) - Gamma * Cur;
    Cur += H * Dx;
    Buf.push_back(Cur);
    if (Buf.size() > Tau * Sub * 2) {
      // Compact the buffer occasionally so it doesn't grow unbounded.
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

static Windowed WindowSeries(const std::vector<float> &S, size_t InLen,
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

// Sparse recurrent network: in -> Hin (tanh) + Hrec @ Hin (sparse) -> tanh
// -> out (linear).
struct SparseRecurrent {
  raw::Linear LIn;  // in -> H
  raw::Linear LRec; // H -> H (sparse via mask)
  raw::Linear LOut; // H -> out
  size_t Hidden = 0;
  size_t MaxHidden = 0;
  size_t InDim = 0;
  size_t OutDim = 0;
  // Mask of which hidden units are "active" (currently within Hidden). The
  // first Hidden positions are active; the rest are inactive. Shrinking
  // pulls Hidden back; growing pushes it forward.
  size_t Rewires = 0, Grows = 0, Shrinks = 0, Bursts = 0;

  // Forward scratch buffers.
  std::vector<float> H1;  // batch x H, tanh(L_in(x))
  std::vector<float> H2;  // batch x H, tanh(H1 + L_rec(H1))

  void Init(size_t InDim_, size_t OutDim_, size_t InitHidden,
            size_t MaxHidden_, float RecurDensity, std::mt19937 &Rng) {
    InDim = InDim_;
    OutDim = OutDim_;
    Hidden = InitHidden;
    MaxHidden = MaxHidden_;
    LIn.Init(InDim, InitHidden, InDim, MaxHidden);
    LRec.Init(InitHidden, InitHidden, MaxHidden, MaxHidden);
    LOut.Init(InitHidden, OutDim, MaxHidden, OutDim);
    // Init live blocks.
    float Lim0 = raw::XavierLimit(InDim, InitHidden);
    float Lim1 = raw::XavierLimit(InitHidden, InitHidden);
    float Lim2 = raw::XavierLimit(InitHidden, OutDim);
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
    // Sparse recurrent mask, no self-loops.
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

  const float *Forward(const float *X, size_t Batch) {
    // h = tanh(L_in(x))
    H1.assign(Batch * LIn.OutDim, 0.0f);
    LIn.Forward(X, H1.data(), Batch);
    raw::ApplyTanh(H1.data(), Batch * LIn.OutDim);
    // rec = L_rec(h) (sparse)
    std::vector<float> Rec(Batch * LRec.OutDim, 0.0f);
    LRec.Forward(H1.data(), Rec.data(), Batch);
    // h = tanh(h + rec); we keep H1 as the *pre*-step activation for the
    // backward path (need both pre-rec and post-rec for backprop through the
    // tanh).
    H2.assign(Batch * LRec.OutDim, 0.0f);
    for (size_t I = 0; I < Batch * LRec.OutDim; ++I)
      H2[I] = std::tanh(H1[I] + Rec[I]);
    // out = L_out(h)
    static thread_local std::vector<float> Out;
    Out.assign(Batch * LOut.OutDim, 0.0f);
    LOut.Forward(H2.data(), Out.data(), Batch);
    return Out.data();
  }

  void Backward(const float *X, const float *TopGrad, size_t Batch) {
    // d_out -> linear; d/dh2 = L_out.W^T * d_out; weight grad := d_out^T * h2
    LOut.BackwardWeights(H2.data(), TopGrad, Batch, 1.0f);
    std::vector<float> Gh2(Batch * LRec.OutDim, 0.0f);
    LOut.BackwardInput(TopGrad, Gh2.data(), Batch);
    // Through h2 = tanh(h1 + rec): d/dArg = (1 - h2^2) * Gh2.
    for (size_t I = 0; I < Batch * LRec.OutDim; ++I)
      Gh2[I] *= (1.0f - H2[I] * H2[I]);
    // Both h1 and rec receive Gh2.
    // L_rec weight grad: Gh2^T @ h1; d/dh1_via_rec: Gh2 @ L_rec.W
    std::vector<float> GhFromRec(Batch * LRec.OutDim, 0.0f);
    LRec.BackwardWeights(H1.data(), Gh2.data(), Batch, 1.0f);
    LRec.BackwardInput(Gh2.data(), GhFromRec.data(), Batch);
    // d/dh1 = Gh2 + GhFromRec (h1 feeds both branches).
    std::vector<float> Gh1(Batch * LIn.OutDim, 0.0f);
    for (size_t I = 0; I < Batch * LIn.OutDim; ++I)
      Gh1[I] = Gh2[I] + GhFromRec[I];
    // Through h1 = tanh(L_in(x)): d/d(L_in.preact) = (1 - h1^2) * Gh1
    for (size_t I = 0; I < Batch * LIn.OutDim; ++I)
      Gh1[I] *= (1.0f - H1[I] * H1[I]);
    LIn.BackwardWeights(X, Gh1.data(), Batch, 1.0f);
  }

  void Update(float Lr) {
    LIn.SGD(Lr);
    LRec.SGD(Lr);
    LOut.SGD(Lr);
  }

  // Grow `n` hidden units: bring [Hidden, Hidden+n) live in L_in (new rows),
  // L_rec (new rows and columns with a sparse seed pattern), L_out (new in
  // cols). New weights ~ N(0, init_scale).
  void Grow(size_t N, float InitScale, std::mt19937 &Rng) {
    if (N == 0)
      return;
    size_t Want = std::min(MaxHidden, Hidden + N);
    if (Want == Hidden)
      return;
    std::normal_distribution<float> Norm(0.0f, InitScale);
    std::bernoulli_distribution Bern(0.05);
    // L_in: new rows [Hidden, Want) x InDim.
    for (size_t I = Hidden; I < Want; ++I) {
      for (size_t J = 0; J < InDim; ++J)
        LIn.Weight[I * LIn.InCap + J] = Norm(Rng);
      LIn.Bias[I] = 0.0f;
    }
    // L_rec: new rows + new cols inside [0, Want) block. Existing block is
    // left intact (mask preserved). New positions get a Bernoulli(0.05)
    // sprinkle of edges.
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
    // L_out: new cols [0, OutDim) x [Hidden, Want).
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

  // Shrink by `n` hidden units (floor at MinHidden). Killed rows/cols beyond
  // the new live block are zeroed so they don't leak gradient or mask state.
  void Shrink(size_t N, size_t MinHidden) {
    if (N == 0)
      return;
    size_t Target = (Hidden > N) ? Hidden - N : MinHidden;
    if (Target < MinHidden)
      Target = MinHidden;
    if (Target >= Hidden)
      return;
    // Zero killed rows in L_in (rows [Target, Hidden)).
    for (size_t I = Target; I < Hidden; ++I) {
      for (size_t J = 0; J < InDim; ++J)
        LIn.Weight[I * LIn.InCap + J] = 0.0f;
      LIn.Bias[I] = 0.0f;
    }
    // Zero killed rows/cols of the recurrent block.
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
    // Zero killed cols of L_out.
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

  // Watts-Strogatz-style rewire on the recurrent block: pick `frac` of alive
  // edges, sever, reconnect each to a random (src, dst) that isn't already
  // alive (single attempt; drop if it lands invalid). Returns commit count.
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
};

static double EvalMseSlice(SparseRecurrent &M, const Windowed &W, size_t Start,
                           size_t Count, size_t Batch) {
  if (Count == 0)
    return 0.0;
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < Count; I += Batch) {
    size_t B = std::min(Batch, Count - I);
    const float *Pred = M.Forward(W.X.data() + (Start + I) * W.InLen, B);
    Sum += raw::MSEEvalMean(Pred, W.Y.data() + (Start + I), B, 1) *
           static_cast<double>(B);
    Cnt += B;
  }
  return Cnt ? Sum / static_cast<double>(Cnt) : 0.0;
}

static double ComputeGradNorm(const SparseRecurrent &M) {
  // L2 norm over all weight + bias gradients in the live blocks.
  double Sum = 0.0;
  auto AccLayer = [&](const raw::Linear &L) {
    for (size_t I = 0; I < L.OutDim; ++I)
      for (size_t J = 0; J < L.InDim; ++J) {
        float G = L.GradW[I * L.InCap + J];
        Sum += static_cast<double>(G) * G;
      }
    for (size_t I = 0; I < L.OutDim; ++I)
      Sum += static_cast<double>(L.GradB[I]) * L.GradB[I];
  };
  AccLayer(M.LIn);
  AccLayer(M.LRec);
  AccLayer(M.LOut);
  return std::sqrt(Sum);
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
  H.MaxDeltaPerStep = static_cast<size_t>(
      Args.GetInt("max-delta-per-step", H.MaxDeltaPerStep));
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

  auto Series = MackeyGlass(H.SeriesLen, H.Tau,
                            static_cast<uint32_t>(Args.Seed));
  // Per-series standardisation.
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
  size_t NTr = static_cast<size_t>(0.7f * W.N);
  size_t NVa = static_cast<size_t>(0.15f * W.N);
  size_t NTe = W.N - NTr - NVa;

  std::cout << "[info] N=" << W.N << " train=" << NTr << " val=" << NVa
            << " test=" << NTe << " steps=" << H.MaxSteps
            << " init_h=" << H.InitHidden
            << " recur_density=" << H.RecurDensity << "\n";

  std::mt19937 InitRng(static_cast<uint32_t>(Args.Seed) * 1000u + 19u);
  SparseRecurrent M;
  M.Init(H.InLen, 1, H.InitHidden, H.MaxHidden, H.RecurDensity, InitRng);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "continuous_large_mg");
  bench::StructuralLog Log(HistPath);

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::uniform_real_distribution<float> Prob(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> PickTrain(0, NTr - 1);
  // Pareto(alpha): X = (1/U)^(1/alpha) - 1 + 1 has min 1; numpy's
  // np.random.pareto(a) returns X-1 (shifted). We replicate numpy's form so
  // np.ceil(np.random.pareto(a)) >= 0; then "1 + " to shift the magnitude to
  // [1, ...).
  auto Pareto = [&]() {
    float U = std::max(1e-9f, Prob(Rng));
    return std::pow(1.0f - U, -1.0f / H.ParetoAlpha) - 1.0f;
  };

  std::vector<float> XBatch(H.Batch * H.InLen);
  std::vector<float> YBatch(H.Batch);
  std::vector<float> Grad(H.Batch);
  std::vector<int> DeltaUnits;
  DeltaUnits.reserve(H.MaxSteps);

  float GrowthMomentum = 0.0f;
  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Step = 1; Step <= H.MaxSteps; ++Step) {
    for (size_t B = 0; B < H.Batch; ++B) {
      size_t Idx = PickTrain(Rng);
      std::memcpy(XBatch.data() + B * H.InLen,
                  W.X.data() + Idx * H.InLen, H.InLen * sizeof(float));
      YBatch[B] = W.Y[Idx];
    }
    Timer.Tick();
    const float *Pred = M.Forward(XBatch.data(), H.Batch);
    Timer.MarkForward();
    raw::MSELossMean(Pred, YBatch.data(), H.Batch, 1, Grad.data());
    Timer.MarkLoss();
    M.Backward(XBatch.data(), Grad.data(), H.Batch);
    Timer.MarkBackward();
    M.Update(H.Lr);
    Timer.MarkUpdate();
    Timer.StepDone();
    float GNorm = static_cast<float>(ComputeGradNorm(M));

    size_t UnitsBefore = M.UnitCount();
    // Heavy-tailed unit delta: sign +/-, magnitude 1 + Pareto.
    int Sign = (Prob(Rng) > 0.5f) ? +1 : -1;
    size_t Mag =
        std::max<size_t>(1, static_cast<size_t>(std::ceil(Pareto())));
    if (Mag > H.MaxDeltaPerStep)
      Mag = H.MaxDeltaPerStep;
    if (Sign > 0 && M.Hidden + Mag <= H.MaxHidden) {
      M.Grow(Mag, 0.05f, Rng);
    } else if (Sign < 0 && M.Hidden > H.MinHidden + Mag - 1) {
      M.Shrink(Mag, H.MinHidden);
    }
    if (Step % H.RewireEvery == 0 && M.Hidden >= 8)
      M.Rewire(H.RewireFrac, Rng);

    GrowthMomentum += GNorm;
    if (GrowthMomentum > H.MomentumThreshold) {
      size_t Burst = std::max<size_t>(2, H.MomentumBurst);
      if (M.Hidden + Burst <= H.MaxHidden) {
        M.Grow(Burst, 0.05f, Rng);
        ++M.Bursts;
      }
      GrowthMomentum = 0.0f;
    }

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
               {"delta_units",
                static_cast<double>(DeltaUnits.back())}});
    }
  }
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
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
  S.Set("delta_units_p50_abs", Pctile(DeltaUnits, 50.0));
  S.Set("delta_units_p95_abs", Pctile(DeltaUnits, 95.0));
  S.Set("delta_units_max_abs", DuMax);
  S.Set("delta_units_mean_nonzero", DuNzMean);
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_mean", JMean);
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  grows=" << M.Grows
            << "  shrinks=" << M.Shrinks << "  rewires=" << M.Rewires
            << "  hidden=" << M.Hidden << "  test_mse=" << TestMse
            << "  jaccard_mean=" << JMean << "  |du|_p95="
            << Pctile(DeltaUnits, 95.0) << "  |du|_max=" << DuMax << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
