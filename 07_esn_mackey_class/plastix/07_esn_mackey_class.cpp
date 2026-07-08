// Workload 7 — Echo State Network on Mackey-Glass, Plastix translation.
//
// The ESN reservoir is genuinely RECURRENT:
//   x(t) = (1-α) x(t-1) + α tanh(W_in u(t) + W_rec x(t-1))
// Plastix's Topological mode requires a DAG, but **Pipeline** mode does not: it
// sweeps every live connection once per DoStep using the PREVIOUS step's
// activations, then applies each non-input unit. That is exactly the reservoir
// update — a recurrent edge j→i reads x_j(t-1), an input edge reads u(t), and
// Apply does the leaky-tanh integration. So the reservoir is expressed natively
// as a cyclic Pipeline network (no sort, no DAG constraint — see plastix.hpp:116
// where SortConnectionsByLevel is gated on Topological only).
//
// The readout W_out is fit by closed-form ridge regression — a batch linear
// solve, not a per-step local rule — so (as in every ESN impl, incl. the cpp/jax
// ports) it lives host-side: collect reservoir states, solve
// (SᵀS + λI) W_out = SᵀY, then predict y = W_out·x. Plastix expresses the
// interesting part (the recurrent reservoir); the readout is one dense solve.
//
// Policy mapping: ForwardPass only (Map=w·x, Combine=+, Apply=leaky-tanh);
// Loss/Backward/Update/Prune/Add all NoX (the reservoir is fixed, the readout is
// external). Propagation::Pipeline.

#include "plastix/common.hpp"

#include <plastix/alloc.hpp>
#include <plastix/conn.hpp>
#include <plastix/macros.hpp>
#include <plastix/plastix.hpp>
#include <plastix/traits.hpp>
#include <plastix/unit_state.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

// ---- Hyperparameters (match the cpp/jax ESN) --------------------------------
namespace hp {
constexpr int Reservoir = 100;
constexpr float SpectralRadius = 1.25f;
constexpr float Leak = 0.3f;
constexpr float InputScale = 1.0f;
constexpr float Ridge = 1e-5f;
constexpr int SeriesLenDefault = 2000;
constexpr int WarmupSteps = 100;
constexpr float TrainFrac = 0.8f;
constexpr int MGTau = 17;
constexpr uint64_t Seed = 0;
} // namespace hp

// ---- Mackey-Glass generator (same Euler recipe as the cpp impl) -------------
std::vector<float> MackeyGlass(int N, int Tau, uint64_t S) {
  std::mt19937_64 Rng(S);
  std::normal_distribution<float> Jitter(0.0f, 0.01f);
  const float Beta = 0.2f, Gamma = 0.1f, H = 0.1f;
  const int Sub = 10;
  const int SettleSteps = Tau * 20;
  std::vector<float> Buf;
  for (int I = 0; I < Tau + 1; ++I)
    Buf.push_back(1.2f + Jitter(Rng));
  float Cur = Buf.back();
  auto Step = [&]() {
    int Di = Tau * Sub;
    float Delayed = static_cast<int>(Buf.size()) >= Di ? Buf[Buf.size() - Di]
                                                       : Buf.front();
    float Dx = Beta * Delayed / (1.0f + std::pow(Delayed, 10.0f)) - Gamma * Cur;
    Cur += H * Dx;
    Buf.push_back(Cur);
    if (static_cast<int>(Buf.size()) > Tau * Sub * 2)
      Buf.erase(Buf.begin(),
                Buf.begin() + (Buf.size() - static_cast<size_t>(Tau) * Sub * 2));
  };
  for (int I = 0; I < SettleSteps * Sub; ++I)
    Step();
  std::vector<float> Out;
  for (int I = 0; I < N * Sub && static_cast<int>(Out.size()) < N; ++I) {
    Step();
    if ((I % Sub) == 0)
      Out.push_back(Cur);
  }
  return Out;
}

void NormalizeInPlace(std::vector<float> &S) {
  double Mean = 0.0;
  for (float V : S)
    Mean += V;
  Mean /= static_cast<double>(S.size());
  double Var = 0.0;
  for (float V : S)
    Var += (V - Mean) * (V - Mean);
  double Std = std::sqrt(Var / static_cast<double>(S.size())) + 1e-8;
  for (float &V : S)
    V = static_cast<float>((V - Mean) / Std);
}

// ---- Plastix traits: fixed recurrent reservoir ------------------------------
// WeightTag (the connection's reservoir weight) is in the default ExtraConnFields.
struct Forward {
  using Accumulator = float;
  // Pipeline forward: Self=dest, Other=src. Sum over edges of w·x_src(t-1).
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  // Leaky-tanh: x_i(t) = (1-α) x_i(t-1) + α tanh(Sum). Reads its own prior
  // activation (not yet overwritten this step). Input units aren't applied.
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Sum) {
    float Old = plastix::GetActivation(U, Id);
    plastix::GetActivation(U, Id) =
        (1.0f - hp::Leak) * Old + hp::Leak * std::tanh(Sum);
  }
};

struct Traits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = Forward;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizeAdd = false;
  static constexpr bool ReverseAdjForward = false;
  // 1 input + N reservoir units; ~N² recurrent edges + N input edges.
  static constexpr size_t UnitCapacity = 512;
  static constexpr size_t ConnCapacity = 32768;
};
static_assert(plastix::NetworkTraits<Traits>);
using Net = plastix::Network<Traits>;

// ---- Reservoir builder: allocate N units + the fixed W_in / W_rec edges -----
struct ReservoirBuilder {
  int N;
  uint64_t Seed;

  template <typename UnitAlloc, typename ConnAlloc>
  PLASTIX_HOST plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                             plastix::UnitRange Inputs) const {
    using namespace plastix;
    const size_t InputId = Inputs.Begin; // the single input unit (level 0)
    std::mt19937_64 Rng(Seed);
    std::uniform_real_distribution<float> UIn(-hp::InputScale, hp::InputScale);
    float RecStd = hp::SpectralRadius / std::sqrt(static_cast<float>(N));
    std::normal_distribution<float> NRec(0.0f, RecStd);

    // Fixed weights (same recipe as the cpp impl: W_in then W_rec, zero diag).
    std::vector<float> WIn(N);
    for (int I = 0; I < N; ++I)
      WIn[I] = UIn(Rng);
    std::vector<float> WRec(static_cast<size_t>(N) * N);
    for (size_t I = 0; I < WRec.size(); ++I)
      WRec[I] = NRec(Rng);
    for (int I = 0; I < N; ++I)
      WRec[static_cast<size_t>(I) * N + I] = 0.0f;

    // Allocate reservoir units at level 1.
    std::vector<size_t> Res(N);
    for (int I = 0; I < N; ++I) {
      Res[I] = UA.Allocate();
      GetLevel(UA, Res[I]) = 1;
    }
    // Input edge: input -> reservoir i, weight W_in[i].
    for (int I = 0; I < N; ++I) {
      auto C = CA.Allocate();
      GetField<FromIdTag>(CA, C) = static_cast<uint32_t>(InputId);
      GetField<ToIdTag>(CA, C) = static_cast<uint32_t>(Res[I]);
      GetField<SrcLevelTag>(CA, C) = 0;
      GetWeight(CA, C) = WIn[I];
    }
    // Recurrent edge: reservoir j -> reservoir i, weight W_rec[i][j] (i!=j).
    for (int I = 0; I < N; ++I)
      for (int J = 0; J < N; ++J) {
        if (I == J)
          continue;
        float W = WRec[static_cast<size_t>(I) * N + J];
        auto C = CA.Allocate();
        GetField<FromIdTag>(CA, C) = static_cast<uint32_t>(Res[J]);
        GetField<ToIdTag>(CA, C) = static_cast<uint32_t>(Res[I]);
        GetField<SrcLevelTag>(CA, C) = 1;
        GetWeight(CA, C) = W;
      }
    return UnitRange{Res.front(), Res.back() + 1};
  }
};

// ---- Host ridge solve: (SᵀS + λI) W = SᵀY via LU (double, partial pivot) ----
bool SolveLinear(std::vector<double> &A, std::vector<double> &b, int N) {
  for (int col = 0; col < N; ++col) {
    int piv = col;
    double best = std::abs(A[static_cast<size_t>(col) * N + col]);
    for (int r = col + 1; r < N; ++r) {
      double v = std::abs(A[static_cast<size_t>(r) * N + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-30)
      return false;
    if (piv != col) {
      for (int c = 0; c < N; ++c)
        std::swap(A[static_cast<size_t>(col) * N + c],
                  A[static_cast<size_t>(piv) * N + c]);
      std::swap(b[col], b[piv]);
    }
    double d = A[static_cast<size_t>(col) * N + col];
    for (int r = col + 1; r < N; ++r) {
      double f = A[static_cast<size_t>(r) * N + col] / d;
      if (f != 0.0) {
        for (int c = col; c < N; ++c)
          A[static_cast<size_t>(r) * N + c] -=
              f * A[static_cast<size_t>(col) * N + c];
        b[r] -= f * b[col];
      }
    }
  }
  for (int r = N - 1; r >= 0; --r) {
    double s = b[r];
    for (int c = r + 1; c < N; ++c)
      s -= A[static_cast<size_t>(r) * N + c] * b[c];
    b[r] = s / A[static_cast<size_t>(r) * N + r];
  }
  return true;
}

// Ridge fit: S is (T x N) float row-major, Y is (T). Writes W (N) float.
void RidgeFit(const std::vector<float> &S, int T, int N,
              const std::vector<float> &Y, float Lambda,
              std::vector<float> &WOut) {
  std::vector<double> STS(static_cast<size_t>(N) * N, 0.0);
  std::vector<double> STY(N, 0.0);
  for (int t = 0; t < T; ++t) {
    const float *St = &S[static_cast<size_t>(t) * N];
    double y = Y[t];
    for (int i = 0; i < N; ++i) {
      double si = St[i];
      STY[i] += si * y;
      double *row = &STS[static_cast<size_t>(i) * N];
      for (int j = 0; j < N; ++j)
        row[j] += si * St[j];
    }
  }
  for (int i = 0; i < N; ++i)
    STS[static_cast<size_t>(i) * N + i] += static_cast<double>(Lambda);
  if (!SolveLinear(STS, STY, N)) {
    std::cerr << "[fatal] ridge solve failed (singular)\n";
    std::exit(3);
  }
  WOut.resize(N);
  for (int i = 0; i < N; ++i)
    WOut[i] = static_cast<float>(STY[i]);
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);
  const int N = hp::Reservoir;
  int SeriesLen = hp::SeriesLenDefault;
  if (Args.Quick)
    SeriesLen = std::min(SeriesLen, 800);

  bench::MemoryProbe MP;
  MP.Start();

  std::vector<float> Series = MackeyGlass(SeriesLen, hp::MGTau, hp::Seed);
  NormalizeInPlace(Series);
  MP.EndDataset();

  const int NTrainAll = static_cast<int>(Series.size() * hp::TrainFrac);
  if (NTrainAll + 1 >= static_cast<int>(Series.size())) {
    std::cerr << "series too short\n";
    return 1;
  }

  Net Network(1, ReservoirBuilder{N, hp::Seed});
  auto &UA = Network.GetUnitAlloc();
  const size_t ResBegin = 1;                       // reservoir units are [1, 1+N)
  const size_t NEdges = bench::LiveEdgeCount(Network.GetConnAlloc());
  MP.EndWeights();

  std::array<float, 1> In{0.0f};
  auto ReadState = [&](std::vector<float> &Dst) {
    for (int I = 0; I < N; ++I)
      Dst[I] = plastix::GetActivation(UA, ResBegin + I);
  };

  bench::PhaseTimer Timer;
  auto WallT0 = std::chrono::steady_clock::now();

  // Warmup — drive the reservoir, discard states.
  for (int T = 0; T < hp::WarmupSteps && T < NTrainAll; ++T) {
    In[0] = Series[T];
    Network.DoForwardPass(In);
  }

  // Collect training states; target = one-step-ahead value.
  const int TrainRows = NTrainAll - hp::WarmupSteps - 1;
  if (TrainRows <= 0) {
    std::cerr << "warmup consumed the training window\n";
    return 1;
  }
  std::vector<float> S(static_cast<size_t>(TrainRows) * N);
  std::vector<float> Y(TrainRows);
  std::vector<float> St(N);
  for (int T = 0; T < TrainRows; ++T) {
    int Idx = hp::WarmupSteps + T;
    In[0] = Series[Idx];
    Timer.Tick();
    Network.DoForwardPass(In);
    Timer.MarkForward();
    ReadState(St);
    std::copy(St.begin(), St.end(), S.begin() + static_cast<size_t>(T) * N);
    Y[T] = Series[Idx + 1];
    Timer.MarkReset();
    Timer.StepDone();
  }

  // Closed-form ridge readout (the one-shot "update").
  std::vector<float> WOut(N, 0.0f);
  Timer.Tick();
  RidgeFit(S, TrainRows, N, Y, hp::Ridge, WOut);
  Timer.MarkUpdate();

  // Test inference: reservoir step + linear readout.
  const int TestRows = static_cast<int>(Series.size()) - NTrainAll - 1;
  std::vector<float> Preds(TestRows), Truth(TestRows);
  for (int T = 0; T < TestRows; ++T) {
    int Idx = NTrainAll + T;
    In[0] = Series[Idx];
    Network.DoForwardPass(In);
    ReadState(St);
    double P = 0.0;
    for (int I = 0; I < N; ++I)
      P += static_cast<double>(WOut[I]) * St[I];
    Preds[T] = static_cast<float>(P);
    Truth[T] = Series[Idx + 1];
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - WallT0)
          .count();

  // Metrics: RMSE / MSE / R².
  double SumSq = 0.0, Mean = 0.0;
  for (float V : Truth)
    Mean += V;
  Mean /= std::max(1, TestRows);
  double TotSS = 0.0;
  for (int I = 0; I < TestRows; ++I) {
    double D = Preds[I] - Truth[I];
    SumSq += D * D;
    double Dm = Truth[I] - Mean;
    TotSS += Dm * Dm;
  }
  double Mse = SumSq / std::max(1, TestRows);
  double Rmse = std::sqrt(Mse);
  double R2 = 1.0 - SumSq / (TotSS + 1e-12);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "esn_mackey_class");
  bench::StructuralLog Log(HistPath);
  Log.Log(1, N + 1, NEdges, nullptr, &Mse,
          {{"test_mse", Mse}, {"test_rmse", Rmse}, {"test_r2", R2},
           {"n_units", static_cast<double>(N + 1)},
           {"n_edges", static_cast<double>(NEdges)}});
  Log.Flush();

  bench::SummaryWriter Sum;
  Sum.Set("workload", std::string{"07_esn_mackey_class"});
  Sum.Set("dataset", std::string{"mackey-glass"});
  Sum.Set("units", N);
  Sum.Set("sr", static_cast<double>(hp::SpectralRadius));
  Sum.Set("leak", static_cast<double>(hp::Leak));
  Sum.Set("ridge", static_cast<double>(hp::Ridge));
  Sum.Set("wall_seconds", Wall);
  Sum.Set("test_mse", Mse);
  Sum.Set("test_rmse", Rmse);
  Sum.Set("test_r2", R2);
  Sum.Set("metric_kind", std::string{"mse"});
  Sum.Set("n_units", static_cast<int>(N + 1));
  Sum.Set("n_edges", static_cast<long long>(NEdges));
  Sum.Set("seed", Args.Seed);
  Timer.WriteSummary(Sum, Wall);
  MP.WriteSummary(Sum);
  Sum.Write(SummaryPath);

  std::cout << "[done] units=" << N << " edges=" << NEdges
            << " train_rows=" << TrainRows << " test=" << TestRows
            << " test_rmse=" << Rmse << " test_r2=" << R2
            << " wall=" << Wall << "s\n";
  (void)LogPath;
  return 0;
}
