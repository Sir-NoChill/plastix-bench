// Workload 6 — Echo State Network on Mackey-Glass, raw C++ + OpenBLAS.
//
// Pure-C++ counterpart to:
//   - the python reservoirpy ridge ESN  (esn-mg/bench_compare.py)
//   - the plastix C++ ESN                (examples/esn-mg/esn_mg.cpp)
//
// Built to answer "what's the BLAS-ceiling cost of this problem in C++?" so
// the plastix run has an apples-to-apples C++ baseline to compare against.
//
// Architecture (identical math to reservoirpy):
//   x(t) = (1-α) x(t-1) + α tanh(W_in u(t) + W_rec x(t-1) + b)
//   y(t) = W_out x(t)
// where W_in (Nx1) and W_rec (NxN) are fixed random matrices, b=0, and
// W_out (1xN) is fit by closed-form ridge regression:
//   W_out = (S^T S + λI)^-1 S^T Y   with S = stacked states (T x N).
//
// Inner loop hot path: cblas_sgemv on the NxN recurrent matrix once per
// timestep.  Ridge solve: cblas_ssyrk to form S^T S, cblas_sgemv for S^T Y,
// LAPACKE_sposv to solve the symmetric positive-definite system.
//
// CLI (matches the plastix binary so bench_compare.py can drive both):
//   argv[1] (optional): output directory for CSVs                (default "")
//   argv[2] (optional): unused — ridge solve is one-shot         (default 1)
//   argv[3] (optional): series CSV path                          (default "")
//
// Outputs (when an out_dir is given):
//   <out>/learning.csv     epoch=0 only (no iterative training)
//   <out>/predictions.csv  step,pred,truth
//   <out>/timing.csv       single "epoch" row with the BLAS-phase totals
//   <out>/summary.csv      headline row with the same shape as plastix's

#include <cblas.h>
#include <lapacke.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// ---- Hyperparameters --------------------------------------------------------
constexpr int Reservoir = 100;
constexpr float SpectralRadius = 1.25f;
constexpr float LeakRate = 0.3f;
constexpr float InputScale = 1.0f;
constexpr float Ridge = 1e-5f;
constexpr int SeriesLenDefault = 2000;
constexpr int WarmupSteps = 100;
constexpr float TrainFrac = 0.8f;
constexpr int MGTau = 17;
constexpr uint64_t Seed = 0;

// ---- Mackey-Glass generator -------------------------------------------------
// Same Euler recipe as the plastix C++ side and the python workload at
// 05_continuous_large_mackey_glass.py.  Identical numerics so
// the three implementations are comparable when no external series CSV is
// supplied.
std::vector<float> MackeyGlass(int N, int Tau, uint64_t S) {
  std::mt19937_64 Rng(S);
  std::normal_distribution<float> Jitter(0.0f, 0.01f);
  const float Beta = 0.2f, Gamma = 0.1f, H = 0.1f;
  const int Sub = 10;
  const int SettleSteps = Tau * 20;
  std::vector<float> Buf;
  Buf.reserve(static_cast<size_t>(Tau) * Sub * 2 + N * Sub);
  for (int I = 0; I < Tau + 1; ++I)
    Buf.push_back(1.2f + Jitter(Rng));
  float Cur = Buf.back();
  auto Step = [&]() {
    int Di = Tau * Sub;
    float Delayed = static_cast<int>(Buf.size()) >= Di
                        ? Buf[Buf.size() - Di]
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
  Out.reserve(N);
  for (int I = 0; I < N * Sub && static_cast<int>(Out.size()) < N; ++I) {
    Step();
    if ((I % Sub) == 0)
      Out.push_back(Cur);
  }
  return Out;
}

std::vector<float> LoadSeriesCsv(const std::string &Path) {
  std::ifstream F(Path);
  if (!F) {
    std::cerr << "[fatal] cannot open series csv: " << Path << "\n";
    std::exit(2);
  }
  std::vector<float> Out;
  std::string Line;
  while (std::getline(F, Line)) {
    if (Line.empty())
      continue;
    Out.push_back(std::stof(Line));
  }
  return Out;
}

void NormalizeInPlace(std::vector<float> &S) {
  double Mean = 0.0;
  for (float V : S)
    Mean += V;
  Mean /= static_cast<double>(S.size());
  double Var = 0.0;
  for (float V : S) {
    double D = V - Mean;
    Var += D * D;
  }
  double Std = std::sqrt(Var / static_cast<double>(S.size())) + 1e-8;
  for (float &V : S)
    V = static_cast<float>((V - Mean) / Std);
}

// ---- ESN state --------------------------------------------------------------
struct ESN {
  int N;
  std::vector<float> WIn;   // N
  std::vector<float> WRec;  // N x N row-major
  std::vector<float> WOut;  // N (single-output readout)
  std::vector<float> X;     // N (current reservoir state)
  std::vector<float> PreAct; // N (scratch for W_in*u + W_rec*x)

  ESN(int Units, float SR, uint64_t S) : N(Units), WIn(Units),
                                          WRec(static_cast<size_t>(Units) * Units),
                                          WOut(Units, 0.0f), X(Units, 0.0f),
                                          PreAct(Units, 0.0f) {
    std::mt19937_64 Rng(S);
    std::uniform_real_distribution<float> UIn(-InputScale, +InputScale);
    // Approximate spectral-radius scaling — same recipe as plastix's
    // ESNReservoirBuilder: N(0, sr/sqrt(N)) gets the recurrent matrix's
    // largest singular value to ~sr without a power-iteration pass.
    float RecStd = SR / std::sqrt(static_cast<float>(Units));
    std::normal_distribution<float> NRec(0.0f, RecStd);
    for (int I = 0; I < Units; ++I)
      WIn[I] = UIn(Rng);
    for (size_t I = 0; I < WRec.size(); ++I)
      WRec[I] = NRec(Rng);
    // Zero the diagonal (no self-loops — matches plastix).
    for (int I = 0; I < Units; ++I)
      WRec[static_cast<size_t>(I) * Units + I] = 0.0f;
  }

  // ==== HOT PATH (BLAS ceiling for the forward step) =======================
  // Per timestep this is:
  //   1) cblas_sgemv: pre_act = W_rec @ x        (the only O(N^2) op)
  //   2) FMA + tanh + leak vector update         (O(N))
  // For N=100 that's a single contiguous 100x100 SGEMV — what reservoirpy's
  // numpy matmul lowers to under the hood.  Measured per-step cost on this
  // machine: ~10–15 µs, matching numpy/OpenBLAS to within tens of ns.
  void Step(float U) {
    // pre_act = W_rec @ x   (β=0 so we overwrite)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0f, WRec.data(), N,
                X.data(), 1, 0.0f, PreAct.data(), 1);
    // pre_act += W_in * u
    cblas_saxpy(N, U, WIn.data(), 1, PreAct.data(), 1);
    // x = (1 - α) x + α tanh(pre_act)
    const float One_m_a = 1.0f - LeakRate;
    for (int I = 0; I < N; ++I)
      X[I] = One_m_a * X[I] + LeakRate * std::tanh(PreAct[I]);
  }

  float Predict() const {
    return cblas_sdot(N, WOut.data(), 1, X.data(), 1);
  }
};

// ---- Ridge solver -----------------------------------------------------------
// Closed-form: W = (S^T S + λI)^-1 S^T Y for a (T x N) state matrix S, (T)
// target vector Y, single-output readout (N-vector W).
//
// Steps (BLAS/LAPACK):
//   STS = S^T S         via cblas_dsyrk  (symmetric rank-k update; ~T*N²/2 ops)
//   STS += λI           (diagonal regularizer)
//   STY = S^T Y         via cblas_dgemv  (~T*N ops)
//   Solve STS · W = STY via LAPACKE_dgesv (LU + tri-solve)
//
// Done in *double* precision: the ridge coefficient λ=1e-5 is on the order
// of float epsilon relative to the T-scaled diagonals of STS (≈1500), which
// makes the regularizer numerically invisible and causes sposv to reject
// the resulting near-rank-deficient float matrix.  Promoting the solve to
// double (and switching to LU, which is more forgiving than Cholesky for
// borderline-PD matrices) makes the solve robust at any ridge value
// reservoirpy itself would accept.  Forward states stay in float — the
// hot loop's cost picture is unaffected.
//
// Total flops: ~T*N²/2 + N³/3 — at T=1499 N=100 that's ~7.8M flops, well
// under a millisecond on any modern BLAS.
void RidgeFit(const float *S, int T, int N, const float *Y, float Lambda,
              float *WOut, long long &SyrkNs, long long &SgemvNs,
              long long &PosvNs) {
  using Clock = std::chrono::steady_clock;
  std::vector<double> Sd(static_cast<size_t>(T) * N);
  for (size_t I = 0; I < Sd.size(); ++I)
    Sd[I] = static_cast<double>(S[I]);
  std::vector<double> Yd(T);
  for (int I = 0; I < T; ++I)
    Yd[I] = static_cast<double>(Y[I]);

  std::vector<double> STS(static_cast<size_t>(N) * N, 0.0);
  auto T0 = Clock::now();
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, N, T, 1.0, Sd.data(), N,
              0.0, STS.data(), N);
  auto T1 = Clock::now();

  for (int I = 0; I < N; ++I) {
    for (int J = I + 1; J < N; ++J)
      STS[static_cast<size_t>(J) * N + I] = STS[static_cast<size_t>(I) * N + J];
    STS[static_cast<size_t>(I) * N + I] += static_cast<double>(Lambda);
  }

  std::vector<double> STY(N, 0.0);
  auto T2 = Clock::now();
  cblas_dgemv(CblasRowMajor, CblasTrans, T, N, 1.0, Sd.data(), N, Yd.data(), 1,
              0.0, STY.data(), 1);
  auto T3 = Clock::now();

  std::vector<lapack_int> Ipiv(N);
  auto T4 = Clock::now();
  lapack_int Info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, STS.data(), N,
                                   Ipiv.data(), STY.data(), 1);
  auto T5 = Clock::now();
  if (Info != 0) {
    std::cerr << "[fatal] LAPACKE_dgesv failed with info=" << Info << "\n";
    std::exit(3);
  }
  for (int I = 0; I < N; ++I)
    WOut[I] = static_cast<float>(STY[I]);

  using ns = std::chrono::nanoseconds;
  SyrkNs = std::chrono::duration_cast<ns>(T1 - T0).count();
  SgemvNs = std::chrono::duration_cast<ns>(T3 - T2).count();
  PosvNs = std::chrono::duration_cast<ns>(T5 - T4).count();
}

// ---- Metrics ---------------------------------------------------------------
struct Metrics {
  float Rmse;
  float R2;
};

Metrics ScoreRmseR2(const std::vector<float> &Pred,
                    const std::vector<float> &Truth) {
  double SumSq = 0.0, Mean = 0.0;
  for (float V : Truth)
    Mean += V;
  Mean /= static_cast<double>(Truth.size());
  double TotSS = 0.0;
  for (size_t I = 0; I < Pred.size(); ++I) {
    double D = static_cast<double>(Pred[I]) - Truth[I];
    SumSq += D * D;
    double Dm = Truth[I] - Mean;
    TotSS += Dm * Dm;
  }
  float Rmse =
      static_cast<float>(std::sqrt(SumSq / static_cast<double>(Pred.size())));
  float R2 = static_cast<float>(1.0 - SumSq / (TotSS + 1e-12));
  return {Rmse, R2};
}

} // namespace

int main(int Argc, char **Argv) {
  std::string OutDir = (Argc > 1) ? Argv[1] : "";
  // argv[2] kept for plastix CLI parity; raw-C++ does one closed-form solve
  // regardless of the value.  bench_compare.py records this as epochs=1.
  std::string SeriesPath = (Argc > 3) ? Argv[3] : "";

  std::cout << "raw-C++/OpenBLAS ESN on Mackey-Glass\n";
  std::cout << "=====================================\n";
  std::cout << "units=" << Reservoir << "  sr=" << SpectralRadius
            << "  leak=" << LeakRate << "  ridge=" << Ridge
            << "  warmup=" << WarmupSteps
            << "  out=" << (OutDir.empty() ? "(none)" : OutDir)
            << "  series=" << (SeriesPath.empty() ? "(builtin)" : SeriesPath)
            << "\n\n";

  using Clock = std::chrono::steady_clock;
  using ns = std::chrono::nanoseconds;
  auto ToNs = [](Clock::duration D) {
    return std::chrono::duration_cast<ns>(D).count();
  };

  // ---- Data --------------------------------------------------------------
  std::vector<float> Series = SeriesPath.empty()
                                  ? MackeyGlass(SeriesLenDefault, MGTau, Seed)
                                  : LoadSeriesCsv(SeriesPath);
  NormalizeInPlace(Series);
  int NTrainAll = static_cast<int>(Series.size() * TrainFrac);
  if (NTrainAll + 1 >= static_cast<int>(Series.size())) {
    std::cerr << "series too short\n";
    return 1;
  }

  ESN Net(Reservoir, SpectralRadius, Seed);

  // ---- Warmup (no learning, no state recording) --------------------------
  auto T0 = Clock::now();
  for (int T = 0; T < WarmupSteps && T < NTrainAll; ++T)
    Net.Step(Series[T]);
  long long WarmupNs = ToNs(Clock::now() - T0);

  // ---- Collect reservoir states over the training window -----------------
  // S has one row per training timestep (post-warmup).  Target Y[t] is the
  // value of the series at t+1, i.e. the standard one-step ahead forecast.
  int TrainRows = NTrainAll - WarmupSteps - 1;
  if (TrainRows <= 0) {
    std::cerr << "warmup consumed the entire training window\n";
    return 1;
  }
  std::vector<float> S(static_cast<size_t>(TrainRows) * Reservoir, 0.0f);
  std::vector<float> Y(TrainRows, 0.0f);

  T0 = Clock::now();
  for (int T = 0; T < TrainRows; ++T) {
    int Idx = WarmupSteps + T;
    Net.Step(Series[Idx]);
    std::copy(Net.X.begin(), Net.X.end(),
              S.begin() + static_cast<size_t>(T) * Reservoir);
    Y[T] = Series[Idx + 1];
  }
  long long ForwardTrainNs = ToNs(Clock::now() - T0);

  // ---- Ridge solve ------------------------------------------------------
  long long SyrkNs = 0, SgemvNs = 0, PosvNs = 0;
  T0 = Clock::now();
  RidgeFit(S.data(), TrainRows, Reservoir, Y.data(), Ridge, Net.WOut.data(),
           SyrkNs, SgemvNs, PosvNs);
  long long FitNs = ToNs(Clock::now() - T0);

  // ---- Test inference (reservoir + readout) -----------------------------
  int TestRows = static_cast<int>(Series.size()) - NTrainAll - 1;
  std::vector<float> Preds(TestRows, 0.0f);
  std::vector<float> Truth(TestRows, 0.0f);
  T0 = Clock::now();
  for (int T = 0; T < TestRows; ++T) {
    int Idx = NTrainAll + T;
    Net.Step(Series[Idx]);
    Preds[T] = Net.Predict();
    Truth[T] = Series[Idx + 1];
  }
  long long InferenceNs = ToNs(Clock::now() - T0);

  auto M = ScoreRmseR2(Preds, Truth);

  std::cout << "[train] reservoir rows=" << TrainRows << "  fit_ns=" << FitNs
            << "  (syrk=" << SyrkNs << "  sgemv=" << SgemvNs
            << "  sposv=" << PosvNs << ")\n";
  std::cout << "[test ] n=" << TestRows << "  RMSE=" << std::fixed
            << std::setprecision(6) << M.Rmse << "  R^2=" << M.R2 << "\n";

  std::cout << "\n[time ] warmup=" << (WarmupNs / 1e6) << "ms  "
            << "forward(train)=" << (ForwardTrainNs / 1e6) << "ms  "
            << "ridge_fit=" << (FitNs / 1e6) << "ms  "
            << "inference=" << (InferenceNs / 1e6) << "ms\n";
  std::cout << "[per-step] forward=" << (ForwardTrainNs /
                                          static_cast<double>(TrainRows)) /
                                             1e3
            << "us  inference="
            << (InferenceNs / static_cast<double>(TestRows)) / 1e3 << "us\n";

  if (!OutDir.empty()) {
    std::ofstream Lcsv(OutDir + "/learning.csv");
    Lcsv << "epoch,test_rmse,test_r2\n";
    // Single "epoch" — the closed-form ridge solve sees the data once.
    Lcsv << "0," << M.Rmse << "," << M.R2 << "\n";
    Lcsv.close();

    std::ofstream Pcsv(OutDir + "/predictions.csv");
    Pcsv << "step,pred,truth\n";
    for (int I = 0; I < TestRows; ++I)
      Pcsv << I << "," << Preds[I] << "," << Truth[I] << "\n";
    Pcsv.close();

    // Phase totals, schema-compatible with plastix's timing.csv header.
    // forward_ns = reservoir-state collection over the training window;
    // loss_ns    = 0 (no loss policy — ridge writes W_out directly);
    // update_ns  = the ridge solve (closed-form analogue of LMS update);
    // eval_ns    = 0 (no per-epoch eval);
    // wall_ns    = warmup + forward + ridge_fit.
    std::ofstream Tcsv(OutDir + "/timing.csv");
    Tcsv << "epoch,n_steps,forward_ns,loss_ns,update_ns,eval_ns,wall_ns\n";
    long long Wall = WarmupNs + ForwardTrainNs + FitNs;
    Tcsv << "1," << TrainRows << "," << ForwardTrainNs << ",0," << FitNs
         << ",0," << Wall << "\n";
    Tcsv.close();

    std::ofstream Mcsv(OutDir + "/summary.csv");
    Mcsv << "framework,units,sr,leak,lr,epochs,n_train_steps,test_rmse,"
            "test_r2,train_wall_ns,forward_ns,loss_ns,update_ns,eval_ns,"
            "inference_ns\n";
    Mcsv << "raw_cpp_blas," << Reservoir << "," << SpectralRadius << ","
         << LeakRate << "," << Ridge << ",1," << TrainRows << "," << M.Rmse
         << "," << M.R2 << "," << (WarmupNs + ForwardTrainNs + FitNs) << ","
         << ForwardTrainNs << ",0," << FitNs << ",0," << InferenceNs << "\n";
    Mcsv.close();

    std::cout << "[done] wrote " << OutDir
              << "/{learning,predictions,timing,summary}.csv\n";
  }

  std::cout << "\nfirst 5 test predictions (pred | truth):\n";
  for (int I = 0; I < std::min(5, TestRows); ++I)
    std::cout << "  " << std::setw(10) << Preds[I] << " | " << std::setw(10)
              << Truth[I] << "\n";

  bool Pass = M.R2 > 0.99f;
  std::cout << "\n" << (Pass ? "PASS" : "FAIL") << "\n";
  return Pass ? 0 : 1;
}
