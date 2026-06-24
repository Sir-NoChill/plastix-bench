// Workload 6 — Echo State Network on Mackey-Glass, pure-CUDA / cuBLAS + cuSOLVER.
//
// GPU port of 07_esn_mackey_class/cpp/06_esn_mackey_glass.cpp. Same math,
// same Mackey-Glass generator, same hyperparameters, same CLI and same output
// CSVs (so the existing run_benchmark.py sentinel drives it unchanged).
//
// Architecture (identical to the cpp impl / reservoirpy):
//   x(t) = (1-alpha) x(t-1) + alpha tanh(W_in u(t) + W_rec x(t-1) + b)
//   y(t) = W_out x(t)
// W_in (Nx1) and W_rec (NxN) are fixed random matrices, b=0, and W_out (1xN)
// is fit by closed-form ridge regression on the collected reservoir states:
//   W_out = (S^T S + lambda I)^-1 S^T Y    S = stacked states (T x N).
//
// GPU mapping:
//   * forward (hot path): per timestep, pre_act = W_rec @ x via cublasSgemv,
//     += u * W_in via cublasSaxpy, then a tiny leak+tanh kernel. The reservoir
//     state X lives on the device across the whole sweep; collected states are
//     written into a device S matrix (T x N row-major).
//   * ridge solve: S^T S via cublasDsyrk, += lambda I, S^T Y via cublasDgemv,
//     then the SPD/LU solve via cuSOLVER (cusolverDnDgetrf + cusolverDnDgetrs).
//     Done in double precision (and via LU) for the same reason as the cpp
//     impl: lambda=1e-5 is float-epsilon-tiny against the T-scaled diagonals of
//     STS, so a float Cholesky rejects the near-rank-deficient matrix. The
//     forward states are still collected/used in float32; only the normal
//     equations and the solve are promoted to double, matching the cpp baseline
//     exactly.
//
// Build (standalone):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++20 --extended-lambda \
//     --expt-relaxed-constexpr -arch=sm_89 -I common -I/usr/local/cuda/include \
//     07_esn_mackey_class/cuda/06_esn_mackey_glass.cu \
//     -L/usr/local/cuda/lib64 -lcublas -lcusolver -o run_benchmark
//
// CLI (matches the cpp binary so run_benchmark.py drives both):
//   argv[1] (optional): output directory for CSVs                (default "")
//   argv[2] (optional): unused — ridge solve is one-shot         (default 1)
//   argv[3] (optional): series CSV path                          (default "")

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

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

#define CUSOLVER_CHECK(x)                                                      \
  do {                                                                         \
    cusolverStatus_t s_ = (x);                                                 \
    if (s_ != CUSOLVER_STATUS_SUCCESS) {                                       \
      std::cerr << "[cusolver] error " << s_ << " at " << __LINE__ << "\n";    \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

namespace {

// ---- Hyperparameters (verbatim from the cpp impl) ---------------------------
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

// ---- Mackey-Glass generator (verbatim from the cpp impl) --------------------
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

// ---- Leak + tanh update kernel ---------------------------------------------
// x = (1 - alpha) x + alpha tanh(pre_act); pre_act already holds W_rec@x + u*W_in.
__global__ void LeakTanh(float *X, const float *PreAct, float OneMA, float Alpha,
                         int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    X[I] = OneMA * X[I] + Alpha * tanhf(PreAct[I]);
}

inline unsigned Grid(int N, unsigned Block) {
  return static_cast<unsigned>((N + Block - 1) / Block);
}
constexpr unsigned kBlock = 256;

// ---- ESN state (device-resident weights + scratch) --------------------------
// W_rec is generated on the host (identical RNG recipe to the cpp impl) then
// uploaded once. cuBLAS is column-major; the host (NxN) row-major W_rec equals
// the column-major (NxN) transpose, so to compute pre_act = W_rec @ x (the
// math we want) we call cublasSgemv with op_T over the uploaded buffer.
struct ESN {
  int N;
  std::vector<float> WOut; // N (single-output readout, host-side after solve)

  cublasHandle_t Bl;
  float *dWIn = nullptr;   // N
  float *dWRec = nullptr;  // N x N (row-major host buffer, used op_T on device)
  float *dX = nullptr;     // N (current reservoir state)
  float *dPreAct = nullptr;// N (scratch)
  float *dWOut = nullptr;  // N (readout weights, for inference)

  ESN(int Units, float SR, uint64_t S, cublasHandle_t Handle)
      : N(Units), WOut(Units, 0.0f), Bl(Handle) {
    std::mt19937_64 Rng(S);
    std::uniform_real_distribution<float> UIn(-InputScale, +InputScale);
    float RecStd = SR / std::sqrt(static_cast<float>(Units));
    std::normal_distribution<float> NRec(0.0f, RecStd);
    std::vector<float> WIn(Units);
    std::vector<float> WRec(static_cast<size_t>(Units) * Units);
    for (int I = 0; I < Units; ++I)
      WIn[I] = UIn(Rng);
    for (size_t I = 0; I < WRec.size(); ++I)
      WRec[I] = NRec(Rng);
    // Zero the diagonal (no self-loops — matches cpp / plastix).
    for (int I = 0; I < Units; ++I)
      WRec[static_cast<size_t>(I) * Units + I] = 0.0f;

    CUDA_CHECK(cudaMalloc(&dWIn, Units * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWRec, WRec.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX, Units * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPreAct, Units * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWOut, Units * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dWIn, WIn.data(), Units * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWRec, WRec.data(), WRec.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dX, 0, Units * sizeof(float)));
    CUDA_CHECK(cudaMemset(dWOut, 0, Units * sizeof(float)));
  }

  ~ESN() {
    cudaFree(dWIn);
    cudaFree(dWRec);
    cudaFree(dX);
    cudaFree(dPreAct);
    cudaFree(dWOut);
  }

  // Per-timestep reservoir update (the O(N^2) hot path):
  //   pre_act = W_rec @ x            (cublasSgemv, op_T over the row-major buf)
  //   pre_act += u * W_in            (cublasSaxpy)
  //   x = (1-alpha) x + alpha tanh(pre_act)   (LeakTanh kernel)
  void Step(float U) {
    const float Alpha = 1.0f, Beta0 = 0.0f;
    CUBLAS_CHECK(cublasSgemv(Bl, CUBLAS_OP_T, N, N, &Alpha, dWRec, N, dX, 1,
                             &Beta0, dPreAct, 1));
    CUBLAS_CHECK(cublasSaxpy(Bl, N, &U, dWIn, 1, dPreAct, 1));
    LeakTanh<<<Grid(N, kBlock), kBlock>>>(dX, dPreAct, 1.0f - LeakRate, LeakRate,
                                          N);
  }

  // y = W_out . x   (dot product of readout weights and current state)
  float Predict() {
    float Y = 0.0f;
    CUBLAS_CHECK(cublasSdot(Bl, N, dWOut, 1, dX, 1, &Y));
    return Y;
  }

  void UploadWOut() {
    CUDA_CHECK(cudaMemcpy(dWOut, WOut.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
};

// ---- Ridge solver (GPU: cuBLAS normal equations + cuSOLVER LU solve) --------
// Closed-form W = (S^T S + lambda I)^-1 S^T Y for a (T x N) state matrix S
// (device, row-major), (T) target vector Y, single-output readout (N-vector W).
//
// Steps:
//   STS = S^T S         cublasDsyrk     (symmetric rank-k; ~T*N^2/2 ops)
//   STS += lambda I     diagonal regularizer (host fill-in of the symmetric half)
//   STY = S^T Y         cublasDgemv     (~T*N ops)
//   solve STS W = STY   cusolverDnDgetrf + cusolverDnDgetrs (LU + tri-solve)
//
// Double precision + LU, matching the cpp impl's rationale (float Cholesky
// rejects the near-rank-deficient matrix when lambda << diag). Inputs come in
// as float (dS row-major T x N, dY length T); we copy/cast to double on host
// then drive the double-precision GPU path.
void RidgeFit(cublasHandle_t Bl, cusolverDnHandle_t Cs, const float *dS, int T,
              int N, const float *dY, float Lambda, float *WOut,
              long long &SyrkNs, long long &GemvNs, long long &SolveNs) {
  using Clock = std::chrono::steady_clock;
  using ns = std::chrono::nanoseconds;

  // Pull the float S / Y back, cast to double, push as double device buffers.
  std::vector<float> Sf(static_cast<size_t>(T) * N);
  std::vector<float> Yf(T);
  CUDA_CHECK(cudaMemcpy(Sf.data(), dS, Sf.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Yf.data(), dY, T * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<double> Sd(Sf.begin(), Sf.end());
  std::vector<double> Yd(Yf.begin(), Yf.end());

  double *dSd = nullptr, *dYd = nullptr, *dSTS = nullptr, *dSTY = nullptr;
  CUDA_CHECK(cudaMalloc(&dSd, Sd.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dYd, T * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dSTS, static_cast<size_t>(N) * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dSTY, N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dSd, Sd.data(), Sd.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dYd, Yd.data(), T * sizeof(double),
                        cudaMemcpyHostToDevice));

  // STS = S^T S.  S is row-major (T x N) == column-major (N x T), ld=N. With
  // cuBLAS column-major, syrk with op_N on a column-major (N x T) matrix forms
  // A A^T = (N x T)(T x N) = STS (N x N). Lower fill (column-major) corresponds
  // to the upper triangle when re-read row-major; we symmetrize fully on host.
  const double One = 1.0, Zero = 0.0;
  auto T0 = Clock::now();
  CUBLAS_CHECK(cublasDsyrk(Bl, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, T, &One,
                           dSd, N, &Zero, dSTS, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  auto T1 = Clock::now();

  // STY = S^T Y. Column-major S is (N x T) ld=N; op_N gives (N x T)(T) = S^T Y.
  auto T2 = Clock::now();
  CUBLAS_CHECK(cublasDgemv(Bl, CUBLAS_OP_N, N, T, &One, dSd, N, dYd, 1, &Zero,
                           dSTY, 1));
  CUDA_CHECK(cudaDeviceSynchronize());
  auto T3 = Clock::now();

  // Symmetrize STS (syrk only wrote one triangle) and add lambda on the diag.
  std::vector<double> STS(static_cast<size_t>(N) * N);
  CUDA_CHECK(cudaMemcpy(STS.data(), dSTS, STS.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  // dSTS holds the symmetric matrix in column-major; syrk wrote only the lower
  // (col-major) triangle. Mirror lower into upper and add lambda to the diag.
  // Column J, row I element lives at index J*N + I (column-major). For I<J the
  // (J,I) entry (col-major upper) is filled from the (I,J) entry (col-major
  // lower).
  for (int J = 0; J < N; ++J) {
    for (int I = 0; I < J; ++I)
      STS[static_cast<size_t>(J) * N + I] =
          STS[static_cast<size_t>(I) * N + J];
    STS[static_cast<size_t>(J) * N + J] += static_cast<double>(Lambda);
  }
  CUDA_CHECK(cudaMemcpy(dSTS, STS.data(), STS.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  // LU factor + solve: cusolverDnDgetrf / Dgetrs. STS is symmetric so its
  // column-major and row-major layouts coincide after symmetrization.
  int LWork = 0;
  CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(Cs, N, N, dSTS, N, &LWork));
  double *dWork = nullptr;
  int *dIpiv = nullptr, *dInfo = nullptr;
  CUDA_CHECK(cudaMalloc(&dWork, LWork * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dIpiv, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));

  auto T4 = Clock::now();
  CUSOLVER_CHECK(cusolverDnDgetrf(Cs, N, N, dSTS, N, dWork, dIpiv, dInfo));
  CUSOLVER_CHECK(cusolverDnDgetrs(Cs, CUBLAS_OP_N, N, 1, dSTS, N, dIpiv, dSTY, N,
                                  dInfo));
  CUDA_CHECK(cudaDeviceSynchronize());
  auto T5 = Clock::now();

  int Info = 0;
  CUDA_CHECK(cudaMemcpy(&Info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (Info != 0) {
    std::cerr << "[fatal] cusolverDnDgetrf/getrs failed with info=" << Info
              << "\n";
    std::exit(3);
  }

  std::vector<double> WOutD(N);
  CUDA_CHECK(cudaMemcpy(WOutD.data(), dSTY, N * sizeof(double),
                        cudaMemcpyDeviceToHost));
  for (int I = 0; I < N; ++I)
    WOut[I] = static_cast<float>(WOutD[I]);

  SyrkNs = std::chrono::duration_cast<ns>(T1 - T0).count();
  GemvNs = std::chrono::duration_cast<ns>(T3 - T2).count();
  SolveNs = std::chrono::duration_cast<ns>(T5 - T4).count();

  cudaFree(dSd);
  cudaFree(dYd);
  cudaFree(dSTS);
  cudaFree(dSTY);
  cudaFree(dWork);
  cudaFree(dIpiv);
  cudaFree(dInfo);
}

// ---- Metrics (verbatim from the cpp impl) -----------------------------------
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
  std::string SeriesPath = (Argc > 3) ? Argv[3] : "";

  std::cout << "pure-CUDA/cuBLAS+cuSOLVER ESN on Mackey-Glass\n";
  std::cout << "=============================================\n";
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

  // ---- Data (verbatim from the cpp impl) ---------------------------------
  std::vector<float> Series = SeriesPath.empty()
                                  ? MackeyGlass(SeriesLenDefault, MGTau, Seed)
                                  : LoadSeriesCsv(SeriesPath);
  NormalizeInPlace(Series);
  int NTrainAll = static_cast<int>(Series.size() * TrainFrac);
  if (NTrainAll + 1 >= static_cast<int>(Series.size())) {
    std::cerr << "series too short\n";
    return 1;
  }

  cublasHandle_t Bl;
  CUBLAS_CHECK(cublasCreate(&Bl));
  cusolverDnHandle_t Cs;
  CUSOLVER_CHECK(cusolverDnCreate(&Cs));

  ESN Net(Reservoir, SpectralRadius, Seed, Bl);

  // ---- Warmup (no learning, no state recording) --------------------------
  auto T0 = Clock::now();
  for (int T = 0; T < WarmupSteps && T < NTrainAll; ++T)
    Net.Step(Series[T]);
  CUDA_CHECK(cudaDeviceSynchronize());
  long long WarmupNs = ToNs(Clock::now() - T0);

  // ---- Collect reservoir states over the training window -----------------
  int TrainRows = NTrainAll - WarmupSteps - 1;
  if (TrainRows <= 0) {
    std::cerr << "warmup consumed the entire training window\n";
    return 1;
  }
  // S device buffer: TrainRows x Reservoir, row-major (state per row).
  float *dS = nullptr, *dY = nullptr;
  CUDA_CHECK(cudaMalloc(&dS, static_cast<size_t>(TrainRows) * Reservoir *
                                 sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, TrainRows * sizeof(float)));
  std::vector<float> Y(TrainRows, 0.0f);

  T0 = Clock::now();
  for (int T = 0; T < TrainRows; ++T) {
    int Idx = WarmupSteps + T;
    Net.Step(Series[Idx]);
    // Copy current state (Net.dX, length N) into row T of dS.
    CUDA_CHECK(cudaMemcpy(dS + static_cast<size_t>(T) * Reservoir, Net.dX,
                          Reservoir * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    Y[T] = Series[Idx + 1];
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  long long ForwardTrainNs = ToNs(Clock::now() - T0);
  CUDA_CHECK(cudaMemcpy(dY, Y.data(), TrainRows * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ---- Ridge solve ------------------------------------------------------
  long long SyrkNs = 0, GemvNs = 0, SolveNs = 0;
  T0 = Clock::now();
  RidgeFit(Bl, Cs, dS, TrainRows, Reservoir, dY, Ridge, Net.WOut.data(), SyrkNs,
           GemvNs, SolveNs);
  long long FitNs = ToNs(Clock::now() - T0);
  Net.UploadWOut();

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
  CUDA_CHECK(cudaDeviceSynchronize());
  long long InferenceNs = ToNs(Clock::now() - T0);

  auto M = ScoreRmseR2(Preds, Truth);

  std::cout << "[train] reservoir rows=" << TrainRows << "  fit_ns=" << FitNs
            << "  (syrk=" << SyrkNs << "  gemv=" << GemvNs
            << "  getrf/getrs=" << SolveNs << ")\n";
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
    Lcsv << "0," << M.Rmse << "," << M.R2 << "\n";
    Lcsv.close();

    std::ofstream Pcsv(OutDir + "/predictions.csv");
    Pcsv << "step,pred,truth\n";
    for (int I = 0; I < TestRows; ++I)
      Pcsv << I << "," << Preds[I] << "," << Truth[I] << "\n";
    Pcsv.close();

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
    Mcsv << "cuda_cublas_cusolver," << Reservoir << "," << SpectralRadius << ","
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

  cudaFree(dS);
  cudaFree(dY);
  cusolverDnDestroy(Cs);
  cublasDestroy(Bl);

  bool Pass = M.R2 > 0.99f;
  std::cout << "\n" << (Pass ? "PASS" : "FAIL") << "\n";
  return Pass ? 0 : 1;
}
