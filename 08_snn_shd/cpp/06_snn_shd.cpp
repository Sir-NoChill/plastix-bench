// Workload 6 — sparse spiking neural network on SHD, raw C++ + OpenBLAS.
//
// Pure-C++ counterpart to:
//   * the snnTorch BPTT port            (snn_shd/pytorch/)
//   * the Plastix sparse e-prop port    (snn_shd/plastix/)
//
// Same network shape as the Plastix port — 700 input channels → K-sparse
// fan-in into a layer of N_hid LIF neurons → dense readout into N_out
// non-spiking integrators — and the same e-prop learning rule with
// random-feedback alignment for the hidden learning signal. The
// implementation strategy mirrors `ccwc_ncp/cpp`: dense matrices behind
// binary masks for the sparse blocks, BLAS sgemv / sger for every per-
// timestep reduction or rank-1 update. The forward and backward inner
// loops collapse to:
//
//   forward step  : 2 × cblas_sgemv  (W_in · u, W_out · spk)
//   backward step : 1 × cblas_sgemv  (B_out^T · L_out)
//   eligibility   : 2 × cblas_sger   (outer products)
//   weight step   : masked elementwise updates with clipping
//
// On SHD's 700-input dense weight matrix the input→hidden mask is sparse
// (only K × N_hid live entries out of 700 × N_hid), but BLAS sgemv on
// the dense layout still beats sparse iteration at this size because the
// matrix fits in L2 and the multiply-by-zero waste is amortised through
// the vectorised inner loop. The mask is enforced on weight and
// eligibility writes so dead positions stay zero — the e-prop trace
// would otherwise drift them.
//
// Output schema (.history.jsonl + .summary.csv) matches the Plastix port
// so the orchestrator picks the two up uniformly.

#include "common.hpp"

#include <cblas.h>

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

namespace {

// ---------------------------------------------------------------------------
// Hyperparameters
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
// Dataset loader (.plxbin format produced by snn_shd/pytorch/data.py)
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
// Model state — dense + mask, BLAS-driven
// ---------------------------------------------------------------------------

struct Model {
  // Sizes
  size_t NIn = 0, NHid = 0, NOut = 0;

  // Weights (row-major)
  std::vector<float> WIn;       // NHid × NIn   — sparse via mask
  std::vector<float> WOut;      // NOut × NHid  — dense
  std::vector<float> BOut;      // NOut × NHid  — random feedback, dense

  // Masks (1 = live, 0 = dead)
  std::vector<uint8_t> MIn;     // NHid × NIn
  size_t NLiveIn = 0;

  // Eligibility traces, one per weight that we actually learn.
  std::vector<float> EIn;       // NHid × NIn
  std::vector<float> EOut;      // NOut × NHid

  // Per-unit state (between timesteps; zeroed per example)
  std::vector<float> MemHid;    // NHid
  std::vector<float> MemOut;    // NOut
  std::vector<float> SpkHid;    // NHid
  std::vector<float> ZHid;      // NHid, saved pre-act for surrogate
  std::vector<float> LogitAcc;  // NOut
  std::vector<float> LHid;      // NHid, learning signal
  std::vector<float> LOut;      // NOut, learning signal
  std::vector<float> Psi;       // NHid, surrogate(z) at this step

  // Scratch reused every step
  std::vector<float> Drive;     // NHid

  void Build(size_t n_in, size_t n_hid, size_t n_out, size_t fan_in,
             float w_scale, float fb_scale, uint64_t seed) {
    NIn = n_in; NHid = n_hid; NOut = n_out;
    WIn.assign(NHid * NIn, 0.0f);
    WOut.assign(NOut * NHid, 0.0f);
    BOut.assign(NOut * NHid, 0.0f);
    MIn.assign(NHid * NIn, 0);
    EIn.assign(NHid * NIn, 0.0f);
    EOut.assign(NOut * NHid, 0.0f);
    MemHid.assign(NHid, 0.0f);
    MemOut.assign(NOut, 0.0f);
    SpkHid.assign(NHid, 0.0f);
    ZHid.assign(NHid, 0.0f);
    LogitAcc.assign(NOut, 0.0f);
    LHid.assign(NHid, 0.0f);
    LOut.assign(NOut, 0.0f);
    Psi.assign(NHid, 0.0f);
    Drive.assign(NHid, 0.0f);

    std::mt19937_64 Rng(seed);
    // Xavier-uniform-ish init scaled to the fan-in (matches Plastix port).
    float BoundIn =
        w_scale * std::sqrt(6.0f / static_cast<float>(fan_in + NHid));
    float BoundOut =
        std::sqrt(6.0f / static_cast<float>(NHid + NOut));
    float BoundFb =
        fb_scale * std::sqrt(6.0f / static_cast<float>(NHid + NOut));
    std::uniform_real_distribution<float> Win(-BoundIn, BoundIn);
    std::uniform_real_distribution<float> Wout(-BoundOut, BoundOut);
    std::uniform_real_distribution<float> Bout(-BoundFb, BoundFb);

    // K-sparse fan-in: for each hidden unit, sample `fan_in` distinct input
    // channels via partial Fisher–Yates. Same shape as the Plastix port; the
    // seed isn't bit-identical (the per-call PRNG state differs from the
    // framework's), but the topology distribution is the same.
    std::vector<size_t> Pool(NIn);
    std::iota(Pool.begin(), Pool.end(), 0);
    for (size_t H = 0; H < NHid; ++H) {
      size_t K = std::min(fan_in, NIn);
      for (size_t I = 0; I < K; ++I) {
        std::uniform_int_distribution<size_t> Pick(I, NIn - 1);
        std::swap(Pool[I], Pool[Pick(Rng)]);
        size_t Src = Pool[I];
        MIn[H * NIn + Src] = 1;
        WIn[H * NIn + Src] = Win(Rng);
      }
    }
    NLiveIn = NHid * fan_in;
    for (auto &v : WOut) v = Wout(Rng);
    for (auto &v : BOut) v = Bout(Rng);
  }

  void ResetSequence() {
    std::fill(MemHid.begin(), MemHid.end(), 0.0f);
    std::fill(MemOut.begin(), MemOut.end(), 0.0f);
    std::fill(SpkHid.begin(), SpkHid.end(), 0.0f);
    std::fill(ZHid.begin(), ZHid.end(), 0.0f);
    std::fill(LogitAcc.begin(), LogitAcc.end(), 0.0f);
    std::fill(LHid.begin(), LHid.end(), 0.0f);
    std::fill(LOut.begin(), LOut.end(), 0.0f);
    std::fill(EIn.begin(), EIn.end(), 0.0f);
    std::fill(EOut.begin(), EOut.end(), 0.0f);
  }

  // -------------------------------------------------------------------------
  // Forward step: subtract-reset LIF for hidden, integrator for readout.
  // `LogitScale` is folded into the readout-membrane accumulation so the
  // softmax sees the per-step mean — same convention as the Plastix port.
  // -------------------------------------------------------------------------
  void ForwardStep(const float *u, float beta, float thr, float logit_scale) {
    // Drive_hid = W_in · u
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(NHid), static_cast<int>(NIn),
                1.0f, WIn.data(), static_cast<int>(NIn),
                u, 1, 0.0f, Drive.data(), 1);
    // Hidden LIF subtract-reset.
    for (size_t H = 0; H < NHid; ++H) {
      float Mem = beta * MemHid[H] + Drive[H];
      float Z = Mem - thr;
      float Spk = (Z >= 0.0f) ? 1.0f : 0.0f;
      Mem -= thr * Spk;
      MemHid[H] = Mem;
      SpkHid[H] = Spk;
      ZHid[H] = Z;
    }
    // Drive_out = W_out · spk
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(NOut), static_cast<int>(NHid),
                1.0f, WOut.data(), static_cast<int>(NHid),
                SpkHid.data(), 1, 0.0f, Drive.data(), 1);
    // Output integrator + scaled logit accumulation.
    for (size_t O = 0; O < NOut; ++O) {
      MemOut[O] = beta * MemOut[O] + Drive[O];
      LogitAcc[O] += MemOut[O] * logit_scale;
    }
  }

  // -------------------------------------------------------------------------
  // Backward + eligibility + weight step for this timestep. `target` is
  // non-null only on the final timestep (the rest accumulate elig only).
  // -------------------------------------------------------------------------
  void EpropStep(const float *u, const float *target,
                 const HP &H, float beta_trace, float lr,
                 bool do_update) {
    // 1. Loss signal (only at terminal step; else L_out = 0).
    if (target != nullptr) {
      // softmax + dL/dlogit = softmax - one_hot
      float maxl = -1e30f;
      for (size_t O = 0; O < NOut; ++O)
        if (LogitAcc[O] > maxl) maxl = LogitAcc[O];
      double Z = 0.0;
      for (size_t O = 0; O < NOut; ++O)
        Z += std::exp(LogitAcc[O] - maxl);
      for (size_t O = 0; O < NOut; ++O) {
        float P = static_cast<float>(std::exp(LogitAcc[O] - maxl) / Z);
        LOut[O] = P - target[O];
      }
    } else {
      std::fill(LOut.begin(), LOut.end(), 0.0f);
    }

    // 2. Hidden learning signal: L_hid = (B_out^T · L_out) ⊙ surrogate(z_hid)
    //    Surrogate first (it's per-unit and reusable for the elig step).
    for (size_t H_ = 0; H_ < NHid; ++H_) {
      float den = 1.0f + H.SurrogateSlope * std::fabs(ZHid[H_]);
      Psi[H_] = 1.0f / (den * den);
    }
    cblas_sgemv(CblasRowMajor, CblasTrans,
                static_cast<int>(NOut), static_cast<int>(NHid),
                1.0f, BOut.data(), static_cast<int>(NHid),
                LOut.data(), 1, 0.0f, LHid.data(), 1);
    for (size_t H_ = 0; H_ < NHid; ++H_)
      LHid[H_] *= Psi[H_];

    // 3. Eligibility traces — outer products via cblas_sger (rank-1 update).
    //    E_in[h, i]   = beta_trace · E_in[h, i]   + Psi[h]   · u[i]
    //    E_out[o, h]  = beta_trace · E_out[o, h]  + 1        · spk[h]
    //    (Readout integrator has linear gradient → Psi_o = 1.)
    cblas_sscal(static_cast<int>(NHid * NIn), beta_trace, EIn.data(), 1);
    cblas_sger(CblasRowMajor,
               static_cast<int>(NHid), static_cast<int>(NIn),
               1.0f, Psi.data(), 1, u, 1,
               EIn.data(), static_cast<int>(NIn));
    cblas_sscal(static_cast<int>(NOut * NHid), beta_trace, EOut.data(), 1);
    {
      static thread_local std::vector<float> Ones;
      if (Ones.size() < NOut) Ones.assign(NOut, 1.0f);
      cblas_sger(CblasRowMajor,
                 static_cast<int>(NOut), static_cast<int>(NHid),
                 1.0f, Ones.data(), 1, SpkHid.data(), 1,
                 EOut.data(), static_cast<int>(NHid));
    }

    if (!do_update) return;

    // 4. Weight step. Only fire when there's a non-zero learning signal,
    //    i.e. at the terminal step. Clip both per-step delta and the
    //    weight magnitude — same shape as ccwc_ncp/cpp.
    bool any_l_out = false;
    for (size_t O = 0; O < NOut; ++O)
      if (LOut[O] != 0.0f) { any_l_out = true; break; }
    if (!any_l_out) return;

    // W_in step (masked): dW[h, i] = L_hid[h] · E_in[h, i]
    for (size_t Hh = 0; Hh < NHid; ++Hh) {
      float Lh = LHid[Hh];
      if (Lh == 0.0f) continue;
      const uint8_t *MaskRow = MIn.data() + Hh * NIn;
      float *Wrow = WIn.data() + Hh * NIn;
      float *Erow = EIn.data() + Hh * NIn;
      for (size_t Ii = 0; Ii < NIn; ++Ii) {
        if (!MaskRow[Ii]) continue;
        float Delta = lr * Lh * Erow[Ii];
        if (Delta >  H.ClipDelta) Delta =  H.ClipDelta;
        if (Delta < -H.ClipDelta) Delta = -H.ClipDelta;
        float Wn = Wrow[Ii] - Delta;
        if (Wn >  H.WMax) Wn =  H.WMax;
        if (Wn < -H.WMax) Wn = -H.WMax;
        Wrow[Ii] = Wn;
      }
    }
    // W_out step (dense): dW[o, h] = L_out[o] · E_out[o, h]
    for (size_t Oo = 0; Oo < NOut; ++Oo) {
      float Lo = LOut[Oo];
      if (Lo == 0.0f) continue;
      float *Wrow = WOut.data() + Oo * NHid;
      float *Erow = EOut.data() + Oo * NHid;
      for (size_t Hh = 0; Hh < NHid; ++Hh) {
        float Delta = lr * Lo * Erow[Hh];
        if (Delta >  H.ClipDelta) Delta =  H.ClipDelta;
        if (Delta < -H.ClipDelta) Delta = -H.ClipDelta;
        float Wn = Wrow[Hh] - Delta;
        if (Wn >  H.WMax) Wn =  H.WMax;
        if (Wn < -H.WMax) Wn = -H.WMax;
        Wrow[Hh] = Wn;
      }
    }
  }

  int Argmax() const {
    int best = 0;
    float bestv = LogitAcc[0];
    for (size_t O = 1; O < NOut; ++O)
      if (LogitAcc[O] > bestv) { bestv = LogitAcc[O]; best = static_cast<int>(O); }
    return best;
  }
};

// ---------------------------------------------------------------------------
// Eval helpers
// ---------------------------------------------------------------------------

static double EvalAccuracy(Model &M, const Dataset &D, const HP &H,
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
  std::vector<float> buf;
  if (time_shuffle) buf.assign(D.NChannels, 0.0f);
  for (size_t i = 0; i < N; ++i) {
    M.ResetSequence();
    if (time_shuffle) std::shuffle(perm.begin(), perm.end(), sr);
    for (size_t T = 0; T < D.NBins; ++T) {
      const float *src = time_shuffle ? D.Sample(i, perm[T])
                                       : D.Sample(i, T);
      M.ForwardStep(src, H.Beta, H.Threshold, logit_scale);
    }
    if (M.Argmax() == static_cast<int>(D.Y[i])) ++correct;
  }
  return N ? static_cast<double>(correct) / N : 0.0;
}

static double MeanFiringRate(Model &M, const Dataset &D, const HP &H,
                             size_t cap) {
  size_t N = std::min(cap, D.NSamples);
  if (N == 0) return 0.0;
  double total = 0.0;
  size_t slots = 0;
  float logit_scale = 1.0f / static_cast<float>(D.NBins);
  for (size_t i = 0; i < N; ++i) {
    M.ResetSequence();
    for (size_t T = 0; T < D.NBins; ++T) {
      M.ForwardStep(D.Sample(i, T), H.Beta, H.Threshold, logit_scale);
      for (size_t h = 0; h < M.NHid; ++h) total += M.SpkHid[h];
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

  // .plxbin cache is produced by snn_shd/pytorch/data.py --export.
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
  std::cout << "[info] train=" << Train.NSamples << " test=" << Test.NSamples
            << " n_in=" << NIn << " n_hid=" << H.NHid
            << " n_out=" << H.NumClasses << " n_bins=" << H.NBins
            << " fan_in=" << H.FanIn << " epochs=" << H.Epochs
            << " lr=" << H.Lr << "\n";

  Model M;
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 41ull;
  M.Build(NIn, H.NHid, H.NumClasses, H.FanIn,
          H.WeightScale, H.FeedbackScale, SeedBase + 1);
  size_t NConns = M.NLiveIn + M.NOut * M.NHid;
  std::cout << "[info] live conns=" << NConns
            << "  (vs " << (NIn * H.NHid + H.NHid * H.NumClasses)
            << " dense)\n";

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "snn_shd");
  bench::StructuralLog Log(HistPath);

  // Initial point — untrained, pure forward.
  size_t EvalCap = H.MaxEvalRows;
  double InitTest = EvalAccuracy(M, Test, H, EvalCap);
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
  std::vector<float> OneHotV(H.NumClasses, 0.0f);
  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();

  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    double LossSum = 0.0;
    size_t LossCnt = 0;
    size_t Correct = 0;

    for (size_t Idx : Perm) {
      M.ResetSequence();
      std::fill(OneHotV.begin(), OneHotV.end(), 0.0f);
      OneHotV[static_cast<size_t>(Train.Y[Idx])] = 1.0f;

      for (size_t T = 0; T < Train.NBins; ++T) {
        const float *u = Train.Sample(Idx, T);
        bool final = (T + 1 == Train.NBins);
        Timer.Tick();
        M.ForwardStep(u, H.Beta, H.Threshold, logit_scale);
        Timer.MarkForward();
        // EpropStep bundles loss + surrogate-gradient backward + eligibility
        // + weight update -- the e-prop trace and weight step are
        // interleaved. Reported under `backward_ns_mean` so it lines up with
        // PyTorch's `loss.backward() + optimizer.step()` bundle on the
        // same workload.
        M.EpropStep(u, final ? OneHotV.data() : nullptr,
                    H, H.BetaTrace, H.Lr, /*do_update=*/true);
        Timer.MarkBackward();
        Timer.StepDone();
      }

      // Cross-entropy loss for reporting (uses final-step LogitAcc).
      float maxl = *std::max_element(M.LogitAcc.begin(), M.LogitAcc.end());
      double Z = 0.0;
      for (float v : M.LogitAcc) Z += std::exp(v - maxl);
      float P = static_cast<float>(
          std::exp(M.LogitAcc[Train.Y[Idx]] - maxl) / Z);
      LossSum += -std::log(std::max(P, 1e-30f));
      ++LossCnt;
      if (M.Argmax() == static_cast<int>(Train.Y[Idx])) ++Correct;
    }

    double TrLoss = LossCnt ? LossSum / LossCnt : 0.0;
    double TrAcc = TrainCap ? static_cast<double>(Correct) / TrainCap : 0.0;
    double TestAcc = 0.0, Rate = 0.0;
    if (Ep == H.Epochs || (Ep % H.EvalEvery) == 0) {
      TestAcc = EvalAccuracy(M, Test, H, EvalCap);
      Rate = MeanFiringRate(M, Train, H, std::min<size_t>(128, TrainCap));
    }
    if (TestAcc > BestTest) BestTest = TestAcc;

    double NegMetric = 1.0 - TestAcc;   // "loss-like" so existing plot tools
                                        // can put it on the val_loss axis.
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
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  double FinalTest = EvalAccuracy(M, Test, H, EvalCap);
  double ShuffledTest = EvalAccuracy(M, Test, H, EvalCap,
                                     /*time_shuffle=*/true,
                                     /*shuffle_seed=*/Args.Seed + 9999);

  Log.Flush();
  bench::SummaryWriter S;
  S.Set("workload", std::string{"06_snn_shd"});
  S.Set("dataset", std::string{"SHD"});
  S.Set("backend", std::string{"openblas"});
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
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_acc=" << FinalTest
            << "  shuffled=" << ShuffledTest
            << "  drop=" << (FinalTest - ShuffledTest) << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";
  (void)LogPath;
  return 0;
}
