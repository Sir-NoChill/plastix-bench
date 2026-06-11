// Workload 2 / 5 — IDEMPOTENT SHRINKAGE regime, Plastix translation.
//
// Iterative Magnitude Pruning (Frankle & Carbin 2018) of a multi-layer
// classification MLP on a UCR-style synthetic time-series task. Mirrors
// 02_idempotent_imp.py and exhibits the same regime signature:
//
//   - alive-edge count decreases monotonically across rounds
//   - the dead-edge set is a growing union of prior rounds' kills
//   - the run terminates when a round kills zero edges (fixed point)
//
// Plastix policy mapping:
//
//   ForwardPass        custom    ReLU hidden, linear (logits) output
//   BackwardPass       custom    backprop through ReLU using stored pre-act
//   Loss               SoftmaxCrossEntropyLoss  (built-in)
//   UpdateConn         custom    plain SGD on WeightTag
//   PruneConn          custom    ShouldPrune := Armed && |w| <= Threshold
//   AddUnit / AddConn  NoX       IMP only removes
//
// IMP "rounds" are orchestrated from main():
//   1. Run E0 dense training epochs via N.DoStep(...).
//   2. Walk the connection allocator on host, find the kth-smallest |w|.
//   3. Set PruneConn::Threshold + PruneConn::Armed via static members; call
//      N.DoPruneConnections() directly — the public Do* phase methods expose
//      the same step-by-step API used internally by DoStep.
//   4. Disarm; finetune via N.DoStep(...) for Ef epochs.
//   5. Repeat until a round kills no edges.

#include "plastix/common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace {

struct HP {
  // Sizing matches the PyTorch reference defaults
  // (02_idempotent_imp.py): length=128, n_classes=5, hidden=512,
  // depth=4 (= 3 ReLU hidden + 1 linear output), prune 20% / round.
  size_t InLen = 128;
  size_t NumClasses = 5;
  size_t Hidden = 512;
  size_t Depth = 4;
  size_t NPerClass = 200;
  float Snr = 1.5f;
  size_t InitEpochs = 20;
  size_t FinetuneEpochs = 4;
  size_t MaxRounds = 20;
  float PruneFrac = 0.2f;
  float Lr = 1e-3f;
  float FinetuneLrScale = 0.3f;
};

struct PreActTag {};
struct GradPreActTag {};
struct IsOutputTag {};

struct Forward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Z) {
    plastix::GetField<PreActTag>(U, Id) = Z;
    plastix::GetActivation(U, Id) = plastix::GetField<IsOutputTag>(U, Id)
                                        ? plastix::math::Linear(Z)
                                        : plastix::math::ReLU(Z);
  }
};

struct Backward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t ToId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) *
           plastix::GetField<GradPreActTag>(U, ToId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    bool IsOut = plastix::GetField<IsOutputTag>(U, Id);
    float DLDA = IsOut ? plastix::GetBackwardAcc(U, Id) : Accumulated;
    float Z = plastix::GetField<PreActTag>(U, Id);
    float DPhiDz = IsOut ? plastix::math::LinearGrad(Z)
                         : plastix::math::ReLUGradFromPreact(Z);
    plastix::GetField<GradPreActTag>(U, Id) = DLDA * DPhiDz;
  }
};

struct UpdateConn {
  static float Lr;
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &) {
    float Grad = plastix::GetField<GradPreActTag>(U, DstId);
    float A = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= Lr * Grad * A;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};
float UpdateConn::Lr = 1e-3f;

// PruneConn policy: only fires when armed. The host code arms it before
// calling N.DoPruneConnections() between training rounds, then disarms.
struct PruneConn {
  static bool Armed;
  static float Threshold;
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &C,
                                     size_t ConnId, auto &) {
    if (!Armed)
      return false;
    return std::abs(plastix::GetWeight(C, ConnId)) <= Threshold;
  }
};
bool PruneConn::Armed = false;
float PruneConn::Threshold = 0.0f;

struct IMPTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = Forward;
  using BackwardPass = Backward;
  using Loss = plastix::SoftmaxCrossEntropyLoss;
  using UpdateConn = ::UpdateConn;
  using PruneConn = ::PruneConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<PreActTag, float>,
      plastix::alloc::SOAField<GradPreActTag, float>,
      plastix::alloc::SOAField<IsOutputTag, bool>>;
  // PyTorch defaults: length=128 -> input 128, 3 hidden of 512, 5 outputs.
  // ~1669 units peak, ~592k initial connections.
  static constexpr size_t UnitCapacity = 4096;
  static constexpr size_t ConnCapacity = 1u << 20; // 1M
};
static_assert(plastix::NetworkTraits<IMPTraits>);

using Net = plastix::Network<IMPTraits>;

struct UniformInit {
  uint64_t Seed;
  float Limit;
  void operator()(auto &CA, auto Id) const {
    plastix::GetWeight(CA, Id) =
        plastix::UniformReal(Seed, static_cast<uint64_t>(Id), -Limit, Limit);
  }
};
struct MarkOutput {
  void operator()(auto &UA, auto Id) const {
    plastix::GetField<IsOutputTag>(UA, Id) = true;
  }
};

using FCHidden = plastix::FullyConnected<UniformInit>;
using FCOut = plastix::FullyConnected<UniformInit, MarkOutput>;

// --- synthetic UCR-style multi-class shapelet data ---------------------------

struct Dataset {
  std::vector<std::vector<float>> X;
  std::vector<int> Y;
};

static Dataset SynthUcr(size_t NPerClass, size_t NClasses, size_t Length,
                        float Snr, uint32_t Seed) {
  constexpr float Pi = 3.14159265358979323846f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> N(0.0f, 1.0f / Snr);
  Dataset D;
  D.X.reserve(NPerClass * NClasses);
  D.Y.reserve(NPerClass * NClasses);
  for (size_t K = 0; K < NClasses; ++K) {
    std::vector<float> Base(Length);
    for (size_t T = 0; T < Length; ++T) {
      float Tt = static_cast<float>(T) / Length;
      Base[T] = std::sin(2.0f * Pi * (K + 1) * Tt) +
                0.4f * std::sin(2.0f * Pi * (2 * K + 3) * Tt);
    }
    for (size_t I = 0; I < NPerClass; ++I) {
      std::vector<float> Row = Base;
      for (auto &V : Row)
        V += N(Rng);
      D.X.push_back(std::move(Row));
      D.Y.push_back(static_cast<int>(K));
    }
  }
  // shuffle (preserve seed-dependence)
  std::vector<size_t> Idx(D.X.size());
  for (size_t I = 0; I < Idx.size(); ++I)
    Idx[I] = I;
  std::shuffle(Idx.begin(), Idx.end(), Rng);
  Dataset Out;
  Out.X.reserve(D.X.size());
  Out.Y.reserve(D.Y.size());
  for (size_t I : Idx) {
    Out.X.push_back(std::move(D.X[I]));
    Out.Y.push_back(D.Y[I]);
  }
  return Out;
}

// Per-instance z-score (UCR convention).
static void NormaliseInstance(std::vector<std::vector<float>> &X) {
  for (auto &R : X) {
    double M = 0.0, V = 0.0;
    for (float V0 : R)
      M += V0;
    M /= R.size();
    for (float V0 : R)
      V += (V0 - M) * (V0 - M);
    double Sd = std::sqrt(V / R.size()) + 1e-6;
    for (auto &V0 : R)
      V0 = static_cast<float>((V0 - M) / Sd);
  }
}

// --- helpers ----------------------------------------------------------------

static std::vector<float> OneHot(int Y, size_t NClasses) {
  std::vector<float> T(NClasses, 0.0f);
  T[Y] = 1.0f;
  return T;
}

static double EvalAcc(Net &N, const Dataset &D) {
  size_t Correct = 0;
  for (size_t I = 0; I < D.X.size(); ++I) {
    N.DoForwardPass(D.X[I]);
    auto Out = N.GetOutput();
    size_t Arg = 0;
    for (size_t J = 1; J < Out.size(); ++J)
      if (Out[J] > Out[Arg])
        Arg = J;
    if (static_cast<int>(Arg) == D.Y[I])
      ++Correct;
  }
  return D.X.empty() ? 0.0 : static_cast<double>(Correct) / D.X.size();
}

// Compute the threshold at the requested global sparsity *increment*: kth-
// smallest |w| among currently-alive edges, where k = prune_frac * alive.
static float ComputeThreshold(Net &N, float PruneFrac) {
  auto &CA = N.GetConnAlloc();
  std::vector<float> AliveAbs;
  AliveAbs.reserve(CA.Size());
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (plastix::GetField<plastix::DeadTag>(CA, C))
      continue;
    AliveAbs.push_back(std::abs(plastix::GetWeight(CA, C)));
  }
  if (AliveAbs.empty())
    return 0.0f;
  size_t K = static_cast<size_t>(PruneFrac * AliveAbs.size());
  if (K == 0)
    K = 1;
  if (K > AliveAbs.size())
    K = AliveAbs.size();
  std::nth_element(AliveAbs.begin(), AliveAbs.begin() + K - 1, AliveAbs.end());
  return AliveAbs[K - 1];
}

// `Depth` follows PyTorch's convention: total number of Linear layers, of
// which `Depth - 1` are ReLU hidden and the last is the linear softmax
// readout. Depth=4 = 3 hidden + 1 out (the PyTorch default).
static std::unique_ptr<Net> Build(const HP &H, uint64_t SeedBase, float Limit) {
  if (H.Depth == 1) {
    return std::unique_ptr<Net>(new Net(
        H.InLen,
        FCOut{H.NumClasses, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));
  } else if (H.Depth == 2) {
    return std::unique_ptr<Net>(new Net(
        H.InLen, FCHidden{H.Hidden, UniformInit{SeedBase + 1, Limit}},
        FCOut{H.NumClasses, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));
  } else if (H.Depth == 3) {
    return std::unique_ptr<Net>(new Net(
        H.InLen, FCHidden{H.Hidden, UniformInit{SeedBase + 1, Limit}},
        FCHidden{H.Hidden, UniformInit{SeedBase + 2, Limit}},
        FCOut{H.NumClasses, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));
  }
  // depth 4 — the PyTorch default.
  return std::unique_ptr<Net>(new Net(
      H.InLen, FCHidden{H.Hidden, UniformInit{SeedBase + 1, Limit}},
      FCHidden{H.Hidden, UniformInit{SeedBase + 2, Limit}},
      FCHidden{H.Hidden, UniformInit{SeedBase + 4, Limit}},
      FCOut{H.NumClasses, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));
}

static void TrainEpochs(Net &N, const Dataset &D, size_t Epochs,
                        std::mt19937 &Rng, bench::PhaseTimer &Timer) {
  size_t NClasses =
      static_cast<size_t>(*std::max_element(D.Y.begin(), D.Y.end())) + 1;
  std::vector<size_t> Perm(D.X.size());
  for (size_t I = 0; I < Perm.size(); ++I)
    Perm[I] = I;
  for (size_t E = 0; E < Epochs; ++E) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    for (size_t I : Perm) {
      auto Tgt = OneHot(D.Y[I], NClasses);
      Timer.Tick();
      N.DoForwardPass(D.X[I]);
      Timer.MarkForward();
      N.DoCalculateLoss(Tgt);
      Timer.MarkLoss();
      N.DoBackwardPass();
      Timer.MarkBackward();
      N.DoUpdateUnitState();
      N.DoUpdateConnectionState();
      Timer.MarkUpdate();
      Timer.StepDone();
    }
  }
}

} // namespace

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.InLen = static_cast<size_t>(Args.GetInt("length", H.InLen));
  H.NumClasses = static_cast<size_t>(Args.GetInt("n-classes", H.NumClasses));
  H.Hidden = static_cast<size_t>(Args.GetInt("hidden", H.Hidden));
  H.Depth = static_cast<size_t>(Args.GetInt("depth", H.Depth));
  H.NPerClass =
      static_cast<size_t>(Args.GetInt("n-per-class", H.NPerClass));
  H.InitEpochs =
      static_cast<size_t>(Args.GetInt("init-epochs", H.InitEpochs));
  H.FinetuneEpochs =
      static_cast<size_t>(Args.GetInt("finetune-epochs", H.FinetuneEpochs));
  H.MaxRounds =
      static_cast<size_t>(Args.GetInt("max-rounds", H.MaxRounds));
  H.PruneFrac = Args.GetFloat("prune-frac", H.PruneFrac);
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.FinetuneLrScale = Args.GetFloat("finetune-lr-scale", H.FinetuneLrScale);
  if (Args.Quick) {
    // PyTorch reference: init_epochs //= 4, max_rounds //= 3 under --quick.
    H.InitEpochs = std::max<size_t>(1, H.InitEpochs / 4);
    H.FinetuneEpochs = std::max<size_t>(1, H.FinetuneEpochs / 2);
    H.MaxRounds = std::max<size_t>(2, H.MaxRounds / 3);
  }
  UpdateConn::Lr = H.Lr;

  auto Train = SynthUcr(H.NPerClass, H.NumClasses, H.InLen, H.Snr,
                        static_cast<uint32_t>(Args.Seed));
  auto Test =
      SynthUcr(std::max<size_t>(16, H.NPerClass / 4), H.NumClasses, H.InLen,
               H.Snr, static_cast<uint32_t>(Args.Seed + 1000));
  NormaliseInstance(Train.X);
  NormaliseInstance(Test.X);
  std::cout << "[info] InLen=" << H.InLen << " Classes=" << H.NumClasses
            << " Hidden=" << H.Hidden << " Depth=" << H.Depth
            << " Train=" << Train.X.size() << " Test=" << Test.X.size()
            << "\n";

  float Limit = std::sqrt(6.0f / static_cast<float>(H.InLen + H.Hidden));
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 11ull;
  auto N = Build(H, SeedBase, Limit);

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "idempotent_imp");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_acc");

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  bench::PhaseTimer Timer;

  // Round 0 — dense training.
  std::cout << "[imp] initial dense training, " << H.InitEpochs
            << " epochs\n";
  TrainEpochs(*N, Train, H.InitEpochs, Rng, Timer);
  double Acc0 = EvalAcc(*N, Test);
  size_t Edges0 = bench::LiveEdgeCount(N->GetConnAlloc());
  auto Edges = bench::LiveEdgeSet(N->GetConnAlloc());
  double Nll0 = 1.0 - Acc0;
  Log.Log(0, N->GetUnitAlloc().Size(), Edges0, &Edges, &Nll0,
          {{"val_acc", Acc0},
           {"round", 0.0},
           {"sparsity", 0.0},
           {"killed_this_round", 0.0},
           {"test_acc", Acc0}});
  TestCsv.Add(0, Acc0, N->GetUnitAlloc().Size(), Edges0);
  std::cout << "[imp] round 0  edges=" << Edges0 << "  acc=" << Acc0 << "\n";

  auto T0 = std::chrono::steady_clock::now();
  size_t FixedPoint = 0;
  size_t FinalRound = 0;
  for (size_t R = 1; R <= H.MaxRounds; ++R) {
    size_t AliveBefore = bench::LiveEdgeCount(N->GetConnAlloc());

    // Step 1 — compute global threshold.
    float Thresh = ComputeThreshold(*N, H.PruneFrac);

    // Step 2 — arm PruneConn and call its dispatch phase directly. This is
    // the "step-by-step declarative API" in action: the framework exposes
    // each Do<Phase> as a public method on Network so the host can stage
    // them outside the canonical DoStep ordering.
    PruneConn::Threshold = Thresh;
    PruneConn::Armed = true;
    N->DoPruneConnections();
    PruneConn::Armed = false;

    size_t AliveAfter = bench::LiveEdgeCount(N->GetConnAlloc());
    size_t Killed = AliveBefore - AliveAfter;

    // Step 3 — finetune with a smaller LR so the survivors don't drift far
    // from the lottery-ticket basin.
    UpdateConn::Lr = H.Lr * H.FinetuneLrScale;
    TrainEpochs(*N, Train, H.FinetuneEpochs, Rng, Timer);

    double Acc = EvalAcc(*N, Test);
    double Sparsity = 1.0 - static_cast<double>(AliveAfter) /
                                static_cast<double>(Edges0);
    auto Ed = bench::LiveEdgeSet(N->GetConnAlloc());
    double Nll = 1.0 - Acc;
    Log.Log(R, N->GetUnitAlloc().Size(), AliveAfter, &Ed, &Nll,
            {{"val_acc", Acc},
             {"round", static_cast<double>(R)},
             {"sparsity", Sparsity},
             {"killed_this_round", static_cast<double>(Killed)},
             {"test_acc", Acc}});
    TestCsv.Add(R, Acc, N->GetUnitAlloc().Size(), AliveAfter);
    std::cout << "[imp] round " << R << "  killed=" << Killed
              << "  alive=" << AliveAfter << "  sparsity=" << Sparsity
              << "  acc=" << Acc << "\n";
    FinalRound = R;
    if (Killed == 0) {
      FixedPoint = 1;
      std::cout << "[imp] reached fixed point at round " << R << "\n";
      break;
    }
  }
  double Wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
          .count();

  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "idempotent_imp"));
  bench::SummaryWriter S;
  S.Set("workload", std::string{"02_idempotent_imp"});
  S.Set("dataset", std::string{"synthetic-ucr"});
  S.Set("n_classes", static_cast<int>(H.NumClasses));
  S.Set("length", static_cast<int>(H.InLen));
  S.Set("hidden", static_cast<int>(H.Hidden));
  S.Set("depth", static_cast<int>(H.Depth));
  S.Set("init_epochs", static_cast<int>(H.InitEpochs));
  S.Set("finetune_epochs", static_cast<int>(H.FinetuneEpochs));
  S.Set("prune_frac", static_cast<double>(H.PruneFrac));
  S.Set("max_rounds", static_cast<int>(H.MaxRounds));
  S.Set("wall_seconds", Wall);
  S.Set("rounds_run", static_cast<int>(FinalRound));
  S.Set("edges_initial", static_cast<int>(Edges0));
  S.Set("edges_final",
        static_cast<int>(bench::LiveEdgeCount(N->GetConnAlloc())));
  double FinalSpars = 1.0 - static_cast<double>(
                                bench::LiveEdgeCount(N->GetConnAlloc())) /
                                static_cast<double>(Edges0);
  S.Set("sparsity_final", FinalSpars);
  S.Set("val_acc_initial", Acc0);
  S.Set("val_acc_final", EvalAcc(*N, Test));
  S.Set("fixed_point_reached", static_cast<int>(FixedPoint));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  rounds=" << FinalRound
            << "  sparsity=" << FinalSpars << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}
