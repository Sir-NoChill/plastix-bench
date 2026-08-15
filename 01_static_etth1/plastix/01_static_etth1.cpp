// Workload 1 / 5 — STATIC regime, Plastix translation.
//
// Mirrors 01_static_etth1.py: a fixed feedforward MLP trained
// against the ETTh1 long-horizon forecasting target with mean-squared error.
// Topology never changes; the topology hash and edge Jaccard from
// `bench::StructuralLog` sit at constants for the entire run.
//
// Plastix policy mapping (matches the comment block at the bottom of the
// Python version):
//
//   ForwardPass        custom    GeLU-hidden / linear-output
//   BackwardPass       custom    backprop through GeLU using stored pre-act
//   Loss               MSELoss   (built-in)
//   UpdateConn         custom    plain SGD on WeightTag
//   PruneUnit / Conn   NoX       static topology
//   AddUnit  / Conn    NoX       static topology
//   ResetGlobal        NoX
//
// Sizing matches the PyTorch reference: in_len=96, out_len=24, hidden=256,
// depth=3, multivariate over all 7 ETTh1 channels (in_dim=672, out_dim=168,
// ~280k connections). The allocator capacity is bumped via the per-traits
// UnitCapacity / ConnCapacity overrides.

#include "plastix/common.hpp"

#include <plastix/math.hpp>
#include <plastix/plastix.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace {

struct HP {
  // PyTorch defaults from 01_static_etth1.py.
  size_t InLen = 96;
  size_t OutLen = 24;
  size_t Hidden = 256;
  // PyTorch convention: depth = total number of Linear layers.
  // depth=3 means in->H->H->out (2 GeLU hidden + 1 linear output).
  size_t Depth = 3;
  size_t Epochs = 20;
  float Lr = 1e-3f;
  // Cap on training rows used per epoch (Plastix is per-example SGD, so
  // running the full ~12k-window train split each epoch is plenty slow on
  // a large MLP). 0 means "no cap".
  size_t MaxTrainRows = 0;
};

// Pre-activation z, captured in ForwardPass::Apply so the backward pass can
// recover phi'(z) without re-running the forward sum.
struct PreActTag {};

// Per-unit pre-activation gradient dL/dz, persisted across backward levels.
struct GradPreActTag {};

// `true` for units in the output layer (linear activation, no GeLU).
struct IsOutputTag {};

struct StaticForward {
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
                                        : plastix::math::GeLU(Z);
  }
};

// Backward pass: BackwardAcc holds dL/da for the *destination* unit (loaded
// from MSELoss on output units, accumulated otherwise). Apply converts dL/da
// to dL/dz using phi'(z) — linear for outputs, GeLU' elsewhere — and stores
// the result in GradPreActTag where the next level down reads it. Map for
// layer L reads upstream dL/dz off the destination unit, so the inter-level
// handoff goes via GradPreActTag, not BackwardAcc (which the framework
// clears after each level's Apply).
struct StaticBackward {
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
                         : plastix::math::GeLUGradFromPreact(Z);
    plastix::GetField<GradPreActTag>(U, Id) = DLDA * DPhiDz;
  }
};

// Runtime-tunable hyperparameters read by the (possibly device-side) policies.
// These live in the network's GlobalState — staged from the host through
// Net::Global() — rather than in host-side static members, which device code
// cannot read under CUDA.
struct StaticGlobals {
  float Lr = 1e-3f;
};

// w_ij -= lr * (dL/dz_dst) * a_src. Standard SGD on float weights.
struct StaticUpdateConn {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    float Grad = plastix::GetField<GradPreActTag>(U, DstId);
    float A = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= G.Lr * Grad * A;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

struct StaticTraits : plastix::DefaultNetworkTraits<> {
  using GlobalState = StaticGlobals;
  using ForwardPass = StaticForward;
  using BackwardPass = StaticBackward;
  using Loss = plastix::MSELoss;
  using UpdateConn = StaticUpdateConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<PreActTag, float>,
      plastix::alloc::SOAField<GradPreActTag, float>,
      plastix::alloc::SOAField<IsOutputTag, bool>>;
  // 7-channel ETTh1 with in_len=96 / out_len=24 / hidden=256 / depth=3
  // peaks at 1352 units and ~280k connections. Headroom on top of that.
  static constexpr size_t UnitCapacity = 4096;
  static constexpr size_t ConnCapacity = 524288;
};
static_assert(plastix::NetworkTraits<StaticTraits>);

using Net = plastix::Network<StaticTraits>;

// Two-shard policy for the multi-GPU variant: level 0 (input units) on
// shard 0, every deeper level on shard 1. Same shape as
// tests/test_nccl_executor.cpp -- IsContiguousByLevel keeps the mapping
// dispatch-friendly, since Assign() is a pure level->shard function so
// each shard owns a contiguous level range.
struct TwoShardOnLevel1 {
  static constexpr uint16_t NumShards = 2;
  static constexpr bool IsContiguousByLevel = true;
  static constexpr plastix::ShardId Assign(uint32_t, uint16_t Level) {
    return plastix::ShardId{Level == 0 ? uint16_t{0} : uint16_t{1}};
  }
};

// Sharded traits: identical to StaticTraits except for the sharding
// policy. Constructed as a separate type so the same source can build
// both the single-device Net and a MultiDeviceExecutor-backed NetSharded
// and main() dispatches at runtime on --multi-device.
struct StaticTraitsSharded : StaticTraits {
  using Sharding = TwoShardOnLevel1;
};
static_assert(plastix::NetworkTraits<StaticTraitsSharded>);

using NetSharded = plastix::Network<StaticTraitsSharded>;

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

// --- data ------------------------------------------------------------------

// Series shape: (T, C). Stored as flat row-major: row I of length C starts
// at index I*C. Mirrors the Python implementation's `data[idx]` array slice.
struct Series {
  std::vector<float> Flat; // length T * C
  size_t T = 0;
  size_t C = 0;
  const float *Row(size_t I) const { return Flat.data() + I * C; }
};

static Series SynthesiseEtth1(size_t T, size_t C, uint32_t Seed) {
  constexpr float Pi = 3.14159265358979323846f;
  std::mt19937 Rng(Seed);
  std::normal_distribution<float> N(0.0f, 0.1f);
  Series S;
  S.T = T;
  S.C = C;
  S.Flat.assign(T * C, 0.0f);
  for (size_t I = 0; I < T; ++I) {
    float Tt = static_cast<float>(I);
    float Base = std::sin(2.0f * Pi * Tt / 24.0f) +
                 0.4f * std::sin(2.0f * Pi * Tt / (24.0f * 7.0f));
    for (size_t Ch = 0; Ch < C; ++Ch)
      S.Flat[I * C + Ch] = Base + N(Rng);
  }
  return S;
}

static Series LoadEtth1(const std::filesystem::path &Dir, bool ForceSynth,
                        uint32_t Seed) {
  if (!ForceSynth) {
    auto Path = Dir / "ETTh1.csv";
    if (std::filesystem::exists(Path)) {
      auto Csv = bench::ReadCsv(Path);
      if (!Csv.Header.empty()) {
        // Mirror 01_static_etth1.py: every column except `date`.
        std::vector<size_t> ColIdx;
        for (size_t I = 0; I < Csv.Header.size(); ++I) {
          std::string H = Csv.Header[I];
          for (auto &C : H)
            C = static_cast<char>(std::tolower(C));
          if (H != "date")
            ColIdx.push_back(I);
        }
        Series S;
        S.T = Csv.Rows.size();
        S.C = ColIdx.size();
        S.Flat.reserve(S.T * S.C);
        for (size_t I = 0; I < S.T; ++I) {
          const auto &R = Csv.Rows[I];
          for (size_t J : ColIdx) {
            float V = 0.0f;
            if (J < R.size()) {
              try {
                V = std::stof(R[J]);
              } catch (...) {
                V = 0.0f;
              }
            }
            S.Flat.push_back(V);
          }
        }
        std::cerr << "[data] loaded ETTh1 (" << S.T << " rows x " << S.C
                  << " channels)\n";
        return S;
      }
    }
  }
  std::cerr << "[data] using synthetic ETTh1 stand-in\n";
  return SynthesiseEtth1(17'420, 7, Seed);
}

// Per-channel z-score over the first `Cut` rows. Mirrors the PyTorch
// reference's data[:n_train].mean(0) / std(0).
static void StandardisePerChannel(Series &S, size_t Cut) {
  if (S.T == 0 || S.C == 0)
    return;
  size_t N = std::min(Cut, S.T);
  std::vector<double> Mu(S.C, 0.0), Sd(S.C, 0.0);
  for (size_t I = 0; I < N; ++I)
    for (size_t Ch = 0; Ch < S.C; ++Ch)
      Mu[Ch] += S.Flat[I * S.C + Ch];
  for (auto &M : Mu)
    M /= static_cast<double>(N);
  for (size_t I = 0; I < N; ++I)
    for (size_t Ch = 0; Ch < S.C; ++Ch) {
      double D = S.Flat[I * S.C + Ch] - Mu[Ch];
      Sd[Ch] += D * D;
    }
  for (auto &V : Sd)
    V = std::sqrt(V / static_cast<double>(N)) + 1e-6;
  for (size_t I = 0; I < S.T; ++I)
    for (size_t Ch = 0; Ch < S.C; ++Ch) {
      double X = S.Flat[I * S.C + Ch];
      S.Flat[I * S.C + Ch] =
          static_cast<float>((X - Mu[Ch]) / Sd[Ch]);
    }
}

struct Dataset {
  std::vector<std::vector<float>> X;
  std::vector<std::vector<float>> Y;
};

// Windowed view: X[I] = rows [I, I+InLen) flattened (length InLen*C),
// Y[I] = rows [I+InLen, I+InLen+OutLen) flattened (length OutLen*C).
static Dataset Window(const Series &S, size_t InLen, size_t OutLen) {
  Dataset D;
  if (S.T < InLen + OutLen)
    return D;
  size_t N = S.T - InLen - OutLen + 1;
  D.X.reserve(N);
  D.Y.reserve(N);
  for (size_t I = 0; I < N; ++I) {
    std::vector<float> Xw(InLen * S.C), Yw(OutLen * S.C);
    for (size_t T = 0; T < InLen; ++T)
      for (size_t Ch = 0; Ch < S.C; ++Ch)
        Xw[T * S.C + Ch] = S.Flat[(I + T) * S.C + Ch];
    for (size_t T = 0; T < OutLen; ++T)
      for (size_t Ch = 0; Ch < S.C; ++Ch)
        Yw[T * S.C + Ch] = S.Flat[(I + InLen + T) * S.C + Ch];
    D.X.push_back(std::move(Xw));
    D.Y.push_back(std::move(Yw));
  }
  return D;
}

template <typename NetT>
static double EvalMse(NetT &N, const std::vector<std::vector<float>> &X,
                      const std::vector<std::vector<float>> &Y) {
  double Sum = 0.0;
  size_t Cnt = 0;
  for (size_t I = 0; I < X.size(); ++I) {
    N.DoForwardPass(X[I]);
    auto Out = N.GetOutput();
    for (size_t J = 0; J < Out.size(); ++J) {
      double D = static_cast<double>(Out[J]) - static_cast<double>(Y[I][J]);
      Sum += D * D;
      ++Cnt;
    }
  }
  return Cnt ? Sum / Cnt : 0.0;
}

using FCHidden = plastix::FullyConnected<UniformInit>;
using FCOut = plastix::FullyConnected<UniformInit, MarkOutput>;

// `Depth` follows PyTorch's convention: total number of Linear layers, of
// which `Depth - 1` are GeLU hidden and the last is the linear readout.
// So Depth=1 = single readout, Depth=2 = 1 hidden + 1 out, Depth=3 = 2
// hidden + 1 out, etc. We pass InDim and OutDim explicitly (the network's
// input dimension is multivariate in_len * channels).
//
// Templated on the network type so the same builder works for both the
// single-device Net and the two-shard NetSharded -- the topology is the
// same either way; only the executor differs (attached after construction
// by the caller when --multi-device is on).
template <typename NetT>
static std::unique_ptr<NetT> BuildNetworkT(size_t InDim, size_t OutDim,
                                            const HP &H, uint64_t SeedBase,
                                            float Limit) {
  if (H.Depth == 1) {
    return std::unique_ptr<NetT>(new NetT(
        InDim, FCOut{OutDim, UniformInit{SeedBase + 1, Limit}, MarkOutput{}}));
  } else if (H.Depth == 2) {
    return std::unique_ptr<NetT>(new NetT(
        InDim, FCHidden{H.Hidden, UniformInit{SeedBase + 1, Limit}},
        FCOut{OutDim, UniformInit{SeedBase + 2, Limit}, MarkOutput{}}));
  } else if (H.Depth == 4) {
    return std::unique_ptr<NetT>(new NetT(
        InDim, FCHidden{H.Hidden, UniformInit{SeedBase + 1, Limit}},
        FCHidden{H.Hidden, UniformInit{SeedBase + 2, Limit}},
        FCHidden{H.Hidden, UniformInit{SeedBase + 4, Limit}},
        FCOut{OutDim, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));
  }
  // depth 3 -- the PyTorch default.
  return std::unique_ptr<NetT>(new NetT(
      InDim, FCHidden{H.Hidden, UniformInit{SeedBase + 1, Limit}},
      FCHidden{H.Hidden, UniformInit{SeedBase + 2, Limit}},
      FCOut{OutDim, UniformInit{SeedBase + 3, Limit}, MarkOutput{}}));
}

} // namespace

namespace {

// Templated body of the benchmark. Instantiated with either Net (single
// device, historical path) or NetSharded (2 shards, MultiDeviceExecutor).
// MultiDevice=true triggers the SetExecutor call after network construction;
// everything else is identical -- the whole point of the sharded API is that
// the training-loop dispatch code is the same regardless of executor.
template <typename NetT>
static int RunBench(bench::CliArgs &Args, HP &H, bool MultiDevice) {
  bench::MemoryProbe MP;
  MP.Start();

  auto Raw = LoadEtth1(Args.DataDir, Args.Synthetic,
                       static_cast<uint32_t>(Args.Seed));
  // The PyTorch reference standardises against the *first 70%* of the
  // series (the training portion). Match that.
  StandardisePerChannel(Raw, std::max<size_t>(64, (Raw.T * 7) / 10));
  auto D = Window(Raw, H.InLen, H.OutLen);
  size_t InDim = H.InLen * Raw.C;
  size_t OutDim = H.OutLen * Raw.C;
  size_t NTr = static_cast<size_t>(0.7f * D.X.size());
  size_t NVa = static_cast<size_t>(0.15f * D.X.size());
  if (H.MaxTrainRows > 0)
    NTr = std::min(NTr, H.MaxTrainRows);

  std::cout << "[info] InLen=" << H.InLen << " OutLen=" << H.OutLen
            << " Channels=" << Raw.C << " InDim=" << InDim << " OutDim="
            << OutDim << " Hidden=" << H.Hidden << " Depth=" << H.Depth
            << " Epochs=" << H.Epochs << " Train=" << NTr << " Val=" << NVa
            << " MultiDevice=" << (MultiDevice ? "yes" : "no") << "\n";

  float Limit = std::sqrt(6.0f / static_cast<float>(InDim + H.Hidden));
  uint64_t SeedBase = static_cast<uint64_t>(Args.Seed) * 1000ull + 7ull;
  auto N = BuildNetworkT<NetT>(InDim, OutDim, H, SeedBase, Limit);
  MP.EndWeights();

#ifdef PLASTIX_HAS_CUDA
  if (MultiDevice) {
    // Two-shard MultiDeviceExecutor. Requires two visible CUDA devices;
    // the sbatch that runs this must request --gpus-per-node=a100:2.
    N->SetExecutor(std::make_unique<plastix::MultiDeviceExecutor>(2));
    std::cout << "[info] executor: MultiDeviceExecutor(2)\n";
  }
#else
  if (MultiDevice) {
    std::cerr << "[fatal] --multi-device requires a CUDA-enabled build "
                 "(PLASTIX_BENCH_ENABLE_CUDA=ON)\n";
    return 2;
  }
#endif
  // Stage the learning rate into the managed GlobalState; the UpdateConn
  // policy reads it on host or device through its Globals handle.
  N->Global().Lr = H.Lr;

  std::vector<std::vector<float>> Xva(D.X.begin() + NTr,
                                      D.X.begin() + NTr + NVa);
  std::vector<std::vector<float>> Yva(D.Y.begin() + NTr,
                                      D.Y.begin() + NTr + NVa);
  std::vector<std::vector<float>> Xte(D.X.begin() + NTr + NVa, D.X.end());
  std::vector<std::vector<float>> Yte(D.Y.begin() + NTr + NVa, D.Y.end());
  MP.EndDataset();

  auto [HistPath, SummaryPath, LogPath] =
      bench::OutputPaths(Args, "static_etth1");
  bench::StructuralLog Log(HistPath);
  bench::TestCsvWriter TestCsv("test_mse");

  auto Edges = bench::LiveEdgeSet(N->GetConnAlloc());
  double InitVal = EvalMse(*N, Xva, Yva);
  double InitTest = EvalMse(*N, Xte, Yte);
  Log.Log(0, N->GetUnitAlloc().Size(),
          bench::LiveEdgeCount(N->GetConnAlloc()), &Edges, &InitVal,
          {{"train_loss", 0.0}, {"epoch", 0.0}, {"test_mse", InitTest}});
  TestCsv.Add(0, InitTest, N->GetUnitAlloc().Size(),
              bench::LiveEdgeCount(N->GetConnAlloc()));

  std::mt19937 Rng(static_cast<uint32_t>(Args.Seed));
  std::vector<size_t> Perm(NTr);
  for (size_t I = 0; I < NTr; ++I)
    Perm[I] = I;

  bench::PhaseTimer Timer;
  auto T0 = std::chrono::steady_clock::now();
  for (size_t Ep = 1; Ep <= H.Epochs; ++Ep) {
    std::shuffle(Perm.begin(), Perm.end(), Rng);
    double TrainLoss = 0.0;
    size_t TrainCnt = 0;
    for (size_t I : Perm) {
      Timer.Tick();
      N->DoForwardPass(D.X[I]);
      Timer.MarkForward();
      N->DoCalculateLoss(D.Y[I]);
      Timer.MarkLoss();
      N->DoBackwardPass();
      Timer.MarkBackward();
      N->DoUpdateUnitState();
      N->DoUpdateConnectionState();
      Timer.MarkUpdate();
      Timer.StepDone();
      auto Out = N->GetOutput();
      for (size_t J = 0; J < Out.size(); ++J) {
        double E = static_cast<double>(Out[J]) - static_cast<double>(D.Y[I][J]);
        TrainLoss += E * E;
        ++TrainCnt;
      }
    }
    TrainLoss /= std::max<size_t>(TrainCnt, 1);
    double VaMse = EvalMse(*N, Xva, Yva);
    double EpTest = EvalMse(*N, Xte, Yte);
    auto Ed = bench::LiveEdgeSet(N->GetConnAlloc());
    Log.Log(Ep, N->GetUnitAlloc().Size(),
            bench::LiveEdgeCount(N->GetConnAlloc()), &Ed, &VaMse,
            {{"train_loss", TrainLoss},
             {"epoch", static_cast<double>(Ep)},
             {"test_mse", EpTest}});
    TestCsv.Add(Ep, EpTest, N->GetUnitAlloc().Size(),
                bench::LiveEdgeCount(N->GetConnAlloc()));
    std::cout << "[ep " << Ep << "] train_mse=" << TrainLoss
              << "  val_mse=" << VaMse << "  test_mse=" << EpTest << "\n";
  }
  double Wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - T0)
                    .count();

  double TeMse = EvalMse(*N, Xte, Yte);
  Log.Flush();
  TestCsv.Write(bench::TestCsvPath(Args, "static_etth1"));

  bench::SummaryWriter S;
  S.Set("workload", std::string{"01_static_etth1"});
  S.Set("dataset", Args.Synthetic ? std::string{"synthetic-etth1"}
                                  : std::string{"ETTh1"});
  S.Set("in_len", static_cast<int>(H.InLen));
  S.Set("out_len", static_cast<int>(H.OutLen));
  S.Set("hidden", static_cast<int>(H.Hidden));
  S.Set("depth", static_cast<int>(H.Depth));
  S.Set("epochs", static_cast<int>(H.Epochs));
  S.Set("lr", static_cast<double>(H.Lr));
  S.Set("wall_seconds", Wall);
  S.Set("val_mse_final", Log.Records().back().ValLoss);
  S.Set("test_mse", TeMse);
  S.Set("n_units", static_cast<int>(N->GetUnitAlloc().Size()));
  S.Set("n_edges",
        static_cast<int>(bench::LiveEdgeCount(N->GetConnAlloc())));
  float JMin = 1.0f, JMax = 1.0f;
  for (auto &R : Log.Records()) {
    JMin = std::min(JMin, R.Jaccard);
    JMax = std::max(JMax, R.Jaccard);
  }
  S.Set("jaccard_min", static_cast<double>(JMin));
  S.Set("jaccard_max", static_cast<double>(JMax));
  S.Set("seed", Args.Seed);
  Timer.WriteSummary(S, Wall);
  MP.WriteSummary(S);
  S.Write(SummaryPath);

  std::cout << "[done] wall=" << Wall << "s  test_mse=" << TeMse << "\n";
  std::cout << "[done] wrote " << HistPath << ", " << SummaryPath << "\n";

  (void)LogPath;
  return 0;
}

} // namespace (RunBench template)

int main(int Argc, char **Argv) {
  auto Args = bench::CliArgs::Parse(Argc, Argv);

  HP H;
  H.InLen = static_cast<size_t>(Args.GetInt("in-len", H.InLen));
  H.OutLen = static_cast<size_t>(Args.GetInt("out-len", H.OutLen));
  H.Hidden = static_cast<size_t>(Args.GetInt("hidden", H.Hidden));
  H.Depth = static_cast<size_t>(Args.GetInt("depth", H.Depth));
  H.Epochs = static_cast<size_t>(Args.GetInt("epochs", H.Epochs));
  H.Lr = Args.GetFloat("lr", H.Lr);
  H.MaxTrainRows = static_cast<size_t>(
      Args.GetInt("max-train-rows", static_cast<int>(H.MaxTrainRows)));
  if (Args.Quick) {
    // PyTorch reference: epochs //= 4 in quick mode. Match that here.
    H.Epochs = std::max<size_t>(1, H.Epochs / 4);
  }

  const bool MultiDevice = Args.GetBool("multi-device", false);
  if (MultiDevice) {
    return RunBench<NetSharded>(Args, H, true);
  }
  return RunBench<Net>(Args, H, false);
}
