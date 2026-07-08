#ifndef TRADITIONAL_RAW_COMMON_HPP
#define TRADITIONAL_RAW_COMMON_HPP

// Shared infrastructure for the traditional-raw benchmark suite.
//
// Mirrors traditional-plastix/common.hpp but with no Plastix dependency. The
// .history.jsonl / .summary.csv shapes match both the PyTorch reference under
// ../ and the Plastix translation under ../traditional-plastix/.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bench {

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

struct CliArgs {
  int Seed = 0;
  bool Quick = false;
  std::string Tag;
  std::filesystem::path DataDir = "data";
  std::filesystem::path OutDir = "_results";
  bool Synthetic = false;
  std::unordered_map<std::string, std::string> Extras;

  static CliArgs Parse(int Argc, char **Argv) {
    CliArgs A;
    for (int I = 1; I < Argc; ++I) {
      std::string_view K = Argv[I];
      auto Next = [&]() -> std::string {
        if (I + 1 >= Argc) {
          std::cerr << "missing value for " << K << "\n";
          std::exit(2);
        }
        return Argv[++I];
      };
      if (K == "--seed")
        A.Seed = std::atoi(Next().c_str());
      else if (K == "--quick")
        A.Quick = true;
      else if (K == "--tag")
        A.Tag = Next();
      else if (K == "--data-dir")
        A.DataDir = Next();
      else if (K == "--out-dir")
        A.OutDir = Next();
      else if (K == "--synthetic")
        A.Synthetic = true;
      else if (K.starts_with("--")) {
        std::string Key{K.substr(2)};
        A.Extras[Key] = Next();
      } else {
        std::cerr << "unrecognised arg: " << K << "\n";
        std::exit(2);
      }
    }
    return A;
  }

  int GetInt(const std::string &K, int Def) const {
    auto It = Extras.find(K);
    return It == Extras.end() ? Def : std::atoi(It->second.c_str());
  }
  float GetFloat(const std::string &K, float Def) const {
    auto It = Extras.find(K);
    return It == Extras.end()
               ? Def
               : static_cast<float>(std::atof(It->second.c_str()));
  }
  bool GetBool(const std::string &K, bool Def) const {
    auto It = Extras.find(K);
    if (It == Extras.end())
      return Def;
    return It->second == "1" || It->second == "true" || It->second == "yes";
  }
};

inline std::tuple<std::filesystem::path, std::filesystem::path,
                  std::filesystem::path>
OutputPaths(const CliArgs &Args, const std::string &Name) {
  std::filesystem::create_directories(Args.OutDir);
  std::string Suffix = Args.Tag.empty() ? "" : "_" + Args.Tag;
  auto Base = Args.OutDir / (Name + Suffix);
  return {std::filesystem::path(Base.string() + ".history.jsonl"),
          std::filesystem::path(Base.string() + ".summary.csv"),
          std::filesystem::path(Base.string() + ".log.txt")};
}

// ---------------------------------------------------------------------------
// CSV reader (handles a single layer of quoted cells; sufficient for the
// dataset CSVs cached under ../data/).
// ---------------------------------------------------------------------------

struct CsvFile {
  std::vector<std::string> Header;
  std::vector<std::vector<std::string>> Rows;

  std::vector<float> Column(const std::string &Name) const {
    int Idx = -1;
    for (size_t I = 0; I < Header.size(); ++I)
      if (Header[I] == Name) {
        Idx = static_cast<int>(I);
        break;
      }
    if (Idx < 0)
      return {};
    std::vector<float> Out;
    Out.reserve(Rows.size());
    for (const auto &R : Rows) {
      if (static_cast<size_t>(Idx) >= R.size())
        continue;
      try {
        Out.push_back(std::stof(R[Idx]));
      } catch (...) {
      }
    }
    return Out;
  }
};

inline CsvFile ReadCsv(const std::filesystem::path &Path) {
  CsvFile F;
  std::ifstream In(Path);
  if (!In)
    return F;
  std::string Line;
  bool First = true;
  while (std::getline(In, Line)) {
    if (!Line.empty() && Line.back() == '\r')
      Line.pop_back();
    std::vector<std::string> Cells;
    Cells.reserve(16);
    std::string Cur;
    bool InQ = false;
    for (char C : Line) {
      if (C == '"') {
        InQ = !InQ;
      } else if (C == ',' && !InQ) {
        Cells.push_back(std::move(Cur));
        Cur.clear();
      } else {
        Cur.push_back(C);
      }
    }
    Cells.push_back(std::move(Cur));
    if (First) {
      F.Header = std::move(Cells);
      First = false;
    } else {
      F.Rows.push_back(std::move(Cells));
    }
  }
  return F;
}

// ---------------------------------------------------------------------------
// Topology utilities — pack (layer, row, col) into a 64-bit edge id, hash and
// Jaccard over a set of those ids.
// ---------------------------------------------------------------------------

using EdgeSet = std::set<uint64_t>;

inline uint64_t PackEdge(uint32_t Layer, uint32_t Row, uint32_t Col) {
  return (static_cast<uint64_t>(Layer) << 56) |
         (static_cast<uint64_t>(Row) << 28) | static_cast<uint64_t>(Col);
}

inline std::string TopologyHash(const EdgeSet &S) {
  uint64_t H = 0xcbf29ce484222325ull;
  for (uint64_t E : S) {
    for (int B = 0; B < 8; ++B) {
      uint8_t Byte = (E >> (B * 8)) & 0xFF;
      H ^= Byte;
      H *= 0x100000001b3ull;
    }
  }
  std::ostringstream Os;
  Os << std::hex << std::setw(16) << std::setfill('0') << H;
  return Os.str();
}

inline float Jaccard(const EdgeSet &A, const EdgeSet &B) {
  if (A.empty() && B.empty())
    return 1.0f;
  size_t Inter = 0;
  auto It1 = A.begin(), It2 = B.begin();
  while (It1 != A.end() && It2 != B.end()) {
    if (*It1 < *It2)
      ++It1;
    else if (*It2 < *It1)
      ++It2;
    else {
      ++Inter;
      ++It1;
      ++It2;
    }
  }
  size_t Uni = A.size() + B.size() - Inter;
  return Uni == 0 ? 1.0f : static_cast<float>(Inter) / static_cast<float>(Uni);
}

// ---------------------------------------------------------------------------
// StructuralLog — append-only per-step JSONL log of (step, n_units, n_edges,
// topology_hash, jaccard, val_loss, **extras).
// ---------------------------------------------------------------------------

struct LogRecord {
  size_t Step;
  size_t NumUnits;
  size_t NumEdges;
  std::string TopoHash;
  float Jaccard;
  double ValLoss = 0.0;
  bool HasValLoss = false;
  std::vector<std::pair<std::string, double>> Extras;
};

class StructuralLog {
public:
  explicit StructuralLog(std::filesystem::path Path) : Path_(std::move(Path)) {}

  void Log(size_t Step, size_t NumUnits, size_t NumEdges, const EdgeSet *Edges,
           const double *ValLoss,
           std::initializer_list<std::pair<std::string, double>> Extras = {}) {
    LogRecord R;
    R.Step = Step;
    R.NumUnits = NumUnits;
    R.NumEdges = NumEdges;
    if (Edges) {
      R.TopoHash = TopologyHash(*Edges);
      R.Jaccard = HasPrev_ ? bench::Jaccard(Prev_, *Edges) : 1.0f;
      Prev_ = *Edges;
      HasPrev_ = true;
    } else {
      R.Jaccard = 1.0f;
    }
    if (ValLoss) {
      R.HasValLoss = true;
      R.ValLoss = *ValLoss;
    }
    for (const auto &E : Extras)
      R.Extras.push_back(E);
    Records_.push_back(std::move(R));
  }

  const std::vector<LogRecord> &Records() const { return Records_; }

  void Flush() {
    if (Path_.has_parent_path())
      std::filesystem::create_directories(Path_.parent_path());
    std::ofstream Os(Path_);
    for (const auto &R : Records_) {
      Os << "{\"step\":" << R.Step << ",\"n_units\":" << R.NumUnits
         << ",\"n_edges\":" << R.NumEdges << ",\"topology_hash\":\""
         << R.TopoHash << "\"" << ",\"jaccard\":" << R.Jaccard
         << ",\"val_loss\":";
      if (R.HasValLoss)
        Os << R.ValLoss;
      else
        Os << "null";
      for (const auto &E : R.Extras)
        Os << ",\"" << E.first << "\":" << E.second;
      Os << "}\n";
    }
  }

private:
  std::filesystem::path Path_;
  std::vector<LogRecord> Records_;
  EdgeSet Prev_;
  bool HasPrev_ = false;
};

// ---------------------------------------------------------------------------
// SummaryWriter — single-row CSV keyed by column name
// ---------------------------------------------------------------------------

class SummaryWriter {
public:
  void Set(const std::string &Key, const std::string &Val) {
    if (Index_.count(Key) == 0) {
      Index_[Key] = Cols_.size();
      Cols_.push_back(Key);
      Vals_.push_back(Val);
    } else {
      Vals_[Index_[Key]] = Val;
    }
  }
  void Set(const std::string &Key, double V) {
    std::ostringstream Os;
    Os << V;
    Set(Key, Os.str());
  }
  void Set(const std::string &Key, long long V) { Set(Key, std::to_string(V)); }
  void Set(const std::string &Key, int V) { Set(Key, std::to_string(V)); }
  void Set(const std::string &Key, size_t V) { Set(Key, std::to_string(V)); }

  void Write(const std::filesystem::path &Path) const {
    if (Path.has_parent_path())
      std::filesystem::create_directories(Path.parent_path());
    std::ofstream Os(Path);
    for (size_t I = 0; I < Cols_.size(); ++I) {
      if (I)
        Os << ',';
      Os << Cols_[I];
    }
    Os << '\n';
    for (size_t I = 0; I < Vals_.size(); ++I) {
      if (I)
        Os << ',';
      Os << Vals_[I];
    }
    Os << '\n';
  }

private:
  std::vector<std::string> Cols_;
  std::vector<std::string> Vals_;
  std::unordered_map<std::string, size_t> Index_;
};

// ---------------------------------------------------------------------------
// PhaseTimer — accumulate per-step nanoseconds across a fixed phase set.
//
// Usage:
//   bench::PhaseTimer T;
//   for (...) {
//     T.Tick();
//     Forward(...);  T.MarkForward();
//     Loss(...);     T.MarkLoss();
//     Backward(...); T.MarkBackward();
//     Update(...);   T.MarkUpdate();
//     T.StepDone();
//   }
//   T.WriteSummary(SummaryCsv, WallSeconds);
//
// `WriteSummary` emits canonical column names so runs.csv has one schema:
//   step_count, step_ns_mean,
//   forward_ns_mean, loss_ns_mean, backward_ns_mean, update_ns_mean,
//   prune_ns_mean, grow_ns_mean, reset_ns_mean, other_ns_mean
// Unused phases default to 0; `other_ns_mean` absorbs whatever the marks
// don't cover (data loading, eval, .item() syncs, Python loop overhead, etc).
// ---------------------------------------------------------------------------

// Per-phase Welford accumulator: tracks running mean + variance with no
// allocation, so a 60k-step run is just six floats × six phases. Per-sample
// overhead is ~5ns on top of the chrono::now() pair (negligible against the
// smallest phase we measure, which is ~50ns).
struct PhaseAcc {
  uint64_t Count = 0;
  double Mean = 0.0;   // mean ns
  double M2 = 0.0;     // sum of (x - mean)^2

  void Add(double X) {
    ++Count;
    double Delta = X - Mean;
    Mean += Delta / static_cast<double>(Count);
    double Delta2 = X - Mean;
    M2 += Delta * Delta2;
  }
  double StdDev() const {
    if (Count < 2)
      return 0.0;
    return std::sqrt(M2 / static_cast<double>(Count - 1));
  }
};

class PhaseTimer {
public:
  using Clk = std::chrono::steady_clock;
  using Ns = std::chrono::nanoseconds;

  void Tick() { Last_ = Clk::now(); }

  void MarkForward()    { Forward_.Add(DeltaNs()); }
  void MarkLoss()       { Loss_.Add(DeltaNs()); }
  void MarkBackward()   { Backward_.Add(DeltaNs()); }
  void MarkUpdate()     { Update_.Add(DeltaNs()); }
  // The old single `structural` phase is split into prune (Prune*) and grow
  // (Add*) so the two halves of structural adaptation are reported separately.
  void MarkPrune()      { Prune_.Add(DeltaNs()); }
  void MarkGrow()       { Grow_.Add(DeltaNs()); }
  void MarkReset()      { Reset_.Add(DeltaNs()); }
  void StepDone()       { ++StepCount_; }

  uint64_t StepCount() const { return StepCount_; }

  // Writes the canonical mean+std columns into the summary CSV. `WallSeconds`
  // sets `step_ns_mean` (one optimisation step worth of wall time) so the
  // reader can tell what fraction the phases account for; `other_ns_mean`
  // absorbs the rest.
  void WriteSummary(SummaryWriter &S, double WallSeconds) const {
    uint64_t Steps = std::max<uint64_t>(StepCount_, 1);
    double StepNsMean = (WallSeconds * 1e9) / static_cast<double>(Steps);
    auto W = [&](const PhaseAcc &A, const char *Name) {
      S.Set(std::string(Name) + "_ns_mean", A.Mean);
      S.Set(std::string(Name) + "_ns_std",  A.StdDev());
    };
    W(Forward_,    "forward");
    W(Loss_,       "loss");
    W(Backward_,   "backward");
    W(Update_,     "update");
    W(Prune_,      "prune");
    W(Grow_,       "grow");
    W(Reset_,      "reset");
    double Sum = Forward_.Mean + Loss_.Mean + Backward_.Mean +
                 Update_.Mean + Prune_.Mean + Grow_.Mean + Reset_.Mean;
    S.Set("step_count", static_cast<long long>(Steps));
    S.Set("step_ns_mean", StepNsMean);
    S.Set("other_ns_mean", std::max(0.0, StepNsMean - Sum));
  }

private:
  double DeltaNs() {
    auto Now = Clk::now();
    auto Dt = std::chrono::duration_cast<Ns>(Now - Last_).count();
    Last_ = Now;
    return static_cast<double>(Dt);
  }

  Clk::time_point Last_;
  PhaseAcc Forward_, Loss_, Backward_, Update_, Prune_, Grow_, Reset_;
  uint64_t StepCount_ = 0;
};

// ---------------------------------------------------------------------------
// MemoryProbe — breaks resident memory (VmRSS) into milestones so a
// .summary.csv can report where memory goes:
//   mem_overhead_kb  RSS right after startup (libs, .bss)
//   mem_dataset_kb   RSS delta across the dataset-load block
//   mem_weights_kb   RSS delta across the model/network-construction block
// `scratch` (peak − after-model) and `max` (peak) are derived downstream from
// the orchestrator's polled peak_rss_kb, so they are NOT emitted here.
//
// Usage (deltas are order-independent — call each End* right after its block):
//   bench::MemoryProbe MP;
//   MP.Start();                 // after arg-parse / startup
//   ... load dataset ...        MP.EndDataset();
//   ... build model ...         MP.EndWeights();
//   MP.WriteSummary(S);         // before S.Write(...)
// ---------------------------------------------------------------------------
inline long long ReadVmRssKb() {
  std::ifstream Status("/proc/self/status");
  std::string Line;
  while (std::getline(Status, Line)) {
    if (Line.rfind("VmRSS:", 0) == 0) {
      std::istringstream Iss(Line.substr(6));
      long long Kb = 0;
      Iss >> Kb;
      return Kb;
    }
  }
  return 0;
}

class MemoryProbe {
public:
  void Start()       { Overhead_ = Cursor_ = ReadVmRssKb(); }
  void EndDataset()  { long long R = ReadVmRssKb(); Dataset_ += R - Cursor_; Cursor_ = R; }
  void EndWeights()  { long long R = ReadVmRssKb(); Weights_ += R - Cursor_; Cursor_ = R; }

  void WriteSummary(SummaryWriter &S) const {
    S.Set("mem_overhead_kb", Overhead_);
    S.Set("mem_dataset_kb", std::max<long long>(0, Dataset_));
    S.Set("mem_weights_kb", std::max<long long>(0, Weights_));
  }

private:
  long long Overhead_ = 0, Dataset_ = 0, Weights_ = 0, Cursor_ = 0;
};

} // namespace bench

#endif // TRADITIONAL_RAW_COMMON_HPP
