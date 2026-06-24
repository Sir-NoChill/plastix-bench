// Header-only mmap-backed reader for the audio-prediction binary dataset
// produced by examples/audio-pred/prepare-cpp.py.
//
// Wire format (little-endian):
//   Header v1 (16 bytes): magic "APBD" | uint32 version=1 | uint64 n_steps
//                         observation dimension is implicitly 2500.
//   Header v2 (24 bytes): magic "APBD" | uint32 version=2 | uint64 n_steps
//                         | uint64 obs_dim
//                         obs_dim = n_freq_bins * n_mag_bins, configurable.
//   Record (packed_bytes + 1, repeated n_steps times):
//     bytes [0, packed_bytes)  packed observation bits, LSB-first
//                     observation[i] lives at byte (i >> 3), bit (i & 7);
//                     packed_bytes = ceil(obs_dim / 8). Trailing bits in the
//                     final byte are zero padding.
//     byte  [packed_bytes]     int8 reward
//
// The observation dimension is read from the header at runtime (v1 files map to
// 2500), so a single binary supports datasets generated with any box count.
//
// Usage:
//   audio_pred::Dataset DS("output/dataset.bin");
//   const std::size_t Dim = DS.ObservationDim();
//   for (std::size_t T = 0; T < DS.Size(); ++T) {
//     audio_pred::StepView Step = DS[T];
//     bool Bit0 = Step.Test(0);
//     std::int8_t R = Step.Reward();
//   }
//
// POSIX-only (uses mmap). Read-only, zero-copy: the constructor takes a single
// mmap and all StepView accesses are pointer arithmetic into the mapping.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace audio_pred {

// Legacy v1 observation dimension. v1 files carry no obs_dim field, so the
// reader assumes this; v2 files state their own dimension in the header.
inline constexpr std::size_t ObservationDim = 2500;
inline constexpr std::size_t PackedBytes = (ObservationDim + 7) / 8; // 313
inline constexpr std::size_t RecordBytes = PackedBytes + 1;          // 314
// FormatVersion is the version the writer emits; the reader accepts 1 and 2.
inline constexpr std::uint32_t FormatVersion = 2;
inline constexpr std::array<char, 4> Magic = {'A', 'P', 'B', 'D'};

// Base header common to every version. v2 appends a uint64 obs_dim after this.
struct DatasetHeader {
  char Magic[4];
  std::uint32_t Version;
  std::uint64_t NSteps;
};
static_assert(sizeof(DatasetHeader) == 16,
              "DatasetHeader must be 16 bytes for the wire format");

// Cheap view into a single 314-byte record inside an mmap'd dataset.
// Copying a StepView copies a single pointer; it does not own the storage.
class StepView {
public:
  // PackedBytes/ObsDim come from the owning Dataset's header so a StepView
  // works for any observation dimension.
  StepView(const std::uint8_t *Ptr, std::size_t PackedBytes,
           std::size_t ObsDim)
      : Ptr_(Ptr), PackedBytes_(PackedBytes), ObsDim_(ObsDim) {}

  // observation[Index] in {false, true}. Caller must keep Index < ObservationDim.
  bool Test(std::size_t Index) const {
    return (Ptr_[Index >> 3] >> (Index & 7)) & 1U;
  }

  // Signed reward; the default benchmark uses {-1, 0, +1}.
  std::int8_t Reward() const {
    return static_cast<std::int8_t>(Ptr_[PackedBytes_]);
  }

  // Number of set bits (n_freq_bins for a well-formed observation).
  std::size_t Popcount() const {
    std::size_t Count = 0;
    for (std::size_t I = 0; I < PackedBytes_; ++I) {
      Count += static_cast<std::size_t>(__builtin_popcount(Ptr_[I]));
    }
    return Count;
  }

  // Expand bits into a contiguous float buffer (1.0f for set bits, 0.0f otherwise).
  // Out.size() must equal the dataset's observation dimension. Useful to feed
  // the network's std::span<const float> input.
  void ExpandTo(std::span<float> Out) const {
    if (Out.size() != ObsDim_)
      throw std::runtime_error(
          "audio_pred::StepView::ExpandTo: span size must equal ObservationDim");
    for (std::size_t I = 0; I < ObsDim_; ++I)
      Out[I] = Test(I) ? 1.0f : 0.0f;
  }

  const std::uint8_t *RawBytes() const { return Ptr_; }

private:
  const std::uint8_t *Ptr_;
  std::size_t PackedBytes_;
  std::size_t ObsDim_;
};

// Owning, mmap-backed view of an APBD file. Move-only.
class Dataset {
public:
  explicit Dataset(const std::filesystem::path &Path) { OpenAndMap(Path); }

  ~Dataset() noexcept { Close(); }

  Dataset(const Dataset &) = delete;
  Dataset &operator=(const Dataset &) = delete;

  Dataset(Dataset &&Other) noexcept { MoveFrom(Other); }
  Dataset &operator=(Dataset &&Other) noexcept {
    if (this != &Other) {
      Close();
      MoveFrom(Other);
    }
    return *this;
  }

  std::size_t Size() const { return NSteps_; }

  // Observation dimension for this dataset (read from the header).
  std::size_t ObservationDim() const { return ObsDim_; }
  std::size_t PackedBytes() const { return PackedBytes_; }

  StepView operator[](std::size_t Index) const {
    return StepView(Records_ + Index * RecordBytes_, PackedBytes_, ObsDim_);
  }

  // Iterator support so range-for and STL algorithms work.
  class Iterator {
  public:
    using value_type = StepView;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using reference = StepView;
    using pointer = void;

    Iterator() = default;
    Iterator(const std::uint8_t *Base, std::size_t Index,
             std::size_t RecordBytes, std::size_t PackedBytes,
             std::size_t ObsDim)
        : Base_(Base), Index_(Index), RecordBytes_(RecordBytes),
          PackedBytes_(PackedBytes), ObsDim_(ObsDim) {}

    StepView operator*() const {
      return StepView(Base_ + Index_ * RecordBytes_, PackedBytes_, ObsDim_);
    }
    Iterator &operator++() {
      ++Index_;
      return *this;
    }
    Iterator operator++(int) {
      Iterator T = *this;
      ++Index_;
      return T;
    }
    bool operator==(const Iterator &O) const { return Index_ == O.Index_; }
    bool operator!=(const Iterator &O) const { return Index_ != O.Index_; }

  private:
    const std::uint8_t *Base_ = nullptr;
    std::size_t Index_ = 0;
    std::size_t RecordBytes_ = 0;
    std::size_t PackedBytes_ = 0;
    std::size_t ObsDim_ = 0;
  };

  Iterator begin() const {
    return Iterator(Records_, 0, RecordBytes_, PackedBytes_, ObsDim_);
  }
  Iterator end() const {
    return Iterator(Records_, NSteps_, RecordBytes_, PackedBytes_, ObsDim_);
  }

private:
  void OpenAndMap(const std::filesystem::path &Path) {
    Fd_ = ::open(Path.c_str(), O_RDONLY);
    if (Fd_ < 0)
      throw std::runtime_error("audio_pred::Dataset: open failed for " +
                               Path.string() + " (errno=" +
                               std::to_string(errno) + ")");

    struct stat St {};
    if (::fstat(Fd_, &St) < 0) {
      int E = errno;
      ::close(Fd_);
      Fd_ = -1;
      throw std::runtime_error("audio_pred::Dataset: fstat failed (errno=" +
                               std::to_string(E) + ")");
    }
    FileSize_ = static_cast<std::size_t>(St.st_size);

    if (FileSize_ < sizeof(DatasetHeader)) {
      ::close(Fd_);
      Fd_ = -1;
      throw std::runtime_error("audio_pred::Dataset: file too small for header");
    }

    void *Map = ::mmap(nullptr, FileSize_, PROT_READ, MAP_PRIVATE, Fd_, 0);
    if (Map == MAP_FAILED) {
      int E = errno;
      ::close(Fd_);
      Fd_ = -1;
      throw std::runtime_error("audio_pred::Dataset: mmap failed (errno=" +
                               std::to_string(E) + ")");
    }
    Map_ = static_cast<const std::uint8_t *>(Map);

    // The dataset is intended to be streamed top-to-bottom; tell the kernel.
    ::madvise(const_cast<std::uint8_t *>(Map_), FileSize_, MADV_SEQUENTIAL);

    DatasetHeader H{};
    std::memcpy(&H, Map_, sizeof(H));
    if (std::memcmp(H.Magic, Magic.data(), 4) != 0) {
      Close();
      throw std::runtime_error("audio_pred::Dataset: bad magic (expected APBD)");
    }
    NSteps_ = static_cast<std::size_t>(H.NSteps);

    // v1: no obs_dim field, dimension is the legacy 2500. v2: a uint64 obs_dim
    // follows the 16-byte base header.
    std::size_t HeaderBytes = sizeof(DatasetHeader);
    if (H.Version == 1) {
      ObsDim_ = audio_pred::ObservationDim;
    } else if (H.Version == 2) {
      HeaderBytes = sizeof(DatasetHeader) + sizeof(std::uint64_t);
      if (FileSize_ < HeaderBytes) {
        Close();
        throw std::runtime_error(
            "audio_pred::Dataset: file too small for v2 header");
      }
      std::uint64_t ObsDim = 0;
      std::memcpy(&ObsDim, Map_ + sizeof(DatasetHeader), sizeof(ObsDim));
      ObsDim_ = static_cast<std::size_t>(ObsDim);
    } else {
      Close();
      throw std::runtime_error("audio_pred::Dataset: unsupported version " +
                               std::to_string(H.Version));
    }
    if (ObsDim_ == 0) {
      Close();
      throw std::runtime_error("audio_pred::Dataset: obs_dim is zero");
    }
    PackedBytes_ = (ObsDim_ + 7) / 8;
    RecordBytes_ = PackedBytes_ + 1;

    const std::size_t Expected = HeaderBytes + NSteps_ * RecordBytes_;
    if (FileSize_ != Expected) {
      Close();
      throw std::runtime_error(
          "audio_pred::Dataset: file size " + std::to_string(FileSize_) +
          " does not match header (expected " + std::to_string(Expected) + ")");
    }

    Records_ = Map_ + HeaderBytes;
  }

  void Close() noexcept {
    if (Map_ != nullptr) {
      ::munmap(const_cast<std::uint8_t *>(Map_), FileSize_);
      Map_ = nullptr;
    }
    if (Fd_ >= 0) {
      ::close(Fd_);
      Fd_ = -1;
    }
    Records_ = nullptr;
    FileSize_ = 0;
    NSteps_ = 0;
    ObsDim_ = 0;
    PackedBytes_ = 0;
    RecordBytes_ = 0;
  }

  void MoveFrom(Dataset &Other) noexcept {
    Fd_ = Other.Fd_;
    Map_ = Other.Map_;
    Records_ = Other.Records_;
    FileSize_ = Other.FileSize_;
    NSteps_ = Other.NSteps_;
    ObsDim_ = Other.ObsDim_;
    PackedBytes_ = Other.PackedBytes_;
    RecordBytes_ = Other.RecordBytes_;
    Other.Fd_ = -1;
    Other.Map_ = nullptr;
    Other.Records_ = nullptr;
    Other.FileSize_ = 0;
    Other.NSteps_ = 0;
    Other.ObsDim_ = 0;
    Other.PackedBytes_ = 0;
    Other.RecordBytes_ = 0;
  }

  int Fd_ = -1;
  const std::uint8_t *Map_ = nullptr;
  const std::uint8_t *Records_ = nullptr;
  std::size_t FileSize_ = 0;
  std::size_t NSteps_ = 0;
  std::size_t ObsDim_ = 0;
  std::size_t PackedBytes_ = 0;
  std::size_t RecordBytes_ = 0;
};

// Synthesize a dataset file on disk. Used by tests and for ad-hoc fixtures;
// the production pipeline goes through prepare-cpp.py.
//
// Observations[i] must have length ObservationDim with values in {0, 1}.
// Rewards[i] is the int8 reward for step i.
inline void WriteDataset(const std::filesystem::path &Path,
                         std::span<const std::array<std::uint8_t, ObservationDim>>
                             Observations,
                         std::span<const std::int8_t> Rewards) {
  if (Observations.size() != Rewards.size())
    throw std::runtime_error(
        "audio_pred::WriteDataset: observation/reward size mismatch");

  const std::uint64_t N = Observations.size();

  std::FILE *F = std::fopen(Path.c_str(), "wb");
  if (!F)
    throw std::runtime_error("audio_pred::WriteDataset: fopen failed for " +
                             Path.string());

  DatasetHeader H{};
  std::memcpy(H.Magic, Magic.data(), 4);
  H.Version = FormatVersion;
  H.NSteps = N;
  if (std::fwrite(&H, sizeof(H), 1, F) != 1) {
    std::fclose(F);
    throw std::runtime_error("audio_pred::WriteDataset: header write failed");
  }
  // v2: emit obs_dim right after the base header.
  const std::uint64_t ObsDim = ObservationDim;
  if (std::fwrite(&ObsDim, sizeof(ObsDim), 1, F) != 1) {
    std::fclose(F);
    throw std::runtime_error("audio_pred::WriteDataset: obs_dim write failed");
  }

  std::array<std::uint8_t, RecordBytes> Record{};
  for (std::size_t T = 0; T < N; ++T) {
    Record.fill(0);
    for (std::size_t I = 0; I < ObservationDim; ++I) {
      if (Observations[T][I])
        Record[I >> 3] |= static_cast<std::uint8_t>(1U << (I & 7));
    }
    Record[PackedBytes] = static_cast<std::uint8_t>(Rewards[T]);
    if (std::fwrite(Record.data(), 1, RecordBytes, F) != RecordBytes) {
      std::fclose(F);
      throw std::runtime_error("audio_pred::WriteDataset: record write failed");
    }
  }

  std::fclose(F);
}

} // namespace audio_pred
