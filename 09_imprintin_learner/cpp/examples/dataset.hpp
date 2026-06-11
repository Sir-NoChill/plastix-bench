// Header-only mmap-backed reader for the audio-prediction binary dataset
// produced by examples/audio-pred/prepare-cpp.py.
//
// Wire format (little-endian):
//   Header (16 bytes): magic "APBD" | uint32 version | uint64 n_steps
//   Record (314 bytes, repeated n_steps times):
//     bytes [0, 313)  packed observation bits, LSB-first
//                     observation[i] lives at byte (i >> 3), bit (i & 7).
//                     Bits 2500..2503 in byte 312 are zero padding.
//     byte  [313]     int8 reward
//
// Usage:
//   audio_pred::Dataset DS("output/dataset.bin");
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

inline constexpr std::size_t ObservationDim = 2500;
inline constexpr std::size_t PackedBytes = (ObservationDim + 7) / 8; // 313
inline constexpr std::size_t RecordBytes = PackedBytes + 1;          // 314
inline constexpr std::uint32_t FormatVersion = 1;
inline constexpr std::array<char, 4> Magic = {'A', 'P', 'B', 'D'};

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
  explicit StepView(const std::uint8_t *Ptr) : Ptr_(Ptr) {}

  // observation[Index] in {false, true}. Caller must keep Index < ObservationDim.
  bool Test(std::size_t Index) const {
    return (Ptr_[Index >> 3] >> (Index & 7)) & 1U;
  }

  // Signed reward; the default benchmark uses {-1, 0, +1}.
  std::int8_t Reward() const {
    return static_cast<std::int8_t>(Ptr_[PackedBytes]);
  }

  // Number of set bits (50 for a well-formed observation).
  std::size_t Popcount() const {
    std::size_t Count = 0;
    for (std::size_t I = 0; I < PackedBytes; ++I) {
      Count += static_cast<std::size_t>(__builtin_popcount(Ptr_[I]));
    }
    return Count;
  }

  // Expand bits into a contiguous float buffer (1.0f for set bits, 0.0f otherwise).
  // Out.size() must equal ObservationDim. Useful to feed the network's
  // std::span<const float> input.
  void ExpandTo(std::span<float> Out) const {
    if (Out.size() != ObservationDim)
      throw std::runtime_error(
          "audio_pred::StepView::ExpandTo: span size must equal ObservationDim");
    for (std::size_t I = 0; I < ObservationDim; ++I)
      Out[I] = Test(I) ? 1.0f : 0.0f;
  }

  const std::uint8_t *RawBytes() const { return Ptr_; }

private:
  const std::uint8_t *Ptr_;
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

  StepView operator[](std::size_t Index) const {
    return StepView(Records_ + Index * RecordBytes);
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
    Iterator(const std::uint8_t *Base, std::size_t Index)
        : Base_(Base), Index_(Index) {}

    StepView operator*() const {
      return StepView(Base_ + Index_ * RecordBytes);
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
  };

  Iterator begin() const { return Iterator(Records_, 0); }
  Iterator end() const { return Iterator(Records_, NSteps_); }

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
    if (H.Version != FormatVersion) {
      Close();
      throw std::runtime_error("audio_pred::Dataset: unsupported version " +
                               std::to_string(H.Version));
    }
    NSteps_ = static_cast<std::size_t>(H.NSteps);

    const std::size_t Expected = sizeof(DatasetHeader) + NSteps_ * RecordBytes;
    if (FileSize_ != Expected) {
      Close();
      throw std::runtime_error(
          "audio_pred::Dataset: file size " + std::to_string(FileSize_) +
          " does not match header (expected " + std::to_string(Expected) + ")");
    }

    Records_ = Map_ + sizeof(DatasetHeader);
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
  }

  void MoveFrom(Dataset &Other) noexcept {
    Fd_ = Other.Fd_;
    Map_ = Other.Map_;
    Records_ = Other.Records_;
    FileSize_ = Other.FileSize_;
    NSteps_ = Other.NSteps_;
    Other.Fd_ = -1;
    Other.Map_ = nullptr;
    Other.Records_ = nullptr;
    Other.FileSize_ = 0;
    Other.NSteps_ = 0;
  }

  int Fd_ = -1;
  const std::uint8_t *Map_ = nullptr;
  const std::uint8_t *Records_ = nullptr;
  std::size_t FileSize_ = 0;
  std::size_t NSteps_ = 0;
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
