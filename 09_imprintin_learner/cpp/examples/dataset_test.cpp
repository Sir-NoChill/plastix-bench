// Standalone sanity test for dataset.hpp. Writes a synthetic dataset to a
// temporary path, reloads it through the mmap reader, and verifies that bits
// and rewards round-trip exactly.

#include "dataset.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <vector>

namespace {

#define EXPECT(Cond)                                                           \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      std::cerr << "EXPECT failed at " << __FILE__ << ":" << __LINE__ << ": "  \
                << #Cond << "\n";                                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Build a deterministic observation pattern: bit I is set iff
// (I * 2654435761u + Step * 0x9E3779B97F4A7C15ull) is odd. The exact rule does
// not matter; we just need something that exercises many bytes.
std::array<std::uint8_t, audio_pred::ObservationDim> MakeObservation(std::size_t Step) {
  std::array<std::uint8_t, audio_pred::ObservationDim> Obs{};
  for (std::size_t I = 0; I < audio_pred::ObservationDim; ++I) {
    auto H = static_cast<std::uint64_t>(I) * 2654435761ull +
             static_cast<std::uint64_t>(Step) * 0x9E3779B97F4A7C15ull;
    Obs[I] = static_cast<std::uint8_t>(H & 1ull);
  }
  return Obs;
}

void TestRoundTrip() {
  constexpr std::size_t N = 7;

  std::vector<std::array<std::uint8_t, audio_pred::ObservationDim>> Observations;
  std::vector<std::int8_t> Rewards;
  Observations.reserve(N);
  Rewards.reserve(N);
  for (std::size_t T = 0; T < N; ++T) {
    Observations.push_back(MakeObservation(T));
    // Cover the full int8 range, including negatives.
    Rewards.push_back(static_cast<std::int8_t>(static_cast<int>(T) - 3));
  }

  auto Path = std::filesystem::temp_directory_path() /
              ("audio_pred_dataset_test_" +
               std::to_string(::getpid()) + ".bin");

  audio_pred::WriteDataset(Path, std::span{Observations}, std::span{Rewards});

  {
    audio_pred::Dataset DS(Path);
    EXPECT(DS.Size() == N);

    for (std::size_t T = 0; T < N; ++T) {
      audio_pred::StepView Step = DS[T];
      EXPECT(Step.Reward() == Rewards[T]);

      // Spot-check every bit; cheap enough for N=7.
      for (std::size_t I = 0; I < audio_pred::ObservationDim; ++I) {
        EXPECT(Step.Test(I) == (Observations[T][I] != 0));
      }

      std::size_t Expected = 0;
      for (auto B : Observations[T])
        Expected += (B != 0);
      EXPECT(Step.Popcount() == Expected);

      std::array<float, audio_pred::ObservationDim> Floats{};
      Step.ExpandTo(Floats);
      for (std::size_t I = 0; I < audio_pred::ObservationDim; ++I) {
        EXPECT((Floats[I] == 1.0f) == (Observations[T][I] != 0));
      }
    }

    // Iterator traversal yields the same rewards in order.
    std::size_t T = 0;
    for (audio_pred::StepView Step : DS) {
      EXPECT(Step.Reward() == Rewards[T]);
      ++T;
    }
    EXPECT(T == N);
  }

  std::error_code Ec;
  std::filesystem::remove(Path, Ec); // best-effort
}

void TestBadMagicRejected() {
  auto Path = std::filesystem::temp_directory_path() /
              ("audio_pred_bad_magic_" +
               std::to_string(::getpid()) + ".bin");

  std::FILE *F = std::fopen(Path.c_str(), "wb");
  EXPECT(F != nullptr);
  audio_pred::DatasetHeader H{};
  std::memcpy(H.Magic, "XXXX", 4);
  H.Version = audio_pred::FormatVersion;
  H.NSteps = 0;
  EXPECT(std::fwrite(&H, sizeof(H), 1, F) == 1);
  std::fclose(F);

  bool Threw = false;
  try {
    audio_pred::Dataset DS(Path);
  } catch (const std::exception &E) {
    Threw = true;
    std::cerr << "  (expected) caught: " << E.what() << "\n";
  }
  EXPECT(Threw);

  std::error_code Ec;
  std::filesystem::remove(Path, Ec);
}

void TestSizeMismatchRejected() {
  auto Path = std::filesystem::temp_directory_path() /
              ("audio_pred_size_mismatch_" +
               std::to_string(::getpid()) + ".bin");

  std::FILE *F = std::fopen(Path.c_str(), "wb");
  EXPECT(F != nullptr);
  audio_pred::DatasetHeader H{};
  std::memcpy(H.Magic, audio_pred::Magic.data(), 4);
  H.Version = audio_pred::FormatVersion;
  H.NSteps = 5; // claim 5 records but write none
  EXPECT(std::fwrite(&H, sizeof(H), 1, F) == 1);
  std::fclose(F);

  bool Threw = false;
  try {
    audio_pred::Dataset DS(Path);
  } catch (const std::exception &E) {
    Threw = true;
    std::cerr << "  (expected) caught: " << E.what() << "\n";
  }
  EXPECT(Threw);

  std::error_code Ec;
  std::filesystem::remove(Path, Ec);
}

} // namespace

int main() {
  std::cout << "audio_pred dataset round-trip test\n";
  TestRoundTrip();
  std::cout << "  round-trip OK\n";
  TestBadMagicRejected();
  std::cout << "  bad-magic rejection OK\n";
  TestSizeMismatchRejected();
  std::cout << "  size-mismatch rejection OK\n";
  std::cout << "All dataset tests passed.\n";
  return 0;
}
