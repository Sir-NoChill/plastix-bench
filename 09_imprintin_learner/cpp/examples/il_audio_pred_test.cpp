// Integration test: drive il::ImprintingLearner over a synthesized APBD dataset
// (no audio pipeline needed), exercising the same path as the il_audio_pred
// harness. Verifies the dataset<->learner glue, the observation-always-tenured
// invariant, finite predictions, and that generation fires.

#include "dataset.hpp"
#include "imprinting/imprinting_learner.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <vector>

namespace {

// One 50-hot observation (matching the benchmark's exactly-50-ones property).
std::array<std::uint8_t, audio_pred::ObservationDim> MakeObservation(std::mt19937& rng) {
    std::array<std::uint8_t, audio_pred::ObservationDim> obs{};
    std::uniform_int_distribution<std::size_t> pick(0, audio_pred::ObservationDim - 1);
    std::size_t set = 0;
    while (set < 50) {
        const std::size_t i = pick(rng);
        if (obs[i] == 0) {
            obs[i] = 1;
            ++set;
        }
    }
    return obs;
}

std::filesystem::path WriteSyntheticDataset(std::size_t n_steps) {
    std::mt19937 rng(7);
    std::vector<std::array<std::uint8_t, audio_pred::ObservationDim>> observations;
    std::vector<std::int8_t> rewards;
    observations.reserve(n_steps);
    rewards.reserve(n_steps);
    for (std::size_t t = 0; t < n_steps; ++t) {
        observations.push_back(MakeObservation(rng));
        rewards.push_back(t % 25 == 0 ? std::int8_t{1} : std::int8_t{0});  // sparse reward
    }
    const auto path = std::filesystem::temp_directory_path() /
                      ("il_audio_pred_test_" + std::to_string(::getpid()) + ".bin");
    audio_pred::WriteDataset(path, std::span{observations}, std::span{rewards});
    return path;
}

TEST(AudioPredHarness, StreamsAndGrowsOnSyntheticData) {
    constexpr std::size_t kSteps = 200;
    const auto path = WriteSyntheticDataset(kSteps);
    audio_pred::Dataset ds(path);
    ASSERT_EQ(ds.Size(), kSteps);

    il::HyperParams hp;
    hp.capacity = 4096;
    hp.gamma = 1.0f;
    // eta bounds the rate of learning AND the generation budget; with ~50 active
    // observations at alpha=0.01 the base tau ~ 0.5, so use a generous eta here
    // to actually exercise generation (the reference eta=0.05 throttles it).
    hp.eta = 2.0f;
    hp.alpha = 0.01f;
    hp.k_pattern = 2;
    hp.k_memory = 2;
    hp.pattern_max_connections = 6;
    hp.epsilon_z = 0.01f;  // generation + removal both on

    il::ImprintingLearner learner(hp);
    learner.addObservations(audio_pred::ObservationDim);

    // Observation features must be tenured (so generation can build on them).
    EXPECT_EQ(learner.arena()[0].getStatus(), il::Status::Tenure);
    EXPECT_EQ(learner.arena()[1234].getStatus(), il::Status::Tenure);
    EXPECT_EQ(learner.arena()[audio_pred::ObservationDim - 1].getStatus(),
              il::Status::Tenure);

    std::array<std::uint8_t, audio_pred::ObservationDim> obs{};
    bool all_finite = true;
    std::size_t peak = learner.arena().size();
    for (std::size_t t = 0; t < kSteps; ++t) {
        const audio_pred::StepView step = ds[t];
        for (std::size_t i = 0; i < audio_pred::ObservationDim; ++i) {
            obs[i] = step.Test(i) ? std::uint8_t{1} : std::uint8_t{0};
        }
        const float v = learner.step(obs, static_cast<float>(step.Reward()));
        all_finite = all_finite && std::isfinite(v);
        peak = std::max(peak, learner.arena().size());
        EXPECT_LE(learner.arena().size(), hp.capacity);  // capacity respected
    }

    EXPECT_TRUE(all_finite);
    EXPECT_GT(peak, audio_pred::ObservationDim);  // generation fired at some point

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace
