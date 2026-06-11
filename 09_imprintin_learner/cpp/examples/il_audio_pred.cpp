// Harness driving il::ImprintingLearner on the audio-prediction benchmark.
//
// Reads an APBD dataset (see dataset.hpp), feeds each step's 2500-dim binary
// observation + scalar reward to the learner, and records the GVF prediction.
// Observation features are always tenured, so feature generation can build
// pattern/memory units on top of them from the start.
//
//   il_audio_pred <dataset.bin> [max_steps] [out.csv]
//
// Produce dataset.bin via examples/prepare-cpp.py (see examples/README.md).

#include "dataset.hpp"
#include "imprinting/imprinting_learner.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>

namespace {

// Starting hyperparameters for the benchmark (to be tuned). SwiftTD values
// follow the thesis/reference harness; generation/removal/tenure are this
// project's settings.
il::HyperParams MakeHyperParams() {
    il::HyperParams hp;
    // Room for the 2500 observation features plus generated ones, under the
    // int16 feature-id cap (32768).
    hp.capacity = 16384;
    hp.trace_dim = 1;

    // Tenure (magnitude, hysteresis) — thesis-style thresholds.
    hp.tenure_threshold = 0.01f;
    hp.tenure_track_threshold = 3e-4f;
    hp.demotion_factor = 0.5f;

    // SwiftTD.
    hp.gamma = 0.99f;  // not specified by the thesis snippet; kept for interpretability
    hp.lambda = 0.9f;
    hp.alpha = 3e-3f;  // initial step size
    hp.eta = 0.1f;
    hp.eta_min = 1e-7f;
    hp.decay = 0.999f;
    hp.meta_step_size = 1e-3f;
    hp.epsilon = 1e-5f;

    // Feature generation (up to 10 of each type per step).
    hp.k_pattern = 10;
    hp.k_memory = 10;
    hp.pattern_fractions = {0.6f, 0.7f, 0.8f, 0.9f};  // sampled per pattern
    hp.pattern_min_connections = 2;
    hp.pattern_max_connections = 8;
    // Memory units sample delay (k1) and active window (k2) from these ranges.
    // delay starts at 0 (a delay-0 unit fires immediately, like a short pattern).
    hp.memory_delay_min = 0;
    hp.memory_delay_max = 20;
    hp.memory_window_min = 1;
    hp.memory_window_max = 3;
    hp.rng_seed = 42;

    // Feature removal.
    hp.epsilon_z = 0.01f;

    // Env-var overrides so hyperparameters can be swept without recompiling.
    if (const char* e = std::getenv("IL_GAMMA")) hp.gamma = std::stof(e);
    if (const char* e = std::getenv("IL_LAMBDA")) hp.lambda = std::stof(e);
    if (const char* e = std::getenv("IL_ALPHA")) hp.alpha = std::stof(e);
    if (const char* e = std::getenv("IL_ETA")) hp.eta = std::stof(e);
    if (const char* e = std::getenv("IL_EPSILON_Z")) hp.epsilon_z = std::stof(e);
    if (const char* e = std::getenv("IL_K_PATTERN"))
        hp.k_pattern = static_cast<std::uint32_t>(std::stoul(e));
    if (const char* e = std::getenv("IL_K_MEMORY"))
        hp.k_memory = static_cast<std::uint32_t>(std::stoul(e));
    if (const char* e = std::getenv("IL_MEMORY_DELAY_MIN"))
        hp.memory_delay_min = static_cast<std::uint16_t>(std::stoul(e));
    if (const char* e = std::getenv("IL_MEMORY_DELAY_MAX"))
        hp.memory_delay_max = static_cast<std::uint16_t>(std::stoul(e));
    if (const char* e = std::getenv("IL_MEMORY_WINDOW_MIN"))
        hp.memory_window_min = static_cast<std::uint16_t>(std::stoul(e));
    if (const char* e = std::getenv("IL_MEMORY_WINDOW_MAX"))
        hp.memory_window_max = static_cast<std::uint16_t>(std::stoul(e));
    return hp;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset.bin> [max_steps] [out.csv]\n"
                     "  Generate dataset.bin via examples/prepare-cpp.py.\n";
        return EXIT_FAILURE;
    }

    const std::string dataset_path = argv[1];
    audio_pred::Dataset ds(dataset_path);
    // max_steps: omit or pass 0 to stream the whole dataset.
    std::size_t requested = (argc >= 3) ? std::stoull(argv[2]) : ds.Size();
    if (requested == 0) {
        requested = ds.Size();
    }
    const std::size_t max_steps = std::min(requested, ds.Size());
    const std::string out_csv = (argc >= 4) ? argv[3] : "il_predictions.csv";

    std::cout << "il imprinting-learner (Audio Prediction)\n"
              << "  dataset: " << dataset_path << " (" << ds.Size()
              << " steps, using " << max_steps << ")\n";

    const il::HyperParams hp = MakeHyperParams();
    il::ImprintingLearner learner(hp);
    learner.addObservations(audio_pred::ObservationDim);  // all tenured
    std::cout << "  observations: " << audio_pred::ObservationDim
              << " (tenured); capacity " << hp.capacity << "\n"
              << "  hp: gamma=" << hp.gamma << " lambda=" << hp.lambda
              << " alpha=" << hp.alpha << " eta=" << hp.eta
              << " epsilon_z=" << hp.epsilon_z << " k_pattern=" << hp.k_pattern
              << " k_memory=" << hp.k_memory << "\n";

    std::ofstream csv(out_csv);
    if (!csv) {
        std::cerr << "error: cannot open " << out_csv << " for writing\n";
        return EXIT_FAILURE;
    }
    csv << std::setprecision(7);
    csv << "step,reward,prediction,n_features\n";

    std::array<std::uint8_t, audio_pred::ObservationDim> obs{};
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    std::size_t peak_features = learner.arena().size();
    std::size_t nonzero_rewards = 0;

    constexpr std::size_t kLogEvery = 5000;
    for (std::size_t t = 0; t < max_steps; ++t) {
        const audio_pred::StepView step = ds[t];
        for (std::size_t i = 0; i < audio_pred::ObservationDim; ++i) {
            obs[i] = step.Test(i) ? std::uint8_t{1} : std::uint8_t{0};
        }
        const float reward = static_cast<float>(step.Reward());
        const float v = learner.step(obs, reward);

        const std::size_t n = learner.arena().size();
        if (step.Reward() != 0) {
            ++nonzero_rewards;
        }
        peak_features = std::max(peak_features, n);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);

        csv << t << ',' << int(step.Reward()) << ',' << v << ',' << n << '\n';

        if (t % kLogEvery == 0) {
            std::cout << std::fixed << std::setprecision(4) << "  step "
                      << std::setw(7) << t << "  reward " << std::setw(2)
                      << int(step.Reward()) << "  V " << std::setw(10) << v
                      << "  features " << std::setw(6) << n << "\n";
        }
    }

    std::cout << "done.\n"
              << "  steps processed : " << max_steps << "\n"
              << "  nonzero rewards : " << nonzero_rewards << "\n"
              << "  final features  : " << learner.arena().size() << "\n"
              << "  peak features   : " << peak_features << "\n"
              << "  V range         : [" << min_v << ", " << max_v << "]\n"
              << "  predictions -> " << out_csv << "\n";
    return 0;
}
