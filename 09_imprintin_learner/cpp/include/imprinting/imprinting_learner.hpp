#pragma once

#include "imprinting/feature_arena.hpp"

#include <swifttd/SwiftTD.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace il {

// Single place to set every tunable knob of the agent, separate from any
// ad-hoc test configuration. Defaults are reasonable starting points, not
// tuned values.
struct HyperParams {
    // --- arena / capacity ---
    std::uint32_t capacity = 1024;  // max features = SwiftTD weight slots
    std::size_t trace_dim = 1;      // eligibility-trace width per feature

    // --- tenure thresholds (on |weight|) ---
    float tenure_threshold = 1.0f;
    float tenure_track_threshold = 0.5f;
    // Demotion thresholds are these times demotion_factor (< 1), so a unit must
    // fall well below its promotion threshold before demoting (hysteresis): the
    // bar to demote is harder to hit than the bar to promote.
    float demotion_factor = 0.5f;

    // --- SwiftTD ---
    float lambda = 0.95f;          // eligibility-trace decay
    float alpha = 1e-2f;           // initial per-feature step size
    float gamma = 0.99f;           // discount factor
    float epsilon = 1e-5f;         // numerical-stability constant
    float eta = 0.1f;              // bound on the rate of learning
    float decay = 0.999f;          // step-size decay
    float meta_step_size = 1e-3f;  // meta learning rate
    float eta_min = 1e-10f;        // floor on the step size

    // --- feature generation (0 disables that type) ---
    std::uint32_t k_pattern = 0;   // max pattern features generated per step
    std::uint32_t k_memory = 0;    // max memory features generated per step
    // Activation fraction k/n: a pattern fires when >= ceil(fraction*n) members
    // are active. Generation draws one fraction (per neuron or per generation;
    // see kSampleGenerationParamsPerNeuron) uniformly from this set.
    std::vector<float> pattern_fractions{0.5f};
    std::uint16_t pattern_min_connections = 2;  // n in [min, max]
    std::uint16_t pattern_max_connections = 8;  // <= kMaxPatternConnections
    // Generated memory features sample delay (k1) and active window (k2)
    // uniformly from these inclusive ranges (min == max gives a fixed value).
    std::uint16_t memory_delay_min = 1;
    std::uint16_t memory_delay_max = 1;
    std::uint16_t memory_window_min = 1;
    std::uint16_t memory_window_max = 1;
    std::uint64_t rng_seed = 0xC0FFEEULL;  // seeds the generation sampler

    // --- feature removal ---
    // After SwiftTD decays z, an Idle (non-observation, non-referenced) feature
    // is removed when z[i] < e^beta[i] * epsilon_z. 0 disables removal.
    float epsilon_z = 0.0f;
};

// Compile-time choice for how generation samples its randomized parameters
// (pattern fraction, memory delay/window): true -> an independent draw per
// generated neuron; false -> one draw shared by all neurons spawned in a step.
// Override at build time with -DIL_SAMPLE_GENERATION_PARAMS_PER_NEURON=0.
#ifndef IL_SAMPLE_GENERATION_PARAMS_PER_NEURON
#define IL_SAMPLE_GENERATION_PARAMS_PER_NEURON 1
#endif
inline constexpr bool kSampleGenerationParamsPerNeuron =
    IL_SAMPLE_GENERATION_PARAMS_PER_NEURON;

// Ties the SOA FeatureArena (forward pass: activations + GVF prediction) to
// SwiftTD (backward pass: TD weight update). Each step:
//   1. SwiftTD's weight span is bound to the arena's weight column (in place),
//   2. arena recomputes activations from the binary input,
//   3. active feature ids + reward are handed to SwiftTD, which updates the
//      arena's weights directly (no copy-back),
//   4. the tenure policy promotes features whose |weight| crosses a threshold.
//
// The arena owns the weights; SwiftTD writes through the bound span. (SwiftTD
// still houses the other per-feature learning vectors — z, beta, ... — which
// the learner will manage for feature removal in a later pass.)
class ImprintingLearner {
public:
    explicit ImprintingLearner(HyperParams params = {});

    // Population growth (delegates to the arena, guarding the SwiftTD capacity).
    // Observation features are always tenured (never idle, never removed).
    std::uint32_t addObservation(std::uint32_t input_index);
    void addObservations(std::size_t input_dim);
    // Generated features start Idle and earn tenure as their |weight| grows.
    std::uint32_t addPattern(std::span<const std::uint32_t> members, float fraction,
                             Status status = Status::Idle);
    std::uint32_t addMemory(std::uint32_t source, std::uint16_t delay,
                            std::uint16_t window, Status status = Status::Idle);

    // Advance one timestep with a binary input and a reward; returns the GVF
    // prediction (value before this step's weight update).
    float step(std::span<const std::uint8_t> input, float reward);

    const HyperParams& params() const noexcept { return params_; }
    const FeatureArena& arena() const noexcept { return arena_; }
    FeatureArena& arena() noexcept { return arena_; }

    // Per-phase timing collected by step(). Always populated -- two
    // steady_clock::now() calls per phase add ~40-80ns/step, negligible
    // against the SwiftTD + arena work. Counters use Welford's online
    // algorithm so we get mean and std without storing per-step samples.
    // Clear them via resetProfile() between measurement windows.
    struct PhaseAcc {
        std::uint64_t count = 0;
        double mean = 0.0;
        double m2 = 0.0;        // sum of (x - mean)^2
        void add(double x) {
            ++count;
            double delta = x - mean;
            mean += delta / static_cast<double>(count);
            double delta2 = x - mean;
            m2 += delta * delta2;
        }
        double stddev() const {
            return count < 2 ? 0.0
                             : std::sqrt(m2 / static_cast<double>(count - 1));
        }
    };
    struct Profile {
        PhaseAcc forward;        // arena.step + active-id collection
        PhaseAcc backward;       // SwiftTD weight update (TD δ + step)
        PhaseAcc prune;          // tenure policy + remove idle/decayed features
        PhaseAcc grow;           // generate new features + activation snapshot
        std::uint64_t step_count = 0;
    };
    const Profile& profile() const noexcept { return profile_; }
    void resetProfile() noexcept { profile_ = {}; }

private:
    void ensureCapacity(std::size_t additional) const;
    void applyTenurePolicy();
    void generateFeatures();  // spawn up to k_pattern/k_memory features this step
    void removeFeatures();    // swap-pop idle, unreferenced, decayed features

    HyperParams params_;
    FeatureArena arena_;
    SwiftTDBinaryFeatures td_;
    std::mt19937_64 rng_;  // generation sampler

    // reused scratch buffers (avoid per-step allocation)
    std::vector<int> active_scratch_;            // active feature ids for SwiftTD
    std::vector<std::uint8_t> prev_activation_;  // phi_{t-1} for the generation gate
    std::vector<std::uint32_t> pool_scratch_;    // active-tenured candidate pool
    std::vector<std::uint32_t> members_scratch_;  // sampled pattern members
    Profile profile_;
};

}  // namespace il
