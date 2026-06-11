#include "imprinting/imprinting_learner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

// Promotion rank so the tenure policy only ever moves a feature upward.
int statusRank(il::Status s) {
    switch (s) {
        case il::Status::Idle:
            return 0;
        case il::Status::TenureTrack:
            return 1;
        case il::Status::Tenure:
            return 2;
    }
    return 0;
}

}  // namespace

namespace il {

namespace {

ArenaConfig arenaConfigFrom(const HyperParams& p) {
    ArenaConfig cfg;
    cfg.tenure_threshold = p.tenure_threshold;
    cfg.tenure_track_threshold = p.tenure_track_threshold;
    cfg.default_step_size = p.alpha;
    cfg.trace_dim = p.trace_dim;
    return cfg;
}

}  // namespace

ImprintingLearner::ImprintingLearner(HyperParams params)
    : params_(params),
      arena_(arenaConfigFrom(params)),
      td_(static_cast<int>(params.capacity), params.lambda, params.alpha,
          params.gamma, params.epsilon, params.eta, params.decay,
          params.meta_step_size, params.eta_min),
      rng_(params.rng_seed) {
    // Reserve so the weight column never reallocates; SwiftTD binds a span to it.
    arena_.reserve(params_.capacity);
}

void ImprintingLearner::ensureCapacity(std::size_t additional) const {
    if (arena_.size() + additional > params_.capacity) {
        throw std::length_error(
            "ImprintingLearner: feature capacity exceeded (raise HyperParams.capacity)");
    }
}

std::uint32_t ImprintingLearner::addObservation(std::uint32_t input_index) {
    ensureCapacity(1);
    return arena_.addObservation(input_index, Status::Tenure);
}

void ImprintingLearner::addObservations(std::size_t input_dim) {
    ensureCapacity(input_dim);
    arena_.addObservations(input_dim, Status::Tenure);
}

std::uint32_t ImprintingLearner::addPattern(std::span<const std::uint32_t> members,
                                            float fraction, Status status) {
    ensureCapacity(1);
    return arena_.addPattern(members, fraction, status);
}

std::uint32_t ImprintingLearner::addMemory(std::uint32_t source, std::uint16_t delay,
                                           std::uint16_t window, Status status) {
    ensureCapacity(1);
    return arena_.addMemory(source, delay, window, status);
}

float ImprintingLearner::step(std::span<const std::uint8_t> input, float reward) {
    using Clk = std::chrono::steady_clock;
    using Ns = std::chrono::nanoseconds;
    auto t0 = Clk::now();

    // Keep phi_{t-1} aligned with the arena: features added since the last
    // snapshot count as inactive last step (pad 0).
    if (prev_activation_.size() < arena_.size()) {
        prev_activation_.resize(arena_.size(), 0);
    }

    // Bind SwiftTD's weights to the arena's column so Step() updates it in place.
    // (Rebound each step because the span length tracks the current feature
    // count; the backing storage is stable thanks to reserve().)
    td_.bindWeights(arena_.weights());

    // 1. Forward pass: refresh activations from the input and feature dynamics.
    arena_.step(input);

    // 2. Collect the ids of the features that fired this step.
    active_scratch_.clear();
    const std::span<const std::uint8_t> act = arena_.activations();
    for (std::size_t i = 0; i < act.size(); ++i) {
        if (act[i] != 0) {
            active_scratch_.push_back(static_cast<int>(i));
        }
    }
    auto t_forward = Clk::now();

    // 3. Backward pass: SwiftTD updates the arena's weights in place and returns
    //    the prediction (computed with the pre-update weights).
    const float prediction = td_.Step(active_scratch_, reward);
    auto t_backward = Clk::now();

    // 4. Promote/demote features by |weight| (hysteresis).
    applyTenurePolicy();

    // 5. Remove idle, decayed, unreferenced features (before generation, so
    //    features spawned this step are not immediately culled).
    removeFeatures();

    // 6. Generate new features (gated on phi_{t-1} and the tau budget).
    generateFeatures();

    // 7. Snapshot this step's activations (incl. just-generated features) so the
    //    next step's generation gate can read phi_t as phi_{t-1}.
    const std::span<const std::uint8_t> a = arena_.activations();
    prev_activation_.assign(a.begin(), a.end());
    auto t_structural = Clk::now();

    profile_.forward.add(static_cast<double>(
        std::chrono::duration_cast<Ns>(t_forward - t0).count()));
    profile_.backward.add(static_cast<double>(
        std::chrono::duration_cast<Ns>(t_backward - t_forward).count()));
    profile_.structural.add(static_cast<double>(
        std::chrono::duration_cast<Ns>(t_structural - t_backward).count()));
    profile_.step_count += 1;

    return prediction;
}

void ImprintingLearner::removeFeatures() {
    if (params_.epsilon_z <= 0.0f) {
        return;  // removal disabled
    }
    // z and beta live in SwiftTD (capacity-sized, stable spans); the underlying
    // values are mutated in place by swapPopFeature during the loop.
    const std::span<const float> z = td_.zTraces();
    const std::span<const float> betas = td_.betas();

    std::uint32_t i = 0;
    while (i < arena_.size()) {
        const FeatureRef f = arena_[i];
        const bool removable =
            f.getFeatureType() != FeatureType::Observation &&  // observations stay
            f.getStatus() == Status::Idle &&
            !arena_.isMemoryArmed(i) &&  // a delayed memory hasn't fired yet
            z[i] < std::exp(betas[i]) * params_.epsilon_z &&
            !arena_.isReferenced(i);  // v1: defer features that still feed others
        if (removable) {
            const auto last = static_cast<std::uint32_t>(arena_.size() - 1);
            arena_.swapPopRemove(i);       // arena columns + reference remap
            td_.swapPopFeature(i, last);   // SwiftTD's per-feature vectors
            if (i != last) {
                prev_activation_[i] = prev_activation_[last];  // keep phi_{t-1} aligned
            }
            prev_activation_.pop_back();
            // Re-examine slot i, which now holds the feature moved from `last`.
        } else {
            ++i;
        }
    }
}

void ImprintingLearner::generateFeatures() {
    if (params_.k_pattern == 0 && params_.k_memory == 0) {
        return;
    }

    // Gate + candidate pool: features active on the previous step that are
    // currently tenured. Empty pool -> no generation this step.
    pool_scratch_.clear();
    for (std::size_t id = 0; id < prev_activation_.size(); ++id) {
        if (prev_activation_[id] != 0 &&
            arena_[static_cast<std::uint32_t>(id)].getStatus() == Status::Tenure) {
            pool_scratch_.push_back(static_cast<std::uint32_t>(id));
        }
    }
    if (pool_scratch_.empty()) {
        return;
    }

    // Base learning rate tau = sum of step sizes (e^beta) over active features.
    const std::span<const std::uint8_t> act = arena_.activations();
    const std::span<const float> betas = td_.betas();
    float tau = 0.0f;
    for (std::size_t id = 0; id < act.size(); ++id) {
        if (act[id] != 0) {
            tau += std::exp(betas[id]);
        }
    }

    const float a0 = params_.alpha;  // new features get step size alpha_init

    // Randomized generation parameters, drawn per-neuron or once per generation
    // step depending on kSampleGenerationParamsPerNeuron (compile-time).
    auto pick_fraction = [&]() -> float {
        if (params_.pattern_fractions.empty()) {
            return 0.5f;
        }
        std::uniform_int_distribution<std::size_t> d(0, params_.pattern_fractions.size() - 1);
        return params_.pattern_fractions[d(rng_)];
    };
    auto pick_delay = [&]() -> std::uint16_t {
        return std::uniform_int_distribution<std::uint16_t>(
            params_.memory_delay_min, params_.memory_delay_max)(rng_);
    };
    auto pick_window = [&]() -> std::uint16_t {
        return std::uniform_int_distribution<std::uint16_t>(
            params_.memory_window_min, params_.memory_window_max)(rng_);
    };

    [[maybe_unused]] float gen_fraction = 0.5f;
    [[maybe_unused]] std::uint16_t gen_delay = 0;
    [[maybe_unused]] std::uint16_t gen_window = 0;
    if constexpr (!kSampleGenerationParamsPerNeuron) {
        gen_fraction = pick_fraction();
        gen_delay = pick_delay();
        gen_window = pick_window();
    }

    // Interleave pattern/memory generation (random type per slot, like the
    // reference) so neither starves the other under a tight tau budget; each
    // type keeps its own cap. Stop on capacity, the tau budget, or no candidate.
    std::uint32_t made_pattern = 0;
    std::uint32_t made_memory = 0;
    std::bernoulli_distribution coin(0.5);
    while (true) {
        if (arena_.size() >= params_.capacity || tau + a0 > params_.eta) {
            break;
        }
        const std::size_t pool = pool_scratch_.size();
        const auto pat_max_n = static_cast<std::uint16_t>(
            std::min<std::size_t>(pool, params_.pattern_max_connections));
        const bool pattern_ok = made_pattern < params_.k_pattern &&
                                pat_max_n >= params_.pattern_min_connections;
        const bool memory_ok = made_memory < params_.k_memory;  // pool is non-empty
        if (!pattern_ok && !memory_ok) {
            break;
        }
        const bool make_memory =
            (pattern_ok && memory_ok) ? coin(rng_) : memory_ok;

        if (make_memory) {
            std::uniform_int_distribution<std::size_t> pick(0, pool - 1);
            const std::uint32_t src = pool_scratch_[pick(rng_)];
            const std::uint16_t delay =
                kSampleGenerationParamsPerNeuron ? pick_delay() : gen_delay;
            const std::uint16_t window =
                kSampleGenerationParamsPerNeuron ? pick_window() : gen_window;
            const std::uint32_t id =
                arena_.addMemory(src, delay, window, Status::Idle);
            if (arena_.triggerGenerated(id)) {
                tau += a0;
            }
            ++made_memory;
        } else {
            std::uniform_int_distribution<std::uint16_t> dist_n(
                params_.pattern_min_connections, pat_max_n);
            const std::uint16_t n = dist_n(rng_);
            // Partial Fisher-Yates: first n entries are n distinct random members.
            members_scratch_.clear();
            for (std::uint16_t i = 0; i < n; ++i) {
                std::uniform_int_distribution<std::size_t> pick(i, pool - 1);
                std::swap(pool_scratch_[i], pool_scratch_[pick(rng_)]);
                members_scratch_.push_back(pool_scratch_[i]);
            }
            const float frac =
                kSampleGenerationParamsPerNeuron ? pick_fraction() : gen_fraction;
            const std::uint32_t id =
                arena_.addPattern(members_scratch_, frac, Status::Idle);
            if (arena_.triggerGenerated(id)) {
                tau += a0;
            }
            ++made_pattern;
        }
    }
}

void ImprintingLearner::applyTenurePolicy() {
    const std::span<const float> w = arena_.weights();

    // Promotion thresholds (rising) and demotion thresholds (lower, by
    // demotion_factor) form a hysteresis band, so units don't oscillate.
    // Everything is on |weight| (magnitude): a strongly negative weight matters
    // as much as a positive one.
    const float promote_tenure = params_.tenure_threshold;
    const float promote_track = params_.tenure_track_threshold;
    const float demote_tenure = promote_tenure * params_.demotion_factor;
    const float demote_track = promote_track * params_.demotion_factor;

    for (std::uint32_t id = 0; id < w.size(); ++id) {
        FeatureRef f = arena_[id];
        if (f.getFeatureType() == FeatureType::Observation) {
            continue;  // observations are always tenured
        }
        const float m = std::fabs(w[id]);
        const int current = statusRank(f.getStatus());

        // Highest rank justified by the (rising) promotion thresholds, and the
        // lowest rank still held by the (lower) demotion thresholds.
        const int promote = (m > promote_tenure) ? 2 : (m > promote_track) ? 1 : 0;
        const int demote_floor = (m >= demote_tenure) ? 2 : (m >= demote_track) ? 1 : 0;

        int target = current;
        if (promote > current) {
            target = promote;            // promote up (possibly multiple ranks)
        } else if (demote_floor < current) {
            target = demote_floor;       // demote down past the hysteresis band
        }
        if (target != current) {
            f.setStatus(target == 2   ? Status::Tenure
                        : target == 1 ? Status::TenureTrack
                                      : Status::Idle);
        }
    }
}

}  // namespace il
