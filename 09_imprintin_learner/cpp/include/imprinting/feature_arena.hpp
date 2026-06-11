#pragma once

#include "imprinting/feature.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace il {

// Upper bound on the tenured features a single pattern feature may connect to.
inline constexpr std::size_t kMaxPatternConnections = 256;

// Sentinel for an empty membership slot, and for an unattached memory source.
inline constexpr std::int16_t kNoFeature = -1;

struct ArenaConfig {
    float tenure_threshold = 1.0f;
    float tenure_track_threshold = 0.5f;
    float default_step_size = 1e-2f;
    std::size_t trace_dim = 1;  // eligibility-trace width per feature
};

class FeatureRef;

// Struct-of-arrays storage for an agent's entire feature population.
class FeatureArena {
public:
    explicit FeatureArena(ArenaConfig config = {});

    // --- population growth (each returns the new feature's global id) -------

    // Observation feature: active when input[input_index] is set.
    std::uint32_t addObservation(std::uint32_t input_index,
                                 Status status = Status::Tenure);
    void addObservations(std::size_t input_dim, Status status = Status::Tenure);

    // Pattern feature over n = members.size() tenured features
    std::uint32_t addPattern(std::span<const std::uint32_t> members,
                             float fraction,
                             Status status = Status::Idle);

    // Memory feature: active for `window` steps starting `delay` steps after
    // its `source` (a tenured feature) activates.
    std::uint32_t addMemory(std::uint32_t source, std::uint16_t delay,
                            std::uint16_t window,
                            Status status = Status::Idle);

    // Advance one timestep: recompute every feature's activation from `input`
    // (a binary vector) and the feature dynamics, then return the GVF
    // prediction.
    float step(std::span<const std::uint8_t> input);

    // GVF prediction = sum over features of weight * activation.
    float predict() const;

    // Set the activation of a just-generated feature by "immediately triggering
    // it from its input"
    bool triggerGenerated(std::uint32_t id);

    std::size_t size() const noexcept { return type_.size(); }
    FeatureRef operator[](std::uint32_t id) noexcept;

    // Reserve column capacity
    void reserve(std::size_t capacity);

    // Remove the feature at `id` by moving the last feature into its slot
    // (swap-pop) and shrinking by one; references to the moved feature are
    // remapped. The caller must swap-pop SwiftTD's per-feature state with the
    // same (id, old-last) indices. Caller must ensure `id` has no dependents.
    void swapPopRemove(std::uint32_t id);

    // True if any live pattern/memory feature references `id` (as a member or
    // source). Used to defer removal of depended-upon features.
    bool isReferenced(std::uint32_t id) const;

    // True if `id` is a memory feature still counting down its delay or active
    // window (it has not finished firing yet), so removal must leave it alone —
    // otherwise a delayed memory unit is culled before it can ever activate.
    bool isMemoryArmed(std::uint32_t id) const;

    // Bulk views, e.g. to drive the SwiftTD weight update from the activation
    // vector and write the updated weights back.
    std::span<const std::uint8_t> activations() const noexcept { return activation_; }
    std::span<float> weights() noexcept { return weight_; }
    std::span<const float> weights() const noexcept { return weight_; }

private:
    friend class FeatureRef;

    std::uint32_t allocateFeature(FeatureType type, Status status);
    // Advance a memory feature's delay/window state machine one tick and return
    // whether it is active. Shared by step() and triggerGenerated().
    bool advanceMemory(std::uint32_t slot, bool src_active);
    // Repoint every member/source reference equal to `from` to `to`.
    void remapReferences(std::uint32_t from, std::uint32_t to);

    // --- type-specific compact stores, indexed by type_slot_ ---------------
    struct ObservationStore {
        std::vector<std::uint32_t> input_index;
    };
    struct PatternStore {
        std::vector<std::int16_t> members;       // flat: slot*kMaxPatternConnections + j
        std::vector<std::uint16_t> member_count;  // n
        std::vector<std::uint16_t> threshold;     // k = ceil(fraction * n)
        std::vector<std::uint8_t> fresh;          // forced active on creation step
    };
    struct MemoryStore {
        std::vector<std::int16_t> source;
        std::vector<std::uint16_t> delay;
        std::vector<std::uint16_t> window;
        std::vector<std::int32_t> delay_counter;
        std::vector<std::int32_t> active_counter;
    };

    // --- common columns (size == feature count) ----------------------------
    std::vector<float> weight_;
    std::vector<float> step_size_;
    std::vector<float> trace_;  // flat, config_.trace_dim floats per feature
    std::vector<Status> status_;
    std::vector<FeatureType> type_;
    std::vector<std::uint8_t> activation_;
    std::vector<std::uint8_t> prev_activation_;  // phi_{t-1}; patterns read members lagged
    std::vector<std::uint32_t> type_slot_;

    ObservationStore observations_;
    PatternStore patterns_;
    MemoryStore memory_;

    ArenaConfig config_;
};

// A lightweight handle exposing one feature (by id) through the il::Feature
// interface while the data stays in the arena's columns.
class FeatureRef {
public:
    FeatureRef(FeatureArena& arena, std::uint32_t id) noexcept
        : arena_(&arena), id_(id) {}

    std::uint32_t id() const noexcept { return id_; }

    float getWeight() const { return arena_->weight_[id_]; }
    std::span<float> getEligibilityTrace() const {
        const std::size_t w = arena_->config_.trace_dim;
        return std::span<float>(arena_->trace_.data() + id_ * w, w);
    }
    float getStepSize() const { return arena_->step_size_[id_]; }
    Status getStatus() const { return arena_->status_[id_]; }
    float getTenureThreshold() const { return arena_->config_.tenure_threshold; }
    float getTenureTrackThreshold() const { return arena_->config_.tenure_track_threshold; }
    FeatureType getFeatureType() const { return arena_->type_[id_]; }
    bool getActivation() const { return arena_->activation_[id_] != 0; }

    void setWeight(float w) const { arena_->weight_[id_] = w; }
    void setStatus(Status s) const { arena_->status_[id_] = s; }
    void setActivation(bool a) const {
        arena_->activation_[id_] = static_cast<std::uint8_t>(a ? 1 : 0);
    }

private:
    FeatureArena* arena_;
    std::uint32_t id_;
};

static_assert(Feature<FeatureRef>,
              "FeatureRef must model il::Feature so the SOA arena stays type-checked");

inline FeatureRef FeatureArena::operator[](std::uint32_t id) noexcept {
    return FeatureRef(*this, id);
}

}  // namespace il
