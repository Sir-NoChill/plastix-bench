#pragma once

#include <concepts>
#include <span>

namespace il {

// Lifecycle state of a feature within the imprinting learner.
enum class Status {
    Tenure,       // promoted; protected from recycling
    TenureTrack,  // under evaluation for promotion
    Idle,         // inactive; candidate for recycling
};

// Role a feature plays in the network.
enum class FeatureType {
    Observation,
    Pattern,
    Memory,
    Output,
};

// A Feature is a learnable unit
template <typename T>
concept Feature = requires(T& f) {
    { f.getWeight() } -> std::same_as<float>;
    { f.getEligibilityTrace() } -> std::same_as<std::span<float>>;
    { f.getStepSize() } -> std::same_as<float>;
    { f.getStatus() } -> std::same_as<Status>;
    { f.getTenureThreshold() } -> std::same_as<float>;
    { f.getTenureTrackThreshold() } -> std::same_as<float>;
    { f.getFeatureType() } -> std::same_as<FeatureType>;
    // Whether the feature fires this timestep; true == 1, false == 0
    { f.getActivation() } -> std::same_as<bool>;
};
}  // namespace il
