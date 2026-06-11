#include "imprinting/feature_arena.hpp"

#include <cmath>
#include <stdexcept>

namespace il {

FeatureArena::FeatureArena(ArenaConfig config) : config_(config) {
    if (config_.trace_dim == 0) {
        throw std::invalid_argument("ArenaConfig.trace_dim must be >= 1");
    }
}

void FeatureArena::reserve(std::size_t capacity) {
    weight_.reserve(capacity);
    step_size_.reserve(capacity);
    trace_.reserve(capacity * config_.trace_dim);
    status_.reserve(capacity);
    type_.reserve(capacity);
    activation_.reserve(capacity);
    prev_activation_.reserve(capacity);
    type_slot_.reserve(capacity);
}

void FeatureArena::remapReferences(std::uint32_t from, std::uint32_t to) {
    const auto from16 = static_cast<std::int16_t>(from);
    const auto to16 = static_cast<std::int16_t>(to);
    for (std::int16_t& m : patterns_.members) {
        if (m == from16) {
            m = to16;
        }
    }
    for (std::int16_t& s : memory_.source) {
        if (s == from16) {
            s = to16;
        }
    }
}

void FeatureArena::swapPopRemove(std::uint32_t id) {
    const auto last = static_cast<std::uint32_t>(size() - 1);
    if (id != last) {
        weight_[id] = weight_[last];
        step_size_[id] = step_size_[last];
        for (std::size_t d = 0; d < config_.trace_dim; ++d) {
            trace_[id * config_.trace_dim + d] = trace_[last * config_.trace_dim + d];
        }
        status_[id] = status_[last];
        type_[id] = type_[last];
        activation_[id] = activation_[last];
        type_slot_[id] = type_slot_[last];
        remapReferences(last, id);  // the moved feature now lives at `id`
    }
    weight_.pop_back();
    step_size_.pop_back();
    trace_.resize(trace_.size() - config_.trace_dim);
    status_.pop_back();
    type_.pop_back();
    activation_.pop_back();
    type_slot_.pop_back();
}

bool FeatureArena::isReferenced(std::uint32_t id) const {
    const auto id16 = static_cast<std::int16_t>(id);
    for (std::uint32_t f = 0; f < size(); ++f) {
        const std::uint32_t slot = type_slot_[f];
        if (type_[f] == FeatureType::Pattern) {
            const std::size_t base = static_cast<std::size_t>(slot) * kMaxPatternConnections;
            const std::uint16_t count = patterns_.member_count[slot];
            for (std::uint16_t j = 0; j < count; ++j) {
                if (patterns_.members[base + j] == id16) {
                    return true;
                }
            }
        } else if (type_[f] == FeatureType::Memory) {
            if (memory_.source[slot] == id16) {
                return true;
            }
        }
    }
    return false;
}

bool FeatureArena::isMemoryArmed(std::uint32_t id) const {
    if (type_[id] != FeatureType::Memory) {
        return false;
    }
    const std::uint32_t slot = type_slot_[id];
    return memory_.delay_counter[slot] > 0 || memory_.active_counter[slot] > 0;
}

std::uint32_t FeatureArena::allocateFeature(FeatureType type, Status status) {
    const auto id = static_cast<std::uint32_t>(type_.size());
    weight_.push_back(0.0f);
    step_size_.push_back(config_.default_step_size);
    trace_.insert(trace_.end(), config_.trace_dim, 0.0f);
    status_.push_back(status);
    type_.push_back(type);
    activation_.push_back(0);
    type_slot_.push_back(0);
    return id;
}

std::uint32_t FeatureArena::addObservation(std::uint32_t input_index, Status status) {
    const std::uint32_t id = allocateFeature(FeatureType::Observation, status);
    type_slot_[id] = static_cast<std::uint32_t>(observations_.input_index.size());
    observations_.input_index.push_back(input_index);
    return id;
}

void FeatureArena::addObservations(std::size_t input_dim, Status status) {
    for (std::size_t i = 0; i < input_dim; ++i) {
        addObservation(static_cast<std::uint32_t>(i), status);
    }
}

std::uint32_t FeatureArena::addPattern(std::span<const std::uint32_t> members,
                                       float fraction, Status status) {
    if (members.size() < 2 || members.size() > kMaxPatternConnections) {
        throw std::invalid_argument(
            "addPattern: member count must be in [2, kMaxPatternConnections]");
    }
    const std::uint32_t id = allocateFeature(FeatureType::Pattern, status);
    const auto slot = static_cast<std::uint32_t>(patterns_.member_count.size());
    type_slot_[id] = slot;

    const std::size_t base = static_cast<std::size_t>(slot) * kMaxPatternConnections;
    patterns_.members.insert(patterns_.members.end(), kMaxPatternConnections, kNoFeature);
    for (std::size_t j = 0; j < members.size(); ++j) {
        if (members[j] > 0x7FFFu) {
            throw std::invalid_argument(
                "addPattern: member id exceeds the int16 range of the membership table");
        }
        patterns_.members[base + j] = static_cast<std::int16_t>(members[j]);
    }

    const auto n = static_cast<std::uint16_t>(members.size());
    auto k = static_cast<std::uint16_t>(std::ceil(fraction * static_cast<float>(n)));
    if (k < 1) {
        k = 1;
    }
    if (k > n) {
        k = n;
    }
    patterns_.member_count.push_back(n);
    patterns_.threshold.push_back(k);
    patterns_.fresh.push_back(1);
    return id;
}

std::uint32_t FeatureArena::addMemory(std::uint32_t source, std::uint16_t delay,
                                      std::uint16_t window, Status status) {
    if (source > 0x7FFFu) {
        throw std::invalid_argument("addMemory: source id exceeds the int16 range");
    }
    const std::uint32_t id = allocateFeature(FeatureType::Memory, status);
    const auto slot = static_cast<std::uint32_t>(memory_.source.size());
    type_slot_[id] = slot;
    memory_.source.push_back(static_cast<std::int16_t>(source));
    memory_.delay.push_back(delay);
    memory_.window.push_back(window);
    memory_.delay_counter.push_back(0);
    memory_.active_counter.push_back(0);
    return id;
}

float FeatureArena::step(std::span<const std::uint8_t> input) {
    const std::size_t n = size();
    // Freeze last step's activations: derived features fire the step *after* their
    // inputs are present, so patterns read their members and memory reads its
    // source with a one-step lag (phi_{t-1}) rather than the values being
    // computed this step. Observations still follow the current input directly.
    prev_activation_.assign(activation_.begin(), activation_.end());
    for (std::size_t id = 0; id < n; ++id) {
        const std::uint32_t slot = type_slot_[id];
        switch (type_[id]) {
            case FeatureType::Observation: {
                const std::uint32_t in = observations_.input_index[slot];
                const bool active = in < input.size() && input[in] != 0;
                activation_[id] = static_cast<std::uint8_t>(active ? 1 : 0);
                break;
            }
            case FeatureType::Pattern: {
                bool active;
                if (patterns_.fresh[slot] != 0) {
                    active = true;  // forced active on the creation step
                    patterns_.fresh[slot] = 0;
                } else {
                    const std::size_t base =
                        static_cast<std::size_t>(slot) * kMaxPatternConnections;
                    const std::uint16_t count_needed = patterns_.threshold[slot];
                    const std::uint16_t member_count = patterns_.member_count[slot];
                    std::uint16_t active_members = 0;
                    for (std::uint16_t j = 0; j < member_count; ++j) {
                        const std::int16_t m = patterns_.members[base + j];
                        if (m != kNoFeature &&
                            prev_activation_[static_cast<std::size_t>(m)] != 0) {
                            ++active_members;
                        }
                    }
                    active = active_members >= count_needed;
                }
                activation_[id] = static_cast<std::uint8_t>(active ? 1 : 0);
                break;
            }
            case FeatureType::Memory: {
                const std::int16_t src = memory_.source[slot];
                const bool src_active =
                    src != kNoFeature &&
                    prev_activation_[static_cast<std::size_t>(src)] != 0;
                activation_[id] =
                    static_cast<std::uint8_t>(advanceMemory(slot, src_active) ? 1 : 0);
                break;
            }
            case FeatureType::Output:
                activation_[id] = 0;
                break;
        }
    }
    return predict();
}

bool FeatureArena::advanceMemory(std::uint32_t slot, bool src_active) {
    std::int32_t dc = memory_.delay_counter[slot];
    std::int32_t ac = memory_.active_counter[slot];

    // Trigger only from the idle state; re-activation while busy is ignored.
    // The trigger step counts as the first delay step.
    if (dc == 0 && ac == 0 && src_active) {
        dc = static_cast<std::int32_t>(memory_.delay[slot]);
        if (dc == 0) {
            ac = static_cast<std::int32_t>(memory_.window[slot]);
        }
    }

    bool active = false;
    if (dc > 0) {
        --dc;
        if (dc == 0) {
            ac = static_cast<std::int32_t>(memory_.window[slot]);
        }
    } else if (ac > 0) {
        active = true;
        --ac;
    }

    memory_.delay_counter[slot] = dc;
    memory_.active_counter[slot] = ac;
    return active;
}

bool FeatureArena::triggerGenerated(std::uint32_t id) {
    const std::uint32_t slot = type_slot_[id];
    bool active = false;
    switch (type_[id]) {
        case FeatureType::Pattern:
            active = true;              // forced active on its creation step
            patterns_.fresh[slot] = 0;  // consume the flag, since it is active now
            break;
        case FeatureType::Memory:
            active = advanceMemory(slot, /*src_active=*/true);  // its source just fired
            break;
        case FeatureType::Observation:
        case FeatureType::Output:
            break;  // not generated; nothing to trigger
    }
    activation_[id] = static_cast<std::uint8_t>(active ? 1 : 0);
    return active;
}

float FeatureArena::predict() const {
    float v = 0.0f;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n; ++i) {
        v += weight_[i] * static_cast<float>(activation_[i]);
    }
    return v;
}

}  // namespace il
