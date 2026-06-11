#include "imprinting/feature.hpp"
#include "imprinting/imprinting_learner.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace {

using il::FeatureType;
using il::HyperParams;
using il::ImprintingLearner;
using il::Status;

TEST(ImprintingLearner, LearnsAndMirrorsWeights) {
    HyperParams hp;
    hp.capacity = 16;
    hp.gamma = 0.9f;
    ImprintingLearner learner(hp);
    learner.addObservations(3);

    const std::array<std::uint8_t, 3> input{1, 0, 1};
    float prediction = 0.0f;
    for (int t = 0; t < 100; ++t) {
        prediction = learner.step(input, /*reward=*/1.0f);
        ASSERT_TRUE(std::isfinite(prediction));
    }

    // The active features' mirrored weights should have moved off zero, so the
    // arena's own prediction matches the running value.
    const auto w = learner.arena().weights();
    EXPECT_GT(w[0], 0.0f);
    EXPECT_GT(w[2], 0.0f);
    EXPECT_FLOAT_EQ(learner.arena().predict(), w[0] + w[2]);
    EXPECT_GT(prediction, 0.0f);
}

TEST(ImprintingLearner, TenurePolicyPromotesByWeightMagnitude) {
    HyperParams hp;
    hp.capacity = 8;
    hp.gamma = 0.9f;
    hp.tenure_track_threshold = 0.01f;
    hp.tenure_threshold = 0.02f;
    ImprintingLearner learner(hp);
    learner.addObservations(2);                          // ids 0,1 (always Tenure)

    // A generated-style pattern starts Idle and must earn tenure by weight.
    const std::array<std::uint32_t, 2> members{0u, 1u};
    const auto p = learner.addPattern(members, 0.5f);    // id 2, Idle
    EXPECT_EQ(learner.arena()[p].getStatus(), Status::Idle);

    // Negative reward drives the weight negative; only magnitude-based promotion
    // (not signed) can ever promote it.
    const std::array<std::uint8_t, 2> on{1, 1};
    Status seen = Status::Idle;
    for (int t = 0; t < 500; ++t) {
        learner.step(on, -1.0f);
        seen = learner.arena()[p].getStatus();
        if (seen == Status::Tenure) {
            break;
        }
    }
    EXPECT_EQ(seen, Status::Tenure);
    EXPECT_LT(learner.arena().weights()[p], 0.0f);  // weight is negative
    EXPECT_GT(std::fabs(learner.arena().weights()[p]), hp.tenure_threshold);
}

TEST(ImprintingLearner, TenurePolicyDemotesWithHysteresis) {
    HyperParams hp;
    hp.capacity = 8;
    hp.gamma = 0.9f;
    hp.alpha = 0.05f;
    hp.tenure_track_threshold = 0.01f;
    hp.tenure_threshold = 0.02f;
    hp.demotion_factor = 0.5f;  // demote below 0.01 (Tenure) / 0.005 (TenureTrack)
    ImprintingLearner learner(hp);
    learner.addObservations(2);
    const std::array<std::uint32_t, 2> members{0u, 1u};
    const auto p = learner.addPattern(members, 0.5f);  // Idle; active while both obs fire
    const std::array<std::uint8_t, 2> on{1, 1};

    // Phase 1: positive reward drives the weight up until the pattern is tenured.
    bool tenured = false;
    for (int t = 0; t < 1000 && !tenured; ++t) {
        learner.step(on, 1.0f);
        tenured = learner.arena()[p].getStatus() == Status::Tenure;
    }
    ASSERT_TRUE(tenured);

    // Phase 2: zero reward -> the value (and weight) decay toward 0 -> demotion.
    Status status = Status::Tenure;
    for (int t = 0; t < 5000; ++t) {
        learner.step(on, 0.0f);
        status = learner.arena()[p].getStatus();
        if (status != Status::Tenure) {
            break;
        }
    }
    EXPECT_NE(status, Status::Tenure);  // hysteresis crossed downward -> demoted
    EXPECT_LT(std::fabs(learner.arena().weights()[p]),
              hp.tenure_threshold * hp.demotion_factor);
}

// A delayed memory feature is idle with z==0 throughout its delay; removal must
// not cull it before it has a chance to fire.
TEST(ImprintingLearner, MemoryNotRemovedWhileArmed) {
    HyperParams hp;
    hp.capacity = 8;
    hp.gamma = 0.9f;
    hp.epsilon_z = 0.01f;  // removal on; generation off (k_pattern/k_memory == 0)
    ImprintingLearner learner(hp);
    const auto src = learner.addObservation(0);                 // id 0, tenured
    const auto mem = learner.addMemory(src, /*delay=*/5, /*window=*/2);  // id 1, Idle
    // Born armed, as generation does (triggerGenerated). With the one-step lag on
    // its source, a hand-added memory would otherwise sit idle and unarmed on its
    // first step and be culled before its delay even begins.
    learner.arena().triggerGenerated(mem);
    ASSERT_EQ(learner.arena().size(), 2u);

    const std::array<std::uint8_t, 1> on{1};
    bool ever_active = false;
    for (int t = 0; t < 12; ++t) {
        learner.step(on, 0.0f);
        ASSERT_EQ(learner.arena().size(), 2u) << "memory culled at step " << t;
        ever_active = ever_active || learner.arena()[mem].getActivation();
    }
    EXPECT_TRUE(ever_active);  // it survived the delay and fired
}

TEST(ImprintingLearner, RejectsGrowthBeyondCapacity) {
    HyperParams hp;
    hp.capacity = 2;
    ImprintingLearner learner(hp);
    learner.addObservations(2);
    EXPECT_THROW(learner.addObservation(2), std::length_error);
}

TEST(ImprintingLearner, GenerationDisabledByDefault) {
    ImprintingLearner learner(HyperParams{});  // k_pattern == k_memory == 0
    learner.addObservations(3);
    const std::array<std::uint8_t, 3> on{1, 1, 1};
    for (int t = 0; t < 10; ++t) {
        learner.step(on, 1.0f);
    }
    EXPECT_EQ(learner.arena().size(), 3u);  // no features generated
}

TEST(ImprintingLearner, NoGenerationWithoutActiveTenured) {
    HyperParams hp;
    hp.capacity = 32;
    hp.k_pattern = 2;
    hp.k_memory = 2;
    ImprintingLearner learner(hp);
    learner.addObservations(3);
    const std::array<std::uint8_t, 3> off{0, 0, 0};  // observations never fire
    for (int t = 0; t < 10; ++t) {
        learner.step(off, 1.0f);
    }
    EXPECT_EQ(learner.arena().size(), 3u);  // gate (no active tenured) blocks generation
}

TEST(ImprintingLearner, GeneratesPatternsAndMemoryUnderTauBudget) {
    HyperParams hp;
    hp.capacity = 64;
    hp.gamma = 0.9f;
    hp.alpha = 0.01f;
    hp.eta = 0.1f;
    hp.k_pattern = 1;
    hp.k_memory = 1;
    hp.pattern_fractions = {0.5f};
    hp.pattern_min_connections = 2;
    hp.pattern_max_connections = 3;
    hp.memory_window_max = 2;  // memory window sampled in [1, 2]
    hp.rng_seed = 123;
    ImprintingLearner learner(hp);
    learner.addObservations(3);  // ids 0,1,2 (always Tenure)

    const std::array<std::uint8_t, 3> on{1, 1, 1};
    learner.step(on, 1.0f);  // first step: no phi_{t-1}, so no generation yet
    EXPECT_EQ(learner.arena().size(), 3u);

    for (int t = 0; t < 20; ++t) {
        learner.step(on, 1.0f);
    }

    const std::size_t after = learner.arena().size();
    EXPECT_GT(after, 3u);             // features were generated
    EXPECT_LE(after, hp.capacity);    // tau budget + capacity keep it bounded

    bool saw_pattern = false;
    bool saw_memory = false;
    for (std::uint32_t id = 3; id < after; ++id) {
        const auto type = learner.arena()[id].getFeatureType();
        EXPECT_NE(type, FeatureType::Observation);  // generated features are not observations
        if (type == FeatureType::Pattern) saw_pattern = true;
        if (type == FeatureType::Memory) saw_memory = true;
    }
    EXPECT_TRUE(saw_pattern);
    EXPECT_TRUE(saw_memory);
}

// A manually-added pattern that never proves useful (idle, inactive so its
// eligibility trace decays) gets removed once z falls below e^beta * epsilon_z.
TEST(ImprintingLearner, RemovesIdleDecayedFeature) {
    HyperParams hp;
    hp.capacity = 8;
    hp.gamma = 0.9f;
    hp.alpha = 0.05f;
    hp.epsilon_z = 0.01f;  // removal enabled
    ImprintingLearner learner(hp);
    learner.addObservations(2);                          // ids 0,1 (Tenure, never removed)
    const std::array<std::uint32_t, 2> members{0u, 1u};
    learner.addPattern(members, 1.0f);                   // id 2, Idle, needs BOTH obs active
    ASSERT_EQ(learner.arena().size(), 3u);

    // Only obs0 fires -> the pattern is inactive after its creation step, so its
    // trace decays; reward 0 keeps its weight ~0 (Idle).
    const std::array<std::uint8_t, 2> in{1, 0};
    for (int t = 0; t < 100; ++t) {
        learner.step(in, 0.0f);
    }
    EXPECT_EQ(learner.arena().size(), 2u);  // the pattern was removed
    EXPECT_EQ(learner.arena()[0].getFeatureType(), FeatureType::Observation);
    EXPECT_EQ(learner.arena()[1].getFeatureType(), FeatureType::Observation);
}

TEST(ImprintingLearner, RemovalDisabledByDefault) {
    HyperParams hp;  // epsilon_z == 0
    hp.gamma = 0.9f;
    ImprintingLearner learner(hp);
    learner.addObservations(2);
    const std::array<std::uint32_t, 2> members{0u, 1u};
    learner.addPattern(members, 1.0f);  // idle, decays, but removal is off
    const std::array<std::uint8_t, 2> in{1, 0};
    for (int t = 0; t < 100; ++t) {
        learner.step(in, 0.0f);
    }
    EXPECT_EQ(learner.arena().size(), 3u);  // nothing removed
}

TEST(ImprintingLearner, NeverRemovesObservations) {
    HyperParams hp;
    hp.epsilon_z = 0.01f;
    ImprintingLearner learner(hp);
    learner.addObservations(3);                  // always Tenure
    const std::array<std::uint8_t, 3> off{0, 0, 0};  // inactive -> z stays 0
    for (int t = 0; t < 50; ++t) {
        learner.step(off, 0.0f);
    }
    EXPECT_EQ(learner.arena().size(), 3u);  // observations are never removed
}

// A feature still referenced by a survivor is deferred (v1 rule) rather than
// removed, even when it would otherwise qualify.
TEST(ImprintingLearner, DefersRemovalOfReferencedFeature) {
    HyperParams hp;
    hp.capacity = 8;
    hp.gamma = 0.9f;
    hp.alpha = 0.05f;
    hp.epsilon_z = 0.01f;
    hp.demotion_factor = 0.0f;  // disable demotion so the referencing memory stays Tenure
    ImprintingLearner learner(hp);
    learner.addObservations(2);                          // ids 0,1
    const std::array<std::uint32_t, 2> members{0u, 1u};
    learner.addPattern(members, 1.0f);                   // id 2: Idle, decays
    learner.addMemory(2, 1, 1, Status::Tenure);          // id 3: Tenure, source = pattern 2

    const std::array<std::uint8_t, 2> in{1, 0};
    for (int t = 0; t < 100; ++t) {
        learner.step(in, 0.0f);
    }
    // Pattern 2 would be removable, but memory 3 references it -> deferred.
    EXPECT_EQ(learner.arena().size(), 4u);
}

// Reproduces the imprinting figure: a pattern feature generated at t_1 that
// imprints on the tenured features active at t_0 (phi2, phi4, phi7) fires
// immediately on its creation step (t_1) -- not the step before (t_0) -- and
// thereafter reads its members with a one-step lag, so it fires the step AFTER
// its imprinted configuration recurs. The config {phi2,phi4,phi7} is fully
// present again at t_3, so the feature fires at t_4. Expected timeline starting
// at t_1: { true, false, false, true }.
TEST(ImprintingLearner, ImprintedPatternFiresOnCreationStep) {
    HyperParams hp;
    hp.capacity = 16;
    hp.gamma = 0.9f;
    hp.k_pattern = 0;   // drive one pattern by hand to isolate creation-step timing
    hp.k_memory = 0;
    hp.epsilon_z = 0.0f;  // no removal
    ImprintingLearner learner(hp);

    // 4 observations + 3 "extra tenured" stand-ins, all observations: ids 0..6 =
    // phi1..phi7.
    learner.addObservations(7);

    // Input rows phi1..phi7 over t_{-1}..t_4.
    const std::array<std::uint8_t, 7> tm1{1, 1, 0, 0, 1, 0, 0};  // t_{-1}
    const std::array<std::uint8_t, 7> t0{0, 1, 0, 1, 0, 0, 1};   // t_0: phi2,phi4,phi7
    const std::array<std::uint8_t, 7> t1{0, 1, 1, 0, 0, 1, 0};   // t_1
    const std::array<std::uint8_t, 7> t2{0, 0, 0, 1, 0, 0, 1};   // t_2
    const std::array<std::uint8_t, 7> t3{0, 1, 0, 1, 0, 1, 1};   // t_3
    const std::array<std::uint8_t, 7> t4{1, 0, 1, 0, 0, 0, 0};   // t_4

    // Observe t_{-1} and t_0; the imprint snapshots t_0's active tenured set.
    learner.step(tm1, 0.0f);
    learner.step(t0, 0.0f);

    // Generate the pattern at the t_0 -> t_1 boundary, imprinting on the features
    // active at t_0: phi2 (id 1), phi4 (id 3), phi7 (id 6). 90% of 3 => all 3.
    const std::array<std::uint32_t, 3> members{1u, 3u, 6u};
    const auto p = learner.addPattern(members, 0.9f);  // id 7, Idle, fresh

    // Just created from t_0, before its first forward pass: it must not have fired
    // at t_0.
    EXPECT_FALSE(learner.arena()[p].getActivation());

    // Expected activation timeline for the new feature, starting at its creation
    // step t_1: { true, false, false, true }.
    learner.step(t1, 0.0f);
    EXPECT_TRUE(learner.arena()[p].getActivation()) << "t_1 (creation step)";
    learner.step(t2, 0.0f);
    EXPECT_FALSE(learner.arena()[p].getActivation()) << "t_2";
    learner.step(t3, 0.0f);
    EXPECT_FALSE(learner.arena()[p].getActivation()) << "t_3";
    learner.step(t4, 0.0f);
    EXPECT_TRUE(learner.arena()[p].getActivation()) << "t_4";
}

// Reproduces the memory-generation figure (thesis Fig 9.2): three memory features
// created from a tenured feature that fires at t_0 and again at t_8. The source is
// read with a one-step lag (like patterns), so each is triggered the step after the
// source fires; with delay k1 / window k2 it is then active over
// [trigger+k1, trigger+k1+k2-1], where trigger = t_1 (first) and t_9 (second):
//   phi[m]: k1=2, k2=2 -> t_3,t_4   and t_11,t_12
//   phi[n]: k1=1, k2=3 -> t_2..t_4  and t_10..t_12
//   phi[o]: k1=3, k2=1 -> t_4       and t_12
// k1+k2 = 4 for all three, so every active window ends together at t_4 / t_12.
TEST(ImprintingLearner, ImprintedMemoriesFireAfterDelayedTrigger) {
    HyperParams hp;
    hp.capacity = 16;
    hp.gamma = 0.9f;
    hp.k_pattern = 0;     // drive three memory features by hand
    hp.k_memory = 0;
    hp.epsilon_z = 0.0f;  // no removal
    ImprintingLearner learner(hp);

    learner.addObservation(0);                  // id 0: tenured source phi[1]
    const auto m = learner.addMemory(0, 2, 2);  // id 1: phi[m]
    const auto n = learner.addMemory(0, 1, 3);  // id 2: phi[n]
    const auto o = learner.addMemory(0, 3, 1);  // id 3: phi[o]

    // phi[1] input over t_{-1}..t_13: fires at t_0 and t_8.
    const std::array<std::uint8_t, 15> in_seq{
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    //  t-1 t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 ...        t13

    // Expected activations over t_1..t_13 (the figure's sequences "start at t_1").
    const std::array<std::uint8_t, 13> exp_m{0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0};
    const std::array<std::uint8_t, 13> exp_n{0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0};
    const std::array<std::uint8_t, 13> exp_o{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};

    std::array<std::uint8_t, 1> in{0};
    for (std::size_t i = 0; i < in_seq.size(); ++i) {
        in[0] = in_seq[i];
        learner.step(in, 0.0f);
        if (i < 2) {
            continue;  // t_{-1}, t_0: before the memory sequences begin
        }
        const std::size_t k = i - 2;  // i == 2 -> t_1
        EXPECT_EQ(learner.arena()[m].getActivation(), exp_m[k] != 0) << "phi[m] t_" << (k + 1);
        EXPECT_EQ(learner.arena()[n].getActivation(), exp_n[k] != 0) << "phi[n] t_" << (k + 1);
        EXPECT_EQ(learner.arena()[o].getActivation(), exp_o[k] != 0) << "phi[o] t_" << (k + 1);
    }
}

}  // namespace
