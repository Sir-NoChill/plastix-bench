#include "imprinting/feature.hpp"
#include "imprinting/feature_arena.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

using il::FeatureArena;
using il::FeatureType;
using il::Status;

// The SOA arena stays type-checked: its element view models il::Feature.
static_assert(il::Feature<il::FeatureRef>);

TEST(FeatureArena, RefExposesFeatureInterface) {
    FeatureArena arena;
    const auto id = arena.addObservation(0, Status::Tenure);
    auto f = arena[id];
    EXPECT_EQ(f.getFeatureType(), FeatureType::Observation);
    EXPECT_EQ(f.getStatus(), Status::Tenure);
    EXPECT_EQ(f.getEligibilityTrace().size(), 1u);
    EXPECT_FALSE(f.getActivation());
}

TEST(FeatureArena, ObservationActivationFollowsInput) {
    FeatureArena arena;
    arena.addObservations(3);  // ids 0,1,2 -> input dims 0,1,2
    const std::array<std::uint8_t, 3> in{1, 0, 1};
    arena.step(in);
    EXPECT_TRUE(arena[0].getActivation());
    EXPECT_FALSE(arena[1].getActivation());
    EXPECT_TRUE(arena[2].getActivation());
}

TEST(FeatureArena, PredictSumsWeightTimesActivation) {
    FeatureArena arena;
    arena.addObservations(3);
    arena[0].setWeight(2.0f);
    arena[1].setWeight(-1.0f);
    arena[2].setWeight(0.5f);
    const std::array<std::uint8_t, 3> in{1, 0, 1};  // features 0 and 2 active
    EXPECT_FLOAT_EQ(arena.step(in), 2.5f);          // 2.0 + 0.5
    EXPECT_FLOAT_EQ(arena.predict(), 2.5f);
}

TEST(FeatureArena, PatternFreshThenLaggedFractionRule) {
    FeatureArena arena;
    arena.addObservations(2);                            // ids 0,1
    const std::array<std::uint32_t, 2> members{0u, 1u};
    const auto p = arena.addPattern(members, 0.5f);      // n=2, k=ceil(1.0)=1
    EXPECT_EQ(arena[p].getFeatureType(), FeatureType::Pattern);

    const std::array<std::uint8_t, 2> none{0, 0};
    const std::array<std::uint8_t, 2> one{1, 0};

    arena.step(none);
    EXPECT_TRUE(arena[p].getActivation());   // forced active on creation step

    arena.step(none);
    EXPECT_FALSE(arena[p].getActivation());  // prev step had 0 active members < k

    // The member turns on now, but a pattern reads its members one step late, so
    // it does not fire until the following step.
    arena.step(one);
    EXPECT_FALSE(arena[p].getActivation());  // lag: prev step still had 0 active

    arena.step(one);
    EXPECT_TRUE(arena[p].getActivation());   // prev step had 1 active member >= k
}

TEST(FeatureArena, PatternRequiresMajorityOneStepLate) {
    FeatureArena arena;
    arena.addObservations(3);                                 // ids 0,1,2
    const std::array<std::uint32_t, 3> members{0u, 1u, 2u};
    const auto p = arena.addPattern(members, 0.6f);           // n=3, k=ceil(1.8)=2

    const std::array<std::uint8_t, 3> none{0, 0, 0};
    arena.step(none);                                         // consume creation step

    arena.step(std::array<std::uint8_t, 3>{1, 0, 0});         // prev had 0 active < 2
    EXPECT_FALSE(arena[p].getActivation());
    arena.step(std::array<std::uint8_t, 3>{1, 1, 0});         // lag: prev had 1 active < 2
    EXPECT_FALSE(arena[p].getActivation());
    arena.step(std::array<std::uint8_t, 3>{1, 1, 0});         // prev had 2 active >= 2
    EXPECT_TRUE(arena[p].getActivation());
}

TEST(FeatureArena, MemoryDelayThenWindowThenReset) {
    FeatureArena arena;
    const auto src = arena.addObservation(0, Status::Tenure);          // id 0
    const auto mem = arena.addMemory(src, /*delay=*/2, /*window=*/3);  // id 1

    const std::array<std::uint8_t, 1> on{1};
    const std::array<std::uint8_t, 1> off{0};

    std::vector<bool> seen;
    arena.step(on);                                  // step 0: source fires
    seen.push_back(arena[mem].getActivation());
    for (int t = 1; t < 7; ++t) {
        arena.step(off);
        seen.push_back(arena[mem].getActivation());
    }
    // The source is read with a one-step lag, so the firing at step 0 is seen at
    // step 1 (which counts as the first delay step). delay 2 -> inactive through
    // step 2; window 3 -> active steps 3,4,5; then reset at step 6.
    const std::vector<bool> expected{false, false, false, true, true, true, false};
    EXPECT_EQ(seen, expected);
}

TEST(FeatureArena, AddPatternRejectsBadMemberCount) {
    FeatureArena arena;
    arena.addObservations(1);
    const std::array<std::uint32_t, 1> one{0u};
    EXPECT_THROW(arena.addPattern(one, 0.5f), std::invalid_argument);  // n < 2
}

}  // namespace
