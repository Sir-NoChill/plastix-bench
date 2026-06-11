#include <swifttd/SwiftTD.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// Verifies the vendored SwiftTD core links and runs: a constant-reward signal
// should drive predictions toward a finite, non-trivial value over time.
TEST(SwiftTD, NonSparseLearnsConstantReward) {
    SwiftTDNonSparse td(/*number_of_features=*/4,
                        /*lambda=*/0.95f, /*alpha=*/1e-2f, /*gamma=*/0.9f,
                        /*epsilon=*/1e-5f, /*eta=*/0.1f, /*decay=*/0.999f,
                        /*meta_step_size=*/1e-3f, /*eta_min=*/1e-10f);

    const std::vector<float> features{1.0f, 0.0f, 0.5f, 0.2f};
    float prediction = 0.0f;
    for (int t = 0; t < 200; ++t) {
        prediction = td.Step(features, /*reward=*/1.0f);
        ASSERT_TRUE(std::isfinite(prediction));
    }

    EXPECT_TRUE(std::isfinite(td.Predict(features)));
    EXPECT_GT(prediction, 0.0f);  // moved off the zero initialization
}

TEST(SwiftTD, SparseBinaryRunsAndPredictsFinite) {
    SwiftTDBinaryFeatures td(/*number_of_features=*/100,
                             0.9f, 1e-2f, 0.9f, 1e-5f, 0.1f, 0.999f, 1e-3f, 1e-10f);

    const std::vector<int> active{1, 42, 99};
    for (int t = 0; t < 50; ++t) {
        ASSERT_TRUE(std::isfinite(td.Step(active, 1.0f)));
    }
    EXPECT_TRUE(std::isfinite(td.Predict(active)));
}

}  // namespace
