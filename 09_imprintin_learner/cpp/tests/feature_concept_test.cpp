#include "imprinting/feature.hpp"

#include <gtest/gtest.h>

#include <array>
#include <span>

namespace {

// A minimal concrete type that models il::Feature.
class ExampleFeature {
public:
    float getWeight() const { return weight_; }
    std::span<float> getEligibilityTrace() { return trace_; }
    float getStepSize() const { return step_size_; }
    il::Status getStatus() const { return status_; }
    float getTenureThreshold() const { return 1.0f; }
    float getTenureTrackThreshold() const { return 0.5f; }
    il::FeatureType getFeatureType() const { return type_; }
    bool getActivation() const { return active_; }

private:
    float weight_ = 0.25f;
    float step_size_ = 0.01f;
    il::Status status_ = il::Status::TenureTrack;
    il::FeatureType type_ = il::FeatureType::Pattern;
    bool active_ = true;
    std::array<float, 3> trace_{0.0f, 0.0f, 0.0f};
};

// Missing most of the interface -> must NOT model the concept.
struct NotAFeature {
    float getWeight() const { return 0.0f; }
};

static_assert(il::Feature<ExampleFeature>,
              "ExampleFeature should satisfy il::Feature");
static_assert(!il::Feature<NotAFeature>,
              "NotAFeature is missing methods and must not satisfy il::Feature");

// A function constrained on the concept; instantiating it is itself a check.
template <il::Feature F>
float effective_update(F& f) {
    return f.getWeight() * f.getStepSize();
}

TEST(FeatureConcept, ExampleFeatureExposesInterface) {
    ExampleFeature f;
    EXPECT_FLOAT_EQ(f.getWeight(), 0.25f);
    EXPECT_FLOAT_EQ(f.getStepSize(), 0.01f);
    EXPECT_EQ(f.getStatus(), il::Status::TenureTrack);
    EXPECT_EQ(f.getFeatureType(), il::FeatureType::Pattern);
    EXPECT_GT(f.getTenureThreshold(), f.getTenureTrackThreshold());
    EXPECT_EQ(f.getEligibilityTrace().size(), 3u);
}

TEST(FeatureConcept, ConstrainedFunctionCompilesAndComputes) {
    ExampleFeature f;
    EXPECT_FLOAT_EQ(effective_update(f), 0.25f * 0.01f);
}

}  // namespace
