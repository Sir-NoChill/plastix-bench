
//
// Created by Khurram Javed on 2024-02-18.
//
// Vendored from SwiftTD (https://github.com/khurramjaved96/SwiftTD), MIT License.
// See third_party/swifttd/LICENSE and third_party/swifttd/CITATION.
// Local additions vs. upstream: include path namespaced; SwiftTDBinaryFeatures
// weights are bindable to external storage for in-place updates, plus
// weights()/betas()/zTraces() accessors and swapPopFeature() for the imprinting
// learner's in-place feature removal (see comments there).
//

#ifndef SWIFTTD_H
#define SWIFTTD_H

#include <cstddef>
#include <span>
#include <vector>

class Math
{
public:
    static float DotProduct(const std::vector<float>& a, const std::vector<float>& b);
};

class SwiftTDNonSparse
{
private:
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;

    std::vector<float> featureVector;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;
    float eta_min;

    float decay;
    float gamma;

public:
    SwiftTDNonSparse(int number_of_features, float lambda_init, float alpha_init, float gamma_init, float epsilon_init,
                     float eta_init,
                     float decay_init, float meta_step_size_init, float eta_min = 1e-10);
    float Step(const std::vector<float>& features, float reward);
    float Predict(const std::vector<float>& features);
};

class SwiftTDBinaryFeatures
{
    std::vector<int> setOfEligibleItems; // set of eligible items
    std::span<float> w;           // bound to externally-owned weights (in place)
    std::vector<float> w_storage; // backing store, used only in standalone mode
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;

    std::vector<float> featureVector;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::vector<float> last_alpha;


    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;
    float eta_min;

    float decay;
    float gamma;

public:
    SwiftTDBinaryFeatures(int number_of_features, float lambda_init, float alpha_init, float gamma_init,
                          float epsilon_init, float eta_init,
                          float decay_init, float meta_step_size_init, float eta_min = 1e-10);
    float Step(const std::vector<int>& feature_indices, float reward);
    float Predict(const std::vector<int>& feature_indices);

    // [local addition, not upstream] In-place weights: bindWeights() points the
    // weight vector at externally-owned storage (the feature arena's column) so
    // Step() updates it directly, with no copy-back. weights()/betas() expose the
    // current weights and log-step-sizes (the latter for the learner's tau).
    void bindWeights(std::span<float> external);
    std::span<const float> weights() const;
    std::span<const float> betas() const;
    std::span<const float> zTraces() const;
    // Remove the feature at index `i` by moving the feature at index `last` into
    // its slot and resetting the freed slot (so feature generation can reuse it).
    // The caller swap-pops the externally-owned weights and any references.
    void swapPopFeature(std::size_t i, std::size_t last);

private:
    float beta_init = 0.0f;  // log(alpha_init); used to reset a freed slot
};


class SwiftTD
{
    std::vector<std::pair<int, float>> setOfEligibleItems; // set of eligible items
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;

    std::vector<float> featureVector;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::vector<float> last_alpha;


    float v_delta;
    float lambda;
    float epsilon;
    float v_old;
    float meta_step_size;

    float eta;
    float eta_min;

    float decay;
    float gamma;

public:
    SwiftTD(int number_of_features, float lambda_init, float alpha_init, float gamma_init,
                          float epsilon_init, float eta_init,
                          float decay_init, float meta_step_size_init, float eta_min = 1e-10);
    float Step(const std::vector<std::pair<int, float>>& feature_indices, float reward);
    float Predict(const std::vector<std::pair<int, float>>& feature_indices);
};

#endif // SWIFTTD_H
