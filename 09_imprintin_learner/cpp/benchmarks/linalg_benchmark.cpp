#include "imprinting/linalg.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <vector>

namespace {

il::Matrix random_matrix(std::size_t rows, std::size_t cols) {
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> data(rows * cols);
    for (auto& v : data) {
        v = dist(rng);
    }
    return il::Matrix::from_row_major(rows, cols, std::move(data));
}

void BM_MatMul(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = random_matrix(n, n);
    const auto b = random_matrix(n, n);
    for (auto _ : state) {
        auto c = il::matmul(a, b);
        benchmark::DoNotOptimize(c.data().data());
        benchmark::ClobberMemory();
    }
    // A dense n x n gemm performs ~2*n^3 floating point operations.
    state.SetItemsProcessed(state.iterations() * 2 * n * n * n);
}
BENCHMARK(BM_MatMul)->RangeMultiplier(2)->Range(64, 512);

void BM_Dot(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const std::vector<double> x(n, 1.5);
    const std::vector<double> y(n, 2.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(il::dot(x, y));
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Dot)->RangeMultiplier(8)->Range(1 << 10, 1 << 20);

}  // namespace
