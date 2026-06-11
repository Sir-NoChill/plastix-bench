#include "imprinting/linalg.hpp"

#include <array>
#include <print>
#include <string_view>

namespace {

void print_matrix(std::string_view name, const il::Matrix& m) {
    std::print("{} = [{} x {}]\n", name, m.rows(), m.cols());
    for (std::size_t r = 0; r < m.rows(); ++r) {
        std::print("  ");
        for (std::size_t c = 0; c < m.cols(); ++c) {
            std::print("{:8.3f} ", m(r, c));
        }
        std::print("\n");
    }
}

}  // namespace

int main() {
    using il::Matrix;

    const auto a = Matrix::from_row_major(2, 3, {1, 2, 3,
                                                 4, 5, 6});
    const auto b = Matrix::from_row_major(3, 2, {7,  8,
                                                 9,  10,
                                                 11, 12});

    print_matrix("A", a);
    print_matrix("B", b);
    print_matrix("A*B", il::matmul(a, b));

    const std::array<double, 3> x{1.0, 0.0, -1.0};
    const auto y = il::matvec(a, x);
    std::print("A*x = [");
    for (std::size_t i = 0; i < y.size(); ++i) {
        std::print("{}{:.3f}", i == 0 ? "" : ", ", y[i]);
    }
    std::print("]\n");

    std::print("dot(x, x) = {:.3f}\n", il::dot(x, x));
    std::print("norm(x)   = {:.3f}\n", il::norm(x));

    return 0;
}
