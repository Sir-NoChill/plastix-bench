#include "imprinting/linalg.hpp"

#include <cblas.h>

#include <stdexcept>
#include <utility>

namespace il {

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        m(i, i) = 1.0;
    }
    return m;
}

Matrix Matrix::from_row_major(std::size_t rows, std::size_t cols,
                              std::vector<double> data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument(
            "from_row_major: data size does not match rows * cols");
    }
    Matrix m;
    m.rows_ = rows;
    m.cols_ = cols;
    m.data_ = std::move(data);
    return m;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul: inner dimensions do not match");
    }

    Matrix c(a.rows(), b.cols(), 0.0);
    const auto m = static_cast<blasint>(a.rows());
    const auto n = static_cast<blasint>(b.cols());
    const auto k = static_cast<blasint>(a.cols());

    // C = 1.0 * A * B + 0.0 * C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, a.data().data(), k,
                b.data().data(), n,
                0.0, c.data().data(), n);
    return c;
}

std::vector<double> matvec(const Matrix& a, std::span<const double> x) {
    if (a.cols() != x.size()) {
        throw std::invalid_argument("matvec: dimension mismatch");
    }
    std::vector<double> y(a.rows(), 0.0);
    const auto m = static_cast<blasint>(a.rows());
    const auto n = static_cast<blasint>(a.cols());

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                m, n,
                1.0, a.data().data(), n,
                x.data(), 1,
                0.0, y.data(), 1);
    return y;
}

double dot(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot: size mismatch");
    }
    return cblas_ddot(static_cast<blasint>(x.size()), x.data(), 1, y.data(), 1);
}

double norm(std::span<const double> x) {
    return cblas_dnrm2(static_cast<blasint>(x.size()), x.data(), 1);
}

}  // namespace il
