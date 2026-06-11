#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace il {

// Row-major dense matrix of doubles, backed by a contiguous std::vector.
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0);

    static Matrix identity(std::size_t n);
    static Matrix from_row_major(std::size_t rows, std::size_t cols,
                                 std::vector<double> data);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }

    double& operator()(std::size_t r, std::size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    double operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    std::span<double> data() noexcept { return data_; }
    std::span<const double> data() const noexcept { return data_; }

    bool operator==(const Matrix&) const = default;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

// C = A * B, computed with cblas_dgemm. Throws std::invalid_argument on a
// shape mismatch (A.cols() != B.rows()).
Matrix matmul(const Matrix& a, const Matrix& b);

// y = A * x, computed with cblas_dgemv.
std::vector<double> matvec(const Matrix& a, std::span<const double> x);

// Inner product of two equal-length vectors (cblas_ddot).
double dot(std::span<const double> x, std::span<const double> y);

// Euclidean (L2) norm of a vector (cblas_dnrm2).
double norm(std::span<const double> x);

}  // namespace il
