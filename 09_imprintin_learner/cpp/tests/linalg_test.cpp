#include "imprinting/linalg.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace {

TEST(Matrix, ConstructFillAndIndex) {
    il::Matrix m(2, 3, 1.5);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    EXPECT_EQ(m.size(), 6u);
    EXPECT_DOUBLE_EQ(m(1, 2), 1.5);
    m(1, 2) = 9.0;
    EXPECT_DOUBLE_EQ(m(1, 2), 9.0);
}

TEST(Matrix, Identity) {
    const auto i = il::Matrix::identity(3);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(i(r, c), r == c ? 1.0 : 0.0);
        }
    }
}

TEST(Matrix, FromRowMajorRejectsWrongSize) {
    EXPECT_THROW(il::Matrix::from_row_major(2, 2, {1.0, 2.0, 3.0}),
                 std::invalid_argument);
}

TEST(MatMul, KnownResult) {
    const auto a = il::Matrix::from_row_major(2, 3, {1, 2, 3, 4, 5, 6});
    const auto b = il::Matrix::from_row_major(3, 2, {7, 8, 9, 10, 11, 12});
    const auto c = il::matmul(a, b);
    ASSERT_EQ(c.rows(), 2u);
    ASSERT_EQ(c.cols(), 2u);
    EXPECT_DOUBLE_EQ(c(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(c(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(c(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 154.0);
}

TEST(MatMul, IdentityIsNeutral) {
    const auto a = il::Matrix::from_row_major(2, 2, {1, 2, 3, 4});
    const auto i = il::Matrix::identity(2);
    EXPECT_EQ(il::matmul(a, i), a);
    EXPECT_EQ(il::matmul(i, a), a);
}

TEST(MatMul, DimensionMismatchThrows) {
    const il::Matrix a(2, 3);
    const il::Matrix b(2, 2);
    EXPECT_THROW(il::matmul(a, b), std::invalid_argument);
}

TEST(MatVec, KnownResult) {
    const auto a = il::Matrix::from_row_major(2, 3, {1, 2, 3, 4, 5, 6});
    const std::array<double, 3> x{1.0, 0.0, -1.0};
    const auto y = il::matvec(a, x);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_DOUBLE_EQ(y[0], -2.0);  // 1*1 + 2*0 + 3*(-1)
    EXPECT_DOUBLE_EQ(y[1], -2.0);  // 4*1 + 5*0 + 6*(-1)
}

TEST(Vector, DotAndNorm) {
    const std::array<double, 3> x{3.0, 4.0, 0.0};
    EXPECT_DOUBLE_EQ(il::dot(x, x), 25.0);
    EXPECT_DOUBLE_EQ(il::norm(x), 5.0);
}

}  // namespace
