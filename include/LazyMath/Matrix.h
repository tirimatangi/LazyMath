#ifndef LAZY_MATRIX_H
#define LAZY_MATRIX_H

#include <cstdint>
#include <cstddef>
#include <utility>
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <limits>
#include <complex>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace LazyMath
{
using std::size_t;
using std::nullptr_t;

template <class T>
struct is_complex
     : std::integral_constant<
         bool,
         std::is_same<std::complex<float>, typename std::remove_cv<T>::type>::value  ||
         std::is_same<std::complex<double>, typename std::remove_cv<T>::type>::value  ||
         std::is_same<std::complex<long double>, typename std::remove_cv<T>::type>::value> {};

template <class T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <class T>
constexpr auto valueTypeSolver(const T&)
{
    if constexpr  (is_complex_v<T>)
        return typename T::value_type();
    else
        return T();
}

// Gives the value type of T if T is complex. Otherwise gives T.
template <class T>
using ValueType = std::decay_t<decltype(valueTypeSolver(T{}))>;

template <class T, size_t Size = 0>
struct Vector;

// You can replace Vector with any container which has random access iterators
// and method size().
// Rows of a Matrix defined below will consist of an std::vector or
// an std::array of objects of type Vector.

// Fixed-size vector made of an array.
template <class T, size_t Size>
struct Vector : std::array<T, Size>
{
    explicit Vector(size_t size) {
        assert(size == Size);
    }
    using std::array<T, Size>::array;
    using std::array<T, Size>::operator=;
};

// Specialization for variable-sized vector
template <class T>
struct Vector<T, 0> : std::vector<T>
{
    using std::vector<T>::vector;
    using std::vector<T>::operator=;
};

template <class T, size_t Size>
void clear(Vector<T, Size>& vec)
{
    std::fill(vec.begin(), vec.end(), T{});
}

template <class T, size_t Rows = 0, size_t Cols = 0>
struct Matrix;

// Fixed-size matrix made of an std::array of Vectors
template <class T, size_t Rows, size_t Cols>
struct Matrix : std::array<Vector<T, Cols>, Rows>
{
    typedef T value_type;
    explicit Matrix(size_t rows, size_t cols) {
        assert(Rows == rows);
        assert(Cols == cols);
    }

    constexpr size_t rows() const noexcept { return Rows; }
    constexpr size_t cols() const noexcept { return Cols; }
    using std::array<Vector<T, Cols>, Rows>::array;
    using std::array<Vector<T, Cols>, Rows>::operator=;
};


// Variable-size matrix made of a std::vector of Vectors
template <class T>
struct Matrix<T, 0, 0> : std::vector<Vector<T, 0>>
{
    typedef T value_type;
    explicit Matrix(size_t rows, size_t cols) {
        this->resize(rows);
        for (auto& row : *this)
            row.resize(cols);
    }
    size_t rows() const noexcept { return this->size(); }
    size_t cols() const noexcept {
        if (this->empty())
            return 0;
        return this->begin()->size();
    }
    using std::vector<Vector<T, 0>>::vector;
    using std::vector<Vector<T, 0>>::operator=;
};

template <class T, size_t Rows, size_t Cols>
void clear(Matrix<T, Rows, Cols>& mat)
{
    for (auto& row : mat)
        std::fill(row.begin(), row.end(), T{});
}

template <class T>
static inline T conjugate(const T& x)
{
    return x;
}

template <class T>
static inline std::complex<T> conjugate(const std::complex<T>& x)
{
    return std::conj(x);
}

// Calculates inner product x'.y
template <class T, size_t Size>
static inline T inner(const Vector<T, Size>& x, const Vector<T, Size>& y)
{
    assert(x.size() == y.size());
    return std::inner_product(x.cbegin(), x.cend(), y.cbegin(), T{},
                                [](auto x, auto y){return x + y;},
                                [](auto x, auto y){return conjugate(x) * y;});
}

// Calculates mat' * mat, where ' is matlab-style transpose or hermitian operator.
// Assumes that Matrix type has operator[][]. Stores the result in *result.
template <class T, size_t Rows, size_t Cols>
void inner(const Matrix<T, Rows, Cols>& matA, Matrix<T, Cols, Cols>* result)
{
    assert(result != nullptr);
    assert(result->rows() == matA.cols());
    assert(result->cols() == matA.cols());
    const auto cols = matA.cols();
    const auto rows = matA.rows();

    clear(*result);

    auto itRowR = result->begin(); // Result matrix R ii'th row
    for (size_t ii = 0; ii < cols; ++ii) {
        auto itRowA = matA.cbegin(); // Input matrix A kk'th row
        for (size_t kk = 0; kk < rows; ++kk) {
            const T coeff = conjugate((*itRowA)[ii]);
            auto itColR = itRowR->begin();
            auto itColA = itRowA->cbegin();
            for (size_t jj = 0; jj <= ii; ++jj) // Update column jj of A and R
                *itColR++ += *itColA++ * coeff;
            ++itRowA;
        } // for kk
        ++itRowR;
    } // for ii
    // Fill in the upper triangle
    for (size_t ii = 0; ii < cols; ++ii) {
        for (size_t jj = 0; jj < ii; ++jj)
            (*result)[jj][ii] = conjugate((*result)[ii][jj]);
        if constexpr (is_complex_v<T>)
            (*result)[ii][ii] = std::real((*result)[ii][ii]);
    }
    // Remove possible imaginary part due to rounding error from the diagonal
}

template <class T, size_t Rows, size_t Cols>
auto inner(const Matrix<T, Rows, Cols>& matA)
{
    Matrix<T, Cols, Cols> result(matA.cols(), matA.cols());
    inner(matA, &result);
    return result;
}


// Calculates mat * mat', where ' is matlab-style transpose or hermitian operator.
// Assumes that Matrix type has operator[][]. Stores the result in *result.
template <class T, size_t Rows, size_t Cols>
void outer(const Matrix<T, Rows, Cols>& matA, Matrix<T, Rows, Rows>* result)
{
    assert(result != nullptr);
    assert(result->rows() == matA.rows());
    assert(result->cols() == matA.rows());
    const auto rows = matA.rows();

    for (size_t ii = 0; ii < rows; ++ii) {
        (*result)[ii][ii] = inner(matA[ii], matA[ii]);
        for (size_t jj = ii + 1; jj < rows; ++jj) {
            T value = inner(matA[ii], matA[jj]);
            (*result)[ii][jj] = conjugate(value);
            (*result)[jj][ii] = value;
        }
    }
}

template <class T, size_t Rows, size_t Cols>
auto outer(const Matrix<T, Rows, Cols>& matA)
{
    Matrix<T, Rows, Rows> result(matA.rows(), matA.rows());
    outer(matA, &result);
    return result;
}


// Calculates squared norm of vec.
template <class T, size_t Size>
static inline double norm2(const Vector<T, Size>& vec)
{
    return std::accumulate(vec.begin(), vec.end(), 0.0, [](double s, T b) { return s + std::norm(b); });
}

// Calculates trace of the matrix
template <class T, size_t Rows, size_t Cols>
T trace(const Matrix<T, Rows, Cols>& matA)
{
    assert(matA.rows() == matA.cols());
    T value{};
    for (size_t i = 0; i < matA.rows(); ++i)
        value += matA[i][i];
    return value;
}

// Calculates Frobenius norm
template <class T, size_t Rows, size_t Cols>
double frobeniusNorm(const Matrix<T, Rows, Cols>& matA)
{
    double fr = 0;
    for (auto row : matA)
        fr += norm2(row);
    return std::sqrt(fr);
}

// Calculates mat * vec. Stores the result into the 3rd argument.
template <class T, size_t Rows, size_t Cols>
void multiply(const Matrix<T, Rows, Cols>& mat, const Vector<T, Cols>& vec, Vector<T, Rows>* result)
{
    assert(result != nullptr);
    assert(vec.size() == mat.cols());
    assert(result->size() == mat.rows());
    auto itResult = result->begin();
    for (const auto& row : mat)
        *itResult++ = std::inner_product(vec.begin(), vec.end(), row.begin(), T{});
}

// Calculates mat * vec. Returns a Vector.
template <class T, size_t Rows, size_t Cols>
auto multiply(const Matrix<T, Rows, Cols>& mat, const Vector<T, Cols>& vec)
{
    assert(vec.size() == mat.cols());
    Vector<T, Rows> result(mat.rows());
    multiply(mat, vec, &result);
    return result;
}

// Calculates matA * matB, dim A = N x M, dim B = M * P, Stores the result into the 3rd argument.
template <class T, size_t N, size_t M, size_t P>
auto multiply(const Matrix<T, N, M>& matA, const Matrix<T, M, P>& matB, Matrix<T, N, P>* result)
{
    assert(matA.cols() == matB.rows());
    assert(result != nullptr && result->rows() == matA.rows() && result->cols() == matB.cols());
    clear(*result);

    for (size_t i = 0; i < matA.rows(); ++i) {
        const auto& rowA = matA[i];
        for (size_t k = 0; k < matA.cols(); ++k) {
            const auto& rowB = matB[k];
            for (size_t j = 0; j < matB.cols(); ++j) {
                (*result)[i][j] += rowA[k] * rowB[j];
            }
        }
    }
    return result;
}

// Calculates matA * matB, dim A = N x M, dim B = M * P
template <class T, size_t N, size_t M, size_t P>
auto multiply(const Matrix<T, N, M>& matA, const Matrix<T, M, P>& matB)
{
    assert(matA.cols() == matB.rows());
    Matrix<T, N, P> result(matA.rows(), matB.cols());
    multiply(matA, matB, &result);
    return result;
}

// Calculates mat' * vec, where ' is matlab-style transpose or hermitian operator.
// Stores the result into the 3rd argument.
template <class T, size_t Rows, size_t Cols>
void multiplyTranspose(const Matrix<T, Rows, Cols>& mat, const Vector<T, Rows>& vec, Vector<T, Cols>* result)
{
    assert(result != nullptr);
    assert(vec.size() == mat.rows());
    assert(result->size() == mat.cols());
    std::fill(result->begin(), result->end(), T{});
    auto itMatRow = mat.cbegin();
    for (T multiplier : vec) {
        auto itCol = itMatRow->cbegin();
        for (T& value : *result) {
            value += conjugate(*itCol++) * multiplier;
        }
        ++itMatRow;
    }
}

// Calculates mat' * vec, where ' is matlab-style transpose or hermitian operator.
template <class T, size_t Rows, size_t Cols>
auto multiplyTranspose(const Matrix<T, Rows, Cols>& mat, const Vector<T, Rows>& vec)
{
    Vector<T, Cols> result(mat.cols());
    multiplyTranspose(mat, vec, &result);
    return result;
}

// Calculates matA' * matB, dim A = M x N, dim B = M * P, Stores the result into the 3rd argument.
template <class T, size_t N, size_t M, size_t P>
auto multiplyTranspose(const Matrix<T, M, N>& matA, const Matrix<T, M, P>& matB, Matrix<T, N, P>* result)
{
    assert(matA.rows() == matB.rows());
    assert(result != nullptr && result->rows() == matA.cols() && result->cols() == matB.cols());
    clear(*result);

    for (size_t k = 0; k < matA.rows(); ++k) {
        const auto& rowA = matA[k];
        const auto& rowB = matB[k];
        for (size_t i = 0; i < matA.cols(); ++i) {
            for (size_t j = 0; j < matB.cols(); ++j) {
                (*result)[i][j] += conjugate(rowA[i]) * rowB[j];
            }
        }
    }
    return result;
}

// Calculates matA' * matB, dim A = M x N, dim B = M * P
template <class T, size_t N, size_t M, size_t P>
auto multiplyTranspose(const Matrix<T, M, N>& matA, const Matrix<T, M, P>& matB)
{
    assert(matA.rows() == matB.rows());
    Matrix<T, N, P> result(matA.cols(), matB.cols());
    multiplyTranspose(matA, matB, &result);
    return result;
}


// Transposes the input matrix M (i.e returns M' where ' is matlab-style transpose or hermitian operator.
template <class T, size_t Rows, size_t Cols>
void transpose(const Matrix<T, Rows, Cols>& mat, Matrix<T, Cols, Rows>* result)
{
    for (size_t i = 0; i < mat.rows(); ++i) {
        const auto& row = mat[i];
        for (size_t j = 0; j < mat.cols(); ++j)
            (*result)[j][i] = conjugate(row[j]);
    }
}

template <class T, size_t Rows, size_t Cols>
auto transpose(const Matrix<T, Rows, Cols>& mat)
{
    Matrix<T, Cols, Rows> result(mat.cols(), mat.rows());
    transpose(mat, &result);
    return result;
}


template <class T, size_t Rows, size_t Cols>
void printMatrix(const Matrix<T, Rows, Cols>& mat, const char* s = nullptr)
{
    if (s)
        std::cout << s << std::endl;
    std::cout << "dim = {" << mat.rows() << " x " << mat.cols() << "}\n";

    int iRow = 0;
    for(const auto& row : mat) {
        std::cout << iRow << ": ";
        for (T col : row)
            std::cout << col << " ";
        std::cout << std::endl;
        ++iRow;
    }
}

template <class T, size_t Size>
std::ostream& operator<<(std::ostream& os, const Vector<T,Size>& vec)
{
    os << '[';
    for (unsigned i = 0; i < vec.size(); ++i) {
        os << vec[i];
        os << ((i < vec.size() - 1) ? ',' : ']');
    }
    return os;
}
} // namespace LazyMath

#endif // LAZY_MATRIX_H
