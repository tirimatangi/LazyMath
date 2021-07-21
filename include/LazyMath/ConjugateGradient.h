#ifndef LAZY_CG_H
#define LAZY_CG_H

#include "Matrix.h"

namespace LazyMath
{
using std::size_t;

enum class SymmetricMode { NonSymmetricMatrix, SymmetricMatrix};

template <class T, size_t Rows, size_t Cols, class Real = ValueType<T>>
auto conjugateGradientSymmetric(const Matrix<T, Rows, Cols>& matA,
                                const Vector<T, Rows>& vecB,
                                const Vector<Real, Cols>& preconditioner,
                                double errorTolerance,
                                int maxIterations,
                                double *error)
{
        static_assert(Rows != Cols);
        assert(Rows == Cols); // Should always fail because you really should not be here at runtime.
        return Vector<T, Cols>(Cols);
}


// Solves x = matA \ vecB assuming that matA is positive definite. Returns x and residual error relative to the norm of vecB.
template <class T, size_t Size, class Real = ValueType<T>>
auto conjugateGradientSymmetric(const Matrix<T, Size, Size>& matA,
                                const Vector<T, Size>& vecB,
                                const Vector<Real, Size>& preconditioner,
                                double errorTolerance,
                                int maxIterations,
                                double *error = nullptr)
{
    constexpr size_t Rows = Size;
    constexpr size_t Cols = Size;
    assert(matA.rows() == matA.cols());
    assert(matA.rows() == vecB.size());
    assert(matA.cols() == preconditioner.size());
    assert(error != nullptr);

    errorTolerance *= errorTolerance;
    Vector<T, Cols> vecX(matA.cols());
    // Squared norm of vec for relative error calculation.
    Real dBb = norm2(vecB);
    if (dBb == 0) { // A*0 = b if b = 0
        *error = 0;
        std::fill(vecX.begin(), vecX.end(), T{});
        return vecX;
    }

    // Calculate initial guess assuming that the matrix is nearly diagonal.
    // vecX = vecB .* preconditioner
    std::transform(vecB.cbegin(), vecB.cend(), preconditioner.cbegin(), vecX.begin(),
                   [](const T& b, const T& pre){ return b * pre; });


    // Initialize residual error vector r=b-A*x
    Vector<T, Rows> vecR(vecB.size()), vecZ(vecB.size());
    Vector<T, Rows> vecAx = multiply(matA, vecX);

    // vecR = vecB - vecAx
    std::transform(vecB.cbegin(), vecB.cend(), vecAx.cbegin(), vecR.begin(),
                   [](const T& b, const T& ax){ return b - ax; });
    // vecZ = M*vecR = vecR .* preconditioner
    std::transform(vecR.cbegin(), vecR.cend(), preconditioner.cbegin(), vecZ.begin(),
                   [](const T& r, const T& pre){ return r * pre; });

    Vector<T, Rows> vecP = vecZ;
    Real dRr = norm2(vecR);
    Real dRz = std::real(inner(vecR, vecZ));
    int iIter = 0;
    while (iIter++ < maxIterations && dRr > 0) {
        Vector<T, Rows> vecAp = multiply(matA, vecP);
        T cxAlpha = dRz / inner(vecP, vecAp);

        // vecX += alpha * vecP
        std::transform(vecP.cbegin(), vecP.cend(), vecX.cbegin(), vecX.begin(),
                       [cxAlpha](const T& p, const T& x){ return x + cxAlpha * p; });
        // vecR -= alpha * vecAp;
        std::transform(vecAp.cbegin(), vecAp.cend(), vecR.cbegin(), vecR.begin(),
                       [cxAlpha](const T& ap, const T& r){ return r - cxAlpha * ap; });
        // vecZ = vecR .* preconditioner and calculate norms
        Real dRzNext = 0, dRrNext = 0;
        std::transform(vecR.cbegin(), vecR.cend(), preconditioner.cbegin(), vecZ.begin(),
                       [&dRzNext, &dRrNext](const T& r, const Real& pre) {
                           Real dR = std::norm(r);
                           dRrNext += dR;         // dRzNext becomes r' * r.
                           dRzNext += (dR * pre); // dRzNext becomes r' * z.
                           return r * pre;
                        });
        if (dRrNext == 0.0 || !std::isfinite(dRrNext)) {
            dRr = dRrNext;
            break; // Exact solution has been found or there has been an error
        }

        Real dBeta = dRzNext / dRz;
        // vecP = vecZ + beta * vecP
        std::transform(vecZ.cbegin(), vecZ.cend(), vecP.cbegin(), vecP.begin(),
                       [dBeta](const T& z, const T& p){ return z + dBeta * p; });
      dRr = dRrNext;
      dRz = dRzNext;
      if (dRr <= errorTolerance * dBb)
        break; // The accuracy has been reached.
    } // while Iter

    *error = std::isfinite(dRr) ? std::sqrt(dRr / dBb) : INFINITY;
    return vecX;
}


// Solves x = matA \ vecB. Returns x and residual error relative to the norm of vecB.
template <class T, size_t Rows, size_t Cols, class Real = ValueType<T>>
auto conjugateGradientLeastSquares(const Matrix<T, Rows, Cols>& matA,
                                   const Vector<T, Rows>& vecB,
                                   const Vector<Real, Cols>& preconditioner,
                                   double errorTolerance,
                                   int maxIterations,
                                   double *error)
{
    assert(matA.rows() == vecB.size());
    assert(matA.cols() == preconditioner.size());
    assert(error != nullptr);

    // Apply preconditioned matrix to make the A-matrix is numerically easier to deal with.
    // So actually we solve problem (A*inv(C))'(A*inv(C)) * (Cx) = (A*inv(C))b
    // where inv(C) is the diagonal preconditioner.
    Matrix<T, Rows, Cols>  matAc(matA.rows(), matA.cols());

    auto itMatRow = matA.begin();
    for(auto itMatAcRow = matAc.begin(); itMatAcRow != matAc.end(); ++itMatAcRow) {
        std::transform(preconditioner.begin(), preconditioner.end(), itMatRow->begin(), itMatAcRow->begin(),
                       [](Real precon, T elem) { return T(precon * elem); } );
        ++itMatRow;
    }

    errorTolerance *= errorTolerance;

    // Initialize the result vector to zero.
    Vector<T, Cols> vecX(matA.cols());
    std::fill(vecX.begin(), vecX.end(), T{});

    // Squared norm of vec for relative error calculation.
    Real dBb = norm2(vecB);
    if (dBb == 0) { // A*0 = b if b = 0
        *error = 0;
        return vecX;
    }

    Vector<T, Rows> vecD = vecB; // Residual assuming that the initial guess vecX=0
    Vector<T, Cols> vecR = multiplyTranspose(matAc, vecD);  // r = A' * residual
    Vector<T, Cols> vecP = vecR;
    Real dRr = norm2(vecR);
    Real dDd = norm2(vecD);
    int iIter = 0;

    // Vector and iteration which have given the so far best result in terms of minimizing d' * d.
    Real dDdMin = INFINITY;
    Vector<T, Cols> vecXmin = vecX;
    int iIterMin = iIter;

    constexpr int iMaxDivergingIterations = 10;
    while (iIter < maxIterations && dDd > 0 && (iIter - iIterMin) < iMaxDivergingIterations) {
        Vector<T, Rows> vecT = multiply(matAc, vecP);

        Real dTt = norm2(vecT);
        Real dAlpha = dRr / dTt;

        // vecX += alpha * vecP
        std::transform(vecP.cbegin(), vecP.cend(), vecX.cbegin(), vecX.begin(),
                       [dAlpha](const T& p, const T& x){ return x + dAlpha * p; });
        // vecD -= alpha * vecT;
        std::transform(vecT.cbegin(), vecT.cend(), vecD.cbegin(), vecD.begin(),
                       [dAlpha](const T& t, const T& d){ return d - dAlpha * t; });
        dDd = norm2(vecD);

        if (dDd < dDdMin) { // vecX is the best candidate so far
          dDdMin = dDd;
          vecXmin = vecX;
          iIterMin = iIter;
        }
        if (dDd <= errorTolerance * dBb)
            break; // The accuracy has been reached.
        vecR = multiplyTranspose(matAc, vecD);
        Real dRrNext = norm2(vecR);

        if (dRrNext == 0 || !std::isfinite(dRrNext))
            break; // Exact solution has been found or there has been an error
        Real dBeta = dRrNext / dRr;
        // vecP = vecR + beta * vecP
        std::transform(vecR.cbegin(), vecR.cend(), vecP.cbegin(), vecP.begin(),
                       [dBeta](const T& r, const T& p){ return r + dBeta * p; });
        dRr = dRrNext;
        ++iIter;
    } // while iIter
    // Un-apply diagonal preconditioner matrix. The actual answer will be inv(C)*x.
    std::transform(preconditioner.begin(), preconditioner.end(), vecXmin.cbegin(), vecXmin.begin(),
                   [](Real precon, T x) { return T(precon * x); } );

    *error = std::isfinite(dDdMin) ? std::sqrt(dDdMin / dBb) : INFINITY;
    return vecXmin;
}


// Calculates least squares solution for x in A*x = b with Conjugate Gradient Method.
// Matrix A may be any N-by-M matrix and b must be a vector of length N.
// The result x is a vector of length M.
// symmetricMode should be set to SymmetricMode::SymmetricMatrix if matrix A is
// a symmetric and positive definite matrix. otherwise, it should be set to SymmetricMode::NonSymmetricMatrix.
// *error is set to ||Ax - b|| / ||b|| which is less than errorTolerance unless the iteration
// has stopped because maxIter limit has been reached.
template <class T, size_t Rows, size_t Cols>
auto linearSolver(const Matrix<T, Rows, Cols>& mat,
                  const Vector<T, Rows>& vec,
                  SymmetricMode symmetricMode, // = SymmetricMode::NonSymmetricMatrix or SymmetricMatrix
                  double errorTolerance = 0,
                  int maxIterations = 0, // <= 0 means default value
                  double *error = nullptr)
{
    using Real = ValueType<T>; // Value type of T if T is complex, otherwise T.
    if (vec.size() != mat.rows())
        throw std::runtime_error("Matrix and vector dimensions don't match.");
    if (symmetricMode == SymmetricMode::SymmetricMatrix && mat.rows() != mat.cols())
        throw std::runtime_error("Matrix must be square and positive definite in symmetric mode.");

    if (maxIterations <= 0)
        maxIterations = 2 * mat.cols();

    if (errorTolerance <= 0)
        errorTolerance = std::sqrt(std::numeric_limits<Real>::epsilon());

    // Calculate the diagonal of the preconditioner matrix into a vector.
    Vector<Real, Cols> vecPreconditioner(mat.cols());
    if (symmetricMode == SymmetricMode::NonSymmetricMatrix) {
        // Calculate the diagonal of A' * A.
        std::fill(vecPreconditioner.begin(), vecPreconditioner.end(), Real(0));
        for (const auto& row : mat)
            for (size_t j = 0; j < mat.cols(); ++j)
                vecPreconditioner[j] += std::norm(row[j]);
    }
    else  {// Symmetric matrix: Just take the diagonal
        for (size_t i = 0; i < mat.cols(); ++i) {
            auto value = mat[i][i];
            vecPreconditioner[i] =  (std::imag(value) == 0) ? std::real(value) : Real(0);
        }
    }

    // Calculate inverses
    for (Real& x : vecPreconditioner) {
        if (x <= 0)
            throw std::runtime_error("Preconditioner failure. Maybe the matrix is not invertible?");
        x = (symmetricMode == SymmetricMode::NonSymmetricMatrix) ? 1 / std::sqrt(x) : 1 / x;
    }

    Vector<T, Cols> vecX;
    double dError = INFINITY;
    vecX = (symmetricMode == SymmetricMode::NonSymmetricMatrix) ?
        conjugateGradientLeastSquares(mat, vec, vecPreconditioner, errorTolerance, maxIterations, &dError) :
        conjugateGradientSymmetric(mat, vec, vecPreconditioner, errorTolerance, maxIterations, &dError);

    if (error)
        *error = dError;
    return vecX;
}



template <class T, size_t N, size_t M, size_t P>
auto constrainedSolverNonSymmetric(const Matrix<T, N, M>& matA,
                                   const Vector<T, N>& vecB,
                                   const Matrix<T, P, M>& matC,
                                   const Vector<T, P>& vecD,
                                   double errorTolerance = 0, // <= 0 means default value
                                   int maxIterations = 0) // <= 0 means default value
{
    /* Remainder:
    matA.rows() means N,
    matA.cols() means M,
    matC.cols() means M,
    matC.rows() means P
    */

    // Apply Schur complement to solve system
    // [A'*A  C'][x] = [A'*b]
    // [C     0 ][z]   [d   ]    (dim z = P, dim x = M)
    // using Schur, z can be solved from system
    // (C*inv(A'*A)*C') * z = C*inv(A'*A)*A'*b - d
    // Now the z is known, x can be solved from system
    // (A'*A) * x = A'*b - C'*z

    // Step 1: Solve matA * vecM_Ab = vecB, dim vecM_Ab = M, meaning vecM_Ab = inv(A'*A)*A'*b.
    Vector<T, M> vecM_Ab = linearSolver(matA, vecB, SymmetricMode::NonSymmetricMatrix, errorTolerance, maxIterations);
    //%% printVector(vecM_Ab);

    // Step 2: Solve (matA' * matA) * matMP = matC', dim matMP = M x P, meaning matMP = inv(A'*A)*C'
    Matrix<T, M, P> matMP(matC.cols(), matC.rows());
    Matrix<T, M, M> matA_inner = inner(matA);
    Vector<T, M> vecC_transposed_col(matC.cols()); // Conjugated column of C
    for (size_t i = 0; i < matC.rows(); ++i) {
        // Copy & conjugate i'th columns of matC to a vector.
        std::transform(matC[i].cbegin(), matC[i].cend(), vecC_transposed_col.begin(), [](const T& x){ return conjugate(x);});
        // vecM_tmp is the i'th column of the result matrix. Solve a symmetric M x M system.
        Vector<T, M> vecM_tmp = linearSolver(matA_inner, vecC_transposed_col, SymmetricMode::SymmetricMatrix, errorTolerance, maxIterations);
        // Copy the vector into column i of matMP.
        for (size_t j = 0; j < matC.cols(); ++j)
            matMP[j][i] = vecM_tmp[j];
    }
    //%% printMatrix(matMP, "matMP");

    // Step 3: Calculate vecP_Cd = C*vecM_Ab - d = C*inv(A'*A)*A'*b - d
    Vector<T, P> vecP_Cd = multiply(matC, vecM_Ab);
    // vecP_Cd = vecP_Cd - vecD
    std::transform(vecP_Cd.cbegin(), vecP_Cd.cend(), vecD.cbegin(), vecP_Cd.begin(), std::minus<T>());
    //%% printVector(vecP_Cd, "vecP_Cd");

    // Step 4: Calculate matPP = C * matMP = C * inv(A'*A)*C'
    Matrix<T, P, P> matPP = multiply(matC, matMP);
    //%% printMatrix(matPP, "matPP");

    // Step 5: Solve matPP * z = vecP_Cd, meaning (C*inv(A'*A)*C') * z = C*inv(A'*A)*A'*b - d
    Vector<T, P> vecZ = linearSolver(matPP, vecP_Cd, SymmetricMode::NonSymmetricMatrix, errorTolerance, maxIterations);
    //%% printVector(vecZ, "vecZ");

    // Step 6: Calculate vecM_Ab = A'*b - C'*z
    vecM_Ab = multiplyTranspose(matA, vecB);
    Vector<T, M> vecM_Cz = multiplyTranspose(matC, vecZ);
    std::transform(vecM_Ab.cbegin(), vecM_Ab.cend(), vecM_Cz.cbegin(), vecM_Ab.begin(), std::minus<T>());
    //%% printVector(vecM_Ab, "A'b-C'z");

    // Step 7: Solve (A'*A) * x = vecM_Ab = A'*b - C'*z
    Vector<T, M> vecX = linearSolver(inner(matA), vecM_Ab, SymmetricMode::SymmetricMatrix, errorTolerance, maxIterations);
    //%% printVector(vecX, "X");

    return vecX;
}

template <class T, size_t N, size_t M, size_t P>
auto constrainedSolverSymmetric(const Matrix<T, N, M>& matA,
                                const Vector<T, N>& vecB,
                                const Matrix<T, P, M>& matC,
                                const Vector<T, P>& vecD,
                                double errorTolerance = 0, // <= 0 means default value
                                int maxIterations = 0) // <= 0 means default value
{
    assert(matA.rows() == matA.cols());
    return Vector<T, M>(matA.cols());
}

template <class T, size_t M, size_t P>
auto constrainedSolverSymmetric(const Matrix<T, M, M>& matA,
                                const Vector<T, M>& vecB,
                                const Matrix<T, P, M>& matC,
                                const Vector<T, P>& vecD,
                                double errorTolerance = 0, // <= 0 means default value
                                int maxIterations = 0) // <= 0 means default value
{
    /* Remainder:
    matA.rows() means M,
    matA.cols() means M,
    matC.cols() means M,
    matC.rows() means P
    */
    assert(matA.rows() == matA.cols());
    // Assume that A' = A and it is invertible.
    // Apply Schur complement to solve system
    // [A'*A  C'][x] = [A'*b]
    // [C     0 ][z]   [d   ]    (dim z = P, dim x = M)
    // which becomes
    // [A  inv(a)*C'][x] = [b]
    // [C         0 ][z]   [d]

    // using Schur, z can be solved from system
    // C*inv(A)*inv(A)*C') * z = C*inv(A)*b - d
    // Now the z is known, x can be solved from system
    // A * x = b - inv(A)*C'*z

    // Step 1. Solve A * matMP = C' --> matMP = inv(A)*C'
    Matrix<T, M, P> matMP(matC.cols(), matC.rows());
    Vector<T, M> vecC_transposed_col(matC.cols()); // Conjugated column of C
    for (size_t i = 0; i < matC.rows(); ++i) {
        // Copy & conjugate i'th columns of matC to a vector.
        std::transform(matC[i].cbegin(), matC[i].cend(), vecC_transposed_col.begin(), [](const T& x){ return conjugate(x);});
        // vecM_tmp is the i'th column of the result matrix. Solve a symmetric M x M system.
        Vector<T, M> vecM_tmp = linearSolver(matA, vecC_transposed_col, SymmetricMode::SymmetricMatrix, errorTolerance, maxIterations);
        // Copy the vector into column i of matMP.
        for (size_t j = 0; j < matC.cols(); ++j)
            matMP[j][i] = vecM_tmp[j];
    }
    //%% printMatrix(matMP, "matMP");

    // Step 2. Calculate vecP = matMP' * b - d = C*inv(A)*b - d
    Vector<T, P> vecP = multiplyTranspose(matMP, vecB);
    // vecP = vecP - vecD
    std::transform(vecP.cbegin(), vecP.cend(), vecD.cbegin(), vecP.begin(), std::minus<T>());
    //%% printVector(vecP, "vecP");

    // Step 3. Solve (matMP'*matMP) z = vecP
    Vector<T, P> vecZ = linearSolver(inner(matMP), vecP, SymmetricMode::SymmetricMatrix, errorTolerance, maxIterations);
    //%% printVector(vecZ, "vecZ");

    // Step 4. Calculate vecM = b - inv(A)*C'*c
    Vector<T, M> vecM = multiply(matMP, vecZ);
    std::transform(vecB.cbegin(), vecB.cend(), vecM.cbegin(), vecM.begin(), std::minus<T>());
    //%% printVector(vecM, "b-inv(A)*C'z");

    // Step 5. Solve A * x = vecM
    Vector<T, M> vecX = linearSolver(matA, vecM, SymmetricMode::SymmetricMatrix, errorTolerance, maxIterations);
    //%% printVector(vecX, "X");

    return vecX;
}

// Minimizes  norm |A*x - b| subject to linear constraints C*x = d.
// Matrix A may be any N-by-M matrix and b must be a vector of length N.
// Matrix C is a P-by-M matrix and d is a vector of length P. So there are P linear constraints.
// The result x is a vector of length M.
// symmetricMode should be set to SymmetricMode::SymmetricMatrix if matrix A is
// a symmetric and positive definite matrix. otherwise, it should be set to SymmetricMode::NonSymmetricMatrix.
// *differenceNorm is set to |Ax - b| if the pointer is not null.
// *constraintViolation is set to |Cx - d| if the pointer is not null.
template <class T, size_t N, size_t M, size_t P>
auto constrainedSolver(const Matrix<T, N, M>& matA,
                       const Vector<T, N>& vecB,
                       const Matrix<T, P, M>& matC,
                       const Vector<T, P>& vecD,
                       SymmetricMode symmetricMode, // = SymmetricMode::NonSymmetricMatrix or SymmetricMatrix
                       double errorTolerance = 0, // <= 0 means default value
                       int maxIterations = 0, // <= 0 means default value
                       double *differenceNorm = nullptr,
                       double *constraintViolation = nullptr)
{
    assert(matA.rows() > 0 && matA.cols() > 0);
    assert(matC.rows() > 0 && matC.cols() > 0);
    assert(matC.cols() == matA.cols());
    assert(matC.rows() == vecD.size());
    assert(matA.rows() == vecB.size());

    Vector<T, M> vecX;
    if (symmetricMode == SymmetricMode::NonSymmetricMatrix)
        vecX = constrainedSolverNonSymmetric(matA, vecB, matC, vecD, errorTolerance, maxIterations);
    else
        vecX = constrainedSolverSymmetric(matA, vecB, matC, vecD, errorTolerance, maxIterations);

    if (differenceNorm) { // Calculate the norm of difference |A*x - b|
        Vector<T, N> vecAxMinusB = multiply(matA, vecX);
        for (int i = 0; i < vecAxMinusB.size(); ++i)
            vecAxMinusB[i] -= vecB[i];
        *differenceNorm = std::sqrt(norm2(vecAxMinusB));
    }

    if (constraintViolation) {// Calculate constraint violation
        Vector<T, P> vecCxMinusD = multiply(matC, vecX);
        for (int i = 0; i < vecCxMinusD.size(); ++i)
            vecCxMinusD[i] -= vecD[i];
        *constraintViolation = std::sqrt(norm2(vecCxMinusD));
    }
    return vecX;
}

} // namespace LazyMath

#endif // LAZY_CG_H