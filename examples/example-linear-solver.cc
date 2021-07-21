#include <iostream>
#include <iomanip>
#include <chrono>
#include <LazyMath/ConjugateGradient.h>

using std::cout;
using std::string;
using std::complex;
using namespace std::complex_literals;
using std::size_t;

using namespace LazyMath;

int main()
{
    try {
        std::cout << "\n*** Example ls.1 *** : Solve real- and complex valued least squares problem Ax = b\n";
        {  // Case A: Heap-allocated real non-symmetric matrix.
            using T = double; // Also float works
            Matrix<T> matA(3, 2);
            matA = { {1, 2},   // A is a 3x2-matrix.
                     {3, 4},
                     {5, 6} };
            Vector<T> vecB(3);
            vecB = { 1, 2, 3 }; // b is a vector of length 3.
            // The result x will be a vector of length 2.
            auto vecX = linearSolver(matA, vecB, SymmetricMode::NonSymmetricMatrix);
            cout << "Case A: X = [ " << vecX[0] << ", " << vecX[1] << " ]\n";
        }
        {  // Case B: Stack-allocated real symmetric positive definite matrix.
            using T = double; // Also float works
            Matrix<T, 2, 2> matA;  // A is a 2x2-matrix.
            matA[0] = {35, 44};
            matA[1] = {44, 56};
            Vector<T, 2> vecB;
            vecB = { 22, 28 }; // b is a vector of length 2.
            auto vecX = linearSolver(matA, vecB, SymmetricMode::SymmetricMatrix);
            cout << "Case B: X = [ " << vecX[0] << ", " << vecX[1] << " ]\n";
        }
        {  // Case C: Complex heap-allocated positive definite matrix.
            using T = std::complex<double>;
            Matrix<T> matA { {28., 10.+30i},
                             {10.-30i, 70.} };
            Vector<T> vecB { -10.+10i, 14.+28i };
            auto vecX = linearSolver(matA, vecB, SymmetricMode::SymmetricMatrix);
            cout << "Case C: X = [ " << std::real(vecX[0]) << " + " << std::imag(vecX[0]) << "i, "
                                     << std::real(vecX[1]) << " + " << std::imag(vecX[1]) << "i ]\n";
        }
        {  // Case D: Complex stack-allocated non-symmetric matrix. Use float instead of double.
            using T = std::complex<float>;
            Matrix<T, 3, 2> matA;
            matA[0] = {3.f-3if, 2.f+1if};
            matA[1] = {2.f-2if, 4.f+2if};
            matA[2] = {1.f-1if, 6.f+3if};
            Vector<T, 3> vecB;
            vecB = { 1if, 2if, 3if };
            auto vecX = linearSolver(matA, vecB, SymmetricMode::NonSymmetricMatrix);
            cout << "Case D: X = [ " << std::real(vecX[0]) << " + " << std::imag(vecX[0]) << "i, "
                                     << std::real(vecX[1]) << " + " << std::imag(vecX[1]) << "i ]\n";

            cout << "\nNote: Using symmetric mode with a non-symmetric or non-positive definite matrix throws:\n";
            vecX = linearSolver(matA, vecB, SymmetricMode::SymmetricMatrix); // throws.
        }
    }
    catch (const std::runtime_error& e)
    {
        cout << "Exception:" << e.what() << "\n";
    }

    {
        using T = double;
        std::cout << "\n*** Example ls.2 *** : Invert a 1000x1000 real-valued symmetric (positive definite) matrix.\n";
        const size_t N = 1000;
        const double invN = 1.0 / N;
        Matrix<T> matA(N, N);
        Matrix<T> matB(N, N); // Will be matB = eye(N);
        // Make a symmetric matrix with strong diagonal dominance.
        for (int i = 0; i < N; ++i) {
            std::fill(matB[i].begin(), matB[i].end(), 0.0);
            matB[i][i] = 1;
            matA[i][i] = (i+1);
            for (int j = 0; j < i-1; ++j) {
                matA[i][j] = (j+1) * invN;
                matA[j][i] = (j+1) * invN;
            }
        }

        Matrix<T> matX(N, N); // Result matrix
        auto start = std::chrono::system_clock::now();

        // Notice that the result matrix X is the tranpose of the inverse of A because
        // std::transform runs once for each row of B and X (e.g. matX.begin() returns a row iterator.)
        std::transform( // std::execution::par_unseq,
            matB.cbegin(), matB.cend(), matX.begin(),
            [&matA](auto b) {
                return linearSolver(matA, b,
                                    SymmetricMode::SymmetricMatrix, // Mode of operation. A is symmetric.
                                    0.01); // Maximum allowed relative error |Ax - b| / |b| is set to 1%
            });

        auto end = std::chrono::system_clock::now();

        // Verify that A*x[i] = b[i] for each i by calculating maximum column error.
        double dErrorMax = 0;
        for (int i = 0; i < N; ++i) {
            Vector<T> vec = multiply(matA, matX[i]);
            for (int j = 0; j < N; ++j)
                vec[j] = vec[j] - matB[i][j];
            dErrorMax = std::max(norm2(vec), dErrorMax);
        }

        std::chrono::duration<double> diff = end-start;
        std::cout << " Matrix size = " << N << ", errorMax = " << dErrorMax << ", Time = " << diff.count() << "\n";
        if (dErrorMax > 0.01)
            std::cout << "!NOTE! errorMax is greater than 1%.\n";
    }

    {
        std::cout << "\n*** Example ls.3 *** : Invert a 500x500 complex-valued non-symmetric matrix. This may take a while.\n";
        using Real = float; // The data type will be complex<Real>
        const size_t N = 500;
        const Real invN = 1.0 / N;
        Matrix<complex<Real>> matA(N, N);
        Matrix<complex<Real>> matB(N, N);

        // Generate matrix A and identity matrix B.
        for (int i = 0; i < N; ++i) {
            std::fill(matB[i].begin(), matB[i].end(), Real(0));
            matB[i][i] = 1;
            matA[i][i] = (i+1);
            for (int j = 0; j < i-1; ++j) {
                matA[i][j] = {(i+j+1) * invN, (i + 1) * invN};
                matA[j][i] = {(i+j+1) * invN, (-j - 1) * invN};
            }
        }
        Matrix<complex<Real>> matX(N, N); // The result will go here.
        std::vector<double> vecError(N);  // Residual errors for each column will go here.
        std::vector<int> sequence(N);  // Vector with values 0,1,...N-1.
        std::iota(sequence.begin(), sequence.end(), 0);

        auto start = std::chrono::system_clock::now();

        // Solve maxA*matX = matB and store the residual errors for each column into a separate vector.
        // Note that matX is actually transpose of the inverse of matA because the results of
        // individual A*x = e problems are stored to rows of matX.
        std::transform(// std::execution::par_unseq,
            sequence.cbegin(), sequence.cend(), matX.begin(),
            [&](int i) {
                return linearSolver(matA, matB[i],
                                    SymmetricMode::NonSymmetricMatrix,
                                    0.001, // Maximum allowed relative error
                                    0,     // Maximum number of iterations (0 = default)
                                    &vecError[i]);  // Store actual relative error here.
            });

        auto end = std::chrono::system_clock::now();

        // How much is the largest relative error and at which column?
        auto pMaxError = std::max_element(vecError.begin(), vecError.end());
        std::chrono::duration<double> diff = end-start;
        std::cout << "Matrix size = " << N << ", errorMax(rel) = " << *pMaxError
                  << " at column " << std::distance(vecError.begin(), pMaxError)
                  << ", Time = " << diff.count() << "\n";
        if (*pMaxError > 0.001)
            std::cout << "!NOTE! errorMax is greater than 0.001.\n";
    }
    {
        std::cout << "\n*** Example ls.4 *** : Like example ls.3 but uses stack-allocated matrices instead of heap.\n";
        std::cout << "(May dump core if your machine runs out of stack.)" << std::endl;
        using Real = float; // The data type will be complex<Real>
        constexpr size_t N = 500;
        constexpr Real invN = 1.0 / N;
        Matrix<complex<Real>, N, N> matA;
        Matrix<complex<Real>, N, N> matB;

        // Generate matrix A and identity matrix B.
        for (int i = 0; i < N; ++i) {
            std::fill(matB[i].begin(), matB[i].end(), Real(0));
            matB[i][i] = 1;
            matA[i][i] = (i+1);
            for (int j = 0; j < i-1; ++j) {
                matA[i][j] = {(i+j+1) * invN, (i + 1) * invN};
                matA[j][i] = {(i+j+1) * invN, (-j - 1) * invN};
            }
        }
        Matrix<complex<Real>, N, N> matX; // The result will go here.
        Vector<double, N> vecError;  // Residual errors for each column will go here.


        auto start = std::chrono::system_clock::now();

        // Solve maxA*matX = matB and store the residual errors for each column into a separate vector.
        // Note that matX is actually transpose of the inverse of matA because the results of
        // individual A*x = e problems are stored to rows of matX.
        for (unsigned i = 0; i < N; ++i) {
            matX[i] = linearSolver(matA, matB[i],
                                   SymmetricMode::NonSymmetricMatrix, // Mode of operation. A is symmetric.
                                   0.001, // Maximum allowed relative error |Ax - b| / |b| is set to 1%
                                   0,     // Maximum number of iterations (0 = default)
                                   &vecError[i]); // Store actual relative error here.
        }

        auto end = std::chrono::system_clock::now();

        // How much is the largest relative error and at which column?
        auto pMaxError = std::max_element(vecError.begin(), vecError.end());
        std::chrono::duration<double> diff = end-start;
        std::cout << "Matrix size = " << N << ", errorMax(rel) = " << *pMaxError
                  << " at column " << std::distance(vecError.begin(), pMaxError)
                  << ", Time = " << diff.count() << "\n";
        if (*pMaxError > 0.001)
            std::cout << "!NOTE! errorMax is greater than 0.001.\n";
    }
    {
        using T = double;
        std::cout << "\n*** Example ls.5 *** : Measure time vs accuracy tradeoff with a huge heap-allocated matrices.\n";
        std::cout << "                         (May take long time, will print the elapsed time after each iteration.)\n";
        const size_t M = 1000;  // Linear system size

        // Solve A*x = b
        Matrix<T> matA(M, M);
        Vector<T> vecB(M);

        // Solve a 1000-by-1000 linear system A*x = b with a constraint
        // which requires that the sum of all elements in vector x is 1.
        // Fill matrix A and vector b with random numbers.
        for (int i = 0; i < M; ++i) {
            vecB[i] = (std::rand() % 2048) - 1024;
            for (int j = 0; j < M; ++j)
                matA[i][j] = (std::rand() % 2048) - 1024;
        }

        std::cout << "Case A: Non-symmetric Linear system, size: " << matA.rows() << " x " << matA.cols() << "\n";

        // Residual error tolerances for conjugate gradient solver.
        std::vector<T> aErrorTol = {1e-2, 1e-3, 1e-6, 1e-9};
        std::vector<double> aResidual(aErrorTol.size());
        std::vector<double> aRunTime(aErrorTol.size());

        for (int iTol = 0; iTol < aErrorTol.size(); ++iTol) {
            std::cout << "Tolerance = " << aErrorTol[iTol] << ": " << std::flush;
            auto start = std::chrono::system_clock::now();
            Vector<T> vecX = linearSolver(matA, vecB, SymmetricMode::NonSymmetricMatrix,
                                          aErrorTol[iTol],
                                          0,  // Use default maximum number of iterations.
                                          &aResidual[iTol]);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end-start;
            std::cout << "relative error |Ax-b|/|b| = " << aResidual[iTol]
                      << ", time = " << diff.count() << "\n";
        }

        // Replace b with A'*b and A with A'*A to get a symmetric system.
        // It should be much faster to solve.
        vecB = multiplyTranspose(matA, vecB);
        matA = inner(matA);
        std::cout << "\nCase B: Symmetric Linear system, size: " << matA.rows() << " x " << matA.cols() << "\n";
        for (int iTol = 0; iTol < aErrorTol.size(); ++iTol) {
            std::cout << "Tolerance = " << aErrorTol[iTol] << ": " << std::flush;
            auto start = std::chrono::system_clock::now();
            Vector<T> vecX = linearSolver(matA, vecB, SymmetricMode::SymmetricMatrix,
                                          aErrorTol[iTol],
                                          0, // Use default maximum number of iterations.
                                          &aResidual[iTol]);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end-start;
            std::cout << "relative error |Ax-b|/|b| = " << aResidual[iTol]
                      << ", time = " << diff.count() << "\n";
        }
    }
}
