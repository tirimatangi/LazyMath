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
    {
        using T = double;
        std::cout << "\n*** Example cs.1 *** : Constrained solver with heap-allocated non-symmetric real matrix.\n";
        // Minimize |A*x - b| subject to C*x = d.
        // Dimensions:
        const size_t N = 6;
        const size_t M = 3;
        const size_t P = 2;

        Matrix<T> matA(N, M);
        matA = { {1,2,3},
                 {-6,-5,-4},
                 {9,7,8},
                 {1,0,0},
                 {0,1,0},
                 {0,0,1} };
        Vector<T> vecB(N);
        vecB = { -3, 2, -1, 1.85, -2.85, 0.285 };

        // ... with constraints C*x = d.
        Matrix<T> matC(P, M);
        matC = { {1, 1, 0},
                 {0, 1, 1} };
        Vector<T> vecD(P);
        vecD = { 2, -1 };

        std::cout << "Non-symmetric Linear system size: " << matA.rows() << " x " << matA.cols() << "\n";
        std::cout << "Constraint matrix size: " << matC.rows() << " x " << matC.cols() << "\n";
        double normAxMinusB = 0, normConstraintViolation = 0;
        Vector<T> vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::NonSymmetricMatrix,
                                           0, // Default relative error tolerance
                                           0, // Default maximum number of iterations
                                           &normAxMinusB, // Residual |A*x - b|
                                           &normConstraintViolation); // Residual |C*x - d|

        std::cout << "Solution x to min |Ax-b| subject to Cx=d: " << vecX << std::endl;
        std::cout << "Object function residual |Ax-b| = " << normAxMinusB
                  << ", constraint violation norm |Cx - d| = " << normConstraintViolation << "\n";
    }
    {
        using T = std::complex<double>;
        std::cout << "\n*** Example cs.2 *** : Constrained solver with stack-allocated non-symmetric complex matrices.\n";
        // Minimize |A*x - b| subject to C*x = d.
        // Dimensions:
        constexpr size_t N = 6;
        constexpr size_t M = 3;
        constexpr size_t P = 2;

        Matrix<T, N, M> matA;
        matA[0] = {1.0+7.0i, 2.0+6.0i, 3.0+5.0i};
        matA[1] = {-6.0-7.0i, -5.0-6.0i, -4.0-5.0i};
        matA[2] = {9.0-2.0i, 7.0+3.0i, 8.0+4.0i};
        matA[3] = {1,0,0};
        matA[4] = {0,1,0};
        matA[5] = {0,0,1};

        Vector<T, N> vecB;
        vecB = { -3.0+1.0i, 2.0-1.0i, -1.0+1.0i, 1.85, -2.85, 0.285 };  // dim = N

        // ... with constraints C*x = d.
        Matrix<T, P, M> matC;
        matC[0] = {1.0i, 1., 0.};
        matC[1] = {0., 1.0i, 1.};
        Vector<T, P> vecD;
        vecD = { 2.0+2.0i, -1.0-1.0i }; // dim = P

        std::cout << "Non-symmetric Linear system size: " << matA.rows() << " x " << matA.cols() << "\n";
        std::cout << "Constraint matrix size: " << matC.rows() << " x " << matC.cols() << "\n";
        // Solve and request residual errors of the object function and the constraint.
        double normAxMinusB = 0, normConstraintViolation = 0;
        Vector<T, M> vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::NonSymmetricMatrix,
                                              0, // Default relative error tolerance
                                              0, // Default maximum number of iterations
                                              &normAxMinusB, // Residual |A*x - b|
                                              &normConstraintViolation); // Residual |C*x - d|
        std::cout << "Solution x to min |Ax-b| subject to Cx=d  (real,img): " << vecX << std::endl;
        std::cout << "Object function residual |Ax-b| = " << normAxMinusB
                  << ", constraint violation norm |Cx - d| = " << normConstraintViolation << "\n";
    }
    {
        using T = double;
        std::cout << "\n*** Example cs.3 *** : Constrained solver with stack-allocated real symmetric matrix.\n";
        // Minimize |A*x - b| subject to C*x = d.
        // Dimensions:
        constexpr size_t N = 3;
        constexpr size_t M = 3;
        constexpr size_t P = 2;

        Matrix<T, N, M> matA;
        matA[0] = {119, 95, 99};
        matA[1] = {95, 79, 82};
        matA[2] = {99, 82, 90};
        Vector<T, N> vecB;
        vecB = { 1, 2, 3 };

        // ... with constraints C*x = d.
        Matrix<T, P, M> matC;
        matC[0] = {1, 0, 0};
        matC[1] = {0, 1, 0};
        Vector<T, P> vecD;
        vecD = { 2, -1 };

        std::cout << "Symmetric Linear system size: " << matA.rows() << " x " << matA.cols() << "\n";
        std::cout << "Constraint matrix size: " << matC.rows() << " x " << matC.cols() << "\n";
        // Solve and request residual errors of the object function and the constraint.
        double normAxMinusB = 0, normConstraintViolation = 0;
        Vector<T, M> vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::SymmetricMatrix,
                                              0, // Default relative error tolerance
                                              0, // Default maximum number of iterations
                                              &normAxMinusB, // Residual |A*x - b|
                                              &normConstraintViolation); // Residual |C*x - d|
        std::cout << "Solution x to min |Ax-b| subject to Cx=d: " << vecX << std::endl;
        std::cout << "Object function residual |Ax-b| = " << normAxMinusB
                  << ", constraint violation norm |Cx - d| = " << normConstraintViolation << "\n";
    }
    {
        using T = std::complex<double>;
        std::cout << "\n*** Example cs.4 *** : Constrained solver with heap-allocated complex symmetric matrix.\n";
        // Minimize |A*x - b| subject to C*x = d.
        // Dimensions:
        const size_t N = 3;
        const size_t M = 3;
        const size_t P = 2;

        Matrix<T> matA(N, M);
        matA[0] = {221.0, 173.0+34.0i, 161.0+38.0i};
        matA[1] = {173.0-34.0i, 160.0, 154.0-3.0i};
        matA[2] = {161.0-38.0i, 154.0+3.0i, 156.0};

        Vector<T> vecB(M);
        vecB = { 1.0+1.0i, 2.0-1.0i, 3.0+1.0i };

        // ... with constraints C*x = d.
        Matrix<T> matC(P, M);
        matC[0] = {1.0i, 0., 0.};
        matC[1] = {0., 1.0i, 0.};
        Vector<T> vecD(P);
        vecD = { 2.0+2.0i, -1.0-1.0i };

        std::cout << "Non-symmetric Linear system size: " << matA.rows() << " x " << matA.cols() << "\n";
        std::cout << "Constraint matrix size: " << matC.rows() << " x " << matC.cols() << "\n";
        // Solve and request residual errors of the object function and the constraint.
        double normAxMinusB = 0, normConstraintViolation = 0;
        Vector<T> vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::SymmetricMatrix,
                                              0, // Default relative error tolerance
                                              0, // Default maximum number of iterations
                                              &normAxMinusB, // Residual |A*x - b|
                                              &normConstraintViolation); // Residual |C*x - d|
        std::cout << "Solution x to min |Ax-b| subject to Cx=d (real, imag): " << vecX << std::endl;
        std::cout << "Object function residual |Ax-b| = " << normAxMinusB
                  << ", constraint violation norm |Cx - d| = " << normConstraintViolation << "\n";
    }
    {
        using T = double;
        std::cout << "\n*** Example cs.5 *** : Measure time vs accuracy tradeoff with a huge heap-allocated matrices.\n";
        std::cout << "                         (May take a while. Prints elapsed time for each relative error tolerance.)\n";
        const size_t M = 1000;  // Linear system size
        const size_t P = 1;     // Number of constraints

        // Solve A*x = b ....
        Matrix<T> matA(M, M);
        Vector<T> vecB(M);
        // ... with constraints C*x = d.
        Matrix<T> matC(P, M);
        Vector<T> vecD(P);

        // Solve a 1000-by-1000 linear system A*x = b with a constraint
        // which requires that the sum of all elements in vector x is 1.
        // Fill matrix A and vector b with random numbers.
        for (int i = 0; i < M; ++i) {
            vecB[i] = (std::rand() % 2048) - 1024;
            for (int j = 0; j < M; ++j)
                matA[i][j] = (std::rand() % 2048) - 1024;
        }

        // Set constraint C = [1, 1, ..., 1] and d = 1, so C*x = 1.
        // So the sum of the elements of x should be 1.
        for (int i = 0; i < M; ++i)
            matC[0][i] = 1;
        vecD[0] = 1;

        std::cout << "Case A: Non-symmetric Linear system, size: " << matA.rows() << " x " << matA.cols() << "\n";
        std::cout << "        Constraint: sum of elements of x should be 1\n";

        // Residual error tolerances for conjugate gradient solver.
        std::vector<T> aErrorTol = {1e-2, 1e-3, 1e-6, 1e-9};
        std::vector<double> aResidual(aErrorTol.size());
        std::vector<double> aConstraintViolation(aErrorTol.size());
        std::vector<double> aRunTime(aErrorTol.size());

        for (int iTol = 0; iTol < aErrorTol.size(); ++iTol) {
            std::cout << "Tolerance = " << aErrorTol[iTol] << ": " << std::flush;
            auto start = std::chrono::system_clock::now();
            Vector<T> vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::NonSymmetricMatrix,
                                               aErrorTol[iTol],
                                               0, // Use default maximum number of iterations.
                                               &aResidual[iTol],
                                               &aConstraintViolation[iTol]);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end-start;

            // Double check: the sum of all elements in X should be 1:
            double sum = 0;
            for (auto x : vecX)
                sum += x;

            std::cout << "residual |Ax-b| = " << aResidual[iTol]
            << ", constraint violation |Cx-d| = " << aConstraintViolation[iTol]
            << ", sum of elements of x = " << sum
            << ", time = " << diff.count() << "\n";
        }

        // Replace b with A'*b and A with A'*A to get a symmetric system.
        // It should be much faster to solve.
        vecB = multiplyTranspose(matA, vecB);
        matA = inner(matA);
        std::cout << "\nCase B: Symmetric Linear system, size: " << matA.rows() << " x " << matA.cols() << "\n";
        std::cout << "        Constraint: sum of elements of x should be 1\n";
        for (int iTol = 0; iTol < aErrorTol.size(); ++iTol) {
            std::cout << "Tolerance = " << aErrorTol[iTol] << ": " << std::flush;
            auto start = std::chrono::system_clock::now();
            Vector<T> vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::SymmetricMatrix,
                                               aErrorTol[iTol],
                                               0, // Use default maximum number of iterations.
                                               &aResidual[iTol],
                                               &aConstraintViolation[iTol]);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end-start;
            // Double check: the sum of all elements in X should be 1:
            double sum = 0;
            for (auto x : vecX)
                sum += x;
            std::cout << "residual |Ax-b| = " << aResidual[iTol]
            << ", constraint violation |Cx-d| = " << aConstraintViolation[iTol]
            << ", sum of elements of x = " << sum
            << ", time = " << diff.count() << "\n";
        }
    }
}
