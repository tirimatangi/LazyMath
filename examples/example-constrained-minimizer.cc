#include <iostream>
#include <iomanip>
#include <chrono>
#include <LazyMath/Optimization.h>

using std::cout;
using namespace LazyMath;


int main()
{
    using T = double;
    // Let us first define the problem to be solved.
    // The object function gives the squared distance of a 3D point to
    // the center of a sphere.
    Vector<T, 3> centerPoint; // Could also be: Vector<T> centerPoint(3);
    centerPoint = {0, 0, 0};
    // The constraints define two planes which are prependicular to the given
    // 3-dimensional direction vectors.
    Matrix<T,2,3> constraintDirection; // Could also be: Matrix<T> constraintDirection(2,3);
    constraintDirection[0] = {1, 0, 0};
    constraintDirection[1] = {0, 1/sqrt(2.0), 1/sqrt(2.0)};
    // These are the distances of the planes from the origin.
    Vector<T,2> constaintDistance;
    constaintDistance = {1, -1};

    auto myFunction = [centerPoint](const auto& x, auto& fx) { // dim x = N, dim fx = M
        fx[0]= 0;
        for (int n = 0; n < x.size(); ++n)
            fx[0] += ((x[n]-centerPoint[n]) * (x[n]-centerPoint[n]));
    };

    auto myConstraint = [constraintDirection, constaintDistance](const auto& x, auto& gx) { // dim x = N, dim gx = P
        for (int p = 0; p < gx.size(); ++p) {
            gx[p] = -constaintDistance[p];  // gx[p] = inner_product(x, constraintDirection[p]) - constraintDistance[p]
            for (int n = 0; n < x.size(); ++n)
                gx[p] += constraintDirection[p][n] * x[n];
        }
    };

    auto myDerivative = [centerPoint](const auto& x, auto& d) { // dim x = N, dim d = NxM
        for (int n = 0; n < x.size(); ++n)
            d[n][0] = 2 * (x[n]-centerPoint[n]);  // df/dx_n
    };

    auto myConstraintDerivative = [constraintDirection](const auto& x, auto& d) { // dim x = N, dim d = NxP
        for (int n = 0; n < d.rows(); ++n)
            for (int p = 0; p < d.cols(); ++p) {
                d[n][p] = constraintDirection[p][n];
        }
    };

    {
        cout << "*** Example constrained.1 ***: Minimize a function with constraints.\n";
        { // Case A: Dimensions are fixed at compile time so only stack is used (i.e. no memory allocation for heap).
            constexpr size_t N = 3;
            constexpr size_t M = 1;
            constexpr size_t P = 2;
            ConstrainedMinimizer<T,N,M,P> minimizer;
            // Configure the objective and constraint functions
            minimizer.function = myFunction;
            minimizer.constraintFunction = myConstraint;
            /* Uncomment the next 2 lines to use analytical derivative instead of numerical approximation. */
            // min.derivative = myDerivative;
            // min.constraintDerivative = myConstraintDerivative;
            /* Uncomment the next line to set an initial guess for the minimum  point other than zero. */
            // minimizer.initX = {1, -1, -1};
            // Run the minimizer
            minimizer.run();
            // Print the results
            cout << "Case A: minimum at x = " << minimizer.minX() << "\n";
            cout << "Case A: Objective function f(x) = " << minimizer.minFx() << "\n";
            cout << "Case A: Constraint g(x) = " << minimizer.minGx() << "  (should be nearly zero)\n";
            cout << "Case A: Iterations done: " << minimizer.iterationsDone() << "\n";
        }
        cout << '\n';
        { // Case B: Dimensions are not fixed at compile time so memory is allocated from the heap.
            size_t N = 3;
            size_t M = 1;
            size_t P = 2;
            ConstrainedMinimizer<T> minimizer(N,M,P);
            // Configure the objective and constraint functions
            minimizer.function = myFunction;
            minimizer.constraintFunction = myConstraint;
            // Comment out the next 2 lines to use numerical derivative if the analytical expression is not available.
            minimizer.derivative = myDerivative;
            minimizer.constraintDerivative = myConstraintDerivative;
            // Comment out the next line if you don't want to set initial guess for the minimum point.
            minimizer.initX = {2, -2, -2};
            // Run the minimizer
            minimizer.run();
            // Print the results
            cout << "Case B: minimum at x = " << minimizer.minX() << "\n";
            cout << "Case B: Objective function f(x) = " << minimizer.minFx() << "\n";
            cout << "Case B: Constraint g(x) = " << minimizer.minGx() << "  (should be nearly zero)\n";
            cout << "Case B: Iterations done: " << minimizer.iterationsDone() << "\n";

        }
        cout << '\n';
        { // Case C: Example of optional configuration parameters given to the minimizer
          //         to get potentially faster convergence.
          //         After the minimization, read out information about the optimality of the solution.
            size_t N = 3;
            size_t M = 1;
            size_t P = 2;
            ConstrainedMinimizer<T> minimizer(N,M,P);
            // Configure the objective and constraint functions. These 2 are the only mandatory parameters.
            minimizer.function = myFunction;
            minimizer.constraintFunction = myConstraint;

            // Assume that the analytical expression for the derivative of the object function is not known,
            // so numerical approximation will be used.
            /* minimizer.derivative = myDerivative; */
            // Assume that the analytical expression for the derivative of the constraint function is known.
            minimizer.constraintDerivative = myConstraintDerivative;
            // Initial guess for the minimum point.
            minimizer.initX = {0.5, 0, -0.5};
            // Maximum number of iterations
            minimizer.maxIterations = 200;
            // Error tolerance for linear solver in Levenberg-Marquardt iteration.
            // The smaller the value, the slower each iteration will be but on the other hand
            // the convergence may be faster. The optimal value depends on the problem so must measure.
            minimizer.errorTolerance = 1e-7;
            // Criteria for stopping the iteration.
            // The solution is deemed as good enough if the squared norm of vector
            // d/dx(0.5*|h(x)|^2) = Dh(x) * h(x) becomes smaller than this value,
            // where h(x) is the object function plus penalty term times the constraint function.
            // The best value depends on the problem. If the value is too small,
            // the number of iterations may be quite big (but always less than maxIterations).
            // If the value is too large, the minimizer gives up too early and gives a wrong answer.
            minimizer.optimalityLimit = 1e-5;
            // Difference values for each N coordinates of input vector x.
            // The differences are used in numerical estimation of Jacobian matrix.
            // Should as as small as possible. However,
            // the closer to zero the expected partial derivates of the object and constraint functions are,
            // the larger the difference should be.
            // This parameter is ignored if the analytical derivates are known for both object and constraint functions.
            minimizer.difference = {0.0001, 0.0001, 0.0001};
            // Booster coefficient for the diagonal load for the linear solver.
            // The larger the boost, the more stable (i.e. less prone to oscillation)
            // the convergence will be but on the other hand
            // the final point may be less optimal. The default value is 1/16.
            minimizer.diagonalBoost = 0.25;
            // Number of successive iterations the minimizer tries to improve the result
            // before giving up. The default value is 8.
            minimizer.maxNonProgressCount = 10;

            // Run the minimizer
            minimizer.run();
            // Print the results
            cout << "Case C: minimum at x = " << minimizer.minX() << "\n";
            cout << "Case C: Objective function f(x) = " << minimizer.minFx() << "\n";
            cout << "Case C: Constraint g(x) = " << minimizer.minGx() << "  (should be nearly zero)\n";

            if (minimizer.derivative)
                cout << "Case C: Used analytical object function derivative.\n";
            else
                cout << "Case C: Used numerical object derivative.\n";

            if (minimizer.constraintDerivative)
                cout << "Case C: Used analytical constraint derivative.\n";
            else
                cout << "Case C: Used numerical constraint derivative.\n";
            // Lagrange coefficients at the optimal point. Rarely useful.
            cout << "Case C: Lagrange Z = " << minimizer.lagrangeZ() << "\n";
            // Acquired optimality condition at the minimum point.
            // See 'minimizer.optimalityLimit' above.
            // If the requested optimality limit is not reached,
            // maxIterations parameter may have to be increased.
            cout << "Case C: Optimality = " << minimizer.optimalityValue() << "\n";
            // Final penalty factor for the constraints. Should not be ridiculously high
            // if everything went fine. Depends on the problem.
            cout << "Case C: Penalty = " << minimizer.penalty() << "\n";
            // Total number of iterations done.
            cout << "Case C: Iterations done: " << minimizer.iterationsDone() << "\n";
        }
    }
    cout << '\n';
    {
        cout << "*** Example constraint.2 ***: Catch an exception if either the object or constraint functions are not set.\n";
        try {  // Both functions missing.
            ConstrainedMinimizer<double> minimizer(3, 1, 2);
            minimizer.run();
        }
        catch (const std::runtime_error& e)
        {
            cout << "--> Exception:" << e.what() << "\n";
        }

        try { // Constraint function missing.
            ConstrainedMinimizer<double> minimizer(3, 1, 2);
            minimizer.function = [](const auto& x, auto& fx) { fx[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]; };
            minimizer.run();
        }
        catch (const std::runtime_error& e)
        {
            cout << "--> Exception:" << e.what() << "\n";
        }
    }
}
