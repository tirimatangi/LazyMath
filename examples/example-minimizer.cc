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
    // The point at which the minimum value is attained.
    T minPoint[3] = {2, 3, 4};
    // The value of the function at the minimum point.
    T minValue[2] = {1, 2};

    // Define the object function to be minimized.
    // Calculate y = f(x) where x is an N-dimensional vector and y is an M-dimensional vector.
    // In this case N = 3 and M = 2.
    auto myFunction = [minPoint, minValue](const auto& x, auto& y) {
        double squares[3] = { };
        for (int i = 0; i < x.size(); ++i)  // Assume that the size of vector x = N = 3
            squares[i] = (x[i]-minPoint[i]) * (x[i]-minPoint[i]);

        y[0] =     10 * squares[0] + 11 * squares[1] + 12 * squares[2]      + minValue[0];
        y[1] = log(20 * squares[0] + 21 * squares[1] + 22 * squares[2] + 1) + minValue[1];
    };

    // Define jacobian matrix for the object function.
    // x is an N-dimensional vector and jacobian is an N-by-M matrix.
    // This can be skipped if the jacobian is not known.
    // In that case, it will automatically approximated numerically.
    auto myJacobian = [minPoint](const auto& x, auto& jacobian) {
        double squares[3] = { };
        for (int i = 0; i < x.size(); ++i)  // Assume that the size of vector x = N = 3.
            squares[i] = (x[i]-minPoint[i]) * (x[i]-minPoint[i]);
        double y1 = 13 * squares[0] + 14 * squares[1] + 15 * squares[2] + 1;

        jacobian[0][0] = 10*2*(x[0]-minPoint[0]);
        jacobian[0][1] = 20*2*(x[0]-minPoint[0]) / y1;

        jacobian[1][0] = 11*2*(x[1]-minPoint[1]);
        jacobian[1][1] = 21*2*(x[1]-minPoint[1]) / y1;

        jacobian[2][0] = 12*2*(x[2]-minPoint[2]);
        jacobian[2][1] = 22*2*(x[2]-minPoint[2]) / y1;
    };

    {
        cout << "*** Example min.1 ***: Minimize a function using default configuration.\n";
        // In this example the minimizer uses default initial values and parameters.
        // Analytical jacobian is used if minimizer.derivative is set. Otherwise jacobian is
        // approximated numerically with difference.
        { // Case A: Dimensions are fixed at compile time so only stack is used (i.e. no memory allocation for heap).
            Minimizer<T,3,2> minimizer;
            minimizer.function = myFunction;
            minimizer.derivative = myJacobian;
            minimizer.run();
            // Get the results
            cout << " A: minimum at x = " << minimizer.minX() << std::endl;
            cout << " A: value f(x) = " << minimizer.minFx() << std::endl;
            cout << std::endl;
        }
        { // Case B: Dimensions are not fixed at compile time so memory is allocated from the heap.
            int inputDimension = 3, outputDimension = 2;
            Minimizer<T> minimizer(inputDimension, outputDimension);
            minimizer.function = myFunction;
            // minimizer.derivative = myJacobian; // Uncomment to use analytical jacobian.
            minimizer.run();
            // Get the results
            cout << " B: minimum at x = " << minimizer.minX() << std::endl;
            cout << " B: value f(x) = " << minimizer.minFx() << std::endl;
            cout << std::endl;
        }
    }
    {
        cout << "*** Example min.2 ***: Minimize a function using manual configuration.\n";
        // This example demonstrates how you can set initial values and parameter manually
        // and get information on the optimality of the solution after the run.
        Minimizer<T> minimizer(3, 2); // or "Minimizer<T, 3, 2> minimizer;" to avoid heap allocations
        minimizer.function = myFunction;
        // minimizer.derivative = myJacobian; // Uncomment to use analytical jacobian.

        // The following parameters can be set to improve convergence and accuracy:
        // Initial guess for the minimum point
        minimizer.initX = {1,2,3};
        // Maximum number of iterations
        minimizer.maxIterations = 100;
        // Error tolerance for linear solver in Levenberg-Marquardt iteration.
        // The smaller the value, the slower each iteration will be but on the other hand
        // the convergence may be faster. The optimal value depends on the problem so must measure.
        minimizer.errorTolerance = 1e-6;
        // Criteria for stopping the iteration.
        // The solution is deemed as good enough if the squared norm of vector
        // d/dx(0.5*|f(x)|^2) = Df(x) * f(x) becomes smaller than this value.
        // The best value depends on the problem. If the value is too small,
        // the number of iterations may be quite big (but always less than maxIterations).
        // If the value is too large, the minimizer gives up too early and gives a wrong answer.
        minimizer.optimalityLimit = 1e-6;
        // Difference values for each N coordinates of input vector x.
        // The differences are used in numerical estimation of Jacobian matrix.
        // Should as as small as possible. However,
        // the closer to zero the expected partial derivates of the object function are,
        // the larger the difference should be.
        // This parameter is ignored if analytical derivate is set.
        minimizer.difference = {0.0001, 0.0001, 0.0001};
        // Estimate for the initial diagonal load in linear solver.
        // Should be as small as possible but not any smaller, so it depends
        // on the problem. Typically this should not be set manually.
        minimizer.lambda = 1;
        // Number of successive iterations the minimizer tries to improve the result
        // before giving up.
        minimizer.maxNonProgressCount = 10;
        // Run the minimizer
        minimizer.run();

        // Print the result and various diagnostics.
        if (minimizer.derivative)
            cout << "Analytical derivative was used.\n";
        else
            cout << "Numerical derivative was used.\n";

        // The minimum value was found at this point
        cout << "Min2: minimum at x = " << minimizer.minX() << std::endl;
        // The value of the object function at the minimum point
        cout << "Min2: value f(x) = " << minimizer.minFx() << std::endl;
        // Norm of the object function at the minimum point.
        cout << "Min2: norm(f(x)) = " << minimizer.minFxNorm() << std::endl;
        // Optimality condition (i.e. squared norm of Df(x) * f(x)) at the minimum point.
        // See 'minimizer.optimalityLimit' above,
        cout << "Min2: Optimality = " << minimizer.optimalityValue() << std::endl;
        // Number of iterations done
        cout << "Iterations done: " << minimizer.iterationsDone() << std::endl;
        // Number of iterations without progress at the end of the run.
        cout << "nonProgressCount = " << minimizer.nonProgressCount() << std::endl;
        cout << std::endl;
    }
    {
        cout << "*** Example min.3 ***: Catch an exception if the object function is not set.\n";
        try {
            Minimizer<double> minimizer(3, 2);
            minimizer.run();
        }
        catch (const std::runtime_error& e)
        {
            cout << "--> Exception: " << e.what() << "\n";
        }
    }
    cout << std::endl;
    {
        cout << "*** Example min.4 ***: Catch an exception if the difference in numerical differentiation is too small.\n";
        try {
            double a = 20, b = 30;
            auto myFunc = [&](const auto& x, auto& fx) {
                fx[0] = (x[0]-a)*(x[0]-a) + (x[1]-b)*(x[1]-b);
            };

            Minimizer<double> minimizer(2, 1);
            minimizer.function = myFunc;

            // Run the minimizer with sensible differences
            minimizer.initX = {10, 10};
            minimizer.difference = {1e-5, 1e-5};
            cout << "Difference vector set to " << minimizer.difference << std::endl;
            minimizer.run();
            // The minimum value was found at this point
            cout << "Min4: minimum at x = " << minimizer.minX() << std::endl;
            // The value of the object function at the minimum point
            cout << "Min4: value f(x) = " << minimizer.minFx() << std::endl;

            // Re-run the minimizer with too small differences. An exception will be thrown.
            minimizer.initX = {10, 10};
            minimizer.difference = {1e-15, 1e-5};
            cout << "Difference vector set to " << minimizer.difference << std::endl;
            minimizer.run();
        }
        catch (const std::runtime_error& e)
        {
            cout << "--> Exception: " << e.what() << "\n";
        }
    }
}
