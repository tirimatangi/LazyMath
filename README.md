
# Complex linear solver and function minimizer in C++

**LazyMath** is a header-only math library written in C++17. The features are

* Linear solver using conjugate gradient algorithm. Both real and complex matrices are supported. Also linear constraints can be given.
* Function minimizer using Levenberg-Marquardt algorithm with or without constraints.

## Examples

Below are examples of a few simple use cases. There are more examples in [examples](https://github.com/tirimatangi/LazyMath/tree/main/examples) folder.

### 1. Linear solver with real and complex matrices

The solver calculates vector _x_ which minimizes &Vert;**A**_x_ &minus; **b**&Vert;<sup>2</sup>.

The dimensions of matrix A and vector b can be either dynamic in which case they are allocated from the heap, or they can be fixed and hence allocated from the stack.

```c++
#include <LazyMath/ConjugateGradient.h>
using namespace LazyMath;
int main()
{
    { // Non-symmetric heap-allocated 3x2 real matrix
      using T = double;
      Matrix<T> matA(3, 2);
      matA = { {1, 2},
               {3, 4},
               {5, 6} };
      Vector<T> vecB(3);
      vecB = { 10, 20, 30 }; // b is a vector of length 3.
      auto vecX = linearSolver(matA, vecB, SymmetricMode::NonSymmetricMatrix);
      std::cout << "X = " << vecX << "\n"; // The answer is x=[0 5]
    }
    { // Symmetric and positive definite stack-allocated 2x2 complex matrix
      using T = std::complex<double>;
      Matrix<T, 2, 2> matA;
      matA[0] = {28.+0i,  10.+30i};
      matA[1] = {10.-30i, 70.+0i};
      Vector<T, 2> vecB;
      vecB = { -100.+100i, 140.+280i }; // b is a complex vector of length 2.
      auto vecX = linearSolver(matA, vecB, SymmetricMode::SymmetricMatrix);
      std::cout << "X = " << vecX << "\n"; // The answer is x=[0 2+4i]
    }
}
```

If the matrix is known to be symmetric and positive definite like in the second example, the solver can be configured to symmetric mode. It is faster and more accurate than the more general non-symmetric mode. However, if the symmetric mode is configured with a non-symmetric matrix, an `std::runtime_error` will be thrown.

The default matrix type defined in [Matrix.h](https://github.com/tirimatangi/LazyMath/blob/main/include/LazyMath/Matrix.h) is simply a vector of vectors. A vector is either an `std::array` or an `std::vector`, depending on whether the size is given as a template argument or as a constructor argument. Both cases are demonstrated above.

Conjugate gradient method lets you trade accuracy for speed. Conjugate gradient iterations continue only until the requested relative error is reached, or the maximum number of iterations is done. The relative error is defined as &Vert;**A**_x_ &minus; **b**&Vert; / &Vert;**b**&Vert;. The error tolerance and maximum number of iterations can be given in 4th and 5th arguments to `LazyMath::linearSolver`. Otherwise, default values are determined by the algorithm.

For much more examples on linear solver, see [example-linear-solver.cc](https://github.com/tirimatangi/LazyMath/blob/main/examples/example-linear-solver.cc).
Examples `ls.3` and `ls.4` invert large matrices and measure running times.
The last example (`ls.5`) shows how the relative error tolerance is configured and measures how changing the tolerance affects the running time. Notice that the solver writes the residual relative error to the variable pointed by the 6th argument so the quality of the result can be monitored.

### 2. Linear solver with linear constraints

Linear constraints can be added to the solver. In this case, the solver calculates vector x which minimizes &Vert;**A**_x_ &minus; **b**&Vert;<sup>2</sup> subject to constraints **C**_x_ = **d**. If the dimension of matrix A is N-by-M, there can be P=1...N-1 constraints. Hence, the dimension of constraint matrix **C** is P-by-M.

Here is a simple example where the a linear problem is solved first without constraints and then with a constraint.

```c++
#include <LazyMath/ConjugateGradient.h>
using namespace LazyMath;
int main()
{
    { //
        using T = std::complex<double>;
        // Solve complex linear system A*x = b
        Matrix<T> matA { {1.+5i, 2.+4i},
                         {3.+3i, 3.-3i},
                         {4.-2i, 5.-1i} };
        Vector<T> vecB { 8.+10i, -6i, 14.-10i };
        auto vecX = linearSolver(matA, vecB, SymmetricMode::NonSymmetricMatrix);
        std::cout << "Unconstrained X = " <<  vecX << "\n";
        // Unconstrained solution: X = [ 1 + 1i, 2 - 2i ]

        // Add complex constraint 0.5 * X[0] + 0.5 * X[1] = i
        // So minimize |A*x - b| subject to C*x = d
        Matrix<T> matC { { 0.5, 0.5 } };
        Vector<T> vecD { 1i };
        vecX = constrainedSolver(matA, vecB, matC, vecD, SymmetricMode::NonSymmetricMatrix);
        std::cout << "Constrained X = " <<  vecX << "\n";
        // Constrained solution: X = [ -1.85 + 1.15i, 1.85 + 0.85i ]
        // Notice that x[0]+x[1]=2i as required by the constraint.
    }
}
```

For much more examples on constrained linear solver, see [example-constrained-linear-solver.cc](https://github.com/tirimatangi/LazyMath/blob/main/examples/example-constrained-linear-solver.cc).
Again the last example (`ls.5`) shows how to configure time vs accuracy trade-off and measures running times with difference accuracy requirements. It also demonstrates how much faster it is to work with symmetric matrices compared to general matrices. So if you know that your matrix A is symmetric and positive definite, remember to set the mode as `SymmetricMode::SymmetricMatrix`.

### 3. Function minimizer

The problem is to find point x which minimizes the squared norm &Vert;f(x)&Vert;<sup>2</sup> of the given function f: **R**<sup>_N_</sup>**&rarr;R**<sup>_M_</sup>. So the object function f(x) maps a point from an _N_ dimensional space to an _M_ dimensional space.

Here is an example of a simple use case where only the object function is specified. In this example the object function is a lambda function but it can also be any callable object like a functor or a function pointer.

```c++
#include <LazyMath/Optimization.h>
using namespace LazyMath;

int main()
{
    // Define an object function with input dimension=3, output dimension=2.
    // The minimum of |f(x)| is found at x=[1/3, 1/3, 0].
    // f(x) at the minimum point is [sqrt(2/9), 1/3].
    auto myFunc = [](const auto& x, auto& fx) {
        fx[0] = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        fx[1] = (x[1] - x[0]) + 1;
    };

    // Run the minimizer
    Minimizer<double> minimizer(3, 2); // input dimension=3, output dimension=2.
    minimizer.function = myFunc;
    minimizer.run();

    // Get the result vectors.
    auto x = minimizer.minX();
    std::cout << "Minimum at x = " << x << "\n";
    auto fx = minimizer.minFx();
    std::cout << "f(x) = " << fx << "\n";
}
```

If the derivative (aka jacobian) of the object function is known, it should be set to minimizer's `derivative` member function. The input to the derivative function is an N-dimensional point _x_ and the output is an N-by-M dimensional jacobian matrix.
This is demonstrated below by minimizing function f(x<sub>0</sub>,x<sub>1</sub>)=exp((x<sub>0</sub>&minus;a)<sup>2</sup> + (x<sub>1</sub>&minus;b)<sup>2</sup>) whose jacobian is <br>
[2 (x<sub>0</sub>&minus;a) f(x<sub>0</sub>,x<sub>1</sub>) ; 2 (x<sub>1</sub>&minus;b) f(x<sub>0</sub>,x<sub>1</sub>)]

```c++
#include <LazyMath/Optimization.h>
using namespace LazyMath;

int main()
{
    // Define the object function f(x0,x1) = exp((x0-a)^2+(x1-b)^2) and its jacobian.
    // The minimum will be at x=(a,b)
    double a = 2, b = 3;
    auto myExp = [](double x, double c){ return exp((x-c)*(x-c)); };
    auto myFunc = [&](const auto& x, auto& fx) {
        fx[0] = myExp(x[0],a) * myExp(x[1],b);
    };
    auto myJacobian = [&](const auto& x, auto& jac) {
        jac[0][0] = 2 * (x[0]-a) * myExp(x[0],a) * myExp(x[1],b);
        jac[1][0] = 2 * (x[1]-b) * myExp(x[0],a) * myExp(x[1],b);
    };

    // Run the minimizer
    Minimizer<double> minimizer(2, 1);
    minimizer.function = myFunc;
    minimizer.derivative = myJacobian;
    minimizer.run();

    // Get the result vectors.
    auto x = minimizer.minX();
    std::cout << "Minimum at x = " << x << "\n";
    auto fx = minimizer.minFx();
    std::cout << "f(x) = " << fx << "\n";
    std::cout << "Optimality = " << minimizer.optimalityValue() << "\n";
    std::cout << "Iterations done: " << minimizer.iterationsDone() << "\n";
}
```
The last two lines print the optimality score of the found minimum point and the number of iterations needed to reach it.

The optimality score is the squared norm of D(f)*f(x), that is, the jacobian matrix multiplied by the object function value at the minimal point. It should be as close to zero as possible. You can trade speed for accuracy by setting member variable `optimalityLimit` to a positive value and testing how it affacts the iteration count and accuracy of the solution. For example, setting `minimizer.optimalityLimit = 1e-3;` in the above example reduces the iterations from 87 to 75. The upper limit to the iteration count can be set with member variable `maxIterations`.

Furthermore, if you have an educated guess of the location of the minimum point, you can tell it to the minimizer by setting member variable `initX`. For example, setting `minimizer.initX = {1, 2};` in the above example reduces the iterations from 87 to 20 because {1, 2} is closer to the minimum point {3, 2} than the default initial point {0, 0} is.

If the jacobian is not known, it is estimated numerically using differences. The difference sizes for each partial derivative can be set separately if you know that the rate of change of the object function varies in different directions. For example, by setting `minimizer.difference = {1e-3, 1e-6};` estimates the first partial derivative using difference 1/1000 and the second partial derivate using difference1/1000000.

The minimizer can be defined in two ways depending on whether it should use heap or stack for storing the results and internal data.
Stack can be used if the dimensions of the object function are known at compile time. If the dimensions are not known until the minimizer is initialized, the heap version must be used. The above examples use the heap version (e.g. `Minimizer<double> minimizer(2, 1);`). Had the stack version been used, the definition would simply have been `Minimizer<double, 2, 1> minimizer;`. So use initializer arguments for heap and template arguments for stack.

For much more examples on function minimizer, see [example-minimizer.cc](https://github.com/tirimatangi/LazyMath/blob/main/examples/example-minimizer.cc). Especially example `min.2` is important because it shows how to configure several optional member variables to improve the speed of the iteration and/or accuracy of the result.

Examples `min.3` and `min.4` demonstrate errors when an exception will be thrown.


### 4. Function minimizer with constraints

Constraints can be added to the minimizer.
In the constrained case, the problem is to find point x which minimizes the squared norm &Vert;_f_(x)&Vert;<sup>2</sup> subject to constraint _g_(x) = 0. An before, object function _f_: **R**<sup>_N_</sup>**&rarr;R**<sup>_M_</sup>. The dimension of constraint function _g_(x) is P, so there are P constraints and hence _g_: **R**<sup>_N_</sup>**&rarr;R**<sup>_P_</sup>.

Here is an example of a simple use case where the object and constraint functions are specified.

```c++
#include <LazyMath/Optimization.h>
using namespace LazyMath;

int main()
{
    // Define an object function with input dimension N=3, output dimension M=1, constraint dimension P=2.
    // The minimum of |f(x)| subject to constraints is found at x=[1/2, 0, -1/2].
    // f(x) at the minimum point is 1/sqrt(2).
    auto myFunc = [](const auto& x, auto& fx) {
        fx[0] = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    };

    // Define the set on constraints in form g(x) = 0
    auto myConstraint = [](const auto& x, auto& gx) {
        gx[0] = (x[0] - x[1] - x[2]) - 1; // 1. Constraint: x[0] - x[1] - x[2] = 1
        gx[1] = (x[0] + x[1] - x[2]) - 1; // 2. Constraint: x[0] + x[1] - x[2] = 1
    };
    // Run the minimizer
    ConstrainedMinimizer<double> minimizer(3, 1, 2); // input dim=3, output dim=1, constraint dim=2
    minimizer.function = myFunc;
    minimizer.constraintFunction = myConstraint;
    minimizer.run();

    // Get the results.
    auto x = minimizer.minX();
    std::cout << "Minimum at x = " << x << "\n";
    auto fx = minimizer.minFx();
    std::cout << "Object function: f(x) = " << fx <<"\n";
    auto gx = minimizer.minGx();
    std::cout << "Constraint: g(x) = " << gx << "\n";
}
```

If the derivatives of the object and/or constraint functions are known, they should be set to minimizer's `derivative` and `constraintDerivative` member functions, respectively.

The input to the derivative of the object function is an N-dimensional point _x_ and the output is an N-by-M_dimensional jacobian matrix.
The input to the derivative of the constraint function is an N-dimensional point _x_ and the output is an P-by-M_dimensional jacobian matrix.

This is demonstrated below by minimizing function f(x<sub>0</sub>,x<sub>1</sub>)=exp((x<sub>0</sub>&minus;a)<sup>2</sup> + (x<sub>1</sub>&minus;b)<sup>2</sup>) whose jacobian is <br>
[2 (x<sub>0</sub>&minus;a) f(x<sub>0</sub>,x<sub>1</sub>) ; 2 (x<sub>1</sub>&minus;b) f(x<sub>0</sub>,x<sub>1</sub>)], subject to affine constraint a x<sub>0</sub> + b x<sub>1</sub> = (a<sup>2</sup> + b<sup>2</sup>)/2 whose jacobian matrix is simply [a; b] for all x.

```c++
#include <LazyMath/Optimization.h>
using namespace LazyMath;

int main()
{
    // Minimize f(x0,x1) = exp((x0-a)^2+(x1-b)^2) subject to |x0*a + x1*b| = (a^2+b^2)/2
    // The minimum will be at x=(a/2,b/2)
    double a = 2, b = 3, l = a*a + b*b;
    auto myExp = [](double x, double c){ return exp((x-c)*(x-c)); };
    auto myFunc = [&](const auto& x, auto& fx) {
        fx[0] = myExp(x[0],a) * myExp(x[1],b);
    };
    auto myConstraint = [&](const auto& x, auto& gx) {
        gx[0] = a*x[0] + b*x[1] - l/2;
    };

    auto myJacobianFunc = [&](const auto& x, auto& jac) {
        jac[0][0] = 2 * (x[0]-a) * myExp(x[0],a) * myExp(x[1],b);
        jac[1][0] = 2 * (x[1]-b) * myExp(x[0],a) * myExp(x[1],b);
    };

    auto myJacobianConstraint = [&](const auto& x, auto& jac) {
        jac[0][0] = a;
        jac[1][0] = b;
    };

    // Run the minimizer
    ConstrainedMinimizer<double> minimizer(2, 1, 1); // input dim=2, output dim=1, constraint dim=1
    minimizer.function = myFunc;
    minimizer.constraintFunction = myConstraint;
    minimizer.derivative = myJacobianFunc;
    minimizer.constraintDerivative = myJacobianConstraint;
    minimizer.optimalityLimit = 0.001;
    minimizer.run();

    // Get the results.
    auto x = minimizer.minX();
    std::cout << "Minimum at x = " << x << "\n";
    auto fx = minimizer.minFx();
    std::cout << "Object function: f(x) = " << fx <<"\n";
    auto gx = minimizer.minGx();
    std::cout << "Constraint: g(x) = " << gx << "\n";
    std::cout << "Optimality = " << minimizer.optimalityValue() << "\n";
    std::cout << "OptimalityLimit = " << minimizer.optimalityLimit << "\n";
    std::cout << "Penalty = " << minimizer.penalty() << "\n";
    std::cout << "Iterations done: " << minimizer.iterationsDone() << "\n";

```
Notice that `minimizer.optimalityLimit` has been manually initialized to a rather large value of 0.001. This reduces the number of iterations and helps avoid divergence of the iteration. In practise you must experiment with a few values to find a good one. Also the penalty factor related to maintaining the constraints is reported. If it explodes into millions, the optimality limit may be too small.
As before, you can set the initial guess of the optimal point (e.g. `minimizer.initX = {a, b};`) and many more optional parameters.

For much more examples on function minimizer, see [example-constrained-minimizer.cc](https://github.com/tirimatangi/LazyMath/blob/main/examples/example-constrained-minimizer.cc). Example `constraint.1` shows how to configure several optional member variables and get diagnostics of the quality of the result after the run. Example `constraint-2` demonstrates that an exception is thrown if the object and constraint functions are not configured properly.

## Compilation

The easiest way to compile all examples is to do
`cmake -DCMAKE_BUILD_TYPE=Release examples` <br>
If you don't want to use cmake, the examples can be compiled manually one by one. For instance, <br>
`g++ examples/example-linear-solver.cc -std=c++17 -I include/ -O3 -march=native -o example-linear-solver` <br>
If you cross-compile binaries for another platform, omit option `-march=native`.

The examples have been compiled with g++ 10.3.0  and clang++ 12.0.0 but any compiler which complies with c++17 standard should do.

## References

The most important reference is the book<br>
Boyd & Vandenberghe: Introduction to Applied Linear Algebra, Cambridge University Press 2018

Especially chapters 16, 18 and 19 are helpful.
A copy of it can be found at [Stanford web site](https://web.stanford.edu/~boyd/vmls/vmls.pdf)

There is a series of lectures of Stanford ENGR108 to go with the book in YouTube. For example here is a lecture on [constrained optimization](https://www.youtube.com/watch?v=00njRSL8WNQ).

