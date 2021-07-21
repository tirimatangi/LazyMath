#ifndef LAZY_OPTIMIZATION_H
#define LAZY_OPTIMIZATION_H

#include "ConjugateGradient.h"
#include <functional>
#include <string>

namespace LazyMath
{
using std::size_t;

// Object function.
// Input is an N-dimension vector and output is an M-dimensional vector.
template <size_t N = 0, size_t M = 0, class T = double>
using ObjectFunction = std::function<void(const Vector<T, N>&, Vector<T, M>&)>;

// Jacobian of the object function.
// Input of the object function an N-dimension vector and output is an M-dimensional vector,
// so the output of the jacobian is an N x M - matrix.
template <size_t N = 0, size_t M = 0, class T = double>
using JacobianFunction = std::function<void(const Vector<T, N>&, Matrix<T, N, M>&)>;

template <class T = double, size_t N = 0, size_t M = 0>
class Minimizer
{
public:
    // Example: Minimizer for an object function with input dimension=3 and output dimension=2
    //   Minimizer<3,2> min1; // A minimizer using fixed size std::arrays as function input/output types.
    //   Minimizer min2(3,2); // A minimizer using std::vectors as function input/output types.
    Minimizer(size_t inputDim = N, size_t outputDim = M)
    {
        static_assert(std::is_floating_point_v<T>, "Only floating point types are supported");
        static_assert((N==0) == (M==0), "Template dimensions must be either both zero or both non-zero.");
        _inputDim = inputDim;
        _outputDim = outputDim;
        // Allocate space if the vector type is dynamic (e.g. std::vector)
        if constexpr (N == 0 && M == 0) {
            initX = Vector<T>(inputDim);
            difference = Vector<T>(inputDim);
            _minFx = Vector<T>(outputDim);
            _jacobian = Matrix<T>(inputDim, outputDim);
            _tmpVectorM = Vector<T>(outputDim);
        } else { // Fixed-size stack allocated vectors and matrices.
            assert(N==inputDim && M==outputDim);
        }
        clear(initX);
        clear(_jacobian);
        _minX = initX;
        // Initialize differences to a default value.
        for (T& diff : difference)
            diff = std::sqrt(std::numeric_limits<T>::epsilon());
    }

    // Object function
    ObjectFunction<N,M,T> function = nullptr;
    // Derivative function a.k.a jacobian
    JacobianFunction<N,M,T> derivative = nullptr;
    // Initial guess for the minimum point.
    Vector<T, N> initX;

    // Difference for derivative approximation if analytical jacobian is not available.
    // One value for each element of the input vector.
    Vector<T, N> difference;
    // Maximum number of iterations
    unsigned maxIterations = 1000;
    // Maximum number of iterations done without progress before stopping
    unsigned maxNonProgressCount = 8;
    // Initial diagonal load. Zero means automatic initialization.
    T lambda = 0;
    // Maximum allowed relative error when LMS problem is solved.
    T errorTolerance = std::sqrt(std::numeric_limits<T>::epsilon());
    // Stop iteration when the norm of vector
    // d/dx(0.5*|f(x)|^2) = Df(x) * f(x) (a.k.a. Optimality Condition)
    // becomes less than this even if there is still progress in the iteration.
    T optimalityLimit = std::numeric_limits<T>::epsilon();


private:
    size_t _inputDim, _outputDim;

    // Number of rounds without progress so far.
    unsigned _nonProgressCount = 0;
    // Iterations run so far
    unsigned _iterationsDone = 0;

    // Current best minimum point
    Vector<T,N> _minX;
    // Current value of the object function
    Vector<T,M> _minFx;
    // Current squared norm of _minFx
    T _minNorm;
    // Current squared norm of optimality condition vector D(f(x))*f(x) = jacobian*_minFx
    T _optimalityCondition = 0;
    // Current value of the jacobian
    Matrix<T,N,M> _jacobian;
    // temporary workspace to avoid allocations.
    Vector<T,M> _tmpVectorM;


    // Calculate the matrix on the left side
    // and the vector on the right side of the iteration equation.
    // Returns squared norm of the optimality condition jacobian * _minFx.
    T updateLeftMatrixAndRightVector(Matrix<T, N, N>* mat, Vector<T, N>* vec)
    {
        // Left Matrix
        outer(_jacobian, mat);
        // Update the diagonal of leftMatrix
        for (size_t i = 0; i < mat->rows(); ++i)
            (*mat)[i][i] += lambda;
        // Right vector
        multiply(_jacobian, _minFx, vec);
        // The optimality condition is the norm of D(f)*f(x).
        return norm2(*vec);
    }

    void _initialize()
    {
        if(initX.size() == 0)
            throw std::runtime_error("Input dimension (dimension of X) is zero.");
        if(_minFx.size() == 0)
            throw std::runtime_error("Output dimension (dimension of f(X)) is zero.");
        if (difference.size() != initX.size())
            throw std::runtime_error("The size of the difference vector must be same as the size of initX.");
        if (!function)
            throw std::runtime_error("Object function has not been defined.");

        _iterationsDone = 0;
        _nonProgressCount = 0;

        // Sanity checks
        errorTolerance = std::max(errorTolerance, T(0));
        optimalityLimit = std::max(optimalityLimit, T(0));

        // Initialize the current function and jacobian values at the init point.
        _minX = initX;
        function(_minX, _minFx);
        _minNorm = norm2(_minFx);
        doJacobian(_minX, _minFx, &_jacobian);

        // Initialize the diagonal load using
        // the inner product of the initial jacobian.
        if (lambda == 0)  { // Skip if the user has specified lambda.
            T sumDiagonals = 0;
            // Calculate average of the diagonal values.
            for (const auto& row : _jacobian)
                sumDiagonals += inner(row, row);
            sumDiagonals /= _inputDim;
            lambda = sumDiagonals * 0.0625; // Just a guess...
            if (lambda == 0) // Safe guard against very small diagonal values.
                lambda = std::numeric_limits<T>::epsilon();
        }
    }

public:
    // Calculate jacobian at the given point x. The result is stored at jac.
    // Use central differences if the analytical function is not defined.
    // The value of the object function at point x is given in fx.
    // The jacobian is an N-by-M matrix, meaning that the gradient (of length N)
    // of i'th function component is stored in column i = 0...M-1.
    void doJacobian(const Vector<T, N>& x, const Vector<T, M>& fx, Matrix<T, N, M>* jac)
    {
        if (derivative) { // Analytical jacobian is defined so use it.
            derivative(x, *jac);
            return;
        }

        Vector<T, N>& tmpX = const_cast<Vector<T, N>&>(x);
        Vector<T, M>& tmpFx = _tmpVectorM;
        size_t underflowAtIndex = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            T diff = difference[i];
            T stashX = tmpX[i];
            tmpX[i] = stashX + diff;  // Calculate f(x+d)
            if (tmpX[i] == stashX)    // Too small difference?
                underflowAtIndex = i+1;
            function(tmpX, (*jac)[i]);
            tmpX[i] = stashX - diff;  // Calculate f(x-d)
            function(tmpX, tmpFx);
            T invDiff = T(0.5) / diff; // Calculate (f(x+d) - f(x-d)) / (2*diff)
            std::transform((*jac)[i].cbegin(), (*jac)[i].cend(), tmpFx.cbegin(), (*jac)[i].begin(),
                           [invDiff](const T& a, const T& b) { return (a - b) * invDiff;});
            tmpX[i] = stashX; // Restore the coordinate
        }
        if (underflowAtIndex > 0) {
            using std::to_string;
            --underflowAtIndex;
            int ex;
            double mant = std::frexp(this->difference[underflowAtIndex], &ex);
            std::string msg = "Underflow in numerical differentiation. "
                              "Suggest increase minimizer.difference["+to_string(underflowAtIndex)+
                              "] which is currently "+to_string(mant)+" * 2^"+to_string(ex);
            throw std::runtime_error(msg);
        }
    }

    void run()
    {
        _initialize();

        Vector<T, N> tmpX = _minX;
        Vector<T, M> tmpFx = _minFx;
        // Outer product of the jacobian (J * J')
        Matrix<T, N, N> leftMatrix(_inputDim, _inputDim);
        Vector<T, N> rightVector(_inputDim);

        T tmpNorm;
        _optimalityCondition = updateLeftMatrixAndRightVector(&leftMatrix, &rightVector);

        bool done = (_minNorm == 0); // Already perfect?
        while (!done) {
            // Solve (D * D' + lambda*eye()) * tmpX = D * f;
            tmpX = linearSolver(leftMatrix, rightVector,
                                SymmetricMode::SymmetricMatrix, // Mode of operation. Matrix is symmetric.
                                errorTolerance); // Maximum allowed relative error |Ax - b| / |b|

            // tmpX = old x - tmpX.
            // So tmpX is the difference to the current point at input and new X candidate at output.
            std::transform(_minX.cbegin(), _minX.cend(), tmpX.cbegin(), tmpX.begin(), std::minus<T>{});

            // Calculate the object function at the new point
            function(tmpX, tmpFx);
            tmpNorm = norm2(tmpFx);

            // Was there enough progress?
            if (tmpNorm * (1 + errorTolerance) < _minNorm) {
                _minX = tmpX;
                _minFx = tmpFx;
                _minNorm = tmpNorm;
                // Re-calculate Jacobian now that we have a new best point.
                doJacobian(_minX, _minFx, &_jacobian);
                lambda *= 0.75;
                _optimalityCondition = updateLeftMatrixAndRightVector(&leftMatrix, &rightVector);
                _nonProgressCount = 0;
            }
            else { // No good progress. Retry with a bigger lambda.
                lambda *= 2;
                // Update the diagonal of the left matrix
                for (size_t i = 0; i < _jacobian.rows(); ++i)
                    leftMatrix[i][i] += lambda;
                ++_nonProgressCount;
            }
            ++_iterationsDone;
            done = (_nonProgressCount >= maxNonProgressCount) ||
                   (_iterationsDone >= maxIterations) ||
                   (_optimalityCondition <= optimalityLimit);
        } // while !done
    } // run()

    // Number of iterations done.
    unsigned iterationsDone() const { return _iterationsDone; }

    // Number of iterations at the end with no progress.
    unsigned nonProgressCount() const { return _nonProgressCount; }

    // Current best minimum point.
    Vector<T, N> minX() const { return _minX; }

    // Current value of the object function.
    Vector<T, M> minFx() const { return _minFx; }

    // Current norm of _minFx.
    T minFxNorm() const { return std::sqrt(_minNorm); }

    // Norm of optimality condition D(f(x))*f(x)
    T optimalityValue() const { return _optimalityCondition; }
};

template <class T = double, size_t N = 0, size_t M = 0, size_t P = 0>
class ConstrainedMinimizer
{
public:

    ConstrainedMinimizer(size_t inputDim = N, size_t outputDim = M, size_t constraintDim = P)
    {
        static_assert(std::is_floating_point_v<T>, "Only floating point types are supported");
        static_assert(((N==0) == (M==0)) && ((N==0) == (P==0)), "Template dimensions must be either all zero or all non-zero.");
        _inputDim = inputDim;
        _outputDim = outputDim;
        _constraintDim = constraintDim;

        if constexpr (N == 0 && M == 0 && P == 0) {
            initX = Vector<T>(inputDim);
            difference = Vector<T>(inputDim);
            _minFx = Vector<T>(outputDim);
            _jacobianF = Matrix<T>(inputDim, outputDim);
            _minZ = Vector<T>(constraintDim);
            _minGx = Vector<T>(constraintDim);
            _jacobianG = Matrix<T>(inputDim, constraintDim);
            _tmpMatrixNbyN = Matrix<T>(inputDim, inputDim);
            _tmpVectorN = Vector<T>(inputDim);
            _tmpVectorM = Vector<T>(outputDim);
            _tmpVectorP = Vector<T>(constraintDim);
        }
        else
            assert(N==inputDim && M==outputDim && P==constraintDim);
        clear(initX);
        _minX = initX;
        // Initialize differences to a default value.
        for (T& diff : difference)
            diff = std::sqrt(std::numeric_limits<T>::epsilon());

        };


    // Object function
    ObjectFunction<N,M,T> function = nullptr;
    // Derivative function a.k.a jacobian
    JacobianFunction<N,M,T> derivative = nullptr;
    // Constraint function
    ObjectFunction<N,P,T> constraintFunction = nullptr;
    // Derivative function a.k.a jacobian
    JacobianFunction<N,P,T> constraintDerivative = nullptr;

    // Initial guess for the minimum point.
    Vector<T, N> initX;
    // Difference for derivative approximation if analytical jacobian is not available.
    // One value for each element of the input vector.
    Vector<T, N> difference;
    // Maximum number of iterations
    unsigned maxIterations = 1000;
    // Maximum number of iterations done without progress before stopping
    unsigned maxNonProgressCount = 8;
    // Initial diagonal load. Zero means automatic initialization.
    T lambda = 0;
    // Booster coefficient for the automatic diagonal load calculator.
    T diagonalBoost = 1.0 / 16;
    // Maximum allowed relative error when LMS problem is solved.
    T errorTolerance = std::sqrt(std::numeric_limits<T>::epsilon());
    // Stop iteration when the norm of vector
    // d/dx{0.5*|f(x)|^2 + 0.5*mu*|g(x)+z/(2*mu)|^2} = Df(x)*f(x)+Dg(x)*(mu*g(x)+z/mu) (a.k.a. Optimality Condition)
    // becomes less than this even if there is still progress in the iteration.
    T optimalityLimit = std::sqrt(std::numeric_limits<T>::epsilon());

private:
    size_t _inputDim, _outputDim, _constraintDim;

    // Penalty coefficient
    T _mu = 1;

    // Iterations run so far
    unsigned _iterationsDone = 0;

    // Current best minimum point
    Vector<T,N> _minX;
    // Current value of the object function
    Vector<T,M> _minFx;
    // Current value of the constraint function
    Vector<T,P> _minGx;
    // Current best Lagrange vector
    Vector<T,P> _minZ;
    // Current squared norm of optimality condition vector D(f(x))*f(x) = jacobian*_minFx
    T _optimalityCondition = 0;
    // Current value of the object function jacobian
    Matrix<T,N,M> _jacobianF;
    // Current value of the constraint function jacobian
    Matrix<T,N,P> _jacobianG;
    // Workspace for calculating the right and left sides of a linear problem Ax=b
    Matrix<T,N,N> _tmpMatrixNbyN;
    Vector<T,N> _tmpVectorN;
    Vector<T,M> _tmpVectorM;
    Vector<T,P> _tmpVectorP;

public:
    void _initialize()
    {
        if(initX.size() == 0)
            throw std::runtime_error("Input dimension (dimension of X) is zero.");
        if(_minFx.size() == 0)
            throw std::runtime_error("Output dimension (dimension of f(X)) is zero.");
        if(_minGx.size() == 0)
            throw std::runtime_error("Constraint dimension (dimension of g(X)) is zero.");
        if (difference.size() != initX.size())
            throw std::runtime_error("The size of the difference vector must be same as the size of initX.");
        if (!function)
            throw std::runtime_error("Object function has not been defined.");
        if (!constraintFunction)
            throw std::runtime_error("Constraint function has not been defined.");

        _iterationsDone = 0;
        clear(_minZ);
        _mu = 1;

        // Sanity checks
        errorTolerance = std::max(errorTolerance, T(0));
        optimalityLimit = std::max(optimalityLimit, T(0));

        // Initialize the current function and jacobian values at the init point.
        _minX = initX;
        function(_minX, _minFx);
        constraintFunction(_minX, _minGx);
        doJacobian<M>(_minX, _minFx, function, derivative, &_jacobianF, &_tmpVectorM);
        doJacobian<P>(_minX, _minGx, constraintFunction, constraintDerivative, &_jacobianG, &_tmpVectorP);

        // Initialize the diagonal load using
        // the inner product of the initial jacobians.
        if (lambda == 0)  // Skip if the user has specified lambda.
            lambda = _estimateDiagonalLoad();
    }

    // Estimates diagonal load (aka lambda) for linear solver.
    // Assumes that jacobian matrices of the object function f and
    // constraint function g are valid.
    T _estimateDiagonalLoad() const
    {
        T sumDiagonals = 0;
        // Calculate average of the diagonal values.
        for (const auto& row : _jacobianF)
            sumDiagonals += inner(row, row);
        for (const auto& row : _jacobianG)
            sumDiagonals += inner(row, row);
        sumDiagonals /= _inputDim;
        sumDiagonals = sumDiagonals * diagonalBoost; // Just a guess...
        if (sumDiagonals == 0) // Safe guard against very small diagonal values.
            sumDiagonals= std::numeric_limits<T>::epsilon();
        return sumDiagonals;
    }

    // Calculate jacobian at the given point x. The result is stored at jac.
    // Use central differences if the analytical function is not defined.
    // The value of the object function at point x is given in fx.
    // The jacobian is an N-by-C matrix, meaning that the gradient (of length N)
    // of i'th function component is stored in column i = 0...C-1.
    // C is either M for object function or P for constraint function
    template <size_t C>
    void doJacobian(const Vector<T, N>& x,
                    const Vector<T, C>& fx,
                    const ObjectFunction<N,C,T>& func,
                    const JacobianFunction<N,C,T>& jacobian,
                    Matrix<T, N, C>* jac,
                    Vector<T, C>* workspace)
    {
        assert(x.size() == _inputDim);
        if (jacobian) { // Analytical jacobian is defined so use it.
            jacobian(x, *jac);
            return;
        }

        // Avoid allocation of temporary workspace by reusing already allocated data.
        Vector<T, N>& tmpX = const_cast<Vector<T, N>&>(x);
        Vector<T, C>& tmpFx = *workspace;
        size_t underflowAtIndex = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            T diff = this->difference[i];
            T stashX = tmpX[i];
            tmpX[i] = stashX + diff;  // Calculate f(x+d)
            if (tmpX[i] == stashX)    // Too small difference?
                underflowAtIndex = i+1;
            func(tmpX, (*jac)[i]);
            tmpX[i] = stashX - diff;  // Calculate f(x-d)
            func(tmpX, tmpFx);
            T invDiff = T(0.5) / diff; // Calculate (f(x+d) - f(x-d)) / (2*diff)
            std::transform((*jac)[i].cbegin(), (*jac)[i].cend(), tmpFx.cbegin(), (*jac)[i].begin(),
                           [invDiff](const T& a, const T& b) { return (a - b) * invDiff;});
            tmpX[i] = stashX; // Restore the coordinate
        }
        if (underflowAtIndex > 0) {
            using std::to_string;
            --underflowAtIndex;
            int ex;
            double mant = std::frexp(this->difference[underflowAtIndex], &ex);
            std::string msg = "Underflow in numerical differentiation. "
                              "Suggest increase minimizer.difference["+to_string(underflowAtIndex)+
                              "] which is currently "+to_string(mant)+" * 2^"+to_string(ex);
            throw std::runtime_error(msg);
        }
    }

    // Calculate the matrix on the left side
    // and the vector on the right side of the iteration equation.
    // Returns squared norm of the optimality condition jacobian * _minFx.
    T updateLeftMatrixAndRightVector(Matrix<T, N, N>* mat, Vector<T, N>* vec)
    {
        auto muScaler = [mu = this->_mu](const T& a, const T& b) { return mu * a + b;};
        // Left Matrix
        outer(_jacobianF, mat);
        outer(_jacobianG, &_tmpMatrixNbyN);
        // Scale Dg(x) with mu and sum up.
        for (size_t i = 0; i < _inputDim; ++i)
            std::transform(_tmpMatrixNbyN[i].cbegin(), _tmpMatrixNbyN[i].cend(), (*mat)[i].cbegin(),(*mat)[i].begin(), muScaler);
        // Update the diagonal of leftMatrix
        for (size_t i = 0; i < _inputDim; ++i)
            (*mat)[i][i] += lambda;

        // Right vector = Df*f(x) + Dg*(mu*g(x)+z)
        multiply(_jacobianG, _minGx, &_tmpVectorN);

        // Set tmpVectorP = mu*_minGx + z
        std::transform(_minGx.cbegin(), _minGx.cend(), _minZ.cbegin(), _tmpVectorP.begin(), muScaler);
        // Set _tmpVectorN = Dg * _tmpVectorP
        multiply(_jacobianG, _tmpVectorP, &_tmpVectorN);
        // Set vec = Df * f(x)
        multiply(_jacobianF, _minFx, vec);
        // The norm of (Df * f(x) + Dg * _tmpVectorP) is the optimality condition.
        std::transform(vec->cbegin(), vec->cend(), _tmpVectorN.cbegin(), vec->begin(), std::plus<T>{});
        return norm2(*vec);
    }


    void run()
    {
        _initialize();

        Matrix<T, N, N> leftMatrix(_inputDim, _inputDim);
        Vector<T, N> rightVector(_inputDim);
        T prevNormGx = norm2(_minGx);
        T tmpNormFxGx;

        T minNormFxGx = std::numeric_limits<T>::max();
        bool outerLoopDone = false;

        // Outer loop runs until the optimality condition is met
        // or the maximum number of iterations is reached.
        while (!outerLoopDone) {
            bool innerLoopDone = false;
            // Number of rounds without progress so far.
            unsigned nonProgressCount = 0;
            _optimalityCondition = updateLeftMatrixAndRightVector(&leftMatrix, &rightVector);
            // If the optimality condition is met without any iteration,
            // there is no point going to the inner loop.
            if (_optimalityCondition < optimalityLimit) {
                outerLoopDone = true;
                break;
            }
            // Minimize function 0.5*|[f(x) sqrt(mu)*g(x)+z/sqrt(mu)]|^2. Dim = M+P.
            while (!innerLoopDone) {
                // Define aliases to avoid temporary allocations.
                Vector<T, N>& tmpX = _tmpVectorN;
                Vector<T, M>& tmpFx = _tmpVectorM;
                Vector<T, P>& tmpGx = _tmpVectorP;
                // Solve (D * D' + lambda*eye()) * tmpX = D * f;
                tmpX = linearSolver(leftMatrix, rightVector,
                                    SymmetricMode::SymmetricMatrix, // Mode of operation. Matrix is symmetric.
                                    errorTolerance); // Maximum allowed relative error |Ax - b| / |b|

                // tmpX = old x - tmpX.
                // So tmpX is the difference to the current point at input and new X candidate at output.
                std::transform(_minX.cbegin(), _minX.cend(), tmpX.cbegin(), tmpX.begin(), std::minus<T>{});

                // Calculate the minimized function at the new point
                function(tmpX, tmpFx);
                constraintFunction(tmpX, tmpGx);
                // Calculate squared norm of the function we are minimizing, ie [f(x) sqrt(mu)*g(x) + z / sqrt(mu)]
                tmpNormFxGx = norm2(tmpFx);
                for (size_t i = 0; i < _constraintDim; ++i)
                    tmpNormFxGx += (_mu * (tmpGx[i] * tmpGx[i]) + 2 * tmpGx[i] * _minZ[i] + (_minZ[i] * _minZ[i]) / _mu);

                // Was there enough progress?
                if (tmpNormFxGx * (1 + errorTolerance) < minNormFxGx) {
                    _minX = tmpX;
                    _minFx = tmpFx;
                    _minGx = tmpGx;
                    minNormFxGx = tmpNormFxGx;
                    // Re-calculate the jacobians now that we have a new best point.
                    // Note: _tmpVectorM and _tmpVectorP will be written over.
                    doJacobian<M>(_minX, _minFx, function, derivative, &_jacobianF, &_tmpVectorM);
                    doJacobian<P>(_minX, _minGx, constraintFunction, constraintDerivative, &_jacobianG, &_tmpVectorP);
                    lambda *= 0.75;
                    _optimalityCondition = updateLeftMatrixAndRightVector(&leftMatrix, &rightVector);
                    nonProgressCount = 0;
                }
                else { // No good progress. Retry with a bigger lambda.
                    lambda *= 2;
                    // Update the diagonal of the left matrix
                    for (size_t i = 0; i < _inputDim; ++i)
                        leftMatrix[i][i] += lambda;
                    ++nonProgressCount;
                }
                ++_iterationsDone;
                innerLoopDone = (nonProgressCount >= maxNonProgressCount) ||
                                (_iterationsDone >= maxIterations) ||
                                (_optimalityCondition <= optimalityLimit);
            } // while !innerLoopDone
            // Now we have the minimum given that _mu and _minZ are as they are.
            // Adjust them and try again.
            if (_iterationsDone < maxIterations) {
                // Calculate penalty at the new X
                T newNormGx = norm2(_minGx);
                // Update z
                for (size_t i = 0; i < _constraintDim; ++i)
                    _minZ[i] += _mu * _minGx[i];

                // Increase the penalty if the constraint norm has not gone down enough.
                // However, penalty increase is not needed if the weighed constraint is already small enough.
                T weighedConstraint = std::sqrt(_mu * newNormGx);
                if (newNormGx > 0.0625 * prevNormGx &&      // Constraint not decreasing fast enough?
                    weighedConstraint > optimalityLimit &&  // Constraint is not small enough?
                    _mu < std::sqrt(std::numeric_limits<T>::max()))  // Mu is not already too large?
                    _mu *= 2;
                prevNormGx = newNormGx;
                lambda = _estimateDiagonalLoad();
                minNormFxGx = std::numeric_limits<T>::max();
            } else  // _iterationsDone == maximum
                outerLoopDone = true;
        } // while !outerLoopDone
    } // run

    // Number of iterations done.
    unsigned iterationsDone() const { return _iterationsDone; }

    // Current best minimum point.
    Vector<T, N> minX() const { return _minX; }

    // Current value of the object function.
    Vector<T, M> minFx() const { return _minFx; }

    // Current value of the constraint function.
    Vector<T, P> minGx() const { return _minGx; }

    // Lagrange Z-vector
    Vector<T, P> lagrangeZ() const { return _minZ; }

    // Norm of optimality condition Df(x)*f(x)+Dg(x)*(mu*g(x)+z/mu)
    T optimalityValue() const { return _optimalityCondition; }

    // Penalty coefficient for the constraint
    T penalty() const { return _mu; }
};

} // namespace LazyMath

#endif // LAZY_OPTIMIZATION_H