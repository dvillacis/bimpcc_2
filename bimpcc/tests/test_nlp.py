import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import coo_array
from bimpcc.nlp import (
    OptimizationProblem,
    ConstraintFn,
    ObjectiveFn,
)


class RosenFn(ObjectiveFn):
    def __call__(self, x):
        return rosen(x)

    def gradient(self, x):
        return rosen_der(x)

    def hessian(self, x):
        return rosen_hess(x)


def test_simple_nlp():
    """Test the correctness of the simple NLP problem."""
    rosen = RosenFn()
    nlp = OptimizationProblem(rosen)
    x0 = np.array([0, 0])
    bounds = [(-10, 10), (-10, 10)]
    res, x, f = nlp.solve(x0, bounds)

    # Expected values
    expected_x = np.ones(2)
    expected_f = 0.0

    # Assert correctness
    np.testing.assert_allclose(x, expected_x, atol=1e-6, err_msg="x is incorrect")
    np.testing.assert_allclose(f, expected_f, atol=1e-6, err_msg="f is incorrect")


def simple_con(x):
    return np.array([10 - x[1] ** 2 - x[2], 100.0 - x[4] ** 2])


def con_jac(x):
    # Dense Jacobian:
    # J = (0  -2*x[1]   -1   0     0     )
    #         (0   0         0   0   -2*x[4] )
    # Sparse Jacobian (COO)
    rows = np.array([0, 0, 1])
    cols = np.array(([1, 2, 4]))
    data = np.array([-2 * x[1], -1, -2 * x[4]])
    return coo_array((data, (rows, cols)))


def con_hess(x, _lambda):
    H1 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    H2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -2],
        ]
    )
    return _lambda[0] * H1 + _lambda[1] * H2


class SimpleConstraint(ConstraintFn):
    def __call__(self, x):
        return simple_con(x)

    def jacobian(self, x):
        return con_jac(x)

    def hessian(self, x, _lambda):
        return con_hess(x, _lambda)


def test_const_nlp():
    """Test the correctness of the constrained NLP problem."""
    c = SimpleConstraint()
    rosen = RosenFn()
    nlp = OptimizationProblem(rosen, [], [c])
    x0 = 1.0 * np.ones(5)
    bounds = None
    res, x, f = nlp.solve(x0, bounds)
    print(res)

    # Expected values
    expected_x = np.array([1, 1, 1, 1, 1])
    expected_f = 0.0

    # Assert correctness
    np.testing.assert_allclose(x, expected_x, atol=1e-6, err_msg="x is incorrect")
    np.testing.assert_allclose(f, expected_f, atol=1e-6, err_msg="f is incorrect")
