import numpy as np
import scipy.sparse as sp
from mpcc import MPCCPenalized, MPCCRelaxed
from nlp import ObjectiveFn, ConstraintFn


class Bard1Objective(ObjectiveFn):
    def __call__(self, x: np.ndarray) -> float:
        return (x[0] - 5) ** 2 + (2 * x[1] + 1) ** 2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * (x[0] - 5), 4 * (2 * x[1] + 1), 0, 0, 0])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [2, 0, 0, 0, 0],
                [0, 8, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )


class BardKKTConstraint(ConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        return np.array([2 * (x[1] - 1) - 1.5 * x[0] + x[2] - 0.5 * x[3] + x[4]])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        row = np.zeros(5, dtype=int)
        col = np.arange(5)
        data = np.array([-1.5, 2, 1, -0.5, 1])
        jac_sp = sp.coo_array((data, (row, col)), shape=(1, 5), dtype=float)
        return jac_sp


class BardIneq(ConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        return np.array([3 * x[0] - x[1] - 3, -x[0] + 0.5 * x[1] + 4, -x[0] - x[1] + 7])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        A = np.array(
            [
                [3, -1, 0, 0, 0],
                [-1, 0.5, 0, 0, 0],
                [-1, -1, 0, 0, 0],
            ]
        )
        rows, cols = np.nonzero(A)
        data = A[rows, cols]
        jac_sp = sp.coo_array((data, (rows, cols)), shape=A.shape, dtype=float)
        return jac_sp


class BardComplementarity1(ConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        return np.array([x[2], x[3], x[4]])

    def jacobian(self, x):
        jac = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        jac_sp = sp.coo_array(jac, dtype=float)
        return jac_sp


class BardComplementarity2(ConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        return np.array([3 * x[0] - x[1] - 3, -x[0] + 0.5 * x[1] + 4, -x[0] - x[1] + 7])

    def jacobian(self, x):
        jac = np.array([[3, -1, 0, 0, 0], [-1, 0.5, 0, 0, 0], [-1, -1, 0, 0, 0]])
        jac_sp = sp.coo_array(jac, dtype=float)
        return jac_sp


class Bard1Relaxed(MPCCRelaxed):
    def __init__(self, relaxation_type="scholtes"):
        objective_func = Bard1Objective()
        eq_constraint_funcs = [BardKKTConstraint()]
        ineq_constraint_funcs = [BardIneq()]
        complementarity_constraints = [
            BardComplementarity1(),
            BardComplementarity2(),
        ]
        bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
        super().__init__(
            objective_func=objective_func,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            complementarity_constraint_funcs=complementarity_constraints,
            bounds=bounds,
            relaxation_type=relaxation_type,
        )

class Bard1Penalized(MPCCPenalized):
    def __init__(self, inner_penalty_init=1.0, outer_penalty_init=1.0):
        objective_func = Bard1Objective()
        eq_constraint_funcs = [BardKKTConstraint()]
        ineq_constraint_funcs = [BardIneq()]
        complementarity_constraints = [
            BardComplementarity1(),
            BardComplementarity2(),
        ]
        bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
        super().__init__(
            objective_func=objective_func,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            complementarity_constraint_funcs=complementarity_constraints,
            bounds=bounds,
            inner_penalty_init=inner_penalty_init,
            outer_penalty_init=outer_penalty_init,
        )
    def initial_guess(self) -> np.ndarray:
        return 10.0*np.ones(5)
