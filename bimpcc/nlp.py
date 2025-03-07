from typing import Protocol, Tuple, List
from cyipopt import minimize_ipopt
import numpy as np


class ConstraintFn(Protocol):
    def __call__(self, x: np.ndarray) -> float:
        pass

    def parse_vars(self, x: np.ndarray) -> Tuple:
        pass

    def jacobian(self, x: np.ndarray) -> float:
        pass

    def hessian(self, x: np.ndarray, _lambda) -> float:
        pass


class ComplementarityConstraintFn(Protocol):
    def __init__(self, t: float = 1.0) -> None:
        self.t = t

    def __call__(self, x: np.ndarray) -> float:
        pass

    def parse_vars(self, x: np.ndarray) -> Tuple:
        pass

    def jacobian(self, x: np.ndarray) -> float:
        pass

    def hessian(self, x: np.ndarray, _lambda) -> float:
        pass


class ObjectiveFn(Protocol):
    def __call__(self, x: np.ndarray) -> float:
        pass

    def parse_vars(self, x: np.ndarray) -> Tuple:
        pass

    def gradient(self, x: np.ndarray) -> float:
        pass

    def hessian(self, x: np.ndarray) -> float:
        pass
    

class PenalizedObjectiveFn(Protocol):
    def __init__(self, pi: float = 1.0) -> None:
        self.pi = pi

    def __call__(self, x: np.ndarray) -> float:
        pass

    def parse_vars(self, x: np.ndarray) -> Tuple:
        pass

    def gradient(self, x: np.ndarray) -> float:
        pass

    def hessian(self, x: np.ndarray) -> float:
        pass


class OptimizationProblem:
    def __init__(
        self,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn] = [],
        ineq_constraint_funcs: List[ConstraintFn] = [],
    ):
        self.objective_func = objective_func
        self.gradient_objective_func = self.objective_func.gradient
        self.hessian_objective_func = self.objective_func.hessian
        self.constraints = []
        if ineq_constraint_funcs is not None:
            for constraint_func in ineq_constraint_funcs:
                self.constraints.append(
                    {
                        "type": "ineq",
                        "fun": constraint_func,
                        "jac": constraint_func.jacobian,
                        # "hess": constraint_func.hessian,
                    }
                )
        if eq_constraint_funcs is not None:
            for constraint_func in eq_constraint_funcs:
                self.constraints.append(
                    {
                        "type": "eq",
                        "fun": constraint_func,
                        "jac": constraint_func.jacobian,
                        # "hess": None,
                    }
                )

    def solve(
        self, x0: List[float], bounds: List[Tuple[float, float]], options: dict = {}
    ) -> Tuple[dict, np.ndarray, float]:
        result = minimize_ipopt(
            fun=self.objective_func,
            jac=self.gradient_objective_func,
            # hess=self.hessian_objective_func,
            x0=x0,
            bounds=bounds,
            constraints=self.constraints,
            options=options,
            # options={
            #     "print_level": print_level,
            #     "max_iter": 5000,
            #     # "alpha_for_y": "safer-min-dual-infeas",
            #     # "dual_inf_tol": 1e-6,
            #     "tol": tol
            #     # "jacobian_approximation": "finite-difference-values",
            # },
        )
        return result, result["x"], result["fun"]


class UnconstrainedOptimizationProblem(OptimizationProblem):
    def __init__(self, objective_func: ObjectiveFn):
        super().__init__(objective_func, [], [])
