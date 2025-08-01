from typing import Protocol, Tuple, List, Union
from cyipopt import minimize_ipopt
import numpy as np
from scipy.optimize import minimize


class ConstraintFn(Protocol):
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


class SumObjectiveFn(ObjectiveFn):
    def __init__(self, f1: ObjectiveFn, f2: ObjectiveFn):
        self.f1 = f1
        self.f2 = f2

    def __call__(self, x):
        return self.f1(x) + self.f2(x)

    def gradient(self, x):
        return self.f1.gradient(x) + self.f2.gradient(x)

    def parse_vars(self, x):
        return self.f1.parse_vars(x)


class OptimizationProblem:
    def __init__(
        self,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn] = [],
        ineq_constraint_funcs: List[ConstraintFn] = [],
        bounds: List[Tuple[Union[int, None], Union[int, None]]] = [],
    ):
        self.objective_func = objective_func
        self.gradient_objective_func = self.objective_func.gradient
        self.hessian_objective_func = self.objective_func.hessian
        self.constraints = []
        self.bounds = bounds
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
        self, x0: List[float], options: dict = {}
    ) -> Tuple[dict, np.ndarray, float]:
        result = minimize_ipopt(
            fun=self.objective_func,
            jac=self.gradient_objective_func,
            # hess=self.hessian_objective_func,
            x0=x0,
            bounds=self.bounds,
            constraints=self.constraints,
            options=options
        )
        # result = minimize(
        #     fun=self.objective_func,
        #     x0=x0,
        #     method="SLSQP",
        #     jac=self.gradient_objective_func,
        #     bounds=self.bounds,
        #     constraints=self.constraints,
        #     options=options,
        # )
        return result, result["x"], result["fun"]


class UnconstrainedOptimizationProblem(OptimizationProblem):
    def __init__(self, objective_func: ObjectiveFn):
        super().__init__(objective_func, [], [])
