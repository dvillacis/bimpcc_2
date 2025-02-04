import numpy as np
from abc import ABC, abstractmethod
from typing import Protocol, Tuple
from cyipopt import minimize_ipopt
from bimpcc.utils import gradient_operator_x, gradient_operator_y


class CostFn(Protocol):
    def __call__(self, x: np.ndarray, *args, **kwargs) -> float:
        pass

    def gradient(self, x: np.ndarray, *args, **kwargs) -> float:
        pass


class L2CostFn(CostFn):
    def __call__(self, x: np.ndarray, x_true: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(x - x_true) ** 2

    def gradient(self, x: np.ndarray, x_true: np.ndarray) -> float:
        return x - x_true


class DataFitFn(Protocol):
    def __call__(self, x: np.ndarray, *args, **kwargs) -> float:
        pass

    def gradient(self, x: np.ndarray, *args, **kwargs) -> float:
        pass


class DenoisingFn(DataFitFn):
    def __call__(self, x: np.ndarray, x_noisy: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(x - x_noisy) ** 2

    def gradient(self, x: np.ndarray, x_noisy: np.ndarray) -> float:
        return x - x_noisy


class AbstractNLP(ABC):
    def __init__(self, cost_fn: CostFn, data_fit_fn: DataFitFn):
        self.cost_fn = cost_fn
        self.data_fit_fn = data_fit_fn

    def objective(self, x, *args, **kwargs):
        return self.cost_fn(x, *args, **kwargs)

    def gradient(self, x, *args, **kwargs):
        return self.cost_fn.gradient(x, *args, **kwargs)

    @abstractmethod
    def setup_problem(self, *args, **kwargs):
        pass

    @abstractmethod
    def getvars(self, x):
        pass

    @abstractmethod
    def constraints(self, x):
        pass

    def solve(self):
        res = minimize_ipopt(self.cost_fn, jac=self.cost_fn.gradient, constraints=self.constraints)

    # @abstractmethod
    # def jacobian(self, x):
    #     pass


class DenoisingTVRegNLP(AbstractNLP):
    def __init__(self, u_true, u_noisy):
        self.u_true = u_true.flatten()
        self.u_noisy = u_noisy.flatten()
        self.M, self.N = u_true.shape
        self.Kx = gradient_operator_x(self.N, self.M)
        self.Ky = gradient_operator_y(self.N, self.M)

        self.nlp = self.setup_problem(self.N)

        datafn = DenoisingFn()
        costfn = L2CostFn()
        super().__init__(costfn, datafn)

    def setup_problem(self, *args, **kwargs):
        n = self.N * self.M
        u0 = self.u_noisy
        qx0 = 0.0 * np.ones(n)
        qy0 = 0.0 * np.ones(n)
        alpha0 = [1e-5] * np.ones(n)
        init_guess = np.concatenate((u0, qx0, qy0, alpha0))

        num_constraints = n
        cl = np.zeros(num_constraints)
        cu = np.zeros(num_constraints)

        num_vars = 4 * n
        # lb_u = np.zeros(n)
        # lb_qx = -1e20 * np.ones(n)
        # lb_qy = -1e20 * np.ones(n)
        # lb_alpha = [1e-10] * np.ones(n)
        lb = np.zeros(num_vars)

        # ub_u = 1e20 * np.ones(n)
        # ub_qx = 1e20 * np.ones(n)
        # ub_qy = 1e20 * np.ones(n)
        # ub_alpha = [1e20] * np.ones(n)
        ub = np.ones(num_vars) * 1e20

        self.nlp = cyipopt.Problem(
            n=num_vars, m=len(cl), problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu
        )
        return self.nlp, init_guess

    def getvars(self, x):
        return x[: self.N], x[self.N : self.N + self.M], x[self.N + self.M :]

    def constraints(self, x):
        u, qx, qy, alpha = self.getvars(x)
        constr1 = u - self.u_noisy + self.Kx.T @ qx + self.Ky.T @ qy
        return constr1
