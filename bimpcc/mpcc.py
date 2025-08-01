from abc import ABC, abstractmethod
from nlp import (
    ConstraintFn,
    ObjectiveFn,
    OptimizationProblem,
)
from typing import List, Tuple, Union, Protocol
from rich import print
import numpy as np
import scipy.sparse as sp


class ComplementarityConstraintFn(Protocol):
    def __init__(self, G: ConstraintFn, H: ConstraintFn, t: float = 1.0) -> None:
        self.t = t
        self.G = G
        self.H = H

    def __call__(self, x: np.ndarray) -> float:
        pass

    def parse_vars(self, x: np.ndarray) -> Tuple:
        pass

    def jacobian(self, x: np.ndarray) -> float:
        pass

    def hessian(self, x: np.ndarray, _lambda) -> float:
        pass


class ScholtesComplementarityConstraintFn(ComplementarityConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        return self.t - self.G(x) * self.H(x)

    def jacobian(self, x):
        G_jacobian = self.G.jacobian(x)
        H_jacobian = self.H.jacobian(x)
        G_diag = sp.diags_array(self.G(x))
        H_diag = sp.diags_array(self.H(x))
        jac = -G_diag @ H_jacobian - H_diag @ G_jacobian
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class FisherComplementarityConstraintFn(ComplementarityConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        norm = np.sqrt(self.G(x) ** 2 + self.H(x) ** 2)
        return self.t - (norm - (self.G(x) + self.H(x)))

    def jacobian(self, x):
        norm = np.sqrt(self.G(x) ** 2 + self.H(x) ** 2)
        alpha = (self.G(x) / norm) - 1
        beta = (self.H(x) / norm) - 1
        G_diag = sp.diags_array(alpha)
        H_diag = sp.diags_array(beta)
        jac = -G_diag @ self.G.jacobian(x) - H_diag @ self.H.jacobian(x)
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class MPCCProblem(ABC):
    def __init__(
        self,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        complementarity_constraint_funcs: Tuple[ConstraintFn, ConstraintFn],
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
        t_init=1.0,
        relaxation_type: str = "scholtes",
        *args,
        **kwargs,
    ):
        self.parse_vars_fn = objective_func.parse_vars
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs
        self.ineq_constraint_funcs = ineq_constraint_funcs

        G, H = complementarity_constraint_funcs
        if relaxation_type.lower() == "scholtes":
            self.complementarity_constraint_func = ScholtesComplementarityConstraintFn(
                G, H, t_init
            )
        elif relaxation_type.lower() == "fisher":
            self.complementarity_constraint_func = FisherComplementarityConstraintFn(
                G, H, t_init
            )
        else:
            raise ValueError(f"Unknown relaxation type: {relaxation_type}")
        self.bounds = bounds
        self.t = t_init

    @abstractmethod
    def compute_complementarity(self, x: np.ndarray) -> float:
        pass

    def _solve_nlp(self, x0, bounds, t, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        tol = kwargs.get("tol", 1e-4)
        max_iter = kwargs.get("max_iter", 5000)
        self.complementarity_constraint_func.t = t
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs + [self.complementarity_constraint_func],
        )
        options = {
            "print_level": print_level,
            "tol": tol,
            "max_iter": max_iter,
            "acceptable_tol": 1e-5,
            "constr_viol_tol": 1e-5,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
        }
        return nlp.solve(x0, bounds, options=options)

    def solve(
        self,
        t_min: float = 1e-5,
        max_iter: int = 10,
        tol: float = 1e-3,
        nlp_tol: float = 1e-6,
        nlp_max_iter: int = 5000,
        verbose: bool = False,
        print_level: int = 0,
        beta=0.5,
        *args,
        **kwargs,
    ):
        x = self.x0
        t = self.t
        res = None
        fn = None
        print(
            f'{"Iter": >5}\t{"Termination_status": >15}\t{"Objective": >15}\t{
          "MPCC_compl": >15}\t{"t": >15}\n'
        )
        for k in range(max_iter):
            if t <= t_min:
                break
            res, x_, fn = self._solve_nlp(
                x,
                self.bounds,
                t,
                tol=nlp_tol,
                print_level=print_level,
                max_iter=nlp_max_iter,
            )
            self.comp = self.compute_complementarity(x_)
            if np.abs(self.comp) < tol:
                print(
                    f'{k: > 5}*\t{res["status"]: > 15}\t{fn: > 15}\t{self.comp: > 15}\t{t: > 15}'
                )
                res["iter"] = k
                return res, x_, fn
            # status = ""
            if res["status"] >= 0:
                t_ = max(t_min, beta * t)
                x = x_
            else:
                t_ = 1.1 * t
                beta = 0.9 * beta
            if verbose:
                print(
                    f'{k: > 5}\t{res["status"]: > 15}\t{fn: > 15}\t{self.comp: > 15}\t{t: > 15}'
                )
            t = t_

        print(
            f"* (STOPPED) Iteration {k+1}: Solving the NLP problem for t = {t} with complementarity: {self.compute_complementarity(x)}"
        )
        res["iter"] = k + 1
        return res, x, fn
