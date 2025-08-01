from abc import ABC, abstractmethod
from nlp import (
    ConstraintFn,
    ObjectiveFn,
    SumObjectiveFn,
    OptimizationProblem,
)
from typing import List, Tuple, Union, Protocol, Callable
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
        jac = jac.tocoo()
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class FisherComplementarityConstraintFn(ComplementarityConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        Gx = self.G(x)
        Hx = self.H(x)
        norm = np.sqrt(Gx**2 + Hx**2)
        return self.t - (norm - (Gx + Hx))

    def jacobian(self, x):
        Gx = self.G(x)
        Hx = self.H(x)
        norm = np.sqrt(Gx**2 + Hx**2) + 1e-8
        alpha = (Gx / norm) - 1
        beta = (Hx / norm) - 1
        G_diag = sp.diags_array(alpha)
        H_diag = sp.diags_array(beta)
        jac = -G_diag @ self.G.jacobian(x) - H_diag @ self.H.jacobian(x)
        jac = jac.tocoo()
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class ComplementarityPenaltyFn(Protocol):
    def __init__(self, G: ConstraintFn, H: ConstraintFn) -> None:
        self.G = G
        self.H = H

    def __call__(self, x: np.ndarray) -> float:
        pass

    def gradient(self, x: np.ndarray) -> float:
        pass


class HyperbolicPenaltyFn(ComplementarityPenaltyFn):
    def __init__(self, G: ConstraintFn, H: ConstraintFn, r: float, v: float) -> None:
        super().__init__(G, H)
        self.r = r
        self.v = v

    def __call__(self, x: np.ndarray) -> float:
        Gx = self.G(x)
        Hx = self.H(x)
        return self.r * np.dot(Gx, Hx) + np.sqrt(
            self.r**2 * np.dot(-Gx, Hx) ** 2 + self.v**2
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        Gx = self.G(x)
        Hx = self.H(x)
        DGx = self.G.jacobian(x)
        DHx = self.H.jacobian(x)
        norm = np.sqrt(self.r**2 * np.dot(-Gx, Hx) ** 2 + self.v**2)
        prod = DGx.T @ Hx + DHx.T @ Gx
        return (self.r + (self.r**2 * np.dot(Gx, Hx) / norm)) * prod


class MPCCProblem(ABC):
    def __init__(
        self,
        parse_vars_fn: Callable,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        complementarity_constraint_funcs: Tuple[ConstraintFn, ConstraintFn],
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
    ):
        self.parse_vars_fn = parse_vars_fn
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs
        self.ineq_constraint_funcs = ineq_constraint_funcs
        self.G, self.H = complementarity_constraint_funcs
        self.bounds = bounds

    def compute_complementarity(self, x: np.ndarray) -> float:
        return np.linalg.norm(self.G(x) * self.H(x))

    @abstractmethod
    def _solve_nlp(self, x0, *args, **kwargs):
        pass

    @abstractmethod
    def solve(self, x0: np.ndarray, *args, **kwargs):
        """
        Solve the MPCC problem.

        Parameters:
        - x0: Initial guess for the solution.
        - args: Additional arguments for the solver.
        - kwargs: Keyword arguments for the solver.

        Returns:
        - res: Result of the optimization.
        - x: Solution to the MPCC problem.
        - fn: Final objective function value.
        """
        pass


class MPCCRelaxed(MPCCProblem):
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
        super().__init__(
            parse_vars_fn=objective_func.parse_vars,
            objective_func=objective_func,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            complementarity_constraint_funcs=complementarity_constraint_funcs,
            bounds=bounds,
        )

        self.G, self.H = complementarity_constraint_funcs
        if relaxation_type.lower() == "scholtes":
            self.complementarity_constraint_func = ScholtesComplementarityConstraintFn(
                self.G, self.H, t_init
            )
        elif relaxation_type.lower() == "fisher":
            self.complementarity_constraint_func = FisherComplementarityConstraintFn(
                self.G, self.H, t_init
            )
        else:
            raise ValueError(f"Unknown relaxation type: {relaxation_type}")
        self.bounds = bounds
        self.t = t_init

    def _solve_nlp(self, x0, t, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        tol = kwargs.get("tol", 1e-4)
        max_iter = kwargs.get("max_iter", 500)
        self.complementarity_constraint_func.t = t
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs + [self.complementarity_constraint_func],
            self.bounds,
        )
        options = {
            "print_level": print_level,
            "tol": tol,
            "max_iter": max_iter,
            # "acceptable_tol": 1e-5,
            # "constr_viol_tol": 1e-5,
            "mu_strategy": "adaptive",
            # "nlp_scaling_method": "gradient-based",
            # "check_derivatives_for_naninf": "yes",
            "compl_inf_tol": 1e-3,
            "bound_relax_factor": 1e-6,
            "sb": "yes",
        }
        return nlp.solve(x0, options=options)

    def solve(
        self,
        x0: np.ndarray,
        t_min: float = 1e-5,
        max_iter: int = 10,
        tol: float = 1e-3,
        nlp_tol: float = 1e-6,
        nlp_max_iter: int = 500,
        verbose: bool = False,
        print_level: int = 0,
        beta=0.5,
        *args,
        **kwargs,
    ):
        x = x0
        t = self.t
        res = None
        fn = None
        print(
            f'{"Iter": >5}\t{"Termination_status": >15}\t{"Objective": >15}\t{
          "MPCC_compl": >15}\t{"t": >15}\n'
        )
        for k in range(max_iter + 1):
            if t <= t_min:
                break
            res, x_, fn = self._solve_nlp(
                x,
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


class MPCCPenalized(MPCCProblem):
    def __init__(
        self,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        complementarity_constraint_funcs: Tuple[ConstraintFn, ConstraintFn],
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
        inner_penalty_init: float = 0.1,
        outer_penalty_init: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            parse_vars_fn=objective_func.parse_vars,
            objective_func=objective_func,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            complementarity_constraint_funcs=complementarity_constraint_funcs,
            bounds=bounds,
        )
        self.inner_penalty = inner_penalty_init
        self.outer_penalty = outer_penalty_init

    def _solve_nlp(self, x0, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        tol = kwargs.get("tol", 1e-4)
        max_iter = kwargs.get("max_iter", 500)

        penalty_func = HyperbolicPenaltyFn(
            self.G, self.H, self.inner_penalty, self.outer_penalty
        )

        nlp = OptimizationProblem(
            SumObjectiveFn(self.objective_func, penalty_func),
            # self.objective_func + [penalty_func],
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs,
            self.bounds,
        )

        options = {
            "print_level": print_level,
            "tol": tol,
            "maxiter": max_iter,
            # "mu_strategy": "adaptive",
            # "compl_inf_tol": 1e-3,
            # "bound_relax_factor": 1e-6,
            "sb": "yes",
        }

        return nlp.solve(x0, options=options)

    def solve(
        self,
        x0: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-3,
        nlp_tol: float = 1e-6,
        nlp_max_iter: int = 500,
        verbose: bool = False,
        print_level: int = 0,
        rho1: float = 0.5,
        rho2: float = 1.5,
        *args,
        **kwargs,
    ):
        x = x0.copy()
        if verbose:
            print(
                f'{"Iter": >5}\t{"Termination_status": >15}\t{"Objective": >15}\t{"MPCC_compl": >15}\t{"Outer_penalty": >15}\t{"Inner_penalty": >15}'
            )
        for k in range(max_iter + 1):
            res, x_, fn = self._solve_nlp(
                x,
                tol=nlp_tol,
                print_level=print_level,
                max_iter=nlp_max_iter,
            )
            self.comp = self.compute_complementarity(x_)
            if np.abs(self.comp) < tol:
                print(
                    f'{k: > 5}*\t{res["status"]: > 15}\t{fn: > 15}\t{self.comp: > 15}\t{self.outer_penalty: > 15}\t{self.inner_penalty: > 15}'
                )
                res["iter"] = k
                return res, x_, fn
            if verbose:
                print(
                    f'{k: > 5}\t{res["status"]: > 15}\t{fn: > 15}\t{self.comp: > 15}\t{self.outer_penalty: > 15}\t{self.inner_penalty: > 15}'
                )
            if np.all(self.G(x_) * self.H(x_) > -1e-3 * self.outer_penalty):
                self.outer_penalty *= rho1
            else:
                self.inner_penalty *= rho2
            x = x_
        return res, x, fn
