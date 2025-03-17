from abc import ABC, abstractmethod
from bimpcc.nlp import (
    ConstraintFn,
    ComplementarityConstraintFn,
    ObjectiveFn,
    PenalizedObjectiveFn,
    OptimizationProblem,
)
from typing import List, Tuple, Union
from rich import print
import numpy as np


class MPCCModel(ABC):
    def __init__(
        self,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        complementarity_constraint_func: ComplementarityConstraintFn,
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
        x0: np.ndarray,
        t_init=1.0,
        *args,
        **kwargs,
    ):
        self.parse_vars_fn = objective_func.parse_vars
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs
        self.ineq_constraint_funcs = ineq_constraint_funcs
        self.complementarity_constraint_func = complementarity_constraint_func
        self.bounds = bounds
        self.t = t_init
        self.x0 = x0

    @abstractmethod
    def compute_complementarity(self, x: np.ndarray) -> float:
        pass

    def _solve_nlp(self, x0, bounds, t, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        tol = kwargs.get("tol", 1e-4)
        self.complementarity_constraint_func.t = t
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs + [self.complementarity_constraint_func],
        )
        options = {
            "print_level": print_level,
            "tol": tol,
        }
        return nlp.solve(x0, bounds, options=options)

    def solve(
        self,
        t_min: float = 1e-5,
        max_iter: int = 10,
        tol: float = 1e-3,
        nlp_tol: float = 1e-8,
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
                print(f"Intermediate result: {res}")
                # print(f"Intermediate x: {x}")
                print(f"Intermediate fn: {fn}")
                print(f"complementarity: {self.compute_complementarity(x)}")
                break
            res, x_, fn = self._solve_nlp(
                x, self.bounds, t, tol=nlp_tol, print_level=print_level
            )
            comp = self.compute_complementarity(x_)
            if np.abs(comp) < tol:
                print(
                    f'{k: > 5}*\t{res["status"]: > 15}\t{fn: > 15}\t{
                comp: > 15}\t{t: > 15}'
                )
                return res, x_, fn
            # status = ""
            if res["status"] >= 0:
                t_ = max(t_min, beta * t)
                x = x_
            else:
                t_ = 1.1 * t
                beta = 0.9 * beta
                # status = f" (FAILED {res['status']})"
            if verbose:
                print(
                    f'{k: > 5}\t{res["status"]: > 15}\t{fn: > 15}\t{
                comp: > 15}\t{t: > 15}'
                )
                # print(
                #     f"* Iteration {k+1} {status}: Solving the NLP problem for t = {t} with fn: {fn}, complementarity: {self.compute_complementarity(x)}"
                # )
            t = t_

        print(
            f"* (STOPPED) Iteration {k+1}: Solving the NLP problem for t = {t} with complementarity: {self.compute_complementarity(x)}"
        )
        return res, x, fn


class MPCCPenalizedModel(ABC):
    def __init__(
        self,
        objective_func: PenalizedObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
        x0: np.ndarray,
        pi_init: float = 1.0,
        *args,
        **kwargs,
    ):
        self.parse_vars_fn = objective_func.parse_vars
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs
        self.ineq_constraint_funcs = ineq_constraint_funcs
        self.bounds = bounds
        self.pi = pi_init
        self.x0 = x0

    @abstractmethod
    def compute_complementarity(self, x: np.ndarray) -> float:
        pass

    def _solve_nlp(self, x0, bounds, pi, mu, tol_c, tol_p, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        max_iter = kwargs.get("max_iter", 5000)
        self.objective_func.pi = pi
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs,
        )
        options = {
            "mu_init": mu,
            "mu_strategy": "monotone",
            "dual_inf_tol": tol_p,
            "constr_viol_tol": tol_p,
            "compl_inf_tol": tol_p,
            "print_level": print_level,
            "max_iter": max_iter,
            "acceptable_tol": tol_p,
            "tol": tol_p,
        }
        return nlp.solve(x0, bounds, options=options)

    def solve(
        self,
        max_iter: int = 10,
        tol: float = 1e-8,
        verbose: bool = False,
        print_level: int = 0,
        mu_init: float = 0.1,
        sigma=10,
        gamma=0.4,
        kappa=0.2,
        nu=10,
    ):
        x = self.x0
        pi = self.pi
        mu = mu_init
        info = None
        fn = None

        print(
            f'{"Iter": >5}\t{"Termination_status": >15}\t{"Objective": >15}\t{
          "MPCC_compl": >15}\t{"lg(mu)": >15}\t{"π": >15}\n'
        )

        for k in range(max_iter):
            tol_c = mu**gamma
            tol_p = nu * mu
            info_, x_, fn_ = self._solve_nlp(
                x, self.bounds, pi, mu, tol_c, tol_p, tol=tol, print_level=print_level
            )
            comp = self.compute_complementarity(x_)
            print(
                f'{k: > 5}\t{info_["status"]: > 15}\t{fn_: > 15}\t{
              comp: > 15}\t{np.log10(mu): > 15}\t{pi: > 15}'
            )
            if (np.abs(comp) < tol) & (info_["status"] == 0):
                print(
                    f"Obtained solution satisfies the complementarity condition at {comp} at {k+1} iterations"
                )
                return info_, x_, fn_
            if (np.abs(comp) <= tol_c) & (info_["status"] >= 0):
                info = info_
                x = x_
                fn = fn_
                #mu *= kappa
            # else:
            #     if pi < 1e10:
            #         pi *= sigma
            #     else:
            #         print("The problem is unbounded")
            #         break
        return info, x, fn
