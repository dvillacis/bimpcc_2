from abc import ABC, abstractmethod
from bimpcc.nlp import (
    ConstraintFn,
    ComplementarityConstraintFn,
    ObjectiveFn,
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
        t_init=1e-2,
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
        return nlp.solve(x0, bounds, print_level=print_level, tol=tol)

    def solve(
        self,
        t_min: float = 1e-5,
        max_iter: int = 10,
        tol: float = 1e-8,
        verbose: bool = False,
        print_level: int = 0,
        *args,
        **kwargs,
    ):
        x = self.x0
        t = self.t
        res = None
        fn = None
        for k in range(max_iter):
            if t <= t_min:
                print(f"Intermediate result: {res}")
                # print(f"Intermediate x: {x}")
                print(f"Intermediate fn: {fn}")
                print(f"complementarity: {self.compute_complementarity(x)}")
                break
            res, x, fn = self._solve_nlp(x, self.bounds, t, tol=tol, print_level=print_level)
            if (res["status"] == 0):
                if verbose:
                    print(f"* Iteration {k+1}: Solving the NLP problem for t = {t} with fn: {fn}, complementarity: {self.compute_complementarity(x)}")
                t = max(t_min, t / 2)
            else:
                print(f"Intermediate result: {res}")
                # print(f"Intermediate x: {x}")
                print(f"Intermediate fn: {fn}")
                print(f"complementarity: {self.compute_complementarity(x)}")
                print(f"* (STOPPED) Iteration {k+1}: Solving the NLP problem for t = {t} with complementarity: {self.compute_complementarity(x)}")
                break
        return res, x, fn
