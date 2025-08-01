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
        