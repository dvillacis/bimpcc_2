from typing import Protocol, Tuple, List, Dict, Any
import cyipopt
from cyipopt import minimize_ipopt
import numpy as np
import scipy.sparse as sp


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

class _CyIpoptSparseJacNLP:
    """
    IPOPT NLP wrapper exposing jacobianstructure(), so IPOPT can exploit sparsity.

    Assumption: each constraint's Jacobian sparsity pattern is constant across iterations.
    """

    def __init__(
        self,
        objective_func,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        x0: np.ndarray,
    ) -> None:
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs or []
        self.ineq_constraint_funcs = ineq_constraint_funcs or []
        
        # Merge all into a flat list of (type, fun)
        self._all_constraints = (
            [("eq", c) for c in self.eq_constraint_funcs]
            + [("ineq", c) for c in self.ineq_constraint_funcs]
        )

        self.n = int(np.asarray(x0).size)
        
        # 1. Calculate bounds and sizes
        cl_list, cu_list, self._c_sizes = [], [], []
        for ctype, cfun in self._all_constraints:
            gx0 = np.atleast_1d(cfun(x0)).astype(float)
            m_i = int(gx0.size)
            self._c_sizes.append(m_i)
            if ctype == "eq":
                cl_list.extend([0.0] * m_i)
                cu_list.extend([0.0] * m_i)
            else:
                cl_list.extend([0.0] * m_i)
                cu_list.extend([np.inf] * m_i)

        self.m = int(sum(self._c_sizes))
        self.cl = np.asarray(cl_list, dtype=float)
        self.cu = np.asarray(cu_list, dtype=float)

        # 2. Build sparsity pattern ONCE at x0
        rows_all, cols_all = [], []
        self._block_meta = [] # Storage for (constraint_func, expected_nnz)

        row_offset = 0
        for (ctype, cfun), m_i in zip(self._all_constraints, self._c_sizes):
            J = cfun.jacobian(x0)
            Jcoo = J.tocoo() if sp.issparse(J) else sp.coo_matrix(J)
            Jcoo.sum_duplicates() # Vital: Merge duplicates

            # Canonical sort order is required for consistent value mapping
            order = np.lexsort((Jcoo.col, Jcoo.row))
            
            # Store global pattern
            rows_all.append(Jcoo.row[order] + row_offset)
            cols_all.append(Jcoo.col[order])
            
            self._block_meta.append({
                "fun": cfun,
                "expected_nnz": Jcoo.nnz,
            })
            row_offset += m_i

        self._jac_rows = np.concatenate(rows_all).astype(int) if rows_all else np.array([], dtype=int)
        self._jac_cols = np.concatenate(cols_all).astype(int) if cols_all else np.array([], dtype=int)

    # IPOPT Callbacks
    def objective(self, x: np.ndarray) -> float:
        return float(self.objective_func(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.objective_func.gradient(x), dtype=float)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        parts = [np.atleast_1d(cfun(x)) for _, cfun in self._all_constraints]
        return np.concatenate(parts).astype(float) if parts else np.array([], dtype=float)

    def jacobianstructure(self):
        return self._jac_rows, self._jac_cols

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        data_blocks = []
        for i, meta in enumerate(self._block_meta):
            cfun = meta["fun"]
            expected_nnz = meta["expected_nnz"]

            J = cfun.jacobian(x)
            Jcoo = J.tocoo() if sp.issparse(J) else sp.coo_matrix(J)
            Jcoo.sum_duplicates()

            # Fast validation check
            if Jcoo.nnz != expected_nnz:
                raise ValueError(
                    f"Sparsity changed in block {i} ({type(cfun).__name__}). "
                    f"Expected {expected_nnz} nnz, got {Jcoo.nnz}. "
                    "Ensure your model returns a FIXED pattern (use supersets)."
                )

            # Sort values to match the pattern order established in __init__
            order = np.lexsort((Jcoo.col, Jcoo.row))
            data_blocks.append(Jcoo.data[order])

        return np.concatenate(data_blocks).astype(float) if data_blocks else np.array([], dtype=float)


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
        self,
        x0: List[float],
        bounds: List[Tuple[float, float]],
        options: dict = {},
        use_jacobian_sparsity: bool = False,
    ):
        x0 = np.asarray(x0, dtype=float)

        if not use_jacobian_sparsity:
            result = minimize_ipopt(
                fun=self.objective_func,
                jac=self.gradient_objective_func,
                x0=x0,
                bounds=bounds,
                constraints=self.constraints,
                options=options,
            )
            return result, result["x"], result["fun"]

        # --- Sparse-jacobian IPOPT path (uses jacobianstructure) ---
        lb = np.array([(-np.inf if b[0] is None else float(b[0])) for b in bounds], dtype=float)
        ub = np.array([(np.inf if b[1] is None else float(b[1])) for b in bounds], dtype=float)

        nlp = _CyIpoptSparseJacNLP(
            objective_func=self.objective_func,
            eq_constraint_funcs=[c["fun"] for c in self.constraints if c["type"] == "eq"],
            ineq_constraint_funcs=[c["fun"] for c in self.constraints if c["type"] == "ineq"],
            x0=x0,
        )

        prob = cyipopt.Problem(
            n=nlp.n,
            m=nlp.m,
            problem_obj=nlp,
            lb=lb,
            ub=ub,
            cl=nlp.cl,
            cu=nlp.cu,
        )

        # If you don't provide a Hessian callback, IPOPT typically needs limited-memory
        if "hessian_approximation" not in options:
            prob.add_option("hessian_approximation", "limited-memory")

        for k, v in (options or {}).items():
            prob.add_option(k, v)

        x_opt, info = prob.solve(x0)
        result = {"x": x_opt, "fun": info.get("obj_val", np.nan), "info": info, "status": info.get("status", None)}
        return result, result["x"], result["fun"]


class UnconstrainedOptimizationProblem(OptimizationProblem):
    def __init__(self, objective_func: ObjectiveFn):
        super().__init__(objective_func, [], [])
