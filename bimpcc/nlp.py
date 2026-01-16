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
        self._all_constraints: List[Tuple[str, ConstraintFn]] = (
            [("eq", c) for c in self.eq_constraint_funcs]
            + [("ineq", c) for c in self.ineq_constraint_funcs]
        )

        self.n = int(np.asarray(x0).size)

        # Constraint sizes and bounds (cl <= g(x) <= cu)
        cl_list: List[float] = []
        cu_list: List[float] = []
        self._c_sizes: List[int] = []

        for ctype, cfun in self._all_constraints:
            gx0 = np.atleast_1d(cfun(x0)).astype(float)
            m_i = int(gx0.size)
            self._c_sizes.append(m_i)
            if ctype == "eq":
                cl_list.extend([0.0] * m_i)
                cu_list.extend([0.0] * m_i)
            else:  # scipy-style "ineq": g(x) >= 0
                cl_list.extend([0.0] * m_i)
                cu_list.extend([np.inf] * m_i)

        self.m = int(sum(self._c_sizes))
        self.cl = np.asarray(cl_list, dtype=float)
        self.cu = np.asarray(cu_list, dtype=float)

        # Build global Jacobian sparsity pattern at x0
        rows_all: List[np.ndarray] = []
        cols_all: List[np.ndarray] = []
        self._jac_blocks: List[Dict[str, Any]] = []

        row_offset = 0
        for (ctype, cfun), m_i in zip(self._all_constraints, self._c_sizes):
            J = cfun.jacobian(x0)
            Jcoo = J.tocoo() if sp.issparse(J) else sp.coo_matrix(J)
            Jcoo.sum_duplicates()

            # Canonical order: sort by (row, col)
            order = np.lexsort((Jcoo.col, Jcoo.row))
            r = np.asarray(Jcoo.row[order], dtype=int)
            c = np.asarray(Jcoo.col[order], dtype=int)

            rows_all.append(r + row_offset)
            cols_all.append(c)

            # Store block pattern in GLOBAL coordinates and also as sortable "keys"
            r_g = r + row_offset
            c_g = c
            pat_keys = (r_g.astype(np.int64) * np.int64(self.n) + c_g.astype(np.int64))

            self._jac_blocks.append(
                {
                    "fun": cfun,
                    "row_offset": row_offset,
                    "r_pattern": r_g.astype(int),
                    "c_pattern": c_g.astype(int),
                    "pat_keys": pat_keys.astype(np.int64),
                }
            )

            row_offset += m_i

        self._jac_rows = np.concatenate(rows_all).astype(int) if rows_all else np.array([], dtype=int)
        self._jac_cols = np.concatenate(cols_all).astype(int) if cols_all else np.array([], dtype=int)

    # IPOPT callbacks
    def objective(self, x: np.ndarray) -> float:
        return float(self.objective_func(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.objective_func.gradient(x), dtype=float)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        parts: List[np.ndarray] = []
        for _, cfun in self._all_constraints:
            parts.append(np.atleast_1d(cfun(x)).astype(float))
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    def jacobianstructure(self):
        return self._jac_rows, self._jac_cols

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        data_blocks: List[np.ndarray] = []

        for blk_i, blk in enumerate(self._jac_blocks):
            cfun = blk["fun"]
            row_offset = blk["row_offset"]
            pat_keys = blk["pat_keys"]  # sorted because (row,col) pattern is sorted

            J = cfun.jacobian(x)
            Jcoo = J.tocoo() if sp.issparse(J) else sp.coo_matrix(J)
            Jcoo.sum_duplicates()

            # Sort current Jacobian by (row, col) in LOCAL coords first
            order = np.lexsort((Jcoo.col, Jcoo.row))
            r_loc = np.asarray(Jcoo.row[order], dtype=int)
            c = np.asarray(Jcoo.col[order], dtype=int)
            v = np.asarray(Jcoo.data[order], dtype=float)

            # Convert LOCAL rows -> GLOBAL rows and build sortable keys
            r_g = (r_loc + row_offset).astype(np.int64)
            c_g = c.astype(np.int64)
            cur_keys = r_g * np.int64(self.n) + c_g

            # Ensure the constraint does NOT introduce new nonzeros outside the declared structure
            # (If it does, you must enlarge jacobianstructure.)
            idx_in_pat = np.searchsorted(pat_keys, cur_keys)
            outside = (idx_in_pat >= pat_keys.size) | (pat_keys[idx_in_pat] != cur_keys)
            if np.any(outside):
                raise ValueError(
                    "Constraint Jacobian has entries outside jacobianstructure(). "
                    f"Block={blk_i}, fun={type(cfun).__name__}, extras={int(np.count_nonzero(outside))}."
                )

            # Fill values aligned to the fixed pattern; missing entries stay at 0.0
            data = np.zeros(pat_keys.size, dtype=float)
            idx = np.searchsorted(cur_keys, pat_keys)

            # idx points into cur_keys (which is sorted because order sorted by row/col)
            mask = (idx < cur_keys.size) & (cur_keys[idx] == pat_keys)
            data[mask] = v[idx[mask]]

            data_blocks.append(data)

        return np.concatenate(data_blocks) if data_blocks else np.array([], dtype=float)


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
