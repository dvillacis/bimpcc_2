import numpy as np
import scipy.sparse as sp
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.utils_reg_sinbeta import build_index_sets, build_jacobian_matrices
from bimpcc.nlp import ObjectiveFn, ConstraintFn, OptimizationProblem
from bimpcc.models.typings import Image
from bimpcc.utils_tvq_sinbeta import diagonal_j_rho, build_nabla_u


def _parse_vars(x: np.ndarray, N: int, M: int):
    return (
        x[:N],
        x[N : N + M],
        x[N + M :],
    )


class TVDenRegObjectiveFn(ObjectiveFn):
    def __init__(
        self,
        true_img: np.ndarray,
        gradient_op: np.ndarray,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
    ):
        self.true_img = true_img.flatten()
        self.K = gradient_op
        self.M, self.N = gradient_op.shape
        self.epsilon = epsilon
        self.parameter_size = parameter_size

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return 0.5 * np.linalg.norm(u - self.true_img) ** 2
        # return 0.5 * np.linalg.norm(u - self.true_img) ** 2 + self.epsilon * np.linalg.norm(alpha) ** 2

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def gradient(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return np.concatenate(
            (u - self.true_img, np.zeros(self.M + self.parameter_size))
        )

    def hessian(self, x: np.ndarray) -> float:
        """
        The Hessian of the objective function.

        Must return a full matrix dont know why exactly.
        """
        d = np.concatenate(
            (
                np.ones(self.N),
                np.zeros(self.M + self.parameter_size),
            )
        )
        return np.diag(d)


class StateConstraintFn(ConstraintFn):
    def __init__(
        self,
        noisy_img: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
        gamma: int = 100,
        rho: float = 1e-3,
        q_param: float = 0.99,
    ):
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.parameter_size = parameter_size
        self.gamma = gamma
        self.rho = rho
        self.q_param = q_param
        self.delta_gamma = (gamma ** (1 - q_param)) * (q_param**q_param)
        self.Id = sp.eye(self.N).tocoo()
        self.KT = (self.gradient_op.T).tocoo()
        self.K = self.gradient_op.tocoo()

        # ---- Conservative sparse superset for W_u (k-hop closure) ----
        KTK = (self.KT @ self.K).tocsr()

        # Powers of KTK (patterns grow: 1-hop, 2-hop, 3-hop, ...)
        KTK1 = KTK.tocoo()
        KTK1.sum_duplicates()

        KTK2 = (KTK @ KTK).tocoo()
        KTK2.sum_duplicates()

        KTK3 = (KTK2.tocsr() @ KTK).tocoo()
        KTK3.sum_duplicates()

        # If you still see extras later, uncomment KTK4
        # KTK4 = (KTK3.tocsr() @ KTK).tocoo()
        # KTK4.sum_duplicates()

        # Representative pattern from build_nabla_u at u0
        u0 = self.noisy_img
        beta0 = 1.0
        W_u0 = build_nabla_u(
            u0,
            self.K,
            self.q_param,
            beta0,
            self.delta_gamma,
            self.gamma,
            self.rho,
            self.N,
            self.M,
        ).tocoo()
        W_u0.sum_duplicates()

        # Union of patterns (KTK1 ∪ KTK2 ∪ KTK3 ∪ W_u0)
        r = np.concatenate([KTK1.row, KTK2.row, KTK3.row, W_u0.row]).astype(np.int64, copy=False)
        c = np.concatenate([KTK1.col, KTK2.col, KTK3.col, W_u0.col]).astype(np.int64, copy=False)

        key = r * np.int64(self.N) + c
        uniq = np.unique(key)
        r_u = (uniq // np.int64(self.N)).astype(int)
        c_u = (uniq % np.int64(self.N)).astype(int)

        self._W_u_pattern_zeros = sp.coo_matrix(
            (np.zeros(r_u.size, dtype=float), (r_u, c_u)),
            shape=(self.N, self.N),
        )
        self._W_u_diag_zeros = sp.coo_matrix(
            (np.zeros(self.N, dtype=float), (np.arange(self.N), np.arange(self.N))),
            shape=(self.N, self.N),
        )

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        Da = diagonal_j_rho(
            self.K @ u, self.delta_gamma, self.q_param, self.gamma, self.rho
        )
        return (-1 / self.delta_gamma) * (
            self.KT @ Da @ self.K @ u - alpha * (u - self.noisy_img)
        ) + self.K.T @ q

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray):
        u, q, alpha = self.parse_vars(x)
        beta = float(np.asarray(alpha).squeeze())

        W_u = build_nabla_u(
            u,
            self.K,
            self.q_param,
            beta,
            self.delta_gamma,
            self.gamma,
            self.rho,
            self.N,
            self.M,
        )

        # Force constant sparsity (even if some values become exactly 0)
        W_u = (W_u + self._W_u_pattern_zeros + self._W_u_diag_zeros).tocoo()
        W_u.sum_duplicates()

        vect = (1 / self.delta_gamma) * (u - self.noisy_img)
        rows = np.arange(self.N, dtype=int)
        cols = np.zeros(self.N, dtype=int)
        vect_s = sp.coo_matrix((vect.astype(float), (rows, cols)), shape=(self.N, 1))
        vect_s.sum_duplicates()

        jac = sp.hstack([W_u, self.KT, vect_s], format="coo").tocoo()
        jac.sum_duplicates()
        return jac


class DualConstraintFn(ConstraintFn):
    def __init__(
        self,
        noisy_img: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
        gamma: int = 100,
    ):
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.parameter_size = parameter_size
        self.M, self.N = gradient_op.shape
        self.gamma = gamma
        self.Id = sp.eye(self.M).tocoo()
        self.Z_P = sp.coo_matrix((self.M, self.parameter_size))

        # ---- Conservative sparse superset for H_u: pattern(K) ----
        Kcoo = self.gradient_op.tocoo()
        Kcoo.sum_duplicates()
        self._H_u_pattern_zeros = sp.coo_matrix(
            (np.zeros_like(Kcoo.data, dtype=float), (Kcoo.row, Kcoo.col)),
            shape=(self.M, self.N),
        )

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        K = self.gradient_op.tocoo()
        Ku = K @ u
        A_gamma, I_gamma, S_gamma, L_1, L_2 = build_index_sets(Ku, self.gamma, self.M)
        return q - (A_gamma @ L_1 + S_gamma @ L_2 + self.gamma * I_gamma) @ Ku

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray):
        u, q, alpha = self.parse_vars(x)
        H_u = build_jacobian_matrices(self.gradient_op, u, q, self.gamma, self.M)

        # Force constant sparsity (active-set logic can zero entries)
        H_u = (H_u + self._H_u_pattern_zeros).tocoo()
        H_u.sum_duplicates()

        jac = sp.hstack([-H_u, self.Id, self.Z_P], format="coo").tocoo()
        jac.sum_duplicates()
        return jac


class TVqRegularized:
    def __init__(
        self,
        true_img: Image,
        noisy_img: Image,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
        x0: np.ndarray = None,
        q_param: float = 0.99,
        *args,
        **kwargs,
    ):
        Kx, Ky, self.K = generate_2D_gradient_matrices(true_img.shape[0])
        true_img = true_img.flatten()
        noisy_img = noisy_img.flatten()

        M, N = self.K.shape

        self.objective_func = TVDenRegObjectiveFn(
            true_img, self.K, epsilon=epsilon, parameter_size=parameter_size
        )
        self.eq_constraint_funcs = [
            StateConstraintFn(
                noisy_img, self.K, parameter_size=parameter_size, q_param=q_param
            ),
            DualConstraintFn(
                noisy_img, self.K, parameter_size=parameter_size, gamma=100
            ),
        ]
        self.ineq_constraint_funcs = []

        u_bounds = [(0, None)] * N
        q_bounds = [(None, None)] * M
        alpha_bounds = [(0.0, None)] * (parameter_size)
        self.bounds = u_bounds + q_bounds + alpha_bounds

        if x0 is None:
            self.x0 = np.concatenate(
                [
                    noisy_img,
                    # np.random.randn(N),
                    1e-3 * np.ones(M),
                    100 * np.ones(parameter_size),
                ]
            )
        else:
            self.x0 = x0

    def solve(self, max_iter: int = 3000, tol: float = 1e-4, print_level: int = 5):
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs,
        )
        options = {
            "print_level": print_level,
            "max_iter": max_iter,
            "tol": tol,
            "check_derivatives_for_naninf": "yes",
        }
        return nlp.solve(self.x0, self.bounds, options=options, use_jacobian_sparsity=True)
