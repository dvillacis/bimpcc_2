import numpy as np
import scipy.sparse as sp
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.utils_reg import build_index_sets, build_jacobian_matrices
from bimpcc.nlp import ObjectiveFn, ConstraintFn, OptimizationProblem
from bimpcc.models.typings import Image

from bimpcc.utils_tvq import diagonal_j_rho, build_nabla_u


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
        self.Z_P = sp.coo_matrix((self.N, self.parameter_size))

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        Da = diagonal_j_rho(
            self.K @ u, alpha, self.delta_gamma, self.q_param, self.gamma, self.rho
        )
        return (-1 / self.delta_gamma) * (
            self.KT @ Da @ self.K @ u - u + self.noisy_img
        ) + self.K.T @ q

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        W_u, W_beta = build_nabla_u(
            u,
            self.K,
            self.q_param,
            alpha,
            self.delta_gamma,
            self.gamma,
            self.rho,
            self.N,
            self.M,
        )
        # W_u = (-1/self.delta_gamma) * (self.K.T @ nabla_u_w - self.Id)
        # W_beta = (-1/self.delta_gamma)*self.K.T@nabla_beta_w
        jac = sp.hstack(
            [
                W_u,  # u
                self.KT,  # q
                W_beta,  # alpha
            ]
        )
        # print(jac.shape)
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


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
        self.M, self.N = gradient_op.shape
        self.gamma = gamma
        self.Id = sp.eye(self.M).tocoo()

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        K = self.gradient_op.tocoo()
        Ku = K @ u
        A_gamma, I_gamma, S_gamma, L_1, L_2 = build_index_sets(
            Ku, alpha, self.gamma, self.M
        )
        return q - (A_gamma @ L_1 + S_gamma @ L_2 + self.gamma * I_gamma) @ Ku

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        H_u, H_alpha = build_jacobian_matrices(
            self.gradient_op, u, q, alpha, self.gamma, self.M
        )
        indices = np.arange(H_alpha.size)
        v_coo = sp.coo_matrix(
            (H_alpha, (indices, np.zeros_like(indices))), shape=(H_alpha.size, 1)
        )

        # Construcción de la jacobiana usando hstack
        jac = sp.hstack(
            [-H_u, self.Id, -v_coo]  # Matrices en columnas
        )
        # Convertir a formato COO para compatibilidad
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


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
                    1e-3 * np.ones(parameter_size),
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
        return nlp.solve(self.x0, self.bounds, options=options)
