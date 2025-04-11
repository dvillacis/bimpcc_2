import numpy as np
import scipy.sparse as sp
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.utils_reg import build_index_sets, build_jacobian_matrices
from bimpcc.nlp import ObjectiveFn, ConstraintFn, OptimizationProblem
from bimpcc.models.typings import Image

# from scipy.sparse import bmat, identity, diags


def _parse_vars(x: np.ndarray, N: int, M: int):
    return (
        x[:N],
        x[N : N + M],
        x[N + M :],
    )


class TVReconsRegObjectiveFn(ObjectiveFn):
    def __init__(
        self,
        forward_map: sp.csr_matrix,
        true_img: np.ndarray,
        M: int,
        N: int,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
    ):
        self.forward_map = forward_map
        self.true_img = true_img.flatten()
        self.M, self.N = M, N
        self.epsilon = epsilon
        self.parameter_size = parameter_size
        self.A = (forward_map.T @ forward_map).tocoo()

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return (
            0.5 * np.linalg.norm(self.forward_map @ u - self.true_img) ** 2
        )

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def gradient(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        residual = self.forward_map @ u - self.true_img
        return np.concatenate(
            (self.forward_map @ residual, np.zeros(self.M + self.parameter_size))
        )

    def hessian(self, x: np.ndarray) -> float:
        """
        The Hessian of the objective function.

        Must return a full matrix dont know why exactly.
        """
        d = np.concatenate(
            (
                self.A,
                np.zeros(self.M + self.parameter_size),
            )
        )
        return np.diag(d)


class ReconsStateConstraintFn(ConstraintFn):
    def __init__(
        self,
        forward_map: sp.csr_matrix,
        noisy_img: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
    ):
        self.forward_map = forward_map
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.parameter_size = parameter_size
        self.Id = sp.eye(self.N).tocoo()
        self.KT = (self.gradient_op.T).tocoo()
        self.Z_P = sp.coo_matrix((self.N, self.parameter_size))
        self.A = (forward_map.T @ forward_map).tocoo()

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        residual = self.forward_map @ u - self.noisy_img
        return self.forward_map @ residual + self.gradient_op.T @ q

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        jac = sp.hstack(
            [
                self.A,  # u
                self.KT,  # q
                self.Z_P,  # alpha
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
        # jac = bmat(
        #     [
        #         [-H_u, identity(self.M), -H_alpha],
        #     ]
        # )
        # # print(f"jacobian nonzero elements: {jac.nnz}")
        # jac_matrix = jac.toarray()
        # # print(f"jac_matrix shape: {jac_matrix.shape}")
        # return jac_matrix.ravel()
        indices = np.arange(H_alpha.size)
        v_coo = sp.coo_matrix((H_alpha, (indices, np.zeros_like(indices))), shape=(H_alpha.size, 1))

        # Construcción de la jacobiana usando hstack
        jac = sp.hstack(
            [-H_u, self.Id, -v_coo]  # Matrices en columnas
        )
        # print(jac.shape)
        # Convertir a formato COO para compatibilidad
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class TVReconsRegularized:
    def __init__(
        self,
        forward_map: sp.csr_matrix,
        true_img: Image,
        noisy_img: Image,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
        x0: np.ndarray = None,
        *args,
        **kwargs,
    ):
        Kx, Ky, self.K = generate_2D_gradient_matrices(true_img.shape[0])
        true_img = true_img.flatten()
        noisy_img = noisy_img.flatten()

        M, N = self.K.shape

        self.objective_func = TVReconsRegObjectiveFn(
            forward_map, true_img, M, N, epsilon=epsilon
        )
        self.eq_constraint_funcs = [
            ReconsStateConstraintFn(forward_map, noisy_img, self.K),
            DualConstraintFn(noisy_img, self.K, parameter_size=parameter_size, gamma=100),
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
