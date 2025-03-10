import numpy as np
import scipy.sparse as sp
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.utils_reg import build_index_sets, build_jacobian_matrices
from bimpcc.nlp import ObjectiveFn, ConstraintFn, OptimizationProblem
from bimpcc.models.typings import Image

# from scipy.sparse import bmat, identity, diags
from scipy.sparse import diags


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
        # v = np.concatenate((q, r, delta, theta, alpha))
        return (
            0.5 * np.linalg.norm(u - self.true_img) ** 2
            # + 0.5 * self.epsilon * np.linalg.norm(v) ** 2
        )
        # return 0.5 * np.linalg.norm(u - self.true_img) ** 2 + self.epsilon * np.linalg.norm(alpha) ** 2

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def gradient(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        # v = np.concatenate((q, r, delta, theta, alpha))
        return np.concatenate(
            (u - self.true_img, np.zeros(self.M + self.parameter_size))
        )
        # return np.concatenate((u - self.true_img, self.epsilon * v))

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
    ):
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.parameter_size = parameter_size
        self.Id = sp.eye(self.N).tocoo()
        self.KT = (self.gradient_op.T).tocoo()
        self.Z_P = sp.coo_matrix((self.N, self.parameter_size))

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return u - self.noisy_img + self.gradient_op.T @ q

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        jac = sp.hstack(
            [
                self.Id,  # u
                self.KT,  # q
                self.Z_P,  # alpha
            ]
        )
        print(jac.shape)
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
        # Jacobian sparsity structure
        o = np.ones(self.M)
        H = self.M // 2
        D = diags((o[:H], o, o[H:]), offsets=(-H, 0, H))
        self.H_u_sparsity_structure = D @ self.gradient_op

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        Ku = self.gradient_op @ u
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
        # Construcción de la jacobiana usando hstack
        jac = sp.hstack(
            [-H_u, self.Id, -H_alpha.reshape((self.M,1))]  # Matrices en columnas
        )
        print(jac.shape)
        # Convertir a formato COO para compatibilidad
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class TVRegularized:
    def __init__(
        self,
        true_img: Image,
        noisy_img: Image,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
        x0: np.ndarray = None,
        *args,
        **kwargs,
    ):
        Kx, Ky, K = generate_2D_gradient_matrices(true_img.shape[0])
        true_img = true_img.flatten()
        noisy_img = noisy_img.flatten()

        M, N = K.shape

        self.objective_func = TVDenRegObjectiveFn(
            true_img, K, epsilon=epsilon, parameter_size=parameter_size
        )
        self.eq_constraint_funcs = [
            StateConstraintFn(noisy_img, K, parameter_size=parameter_size),
            DualConstraintFn(noisy_img, K, parameter_size=parameter_size, gamma=100),
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
        }
        return nlp.solve(self.x0, self.bounds, options=options)
