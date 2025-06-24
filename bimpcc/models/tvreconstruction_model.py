import numpy as np
import scipy.sparse as sp
from .model import MPCCModel, MPCCPenalizedModel
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.nlp import (
    ObjectiveFn,
    PenalizedObjectiveFn,
    ConstraintFn,
    ComplementarityConstraintFn,
)
from bimpcc.models.typings import Image
from bimpcc.utils import gaussian_blur_sparse_matrix_symmetric


def _parse_vars(x: np.ndarray, N: int, M: int):
    R = M // 2
    return (
        x[:N],
        x[N : N + M],
        x[N + M : N + M + R],
        x[N + M + R : N + M + (2 * R)],
        x[N + M + (2 * R) : N + M + (3 * R)],
        x[N + M + (3 * R) :],
    )


class TVRecObjectiveFn(ObjectiveFn):
    def __init__(
        self,
        forward_map:sp.csr_matrix,
        true_img: np.ndarray,
        M:int,
        N:int,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
    ):
        self.forward_map = forward_map
        self.true_img = true_img.flatten()
        self.M, self.N = M,N
        self.R = self.M // 2
        self.epsilon = epsilon
        self.parameter_size = parameter_size
        self.A = (forward_map.T @ forward_map).tocoo()

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        return (
            0.5 * np.linalg.norm(self.forward_map @ u - self.true_img) ** 2
        )

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def gradient(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        residual = self.forward_map @ u - self.true_img
        return np.concatenate(
            (self.forward_map.T@residual, np.zeros(5 * self.R + self.parameter_size))
        )

    def hessian(self, x: np.ndarray) -> float:
        """
        The Hessian of the objective function.

        Must return a full matrix dont know why exactly.
        """
        d = np.concatenate(
            (
                self.A,
                np.zeros(5 * self.R + self.parameter_size),
            )
        )
        # d = np.concatenate(
        #     (
        #         np.ones(self.N),
        #         self.epsilon * np.ones(5 * self.R + self.parameter_size),
        #     )
        # )
        # hess = sp.diags_array(d)
        return np.diag(d)


class StateConstraintFn(ConstraintFn):
    def __init__(
        self,
        forward_map:sp.csr_matrix,
        noisy_img: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
    ):
        self.forward_map = forward_map
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.R = self.M // 2
        self.parameter_size = parameter_size
        self.Id = sp.eye(self.N).tocoo()
        self.KT = (gradient_op.T).tocoo()
        self.Z_R = sp.coo_matrix((self.N, self.R))
        self.Z_P = sp.coo_matrix((self.N, self.parameter_size))
        self.A = (forward_map.T @ forward_map).tocoo()

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        residual = self.forward_map @ u - self.noisy_img
        return self.forward_map.T@residual + self.gradient_op.T @ q

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        jac = sp.hstack(
            [
                self.A,  # u
                self.KT,  # q
                self.Z_R,  # r
                self.Z_R,  # delta
                self.Z_R,  # theta
                self.Z_P,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class PrimalConstraintFn(ConstraintFn):
    def __init__(self, gradient_op: np.ndarray, parameter_size: int = 1):
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.R = self.M // 2
        self.parameter_size = parameter_size
        self.K = gradient_op.tocoo()
        self.Z_M = sp.coo_matrix((self.M, self.M))
        self.Z_R = sp.coo_matrix((self.M, self.R))
        self.Z_P = sp.coo_matrix((self.M, self.parameter_size))

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        r_theta = np.concatenate((r * np.cos(theta), r * np.sin(theta)))
        return self.gradient_op @ u - r_theta

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        ct = np.cos(theta)
        st = np.sin(theta)

        diag1 = sp.coo_matrix((-ct, (np.arange(self.R), np.arange(self.R))))
        diag2 = sp.coo_matrix((-st, (np.arange(self.R), np.arange(self.R))))
        diag3 = sp.coo_matrix((r * st, (np.arange(self.R), np.arange(self.R))))
        diag4 = sp.coo_matrix((-r * ct, (np.arange(self.R), np.arange(self.R))))

        Jr = sp.vstack([diag1, diag2])

        Jtheta = sp.vstack([diag3, diag4])

        jac = sp.hstack(
            [
                self.K,  # u
                self.Z_M,  # q
                Jr,  # r
                self.Z_R,  # delta
                Jtheta,  # theta
                self.Z_P,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class DualConstraintFn(ConstraintFn):
    def __init__(
        self,
        gradient_op: np.ndarray,
        gradient_op_x,
        gradient_op_y,
        parameter_size: int = 1,
    ):
        self.gradient_op = gradient_op
        self.Kx = gradient_op_x.tocoo()
        self.Ky = gradient_op_y.tocoo()
        self.M, self.N = gradient_op.shape
        self.R = self.M // 2
        self.parameter_size = parameter_size
        self.Id = sp.diags(np.ones(self.M), format="coo")
        self.Z_R = sp.coo_matrix((self.M, self.R))
        self.Z_N = sp.coo_matrix((self.M, self.N))
        self.Z_P = sp.coo_matrix((self.M, self.parameter_size))

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        delta_theta = np.concatenate((delta * np.cos(theta), delta * np.sin(theta)))
        return q - delta_theta

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        ct = np.cos(theta)
        st = np.sin(theta)

        q = q.reshape(2, -1).T

        a = (alpha * q[:, 0]) / np.maximum(alpha, np.abs(q[:, 0]))
        b = (alpha * q[:, 1]) / np.maximum(alpha, np.abs(q[:, 1]))
        diaga = sp.coo_matrix((a, (np.arange(self.R), np.arange(self.R))))
        diagb = sp.coo_matrix((b, (np.arange(self.R), np.arange(self.R))))

        diag1 = sp.coo_matrix((-ct, (np.arange(self.R), np.arange(self.R))))
        diag2 = sp.coo_matrix((-st, (np.arange(self.R), np.arange(self.R))))

        Jdelta = sp.vstack([diag1, diag2])

        Jtheta = sp.vstack([diagb, -diaga])

        jac = sp.hstack(
            [
                self.Z_N,  # u
                self.Id,  # q
                self.Z_R,  # r
                Jdelta,  # delta
                Jtheta,  # theta
                self.Z_P,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class BoundConstraintFn(ConstraintFn):
    def __init__(self, M, N, parameter_size: int = 1):
        self.M = M
        self.N = N
        self.R = M // 2
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_M = sp.coo_matrix((self.R, self.M))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.Id = sp.diags(np.ones(self.R), format="coo")

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        return alpha - delta

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        # if self.parameter_size == 1:
        Jalpha = sp.coo_matrix((np.ones(self.R), (np.arange(self.R), [0] * self.R)))
        # else:
        #     Jalpha = sp.coo_matrix(
        #         (np.ones(self.R), (np.arange(self.R), np.arange(self.R)))
        #     )
        jac = sp.hstack(
            [
                self.Z_N,  # u
                self.Z_M,  # q
                self.Z_R,  # r
                -self.Id,  # delta
                self.Z_R,  # theta
                Jalpha,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class TVDenComplementarityConstraintFn(ComplementarityConstraintFn):
    def __init__(self, M: int, N: int, t: float = 1.0, parameter_size: int = 1):
        self.M = M
        self.N = N
        self.R = M // 2
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_M = sp.coo_matrix((self.R, self.M))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.Id = sp.diags(np.ones(self.R), format="coo")
        self.t = t

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        return self.t - (r * (alpha - delta))

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = self.parse_vars(x)
        Jalpha = sp.coo_matrix((r*np.ones(self.R), (np.arange(self.R), [0] * self.R)))
        Jr = sp.coo_matrix((alpha - delta, (np.arange(self.R), np.arange(self.R))))
        Jdelta = sp.coo_matrix((r, (np.arange(self.R), np.arange(self.R))))
        # Jalpha = sp.coo_matrix((r, (np.arange(self.R), np.arange(self.R))))
        jac = sp.hstack(
            [
                self.Z_N,  # u
                self.Z_M,  # q
                -Jr,  # r
                Jdelta,  # delta
                self.Z_R,  # theta
                -Jalpha,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class TVReconstructionMPCC(MPCCModel):
    def __init__(
        self,
        forward_map:sp.csr_matrix,
        true_img: Image,
        noisy_img: Image,
        epsilon: float = 1e-5,
        t_init: float = 1.0,
        x0: np.ndarray = None,
        parameter_size: int = 1,
    ):
        Kx, Ky, self.K = generate_2D_gradient_matrices(true_img.shape[0])
        true_img = true_img.flatten()
        noisy_img = noisy_img.flatten()

        M, N = self.K.shape
        R = M // 2
        objective_func = TVRecObjectiveFn(forward_map,true_img, M, N, epsilon=epsilon)
        eq_constraint_funcs = [
            StateConstraintFn(forward_map,noisy_img, self.K),
            PrimalConstraintFn(self.K),
            DualConstraintFn(self.K, Kx, Ky),
        ]
        ineq_constraint_funcs = [BoundConstraintFn(M, N)]
        # ineq_constraint_funcs = []

        u_bounds = [(0, None)] * N
        q_bounds = [(None, None)] * M
        r_bounds = [(0, None)] * R
        delta_bounds = [(0.001, None)] * R
        theta_bounds = [(None, None)] * R
        alpha_bounds = [(0.0, None)] * (parameter_size)
        bounds = (
            u_bounds + q_bounds + r_bounds + delta_bounds + theta_bounds + alpha_bounds
        )

        if x0 is None:
            x0 = np.concatenate(
                [
                    noisy_img,
                    # np.random.randn(N),
                    1e-3 * np.ones(M),
                    1e-3 * np.ones(R),
                    1e-3 * np.ones(R),
                    1e-3 * np.ones(R),
                    1e-3 * np.ones(parameter_size),
                ]
            )

        super().__init__(
            objective_func=objective_func,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            complementarity_constraint_func=TVDenComplementarityConstraintFn(M, N),
            bounds=bounds,
            t_init=t_init,
            x0=x0,
        )

    def compute_complementarity(self, x):
        u, q, r, delta, theta, alpha = self.objective_func.parse_vars(x)
        return np.linalg.norm(np.minimum(r, alpha - delta))
    
class TVDeblurringMPCC(TVReconstructionMPCC):
    def __init__(self, true_img, blurred_img, x0=None, epsilon=0.0):
        forward_map = gaussian_blur_sparse_matrix_symmetric(true_img.shape)
        super().__init__(
            forward_map=forward_map,
            true_img=true_img,
            noisy_img=blurred_img,
            x0=x0,
            epsilon=epsilon,
        )


class PenalizedTVDenoisingMPCC(MPCCPenalizedModel):
    def __init__(
        self,
        true_img: Image,
        noisy_img: Image,
        epsilon: float = 1e-5,
        pi_init: float = 1.0,
        x0: np.ndarray = None,
        parameter_size: int = 1,
    ):
        Kx, Ky, K = generate_2D_gradient_matrices(true_img.shape[0])
        true_img = true_img.flatten()
        noisy_img = noisy_img.flatten()

        M, N = K.shape
        R = M // 2
        objective_func = PenalizedTVDenObjectiveFn(true_img, K, epsilon=epsilon)
        eq_constraint_funcs = [
            StateConstraintFn(noisy_img, K),
            PrimalConstraintFn(K),
            DualConstraintFn(K, Kx, Ky),
        ]
        ineq_constraint_funcs = [BoundConstraintFn(M, N)]

        u_bounds = [(0, None)] * N
        q_bounds = [(None, None)] * M
        r_bounds = [(0, None)] * R
        delta_bounds = [(0, None)] * R
        theta_bounds = [(None, None)] * R
        alpha_bounds = [(0, None)] * (parameter_size)
        bounds = (
            u_bounds + q_bounds + r_bounds + delta_bounds + theta_bounds + alpha_bounds
        )

        if x0 is None:
            x0 = np.concatenate(
                [
                    noisy_img,
                    # np.random.randn(N),
                    1e-1 * np.ones(M),
                    1e-1 * np.ones(R),
                    1e-1 * np.ones(R),
                    1e-1 * np.ones(R),
                    1e-1 * np.ones(parameter_size),
                ]
            )

        super().__init__(
            objective_func=objective_func,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            bounds=bounds,
            pi_init=pi_init,
            x0=x0,
        )

    def compute_complementarity(self, x):
        u, q, r, delta, theta, alpha = self.objective_func.parse_vars(x)
        # return np.dot(r, alpha - delta)
        return np.linalg.norm(np.minimum(r, alpha - delta))
