import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp
from bimpcc.nlp import OptimizationProblem, ConstraintFn, ObjectiveFn
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.dataset import Dataset


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


def _generate_index(u, gradient_op):
    Ku = (gradient_op @ u).reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=Ku)
    act = np.where(normKu <= 1e-10, 1, 0)
    inact = 1 - act
    return act, inact


def _compute_gradient(u, gradient_op):
    Ku = gradient_op @ u
    Ku_ = (gradient_op @ u).reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=Ku_)
    act = np.where(normKu <= 1e-10, 1, 0)
    inact = 1 - act
    inv_normKu = np.where(normKu <= 1e-10, 0, 1 / normKu)
    return Ku, Ku_, normKu, inv_normKu, act, inact


def _build_tensor(Ku, inv_normKu, Kx, Ky, INACT):
    Kxu2 = Ku[:, 0] ** 2
    Kyu2 = Ku[:, 1] ** 2
    KxuKyu = Ku[:, 0] * Ku[:, 1]

    a = INACT * inv_normKu
    b = INACT * Kxu2 * (inv_normKu**3)
    c = INACT * Kyu2 * (inv_normKu**3)
    d = INACT * KxuKyu * (inv_normKu**3)

    A = Kx.multiply(a[:, None])
    B = Kx.multiply(b[:, None])
    C = Ky.multiply(d[:, None])
    D = Ky.multiply(a[:, None])
    E = Ky.multiply(c[:, None])
    F = Kx.multiply(d[:, None])

    T = sp.vstack([A - B - C, D - E - F]).toarray()

    pattern = sp.vstack([Kx + Ky, Kx + Ky]).toarray()

    rows, cols = np.where(pattern != 0)
    data = T[rows, cols]
    T_sparse = sp.coo_matrix((data, (rows, cols)), shape=(2 * Ku.shape[0], Kx.shape[1]))

    return T_sparse


class TVDenObjectiveFn(ObjectiveFn):
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
        self.R = self.M // 2
        self.epsilon = epsilon
        self.parameter_size = parameter_size

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        v = np.concatenate((q, r, delta, theta, alpha))
        return (
            0.5 * np.linalg.norm(u - self.true_img) ** 2
            # + self.epsilon * np.linalg.norm(v) ** 2
        )
        # return 0.5 * np.linalg.norm(u - self.true_img) ** 2 + self.epsilon * np.linalg.norm(alpha) ** 2

    def gradient(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        # v = np.concatenate((q, r, delta, theta, alpha))
        return np.concatenate(
            (u - self.true_img, np.zeros(5 * self.R + self.parameter_size))
        )

    def hessian(self, x: np.ndarray) -> float:
        """
        The Hessian of the objective function.

        Must return a full matrix dont know why exactly.
        """
        d = np.concatenate(
            (
                np.ones(self.N),
                np.zeros(5 * self.R + self.parameter_size),
            )
        )
        # hess = sp.diags_array(d)
        return np.diag(d)


class StateConstraintFn(ConstraintFn):
    def __init__(
        self, noisy_img: np.ndarray, gradient_op: np.ndarray, parameter_size: int = 1
    ):
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.R = self.M // 2
        self.parameter_size = parameter_size
        self.Id = sp.eye(self.N).tocoo()
        self.KT = (self.gradient_op.T).tocoo()
        self.Z_R = sp.coo_matrix((self.N, self.R))
        self.Z_P = sp.coo_matrix((self.N, self.parameter_size))

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return u - self.noisy_img + self.gradient_op.T @ q

    def jacobian(self, x: np.ndarray) -> float:
        jac = sp.hstack(
            [
                self.Id,  # u
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
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        r_theta = np.concatenate((r * np.cos(theta), r * np.sin(theta)))
        return self.gradient_op @ u - r_theta

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
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
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        delta_theta = np.concatenate((delta * np.cos(theta), delta * np.sin(theta)))
        return q - delta_theta

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
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
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return alpha - delta

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


class ComplementarityConstraintFn(ConstraintFn):
    def __init__(self, M, N, parameter_size: int = 1, t=1e-1):
        self.M = M
        self.N = N
        self.R = M // 2
        self.t = t
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_M = sp.coo_matrix((self.R, self.M))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.Id = sp.diags(np.ones(self.R), format="coo")

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return self.t - r * (alpha - delta)

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        Jalpha = sp.coo_matrix((np.ones(self.R), (np.arange(self.R), [0]*self.R)))
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


class TVDenoising:
    def __init__(self, true_img, noisy_img, gamma=100, epsilon=1e-5):
        self.true_img = true_img.flatten()
        self.noisy_img = noisy_img.flatten()
        self.gamma = gamma
        self.epsilon = epsilon
        self.Kx, self.Ky, self.K = generate_2D_gradient_matrices(true_img.shape[0])
        self.M, self.N = self.K.shape
        self.R = self.M // 2
        self.nlp = OptimizationProblem(
            TVDenObjectiveFn(self.true_img, self.K, epsilon),
            eq_constraint_funcs=[
                StateConstraintFn(self.noisy_img, self.K),
                PrimalConstraintFn(self.K),
                DualConstraintFn(self.K, self.Kx, self.Ky),
            ],
            ineq_constraint_funcs=[
                BoundConstraintFn(self.M, self.N),
                ComplementarityConstraintFn(self.M, self.N, t=1.0),
            ],
        )

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def solve(self, x0=None, bounds=None, parameter_size: int = 1, print_level=0):
        if x0 is None:
            u0 = self.noisy_img
            q0 = 0.01 * np.ones(self.M)
            r0 = 0.01 * np.ones(self.R)
            delta0 = 0.01 * np.ones(self.R)
            theta0 = 0.01 * np.ones(self.R)
            alpha0 = 0.01 * np.ones(1)
            x0 = np.concatenate((u0, q0, r0, delta0, theta0, alpha0))
        if bounds is None:
            u_bounds = [(0, None)] * (self.N)
            q_bounds = [(None, None)] * (self.M)
            r_bounds = [(0, None)] * (self.R)
            delta_bounds = [(0, None)] * (self.R)
            theta_bounds = [(None, None)] * (self.R)
            alpha_bounds = [(0, None)] * (parameter_size)
            bounds = (
                u_bounds
                + q_bounds
                + r_bounds
                + delta_bounds
                + theta_bounds
                + alpha_bounds
            )
        return self.nlp.solve(x0, bounds, print_level=print_level)


@dataclass
class TVDenoisingModel:
    dataset: Dataset
    base_model: TVDenoising
