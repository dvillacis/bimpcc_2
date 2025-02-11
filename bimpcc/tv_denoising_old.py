import numpy as np
import scipy.sparse as sp
from bimpcc.nlp import OptimizationProblem, ConstraintFn, ObjectiveFn
from bimpcc.utils import generate_2D_gradient_matrices


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
        self, true_img: np.ndarray, N, M, epsilon: float = 1e-5, parameter_size: int = 1
    ):
        self.true_img = true_img.flatten()
        self.N = N
        self.M = M
        self.R = M // 2
        self.epsilon = epsilon
        self.parameter_size = parameter_size

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return (
            0.5 * np.linalg.norm(u - self.true_img) ** 2
            + 0.5 * self.epsilon * np.linalg.norm(alpha) ** 2
        )

    def gradient(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return np.concatenate(
            (u - self.true_img, np.zeros(5 * self.R), self.epsilon * alpha)
        )

    def hessian(self, x: np.ndarray) -> float:
        """
        The Hessian of the objective function.

        Must return a full matrix dont know why exactly.
        """
        d = np.concatenate(
            (
                np.ones(self.N),
                0.0 * np.ones(5 * self.R),
                self.epsilon * np.ones(self.parameter_size),
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

    def hessian(self, x: np.ndarray, _lambda) -> float:
        return None


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
        _, INACT = _generate_index(u, self.gradient_op)
        r_theta_ = np.concatenate(
            (INACT * r * np.cos(theta), INACT * r * np.sin(theta))
        )
        return self.gradient_op @ u - r_theta_

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        ct = np.cos(theta)
        st = np.sin(theta)
        _, INACT = _generate_index(u, self.gradient_op)
        diag1 = sp.coo_matrix((-INACT * ct, (np.arange(self.R), np.arange(self.R))))
        diag2 = sp.coo_matrix((-INACT * st, (np.arange(self.R), np.arange(self.R))))
        diag3 = sp.coo_matrix((INACT * r * st, (np.arange(self.R), np.arange(self.R))))
        diag4 = sp.coo_matrix((-INACT * r * ct, (np.arange(self.R), np.arange(self.R))))

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

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        _, Ku, normKu, inv_normKu, ACT, INACT = _compute_gradient(u, self.gradient_op)
        gamma = np.concatenate(
            (
                alpha[0] * INACT * Ku[:, 0] * inv_normKu,
                alpha[0] * INACT * Ku[:, 1] * inv_normKu,
            )
        )
        delta_theta_ = np.concatenate(
            (ACT * delta * np.cos(theta), ACT * delta * np.sin(theta))
        )
        return q - gamma - delta_theta_

    def jacobian(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        ct = np.cos(theta)
        st = np.sin(theta)
        _, Ku, normKu, inv_normKu, ACT, INACT = _compute_gradient(u, self.gradient_op)
        data_alpha = np.concatenate(
            (-INACT * Ku[:, 0] * inv_normKu, -INACT * Ku[:, 1] * inv_normKu)
        )
        Jalpha = sp.coo_matrix((data_alpha, (np.arange(self.M), [0]*self.M)))
        T = _build_tensor(Ku, normKu, self.Kx, self.Ky, INACT)
        diag1 = sp.coo_matrix((-ACT * ct, (np.arange(self.R), np.arange(self.R))))
        diag2 = sp.coo_matrix((-ACT * st, (np.arange(self.R), np.arange(self.R))))
        diag3 = sp.coo_matrix((ACT * delta * st, (np.arange(self.R), np.arange(self.R))))
        diag4 = sp.coo_matrix((-ACT * delta * ct, (np.arange(self.R), np.arange(self.R))))

        Jdelta = sp.vstack([diag1, diag2])

        Jtheta = sp.vstack([diag3, diag4])

        jac = sp.hstack(
            [
                -alpha[0]*T,  # u
                self.Id,  # q
                self.Z_R,  # r
                Jdelta,  # delta
                Jtheta,  # theta
                Jalpha,  # alpha
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
            TVDenObjectiveFn(self.true_img, self.N, self.M, epsilon),
            eq_constraint_funcs=[
                StateConstraintFn(self.noisy_img, self.K),
                # PrimalConstraintFn(self.K),
                # DualConstraintFn(self.K, self.Kx, self.Ky),
            ],
            ineq_constraint_funcs=[],
        )

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def solve(self, x0=None, bounds=None, print_level=0):
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
            alpha_bounds = [(0, None)] * (1)
            bounds = (
                u_bounds
                + q_bounds
                + r_bounds
                + delta_bounds
                + theta_bounds
                + alpha_bounds
            )
        return self.nlp.solve(x0, bounds, print_level=print_level)
