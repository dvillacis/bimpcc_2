import numpy as np
import scipy.sparse as sp
from nlp import ConstraintFn, ObjectiveFn
from nlp import OptimizationProblem
from utils import generate_2D_gradient_matrices


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
        true_img: np.ndarray,
        forward_op: np.ndarray,
        M,
        N,
        parameter_size: int = 1,
        mu = 1.0
    ):
        self.true_img = true_img.flatten()
        self.forward_op = forward_op
        self.N = N
        self.M = M
        self.R = M // 2
        self.parameter_size = parameter_size
        self.mu = mu

    def __call__(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        data_fidelity = 0.5 * np.linalg.norm(self.forward_op @ u - self.true_img) ** 2
        reg = self.mu * np.sum(r * (alpha - delta))
        return data_fidelity + reg

    def gradient(self, x: np.ndarray) -> float:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        grad_u = self.forward_op.T @ (self.forward_op @ u - self.true_img)
        return np.concatenate(
            (
                grad_u,
                np.zeros(self.M),
                -self.mu*r,
                self.mu*r,
                np.zeros(self.R),
                self.mu*(alpha-delta),
            )
        )


class StateConstraintFn(ConstraintFn):
    def __init__(
        self,
        noisy_img: np.ndarray,
        forward_op: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
    ):
        self.noisy_img = noisy_img.flatten()
        self.forward_op = forward_op
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
        return (
            self.forward_op.T @ (self.forward_op @ u - self.noisy_img)
            + self.gradient_op.T @ q
        )

    def jacobian(self, x: np.ndarray) -> sp.coo_array:
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
        Jalpha = sp.coo_matrix((np.ones(self.R), (np.arange(self.R), [0] * self.R)))
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


class TVComplementarity1(ConstraintFn):
    def __init__(self, N: int, M: int, parameter_size: int = 1):
        self.N = N
        self.M = M
        self.R = M // 2
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_M = sp.coo_matrix((self.R, self.M))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.Z_P = sp.coo_matrix((self.R, self.parameter_size))
        self.I_R = sp.eye(self.R).tocoo()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return r

    def jacobian(self, x: np.ndarray) -> sp.coo_array:
        J = sp.hstack(
            [
                self.Z_N,  # u
                self.Z_M,  # q
                self.I_R,  # r
                self.Z_R,  # delta
                self.Z_R,  # theta
                self.Z_P,  # alpha
            ]
        )
        return J.tocoo()


class TVComplementarity2(ConstraintFn):
    def __init__(self, N: int, M: int, parameter_size: int = 1):
        self.N = N
        self.M = M
        self.R = M // 2
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_M = sp.coo_matrix((self.R, self.M))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.I_P = sp.coo_matrix((self.R, self.parameter_size))
        self.I_R = sp.eye(self.R).tocoo()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)
        return alpha - delta

    def jacobian(self, x: np.ndarray) -> sp.coo_array:
        J = sp.hstack(
            [
                self.Z_N,  # u
                self.Z_M,  # q
                self.Z_R,  # r
                -self.I_R,  # delta
                self.Z_R,  # theta
                self.I_P,  # alpha
            ]
        )
        return J.tocoo()


class TVRec(OptimizationProblem):
    def __init__(
        self,
        true_img: np.ndarray,
        dmg_img: np.ndarray,
        forward_op: np.ndarray,
        parameter_size: int = 1,
        mu: float = 1.0,
    ):
        self.img_shape = true_img.shape
        self.true_img = true_img.flatten()
        self.dmg_img = dmg_img.flatten()
        self.Kx, self.Ky, self.K = generate_2D_gradient_matrices(true_img.shape[0])
        self.M, self.N = self.K.shape
        self.R = self.M // 2
        self.parameter_size = parameter_size

        # Defining the objective function
        objective_fn = TVRecObjectiveFn(
            true_img=self.true_img,
            forward_op=forward_op,
            M=self.M,
            N=self.N,
            parameter_size=parameter_size,
            mu=mu,
        )

        # Defining equality constraints
        eq_constraint_funcs = [
            StateConstraintFn(
                noisy_img=self.dmg_img,
                forward_op=forward_op,
                gradient_op=self.K,
                parameter_size=parameter_size,
            ),
            PrimalConstraintFn(
                gradient_op=self.K,
                parameter_size=parameter_size,
            ),
            DualConstraintFn(
                gradient_op=self.K,
                gradient_op_x=self.Kx,
                gradient_op_y=self.Ky,
                parameter_size=parameter_size,
            ),
        ]

        # Defining inequality constraints
        ineq_constraint_funcs = [
            BoundConstraintFn(self.M, self.N, parameter_size=parameter_size),
        ]
        # ineq_constraint_funcs = []

        # Defining bounds
        # u, q, r, delta, theta, alpha
        u_bounds = [(0, None)] * self.N
        q_bounds = [(None, None)] * self.M
        r_bounds = [(0, None)] * self.R
        delta_bounds = [(0, None)] * self.R
        theta_bounds = [(None, None)] * self.R
        alpha_bounds = [(0, 10.0)] * (parameter_size)
        bounds = (
            u_bounds + q_bounds + r_bounds + delta_bounds + theta_bounds + alpha_bounds
        )

        super().__init__(
            objective_func=objective_fn,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            bounds=bounds,
        )

    def solve(
        self,
        tol=1e-4,
        max_iter=500,
        verbose=False,
        print_level=0,
        x0=None,
        *args,
        **kwargs,
    ):
        if x0 is None:
            x0 = np.concatenate(
                (
                    self.dmg_img.flatten(),
                    0.01 * np.ones(self.M),
                    0.01 * np.ones(self.R),
                    0.01 * np.ones(self.R),
                    np.zeros(self.R),
                    np.zeros(self.parameter_size),
                )
            )
        if verbose:
            print_level = 5
        options = {
            "print_level": print_level,
            "tol": tol,
            "max_iter": max_iter,
            # "acceptable_tol": 1e-5,
            # "constr_viol_tol": 1e-5,
            "mu_strategy": "adaptive",
            # "nlp_scaling_method": "gradient-based",
            # "check_derivatives_for_naninf": "yes",
            # "compl_inf_tol": 1e-3,
            # "bound_relax_factor": 1e-6,
            # "sb": "yes",
        }
        res, x, fn = super().solve(x0, options=options)

        u, q, r, delta, theta, alpha = _parse_vars(x, self.N, self.M)

        return {"u": u.reshape(self.img_shape), "alpha": alpha, "r": r, "delta": delta, "fn": fn, "res": res}

class TVDenoising(TVRec):
    def __init__(
        self,
        true_img: np.ndarray,
        noisy_img: np.ndarray,
        parameter_size: int = 1,
        mu: float = 1.0,
    ):
        forward_op = sp.eye(true_img.size)
        super().__init__(
            true_img, noisy_img, forward_op, parameter_size, mu
        )
