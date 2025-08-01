import numpy as np
import scipy.sparse as sp
from nlp import ConstraintFn, ObjectiveFn
from mpcc import MPCCProblem
from utils import generate_2D_gradient_matrices


def _parse_vars(x: np.ndarray, N: int, R: int):
    return (
        x[:N],
        x[N : N + R],
        x[N + R : N + 2 * R],
        x[N + 2 * R :],
    )


class TVRecObjectiveFn(ObjectiveFn):
    def __init__(
        self,
        true_img: np.ndarray,
        forward_op: np.ndarray,
        N,
        R,
        parameter_size: int = 1,
    ):
        self.true_img = true_img.flatten()
        self.forward_op = forward_op
        self.N = N
        self.R = R
        self.parameter_size = parameter_size

    def __call__(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        data_fidelity = 0.5 * np.linalg.norm(self.forward_op @ u - self.true_img) ** 2
        reg = 0.5 * 1e-2 * np.linalg.norm(alpha) ** 2
        return data_fidelity + reg

    def gradient(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        grad_u = self.forward_op.T @ (self.forward_op @ u - self.true_img)
        grad_alpha = 1e-2 * alpha
        return np.concatenate(
            (
                grad_u,
                np.zeros(2 * self.R),
                grad_alpha,
            )
        )


class StateConstraintFn(ConstraintFn):
    def __init__(
        self,
        noisy_img: np.ndarray,
        forward_op: np.ndarray,
        gradient_op_x: np.ndarray,
        gradient_op_y: np.ndarray,
        parameter_size: int = 1,
    ):
        self.noisy_img = noisy_img.flatten()
        self.forward_op = forward_op
        self.gradient_op_x = gradient_op_x
        self.gradient_op_y = gradient_op_y
        self.R, self.N = gradient_op_x.shape
        self.parameter_size = parameter_size
        KxT = (self.gradient_op_x.T).tocoo()
        KyT = (self.gradient_op_y.T).tocoo()
        Z_P = sp.coo_matrix((self.N, self.parameter_size))

        self.jac = sp.hstack(
            [
                self.forward_op.T @ self.forward_op,  # u
                KxT,  # qx
                KyT,  # qy
                Z_P,  # alpha
            ]
        )
        self.jac_shape = self.jac.shape

    def __call__(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        return (
            self.forward_op.T @ (self.forward_op @ u - self.noisy_img)
            + self.gradient_op_x.T @ qx
            + self.gradient_op_y.T @ qy
        )

    def jacobian(self, x: np.ndarray) -> sp.coo_array:
        return sp.coo_array(
            (self.jac.data, (self.jac.row, self.jac.col)), shape=self.jac_shape
        )


class DualBallConstraintFn(ConstraintFn):
    def __init__(self, N, R, parameter_size: int = 1):
        self.N = N
        self.R = R
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.Z_P = sp.coo_matrix((self.R, self.parameter_size))

    def __call__(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        return 1 - qx**2 - qy**2

    def jacobian(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)

        rows = np.arange(self.R)
        data_x = -2 * qx
        data_y = -2 * qy
        Jqx = sp.coo_matrix((data_x, (rows, rows)))
        Jqy = sp.coo_matrix((data_y, (rows, rows)))
        jac = sp.hstack(
            [
                self.Z_N,  # u
                Jqx,  # qx
                Jqy,  # qy
                self.Z_P,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class DualConstraintFn(ConstraintFn):
    def __init__(self, N, R, parameter_size: int = 1):
        self.N = N
        self.R = R
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))

    def __call__(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        return alpha - np.sqrt(qx**2 + qy**2)

    def jacobian(self, x: np.ndarray) -> float:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
    
        # qx = (alpha * qx) / np.maximum(alpha, np.abs(qx))
        # qy = (alpha * qy) / np.maximum(alpha, np.abs(qy))

        norm_q = np.sqrt(qx**2 + qy**2 + 1e-6)

        rows = np.arange(self.R)
        data_x = -qx / norm_q
        data_y = -qy / norm_q
        Jqx = sp.coo_matrix((data_x, (rows, rows)))
        Jqy = sp.coo_matrix((data_y, (rows, rows)))
        Jalpha = sp.coo_matrix((np.ones(self.R), (np.arange(self.R), [0] * self.R)))

        jac = sp.hstack(
            [
                self.Z_N,  # u
                Jqx,  # qx
                Jqy,  # qy
                Jalpha,  # alpha
            ]
        )
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class TVComplementarity1(ConstraintFn):
    def __init__(
        self, N: int, R: int, gradient_op_x, gradient_op_y, parameter_size: int = 1
    ):
        self.N = N
        self.R = R
        self.parameter_size = parameter_size
        self.Z_N = sp.coo_matrix((self.R, self.N))
        self.Z_R = sp.coo_matrix((self.R, self.R))
        self.Z_P = sp.coo_matrix((self.R, self.parameter_size))
        self.gradient_op_x = gradient_op_x
        self.gradient_op_y = gradient_op_y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        return np.sqrt((self.gradient_op_x @ u) ** 2 + (self.gradient_op_y @ u) ** 2)

    def jacobian(self, x: np.ndarray) -> sp.coo_array:
        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)
        Kxu = self.gradient_op_x @ u
        Kyu = self.gradient_op_y @ u
        norm_Ku = np.sqrt(Kxu**2 + Kyu**2 + 1e-6)
        # norm_Ku = np.maximum(norm_Ku, 1e-10)
        # Build fixed-pattern diagonal matrices
        eps=1e-10
        rows = np.arange(self.R)
        D  = sp.csr_matrix((1.0 / norm_Ku, (rows, rows)), shape=(self.R, self.R))
        Dx = sp.csr_matrix((Kxu+eps, (rows, rows)), shape=(self.R, self.R))
        Dy = sp.csr_matrix((Kyu+eps, (rows, rows)), shape=(self.R, self.R))

        Ju = D @ (Dx @ self.gradient_op_x + Dy @ self.gradient_op_y)

        jac = sp.hstack(
            [
                Ju,  # u
                self.Z_R,  # qx
                self.Z_R,  # qy
                self.Z_P,  # alpha
            ],
            format="coo",
        )
        # print(jac.nnz)
        return sp.coo_array((jac.data, (jac.row, jac.col)), shape=jac.shape)


class TVRec(MPCCProblem):
    def __init__(
        self,
        true_img: np.ndarray,
        dmg_img: np.ndarray,
        forward_op: np.ndarray,
        parameter_size: int = 1,
        relaxation_type: str = "scholtes",
        t_init: float = 1.0,
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
            N=self.N,
            R=self.R,
            parameter_size=parameter_size,
        )

        # Defining equality constraints
        eq_constraint_funcs = [
            StateConstraintFn(
                noisy_img=self.dmg_img,
                forward_op=forward_op,
                gradient_op_x=self.Kx,
                gradient_op_y=self.Ky,
                parameter_size=parameter_size,
            )
        ]

        # Defining inequality constraints
        ineq_constraint_funcs = [
            DualBallConstraintFn(self.N, self.R, parameter_size=parameter_size),
            DualConstraintFn(self.N, self.R, parameter_size=parameter_size),
        ]

        # Defining complementarity constraints
        complementarity_constraint_funcs = (
            TVComplementarity1(self.N, self.R, self.Kx, self.Ky, parameter_size),
            DualConstraintFn(self.N, self.R, parameter_size),
        )

        # Defining bounds
        # u, qx,qy, alpha
        u_bounds = [(0, None)] * self.N
        qx_bounds = [(None, None)] * self.R
        qy_bounds = [(None, None)] * self.R
        alpha_bounds = [(0, 10.0)] * (parameter_size)
        bounds = u_bounds + qx_bounds + qy_bounds + alpha_bounds

        super().__init__(
            objective_func=objective_fn,
            eq_constraint_funcs=eq_constraint_funcs,
            ineq_constraint_funcs=ineq_constraint_funcs,
            complementarity_constraint_funcs=complementarity_constraint_funcs,
            bounds=bounds,
            t_init=t_init,
            relaxation_type=relaxation_type,
        )

    def solve(
        self,
        t_min=0.00001,
        max_iter=10,
        tol=0.001,
        nlp_tol=0.000001,
        nlp_max_iter=5000,
        verbose=False,
        print_level=0,
        beta=0.5,
        x0=None,
        *args,
        **kwargs,
    ):
        if x0 is None:
            x0 = np.concatenate(
                (
                    self.dmg_img.flatten(),
                    0.01 * np.ones(self.R),
                    0.01 * np.ones(self.R),
                    0.05 * np.ones(self.parameter_size),
                )
            )
        res, x, fn = super().solve(
            x0,
            t_min,
            max_iter,
            tol,
            nlp_tol,
            nlp_max_iter,
            verbose,
            print_level,
            beta,
            x0,
            *args,
            **kwargs,
        )

        u, qx, qy, alpha = _parse_vars(x, self.N, self.R)

        return {"u": u.reshape(self.img_shape), "alpha": alpha, "fn": fn, "res": res}


class TVDenoising(TVRec):
    def __init__(
        self,
        true_img: np.ndarray,
        noisy_img: np.ndarray,
        parameter_size: int = 1,
        relaxation_type: str = "scholtes",
        t_init: float = 1.0,
    ):
        forward_op = sp.eye(true_img.size)
        super().__init__(
            true_img, noisy_img, forward_op, parameter_size, relaxation_type, t_init
        )
