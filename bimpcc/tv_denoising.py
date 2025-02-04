import numpy as np
import scipy.sparse as sp
from bimpcc.nlp import OptimizationProblem, ConstraintFn, ObjectiveFn
from bimpcc.utils import gradient_operator_x, gradient_operator_y


class TVDenObjectiveFn(ObjectiveFn):
    def __init__(self, true_img: np.ndarray, epsilon: float = 1e-5):
        self.true_img = true_img
        self.N, self.M = true_img.shape
        self.num_pixels = self.N * self.M
        self.epsilon = epsilon

    def parse_vars(self, x):
        return x[: self.N], x[self.N : self.N + self.M], x[self.N + self.M :]

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return (
            0.5 * np.linalg.norm(u - self.u_true) ** 2
            + self.epsilon * np.linalg.norm(alpha) ** 2
        )

    def gradient(self, x: np.ndarray) -> float:
        u, q, alpha = self.getvars(x)
        return np.concatenate((u - self.u_true, self.Z, self.epsilon * alpha))

    def hessian(self, x: np.ndarray) -> float:
        return sp.spdiags(np.ones(self.num_pixels), 0, self.num_pixels, self.num_pixels)
    

class StateConstraintFn(ConstraintFn):
    def __init__(self, noisy_img: np.ndarray, gradient_op):
        self.noisy_img = noisy_img
        self.gradient_op = gradient_op
        self.N, self.M = noisy_img.shape
        self.num_pixels = self.N * self.M

    def parse_vars(self, x):
        return x[: self.N], x[self.N : self.N + self.M], x[self.N + self.M :]

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return u - self.noisy_img + self.gradient_op.T @ q

    def jacobian(self, x: np.ndarray) -> float:
        return np.eye(self.num_pixels)

    def hessian(self, x: np.ndarray, _lambda) -> float:
        return np.zeros((self.num_pixels, self.num_pixels))
    
def solve(true_img, noisy_img,gamma = 100,epsilon=1e-5):
    N, M = true_img.shape
    Kx = gradient_operator_x(M, N)
    Ky = gradient_operator_y(M, N)
    K = sp.vstack([Kx, Ky])
    nlp = OptimizationProblem(
        TVDenObjectiveFn(true_img, epsilon),
        [StateConstraintFn(noisy_img, K)],
        []
    )
    x0 = np.zeros(3 * N * M)
    res, x, f = nlp.solve(x0)
    return x
