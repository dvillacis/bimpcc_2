import numpy as np
from bimpcc.models.model import MPCCPenalizedModel
from bimpcc.nlp import PenalizedObjectiveFn, ConstraintFn
from rich import print


class Bard1Objective(PenalizedObjectiveFn):
    def __call__(self, x: np.ndarray) -> float:
        pen = np.sum(
            np.array(
                [
                    x[2] * (3 * x[0] - x[1] - 3),
                    x[3] * (-x[0] + 0.5 * x[1] + 4),
                    x[4] * (-x[0] - x[1] + 7),
                ]
            )
        )
        return (x[0] - 5) ** 2 + (2 * x[1] + 1) ** 2 + self.pi * pen

    def parse_vars(self, x: np.ndarray) -> np.ndarray:
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                2 * (x[0] - 5) + 3 * self.pi * x[2] - self.pi * x[3] - self.pi * x[4],
                4 * (2 * x[1] + 1) + 0.5 * self.pi * x[3] - self.pi * x[4],
                self.pi * (3 * x[0] - x[1] - 3),
                self.pi * (-x[0] + 0.5 * x[1] + 4),
                self.pi * (-x[0] - x[1] + 7),
            ]
        )

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [2, 0, 3 * self.pi, -self.pi, -self.pi],
                [0, 8, 0, 0.5 * self.pi, -self.pi],
                [3 * self.pi, -self.pi, 0, 0, 0],
                [-self.pi, 0.5 * self.pi, 0, 0, 0],
                [-self.pi, -self.pi, 0, 0, 0],
            ]
        )


class BardKKTConstraint(ConstraintFn):
    def __call__(self, x: np.ndarray) -> float:
        return 2 * (x[1] - 1) - 1.5 * x[0] + x[2] - 0.5 * x[3] + x[4]

    def parse_vars(self, x: np.ndarray) -> np.ndarray:
        return x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array([-1.5, 2, 1, -0.5, 1])


class BardIneq(ConstraintFn):
    def __init__(self):
        self.A = np.array(
            [
                [3, -1, 0, 0, 0],
                [-1, 0.5, 0, 0, 0],
                [-1, -1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.b = np.array([-3, 4, 7, 0, 0])

    def __call__(self, x: np.ndarray) -> float:
        return self.A @ x + self.b

    def parse_vars(self, x: np.ndarray) -> np.ndarray:
        return x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.A


# class BardComplementarity(ComplementarityConstraintFn):
#     def __call__(self, x: np.ndarray) -> float:
#         return self.t - np.array(
#             [
#                 x[2] * (3 * x[0] - x[1] - 3),
#                 x[3] * (-x[0] + 0.5 * x[1] + 4),
#                 x[4] * (-x[0] - x[1] + 7),
#             ]
#         )

#     def parse_vars(self, x: np.ndarray) -> np.ndarray:
#         return x

#     def jacobian(self, x: np.ndarray) -> np.ndarray:
#         return np.array(
#             [
#                 [-3 * x[2], x[2], 3, 0, 0],
#                 [x[3], -0.5 * x[3], 0, -4, 0],
#                 [x[4], x[4], 0, 0, -7],
#             ]
#         )


class Bard1(MPCCPenalizedModel):
    def __init__(self, x0, pi_init: float = 1):
        self.x0 = x0
        self.pi = pi_init

        bounds = [(0, None)] * 5
        objective_func = Bard1Objective(pi=pi_init)
        eq_constraint_funcs = [BardKKTConstraint()]
        ineq_constraint_funcs = [BardIneq()]
        super().__init__(
            objective_func,
            eq_constraint_funcs,
            ineq_constraint_funcs,
            bounds,
            x0,
            pi_init,
        )

    def compute_complementarity(self, x: np.ndarray) -> float:
        comp = np.array(
            [
                x[2] * (3 * x[0] - x[1] - 3),
                x[3] * (-x[0] + 0.5 * x[1] + 4),
                x[4] * (-x[0] - x[1] + 7),
            ]
        )
        return np.sum(comp)


class Ralph2Objective(PenalizedObjectiveFn):
    def __call__(self, x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2 - 4 * x[0] * x[1] + self.pi * (x[0] * x[1])

    def parse_vars(self, x: np.ndarray) -> np.ndarray:
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * x[0] - 4 * x[1] + self.pi * x[1], 2 * x[1] - 4 * x[0] + self.pi * x[0]])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2, -4+self.pi], [-4 + self.pi, 2]])


# class Ralph2Complementarity(ComplementarityConstraintFn):
#     def __call__(self, x: np.ndarray) -> float:
#         return self.t - (x[0] * x[1])

#     def parse_vars(self, x: np.ndarray) -> np.ndarray:
#         return x

#     def jacobian(self, x: np.ndarray) -> np.ndarray:
#         return np.array([-x[1], -x[0]])


class Ralph2Model(MPCCPenalizedModel):
    def __init__(self, x0, pi_init: float = 1):
        self.x0 = x0
        self.pi = pi_init

        bounds = [(0, None)] * 2
        objective_func = Ralph2Objective(pi=pi_init)
        eq_constraint_funcs = []
        ineq_constraint_funcs = []
        super().__init__(
            objective_func,
            eq_constraint_funcs,
            ineq_constraint_funcs,
            bounds,
            x0,
            pi_init,
        )

    def compute_complementarity(self, x: np.ndarray) -> float:
        comp = x[0] * x[1]
        return comp


def test_bard1():
    x0 = np.array([0, 0, 0, 0, 0])
    model = Bard1(x0)
    res, x_opt, fun_opt = model.solve(
        print_level=0, tol=1e-9, max_iter=100, verbose=True
    )
    print(f"Final res: {model.parse_vars_fn(x_opt)}")
    print(f"complementarity test: {model.compute_complementarity(x_opt)}")
    assert model.compute_complementarity(x_opt) < 1e-6


def test_ralph2():
    x0 = np.array([0, 0])
    model = Ralph2Model(x0)
    res, x_opt, fun_opt = model.solve(print_level=0, tol=1e-9, max_iter=100)
    print(f"Final res: {model.parse_vars_fn(x_opt)}")
    print(f"complementarity test: {model.compute_complementarity(x_opt)}")
    assert model.compute_complementarity(x_opt) < 1e-6
