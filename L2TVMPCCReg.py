import numpy as np
from AbstractNLP import AbstractNLP
from Dataset import get_dataset
from utils import (
    generate_2D_gradient_matrices,
    plot_experiment,
)
import cyipopt
from rich import print
import pickle
from scipy.sparse import bmat, identity, diags


def build_index_sets(v, alpha, gamma):
    V = v.reshape(2, -1).T
    norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
    res = np.ones_like(norm)
    a_gamma = np.where(gamma * norm >= alpha + 0.5 / gamma, res, 0)
    i_gamma = np.where(gamma * norm <= alpha - 0.5 / gamma, res, 0)
    s_gamma = np.ones_like(a_gamma) - a_gamma - i_gamma
    # a = alpha / norm
    a = np.where(norm < 1e-6, 0, alpha/norm)
    b = np.where(norm < 1e-6, 0, (alpha - 0.5 * gamma * (alpha - gamma * norm + 0.5 / gamma) ** 2) / norm)
    # b = (alpha - 0.5 * gamma * (alpha - gamma * norm + 0.5 / gamma) ** 2) / norm
    A_gamma = diags(np.concatenate((a_gamma, a_gamma)))
    I_gamma = diags(np.concatenate((i_gamma, i_gamma)))
    S_gamma = diags(np.concatenate((s_gamma, s_gamma)))
    L_1 = diags(np.concatenate((a, a)))
    L_2 = diags(np.concatenate((b, b)))
    return A_gamma, I_gamma, S_gamma, L_1, L_2


def build_jacobian_matrices(K, u, q, alpha, gamma):
    Ku = K @ u
    Ku = np.where(Ku == 0, 1e-10, Ku)
    V = Ku.reshape(2, -1).T
    Q = q.reshape(2, -1).T
    norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
    norm_q = np.apply_along_axis(np.linalg.norm, axis=1, arr=Q)
    res = np.ones_like(norm)
    a_gamma = np.where(gamma * norm >= alpha + 0.5 / gamma, res, 0)
    i_gamma = np.where(gamma * norm <= alpha - 0.5 / gamma, res, 0)
    s_gamma = np.ones_like(a_gamma) - a_gamma - i_gamma
    A_gamma = diags(np.concatenate((a_gamma, a_gamma)))
    I_gamma = diags(np.concatenate((i_gamma, i_gamma)))
    S_gamma = diags(np.concatenate((s_gamma, s_gamma)))
    a = alpha / norm
    b = alpha / (norm**2 * np.maximum(alpha, norm_q))
    c = (alpha - 0.5 * gamma * (alpha - gamma * norm + 0.5 / gamma) ** 2)/norm
    d = (gamma**2 / norm**2) * (alpha - gamma * norm + 0.5 / gamma)
    e = 1 / norm
    f = (1 - gamma * (alpha - gamma * norm + 0.5 / gamma)) / norm

    diag_a = diags(np.concatenate((a, a)))
    diag_b = diags(np.concatenate((b, b)))
    diag_c = diags(np.concatenate((c, c)))
    diag_d = diags(np.concatenate((d, d)))
    diag_q = diags(q)
    diag_Ku = diags(Ku)

    diag_e = diags(np.concatenate((e, e)))
    diag_f = diags(np.concatenate((f, f)))

    n = len(norm)
    L = diags((Ku[:n], Ku, Ku[n:]), offsets=(-n, 0, n))
    H_1_1 = (A_gamma @ diag_a + S_gamma @ diag_c + gamma * I_gamma) @ K
    H_1_2 = (
        (
            A_gamma @ diag_b @ diag_q
            + S_gamma @ diag_b @ diag_q
            - S_gamma @ diag_d @ diag_Ku
        )
        @ L
        @ K
    )
    # H_1 = (A_gamma@diag_b@diag_q + S_gamma@diag_b@diag_q + S_gamma@diag_d@diag_Ku)@L@K

    H_2 = (A_gamma @ diag_e + S_gamma @ diag_f) @ Ku
    H_2 = H_2.reshape(len(Ku), 1)
    # H_2 = diags(H_2)
    # H_2 = np.where(H_2 == 0, 1e-10, H_2)

    return H_1_1 - H_1_2, H_2


def h(row: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    norm_row = np.linalg.norm(row)
    a = alpha / norm_row
    b = (alpha - 0.5 * gamma * (alpha - gamma * norm_row + 0.5 / gamma) ** 2) / norm_row
    if gamma * norm_row >= alpha + 0.5 / gamma:
        return a * row
    elif gamma * norm_row <= alpha - 0.5 / gamma:
        return gamma * row
    else:
        return b * row


def h_alpha(row: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    norm_row = np.linalg.norm(row)
    a = 1 / norm_row
    b = (1 - gamma * (alpha - gamma * norm_row + 0.5 / gamma)) / norm_row
    if gamma * norm_row >= alpha + 0.5 / gamma:
        return a * row
    elif gamma * norm_row <= alpha - 0.5 / gamma:
        return np.zeros_like(row)
    else:
        return b * row


# def h_u(row: np.ndarray, alpha: float, gamma: float) -> np.ndarray:


def setup_problem(size, dataset_name="cameraman"):
    u_true, u_noisy = get_dataset(dataset_name, size).get_training_data()
    Kx, Ky, K = generate_2D_gradient_matrices(size)
    M, N = K.shape
    P = 1
    problem = L2TVMPCCReg(u_true.ravel(), u_noisy.ravel(), K)
    u0 = u_noisy.ravel()
    q0 = 0.0 * np.ones(M)
    alpha0 = [1e-5] * np.ones(P)
    init_guess = np.concatenate((u0, q0, alpha0))
    cl_1 = np.zeros(N)
    cl_2 = np.zeros(M)
    cl = np.concatenate((cl_1, cl_2))

    cu_1 = np.zeros(N)
    cu_2 = np.zeros(M)
    cu = np.concatenate((cu_1, cu_2))

    lb_u = np.zeros(N)
    lb_q = -1e20 * np.ones(M)
    lb_alpha = [1e-10] * np.ones(P)
    lb = np.concatenate((lb_u, lb_q, lb_alpha))

    ub_u = 1e20 * np.ones(N)
    ub_q = 1e20 * np.ones(M)
    ub_alpha = [1e20] * np.ones(P)
    ub = np.concatenate((ub_u, ub_q, ub_alpha))

    nlp = cyipopt.Problem(
        n=len(init_guess), m=len(cl), problem_obj=problem, lb=lb, ub=ub, cl=cl, cu=cu
    )
    return nlp, init_guess


class L2TVMPCCReg(AbstractNLP):
    def __init__(self, u_true, u_noisy, K, gamma=1e-1):
        self.u_true = u_true
        self.u_noisy = u_noisy
        self.M, self.N = K.shape
        self.K = K
        self.Z = np.zeros(self.M)
        self.gamma = gamma
        self.epsilon = 0

        # Jacobian sparsity structure
        o = np.ones(self.M)
        H = self.M // 2
        D = diags((o[:H], o, o[H:]), offsets=(-H, 0, H))
        self.H_u_sparsity_structure = D @ self.K

    def getvars(self, x):
        return x[: self.N], x[self.N : self.N + self.M], x[self.N + self.M :]

    def objective(self, x):
        u, q, alpha = self.getvars(x)
        return (
            0.5 * np.linalg.norm(u - self.u_true) ** 2
            + self.epsilon * np.linalg.norm(alpha) ** 2
        )

    def gradient(self, x):
        u, q, alpha = self.getvars(x)
        return np.concatenate((u - self.u_true, self.Z, self.epsilon * alpha))

    def constraints(self, x):
        u, q, alpha = self.getvars(x)
        Ku = self.K @ u
        A_gamma, I_gamma, S_gamma, L_1, L_2 = build_index_sets(Ku, alpha, self.gamma)
        constr1 = u - self.u_noisy + self.K.T @ q
        constr2 = q - (A_gamma @ L_1 + S_gamma @ L_2 + self.gamma * I_gamma) @ Ku
        return np.concatenate((constr1, constr2))

    # def jacobianstructure(self):
    #     struct = bmat(
    #         [
    #             [identity(self.N), self.K.T, None],
    #             [self.H_u_sparsity_structure, identity(self.M), np.ones((self.M, 1))],
    #         ]
    #     )
    #     print(f"jacobian structure nonzero elements: {struct.nnz}")
    #     return struct.row, struct.col

    def jacobian(self, x):
        u, q, alpha = self.getvars(x)
        H_u, H_alpha = build_jacobian_matrices(self.K, u, q, alpha, self.gamma)
        # print(H_u == 0)
        jac = bmat(
            [
                [identity(self.N), self.K.T, None],
                [-H_u, identity(self.M), H_alpha],
            ]
        )
        # print(f"jacobian nonzero elements: {jac.nnz}")
        jac_matrix = jac.toarray()
        # print(f"jac_matrix shape: {jac_matrix.shape}")
        return jac_matrix.ravel()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="cameraman")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    folder_name = f"results/L2TVMPCCReg_{args.dataset}_{args.N}"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print(f"Creating folder {folder_name}")

    if args.plot:
        print(f"Plotting results from {folder_name}/{args.dataset}_{args.N}.pkl")
        plot_experiment(f"{folder_name}/{args.dataset}_{args.N}.pkl")
    else:
        print(f"Running experiment with N={args.N} and dataset={args.dataset}")
        nlp, initial_guess = setup_problem(args.N, args.dataset)
        # nlp.add_option("jacobian_approximation", "finite-difference-values")
        # nlp.add_option("derivative_test", "first-order")

        variables, info = nlp.solve(initial_guess)

        if args.save:
            with open(f"{folder_name}/{args.dataset}_{args.N}.pkl", "wb") as f:
                print(f"Saving results in {folder_name}/{args.dataset}_{args.N}.pkl")
                pickle.dump(info, f)
