import numpy as np

# from utils import generate_2D_gradient_matrices
# from scipy.sparse import bmat, identity, diags
import scipy.sparse as sp
from scipy.sparse import diags
from L2TVMPCCReg import build_jacobian_matrices as bjm


def build_index_sets(v, alpha, gamma, M):
    V = v.reshape(2, -1).T
    norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
    res = np.ones_like(norm)
    a_gamma = np.where(gamma * norm >= alpha + 0.5 / gamma, res, 0)
    i_gamma = np.where(gamma * norm <= alpha - 0.5 / gamma, res, 0)
    s_gamma = np.ones_like(a_gamma) - a_gamma - i_gamma
    # s_gamma_2 = np.where(abs(gamma * norm - alpha) <= 0.5 / gamma, res, 0)
    # a = alpha / norm
    a = np.where(norm <= 1e-8, 0, alpha / norm)
    c = np.where(
        norm <= 1e-8,
        0,
        (alpha - (0.5 * gamma) * (alpha - gamma * norm + (0.5 / gamma)) ** 2) / norm,
    )
    # b = (alpha - 0.5 * gamma * (alpha - gamma * norm + 0.5 / gamma) ** 2) / norm
    A_gamma = sp.coo_matrix(
        (np.concatenate((a_gamma, a_gamma)), (np.arange(M), np.arange(M)))
    )
    I_gamma = sp.coo_matrix(
        (np.concatenate((i_gamma, i_gamma)), (np.arange(M), np.arange(M)))
    )
    S_gamma = sp.coo_matrix(
        (np.concatenate((s_gamma, s_gamma)), (np.arange(M), np.arange(M)))
    )
    L_1 = sp.coo_matrix((np.concatenate((a, a)), (np.arange(M), np.arange(M))))
    L_2 = sp.coo_matrix((np.concatenate((c, c)), (np.arange(M), np.arange(M))))
    return A_gamma, I_gamma, S_gamma, L_1, L_2


# def build_jacobian_matrices(K, u, q, alpha, gamma):
#     Ku = K @ u
#     Ku = np.where(Ku == 0, 1e-10, Ku)
#     V = Ku.reshape(2, -1).T
#     Q = q.reshape(2, -1).T
#     norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
#     norm_q = np.apply_along_axis(np.linalg.norm, axis=1, arr=Q)
#     res = np.ones_like(norm)
#     a_gamma = np.where(gamma * norm >= alpha + 0.5 / gamma, res, 0)
#     i_gamma = np.where(gamma * norm <= alpha - 0.5 / gamma, res, 0)
#     # s_gamma_2 = np.where(np.abs(gamma * norm - alpha) <= 0.5 / gamma, res, 0)
#     s_gamma = np.ones_like(a_gamma) - a_gamma - i_gamma
#     A_gamma = diags(np.concatenate((a_gamma, a_gamma)))
#     I_gamma = diags(np.concatenate((i_gamma, i_gamma)))
#     S_gamma = diags(np.concatenate((s_gamma, s_gamma)))
#     a = np.where(norm <= 1e-10, 0, alpha / norm)
#     b = np.where(norm <= 1e-10, 0, alpha / (norm**2 * np.maximum(alpha, norm_q)))
#     c = np.where(
#         norm <= 1e-10,
#         0,
#         (alpha - (0.5 * gamma) * (alpha - (gamma * norm) + (0.5 / gamma)) ** 2) / norm,
#     )
#     d = np.where(
#         norm <= 1e-8, 0, (gamma**2 / norm**2) * (alpha - (gamma * norm) + (0.5 / gamma))
#     )
#     e = np.where(norm <= 1e-10, 0, 1 / norm)
#     f = np.where(
#         norm <= 1e-10, 0, (1 - gamma * (alpha - (gamma * norm) + (0.5 / gamma))) / norm
#     )

#     diag_a = diags(np.concatenate((a, a)))
#     diag_b = diags(np.concatenate((b, b)))
#     diag_c = diags(np.concatenate((c, c)))
#     diag_d = diags(np.concatenate((d, d)))
#     diag_q = diags(q)
#     diag_Ku = diags(Ku)

#     diag_e = diags(np.concatenate((e, e)))
#     diag_f = diags(np.concatenate((f, f)))

#     n = len(norm)
#     L = diags((Ku[:n], Ku, Ku[n:]), offsets=(-n, 0, n))
#     H_1_1 = (A_gamma @ diag_a + S_gamma @ diag_c + gamma * I_gamma) @ K
#     H_1_2 = (
#         (
#             A_gamma @ diag_b @ diag_q
#             + S_gamma @ diag_b @ diag_q
#             - S_gamma @ diag_d @ diag_Ku
#         )
#         @ L
#         @ K
#     )
#     # H_1 = (A_gamma@diag_b@diag_q + S_gamma@diag_b@diag_q + S_gamma@diag_d@diag_Ku)@L@K

#     H_2 = (A_gamma @ diag_e + S_gamma @ diag_f) @ Ku
#     H_2 = H_2.reshape(len(Ku), 1)
#     # H_2 = diags(H_2)
#     # H_2 = np.where(H_2 == 0, 1e-10, H_2)

#     return H_1_1 - H_1_2, H_2


def build_jacobian_matrices(K, u, q, alpha, gamma, M):
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
    # H_2 = H_2.reshape(len(Ku), 1)
    # H_2 = diags(H_2)
    # H_2 = np.where(H_2 == 0, 1e-10, H_2)

    # Jacobian sparsity structure
    o = np.ones(M)
    H = M // 2
    D = diags((o[:H], o, o[H:]), offsets=(-H, 0, H))
    H_u_sparsity_structure = D @ K
    
    H = H_1_1 - H_1_2

    H_ = H.toarray()

    row,col = np.nonzero(H_u_sparsity_structure)
    values = H_[row,col]

    H_ = sp.coo_matrix((values,(row,col)),shape=H.shape)
    
    return H_, H_2


def h(row: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    norm_row = np.linalg.norm(row)
    a = alpha / norm_row
    b = (
        alpha - (0.5 * gamma) * (alpha - gamma * norm_row + (0.5 / gamma)) ** 2
    ) / norm_row
    if gamma * norm_row >= alpha + 0.5 / gamma:
        return a * row
    elif gamma * norm_row <= alpha - 0.5 / gamma:
        return gamma * row
    else:
        return b * row


def h_alpha(row: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    norm_row = np.linalg.norm(row)
    a = 1 / norm_row
    b = (1 - gamma * (alpha - gamma * norm_row + (0.5 / gamma))) / norm_row
    if gamma * norm_row >= alpha + 0.5 / gamma:
        return a * row
    elif gamma * norm_row <= alpha - 0.5 / gamma:
        return np.zeros_like(row)
    else:
        return b * row
