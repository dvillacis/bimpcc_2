import numpy as np

# from utils import generate_2D_gradient_matrices
# from scipy.sparse import bmat, identity, diags
import scipy.sparse as sp
from scipy.sparse import diags
from L2TVMPCCReg import build_jacobian_matrices as bjm


def build_index_sets_orig(v, gamma, M):
    V = v.reshape(2, -1).T
    norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
    res = np.ones_like(norm)
    a_gamma = np.where(gamma * norm >= 1 + 0.5 / gamma, res, 0)
    i_gamma = np.where(gamma * norm <= 1 - 0.5 / gamma, res, 0)
    s_gamma = np.ones_like(a_gamma) - a_gamma - i_gamma
    # s_gamma_2 = np.where(abs(gamma * norm - 1) <= 0.5 / gamma, res, 0)
    # a = 1 / norm
    a = np.where(norm <= 1e-8, 0, 1 / norm)
    c = np.where(
        norm <= 1e-8,
        0,
        (1 - (0.5 * gamma) * (1 - gamma * norm + (0.5 / gamma)) ** 2) / norm,
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


def build_index_sets(v, gamma, M, eps=1e-12):
    V = v.reshape(2, -1).T
    norm = np.linalg.norm(V, axis=1)

    res = np.ones_like(norm, dtype=float)

    a_gamma = np.where(gamma * norm >= 1 + 0.5 / gamma, res, 0.0)
    i_gamma = np.where(gamma * norm <= 1 - 0.5 / gamma, res, 0.0)
    s_gamma = 1.0 - a_gamma - i_gamma

    # máscara: índices donde es seguro dividir
    nz = norm > eps
    norm_safe = np.maximum(norm, eps)  # evita /0 en cualquier caso

    # a = 1/norm (y 0 si norm es pequeño)
    a = np.zeros_like(norm, dtype=float)
    a[nz] = 1.0 / norm_safe[nz]

    # c = (...) / norm (y 0 si norm es pequeño)
    c = np.zeros_like(norm, dtype=float)
    tmp = 1.0 - gamma * norm + (0.5 / gamma)  # vector (M,)
    numer = 1.0 - 0.5 * gamma * (tmp**2)  # vector (M,)
    c[nz] = numer[nz] / norm_safe[nz]

    idx = np.arange(M)

    A_gamma = sp.coo_matrix(
        (np.concatenate((a_gamma, a_gamma)), (idx, idx)), shape=(M, M)
    )
    I_gamma = sp.coo_matrix(
        (np.concatenate((i_gamma, i_gamma)), (idx, idx)), shape=(M, M)
    )
    S_gamma = sp.coo_matrix(
        (np.concatenate((s_gamma, s_gamma)), (idx, idx)), shape=(M, M)
    )
    L_1 = sp.coo_matrix((np.concatenate((a, a)), (idx, idx)), shape=(M, M))
    L_2 = sp.coo_matrix((np.concatenate((c, c)), (idx, idx)), shape=(M, M))

    return A_gamma, I_gamma, S_gamma, L_1, L_2


def build_jacobian_matrices(K, u, q, gamma, M):
    Ku = K @ u
    Ku = np.where(Ku == 0, 1e-10, Ku)
    V = Ku.reshape(2, -1).T
    Q = q.reshape(2, -1).T
    norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
    norm_q = np.apply_along_axis(np.linalg.norm, axis=1, arr=Q)
    res = np.ones_like(norm)
    a_gamma = np.where(gamma * norm >= 1 + 0.5 / gamma, res, 0)
    i_gamma = np.where(gamma * norm <= 1 - 0.5 / gamma, res, 0)
    s_gamma = np.ones_like(a_gamma) - a_gamma - i_gamma
    A_gamma = diags(np.concatenate((a_gamma, a_gamma)))
    I_gamma = diags(np.concatenate((i_gamma, i_gamma)))
    S_gamma = diags(np.concatenate((s_gamma, s_gamma)))
    a = 1 / norm
    b = 1 / (norm**2 * np.maximum(1, norm_q))
    c = (1 - 0.5 * gamma * (1 - gamma * norm + 0.5 / gamma) ** 2) / norm
    d = (gamma**2 / norm**2) * (1 - gamma * norm + 0.5 / gamma)

    diag_a = diags(np.concatenate((a, a)))
    diag_b = diags(np.concatenate((b, b)))
    diag_c = diags(np.concatenate((c, c)))
    diag_d = diags(np.concatenate((d, d)))
    diag_q = diags(q)
    diag_Ku = diags(Ku)

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

    # Jacobian sparsity structure
    o = np.ones(M)
    H = M // 2
    D = diags((o[:H], o, o[H:]), offsets=(-H, 0, H))
    H_u_sparsity_structure = D @ K

    H = H_1_1 - H_1_2

    H_ = H.toarray()

    row, col = np.nonzero(H_u_sparsity_structure)
    values = H_[row, col]

    H_ = sp.coo_matrix((values, (row, col)), shape=H.shape)

    return H_


def h(row: np.ndarray, gamma: float) -> np.ndarray:
    norm_row = np.linalg.norm(row)
    a = 1 / norm_row
    b = (1 - (0.5 * gamma) * (1 - gamma * norm_row + (0.5 / gamma)) ** 2) / norm_row
    if gamma * norm_row >= 1 + 0.5 / gamma:
        return a * row
    elif gamma * norm_row <= 1 - 0.5 / gamma:
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
