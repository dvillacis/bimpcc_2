import numpy as np
from scipy.sparse import spdiags, kron, diags
import scipy.sparse as sp


def generate_2D_gradient_matrices(N) -> tuple:
    """
    Generate the gradient matrices for a 2D image

    Parameters:
    N: int
        Number of pitels in each dimension

    Returns:
    Kt: np.ndarray
        Gradient matrit in the t-direction
    Ky: np.ndarray
        Gradient matrit in the y-direction
    """
    Kx_temp = spdiags([-np.ones(N), np.ones(N)], [0, 1], N - 1, N, format="csr")
    Kx = kron(spdiags(np.ones(N), [0], N, N), Kx_temp, format="csr")
    Ky_temp = spdiags([-np.ones(N * (N - 1))], [0], N * (N - 1), N**2, format="csr")
    Ky = Ky_temp + spdiags(
        [np.ones(N * (N - 1) + N)], [N], N * (N - 1), N**2, format="csr"
    )

    # factor = 1/(N-1)
    factor = 1
    # CREAR K NO SPARSE
    # Convertir matrices dispersas a arrays densos
    Kx = factor * Kx.toarray()  # Convertir Kt a array denso
    Ky = factor * Ky.toarray()  # Convertir Ky a array denso

    K = np.vstack((Kx, Ky))
    # h = 1/(N-1)

    # return h*Kt, h*Ky, h*K
    return Kx, Ky, K


def coef(delta_gamma, q, gamma, rho):
    A = delta_gamma - q * (q / gamma + rho) ** (q - 1)
    B = q * (q - 1) * (q / gamma + rho) ** (q - 2)
    a = -(gamma / (4 * rho**2 * (1 + gamma * rho))) * (
        ((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B
    )
    b = (A * gamma) / (4 * rho**2 * (1 + gamma * rho)) + (
        gamma / (2 * rho * (1 + gamma * rho))
    ) * (((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B)
    return a, b


def calc_norm_Ku(u, Kx, Ky):
    """
    Compute the norm of Ku for each element.

    Parameters:
    u : np.ndarray
        Input vector (flattened 2D image or similar).
    Kx, Ky : np.ndarray
        Gradient matrices in the x and y directions.

    Returns:
    norm_Ku : np.ndarray
        Vector of norms for each element of Ku.
    """
    Ku_x = Kx @ u
    Ku_y = Ky @ u
    norm_Ku = np.sqrt(Ku_x**2 + Ku_y**2)  # Element-wise norm
    return norm_Ku


def hat_j_rho_orig(normKu, delta_gamma, q_param, gamma, rho):
    normKu = normKu + 0.001
    t1 = 1 / gamma - rho
    t2 = 1 / gamma + rho

    A = delta_gamma - q_param * (q_param / gamma + rho) ** (q_param - 1)
    B = q_param * (q_param - 1) * (q_param / gamma + rho) ** (q_param - 2)

    a = -(gamma / (4 * rho**2 * (1 + gamma * rho))) * (
        ((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B
    )
    b = (A * gamma) / (4 * rho**2 * (1 + gamma * rho)) + (
        gamma / (2 * rho * (1 + gamma * rho))
    ) * (((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B)

    res = np.zeros_like(normKu)
    res = np.where(
        (normKu > t1) & (normKu <= t2),
        a * (normKu - t1) ** 3 + b * (normKu - t1) ** 2,
        res,
    )
    res = np.where(
        normKu > t2,
        delta_gamma / normKu
        - q_param / normKu * (normKu + (q_param - 1) / gamma) ** (q_param - 1),
        res,
    )
    return np.diag(np.concatenate((res, res)))


def hat_j_rho(normKu, delta_gamma, q_param, gamma, rho, eps=1e-12):
    """
    Devuelve diag([res,res]) donde res es la función por tramos definida en tu código,
    pero evitando NaN/inf por:
      - divisiones por cero (normKu ~ 0)
      - base <= 0 en potencias fraccionarias (q_param<1)
      - evaluación innecesaria fuera del tramo 3
    """
    # Seguridad numérica para divisiones
    normKu_safe = np.maximum(normKu, eps)

    t1 = 1.0 / gamma - rho
    t2 = 1.0 / gamma + rho

    A = delta_gamma - q_param * (q_param / gamma + rho) ** (q_param - 1.0)
    B = q_param * (q_param - 1.0) * (q_param / gamma + rho) ** (q_param - 2.0)

    a = -(gamma / (4.0 * rho**2 * (1.0 + gamma * rho))) * (
        ((2.0 * gamma * rho + 1.0) * A) / (rho * (1.0 + gamma * rho)) + B
    )
    b = (A * gamma) / (4.0 * rho**2 * (1.0 + gamma * rho)) + (
        gamma / (2.0 * rho * (1.0 + gamma * rho))
    ) * (((2.0 * gamma * rho + 1.0) * A) / (rho * (1.0 + gamma * rho)) + B)

    res = np.zeros_like(normKu_safe)

    # ----- Tramo 2: t1 < normKu <= t2 -----
    mask2 = (normKu_safe > t1) & (normKu_safe <= t2)
    if np.any(mask2):
        z = normKu_safe[mask2] - t1
        res[mask2] = a * z**3 + b * z**2

    # ----- Tramo 3: normKu > t2 -----
    mask3 = normKu_safe > t2
    if np.any(mask3):
        nk = normKu_safe[mask3]
        base = nk + (q_param - 1.0) / gamma

        # base debe ser >0 si estás realmente en tramo 3 y q_param>0,
        # pero por seguridad numérica lo clampeamos:
        base_safe = np.maximum(base, eps)

        term_pow = base_safe ** (q_param - 1.0)
        res[mask3] = delta_gamma / nk - (q_param / nk) * term_pow

    return np.diag(np.concatenate((res, res)))


def diagonal_j_rho(Ku, delta_gamma, q_param, gamma, rho):
    V = Ku.reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V) + 1e-3
    return hat_j_rho(normKu, delta_gamma, q_param, gamma, rho)


def build_nabla_u_orig(u, K, q_param, beta, delta_gamma, gamma, rho, N, M):
    Ku = K @ u
    V = Ku.reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V) + 0.001
    res = np.ones_like(normKu)
    i1 = np.where(normKu <= 1 / gamma - rho, res, 0)
    i2 = np.where((1 / gamma - rho < normKu) & (normKu <= 1 / gamma + rho), res, 0)
    i3 = np.ones_like(i1) - i1 - i2
    # I1 = diags(np.concatenate((i1, i1)))
    I2 = diags(np.concatenate((i2, i2)))
    I3 = diags(np.concatenate((i3, i3)))
    A = delta_gamma - q_param * (q_param / gamma + rho) ** (q_param - 1)
    B = q_param * (q_param - 1) * (q_param / gamma + rho) ** (q_param - 2)
    a = -(gamma / (4 * rho**2 * (1 + gamma * rho))) * (
        ((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B
    )
    b = (A * gamma) / (4 * rho**2 * (1 + gamma * rho)) + (
        gamma / (2 * rho * (1 + gamma * rho))
    ) * (((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B)
    b_rho = (1 / normKu) * (
        3 * a * (normKu - 1 / gamma + rho) ** 2 + 2 * b * (normKu - 1 / gamma + rho)
    )
    c_rho = (-(delta_gamma) / normKu**3) + q_param * (
        (1 / (normKu**3)) * (normKu + (q_param - 1) / gamma) ** (q_param - 1)
        - ((q_param - 1) / normKu**2)
        * (normKu + (q_param - 1) / gamma) ** (q_param - 2)
    )
    f = a * (normKu - 1 / gamma + rho) ** 3 + b * (normKu - 1 / gamma + rho) ** 2
    e = (delta_gamma) / normKu - (q_param / normKu) * (
        normKu + (q_param - 1) / gamma
    ) ** (q_param - 1)

    diag_b_rho = diags(np.concatenate((b_rho, b_rho)))
    diag_c_rho = diags(np.concatenate((c_rho, c_rho)))
    diag_Ku = diags(Ku)
    diag_e = diags(np.concatenate((e, e)))
    diag_f = diags(np.concatenate((f, f)))

    n = len(normKu)
    L = diags((Ku[:n], Ku, Ku[n:]), offsets=(-n, 0, n))
    nabla_u_w = (I2 @ diag_b_rho @ diag_Ku + I3 @ diag_c_rho @ diag_Ku) @ L @ K + (
        I2 @ diag_f + I3 @ diag_e
    ) @ K

    W_u = (-1 / delta_gamma) * (K.T @ nabla_u_w - beta * sp.eye(N))

    # Jacobian sparsity structure
    o = np.ones(M)
    H = M // 2
    D = diags((o[:H], o, o[H:]), offsets=(-H, 0, H))
    H_u_sparsity_structure = K.T @ D @ K
    # H_beta_sparsity_structure = K.T @ np.ones((M,1))

    H_ = W_u.toarray()

    row, col = np.nonzero(H_u_sparsity_structure)
    values = H_[row, col]

    H_ = sp.coo_matrix((values, (row, col)), shape=W_u.shape)
    return H_


def build_nabla_u(u, K, q_param, beta, delta_gamma, gamma, rho, N, M, eps=1e-12):
    Ku = K @ u
    V = Ku.reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)

    # Seguridad numérica
    normKu_safe = np.maximum(normKu, eps)

    # Indicadores por tramos (usa normKu_safe)
    res = np.ones_like(normKu_safe)
    i1 = np.where(normKu_safe <= 1 / gamma - rho, res, 0)
    i2 = np.where(
        (1 / gamma - rho < normKu_safe) & (normKu_safe <= 1 / gamma + rho), res, 0
    )
    i3 = np.ones_like(i1) - i1 - i2

    I2 = diags(np.concatenate((i2, i2)))
    I3 = diags(np.concatenate((i3, i3)))

    A = delta_gamma - q_param * (q_param / gamma + rho) ** (q_param - 1)
    B = q_param * (q_param - 1) * (q_param / gamma + rho) ** (q_param - 2)

    a = -(gamma / (4 * rho**2 * (1 + gamma * rho))) * (
        ((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B
    )
    b = (A * gamma) / (4 * rho**2 * (1 + gamma * rho)) + (
        gamma / (2 * rho * (1 + gamma * rho))
    ) * (((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B)

    # Tramo 2 (polinómico): solo usa normKu_safe
    shift = normKu_safe - 1 / gamma + rho
    b_rho = (1.0 / normKu_safe) * (3 * a * shift**2 + 2 * b * shift)
    f = a * shift**3 + b * shift**2

    # Tramo 3: proteger base de potencias
    base = normKu_safe + (q_param - 1.0) / gamma
    base_safe = np.maximum(base, eps)

    pow_qm1 = base_safe ** (q_param - 1.0)
    pow_qm2 = base_safe ** (q_param - 2.0)

    c_rho = (-(delta_gamma) / normKu_safe**3) + q_param * (
        (1.0 / (normKu_safe**3)) * pow_qm1
        - ((q_param - 1.0) / normKu_safe**2) * pow_qm2
    )

    e = (delta_gamma) / normKu_safe - (q_param / normKu_safe) * pow_qm1

    diag_b_rho = diags(np.concatenate((b_rho, b_rho)))
    diag_c_rho = diags(np.concatenate((c_rho, c_rho)))
    diag_Ku = diags(Ku)
    diag_e = diags(np.concatenate((e, e)))
    diag_f = diags(np.concatenate((f, f)))

    n = len(normKu_safe)
    L = diags((Ku[:n], Ku, Ku[n:]), offsets=(-n, 0, n))

    nabla_u_w = (I2 @ diag_b_rho @ diag_Ku + I3 @ diag_c_rho @ diag_Ku) @ L @ K + (
        I2 @ diag_f + I3 @ diag_e
    ) @ K

    W_u = (-1 / delta_gamma) * (K.T @ nabla_u_w - beta * sp.eye(N))

    # Jacobian sparsity structure
    o = np.ones(M)
    H = M // 2
    D = diags((o[:H], o, o[H:]), offsets=(-H, 0, H))
    H_u_sparsity_structure = K.T @ D @ K

    H_ = W_u.toarray()
    row, col = np.nonzero(H_u_sparsity_structure)
    values = H_[row, col]
    H_ = sp.coo_matrix((values, (row, col)), shape=W_u.shape)
    return H_
