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


def coef(beta, delta_gamma, q, gamma, rho):
    A = beta * delta_gamma - beta * q * (q / gamma + rho) ** (q - 1)
    B = beta * q * (q - 1) * (q / gamma + rho) ** (q - 2)
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


def hat_j_rho_2(t, beta, delta_gamma, q, gamma, rho):
    t1 = 1 / gamma - rho
    t2 = 1 / gamma + rho

    A = beta * delta_gamma - beta * q * (q / gamma + rho) ** (q - 1)
    B = beta * q * (q - 1) * (q / gamma + rho) ** (q - 2)

    a = -(gamma / (4 * rho**2 * (1 + gamma * rho))) * (
        ((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B
    )
    b = (A * gamma) / (4 * rho**2 * (1 + gamma * rho)) + (
        gamma / (2 * rho * (1 + gamma * rho))
    ) * (((2 * gamma * rho + 1) * A) / (rho * (1 + gamma * rho)) + B)

    j_values = np.piecewise(
        t,
        [t <= t1, (t > t1) & (t <= t2), t > t2],
        [
            lambda t: 0,
            lambda t: a * (t - t1) ** 3 + b * (t - t1) ** 2,
            lambda t: beta * delta_gamma / t
            - beta * q / t * (t + (q - 1) / gamma) ** (q - 1),
        ],
    )
    return j_values


def hat_j_rho(normKu, delta_gamma, q_param, gamma, rho):
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


def diagonal_j_rho(Ku, delta_gamma, q_param, gamma, rho):
    V = Ku.reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V) + 1e-3
    return hat_j_rho(normKu, delta_gamma, q_param, gamma, rho)


def build_nabla_u(u, K, q_param, beta, delta_gamma, gamma, rho, N, M):
    Ku = K @ u
    V = Ku.reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V) + 1e-3
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


def grad_u_j_rho(u, Kx, Ky, beta, delta_gamma, q, gamma, rho):
    a, b = coef(beta, delta_gamma, q, gamma, rho)
    m, n = Kx.shape
    Ku_norm = calc_norm_Ku(u, Kx, Ky)
    nabla_u_x = np.zeros((m, n))
    nabla_u_y = np.zeros((m, n))
    hat_j_rho_val = np.vectorize(
        lambda t: hat_j_rho(t, beta, delta_gamma, q, gamma, rho)
    )(Ku_norm)

    for i in range(m):
        Kx_i = Kx[i, :].reshape(1, -1)
        Ky_i = Ky[i, :].reshape(1, -1)
        Ki = np.vstack((Kx[i, :], Ky[i, :]))
        if Ku_norm[i] <= 1 / gamma - rho:
            nabla_u_x[i, :] = (np.zeros(n)).T
            nabla_u_y[i, :] = (np.zeros(n)).T
        elif 1 / gamma - rho < Ku_norm[i] <= 1 / gamma + rho:
            b_rho_i = (1 / Ku_norm[i]) * (
                3 * a * (Ku_norm[i] - (1 / gamma) + rho) ** 2
                + 2 * b * (Ku_norm[i] - (1 / gamma) + rho)
            )
            nabla_u_x[i, :] = (
                Ki.T @ (b_rho_i * Ki @ u).T * (Kx_i @ u) + hat_j_rho_val[i] * Kx_i
            )
            nabla_u_y[i, :] = (Ki.T @ (b_rho_i * Ki @ u)).T * (
                Ky_i @ u
            ) + hat_j_rho_val[i] * Ky_i
        else:
            c_rho_i = (-beta * delta_gamma / (Ku_norm[i] ** 3)) + beta * q * (
                (1 / Ku_norm[i] ** 3) * ((Ku_norm[i] + (q - 1) / gamma) ** (q - 1))
                - ((q - 1) / Ku_norm[i] ** 2)
                * ((Ku_norm[i] + (q - 1) / gamma) ** (q - 2))
            )
            nabla_u_x[i, :] = (Ki.T @ (c_rho_i * Ki @ u)).T * (
                Kx_i @ u
            ) + hat_j_rho_val[i] * Kx_i
            nabla_u_y[i, :] = (Ki.T @ (c_rho_i * Ki @ u)).T * (
                Ky_i @ u
            ) + hat_j_rho_val[i] * Ky_i
    nabla_u = np.vstack((nabla_u_x, nabla_u_y))
    return nabla_u


def grad_A(u, Kx, Ky, beta, delta_gamma, q, gamma, rho):
    a, b = coef(beta, delta_gamma, q, gamma, rho)
    m, n = Kx.shape
    Ku_norm = calc_norm_Ku(u, Kx, Ky)
    nabla_u_x = np.zeros((m, n))
    nabla_u_y = np.zeros((m, n))

    for i in range(m):
        Kx_i = Kx[i, :].reshape(1, -1)
        Ky_i = Ky[i, :].reshape(1, -1)
        Ki = np.vstack((Kx[i, :], Ky[i, :]))
        if Ku_norm[i] <= 1 / gamma - rho:
            nabla_u_x[i, :] = (np.zeros(n)).T
            nabla_u_y[i, :] = (np.zeros(n)).T
        elif 1 / gamma - rho < Ku_norm[i] <= 1 / gamma + rho:
            b_rho_i = (1 / Ku_norm[i]) * (
                3 * a * (Ku_norm[i] - (1 / gamma) + rho) ** 2
                + 2 * b * (Ku_norm[i] - (1 / gamma) + rho)
            )
            nabla_u_x[i, :] = Ki.T @ (b_rho_i * Ki @ u).T * (Kx_i @ u)
            nabla_u_y[i, :] = (Ki.T @ (b_rho_i * Ki @ u)).T * (Ky_i @ u)
        else:
            c_rho_i = (-beta * delta_gamma / (Ku_norm[i] ** 3)) + beta * q * (
                (1 / Ku_norm[i] ** 3) * ((Ku_norm[i] + (q - 1) / gamma) ** (q - 1))
                - ((q - 1) / Ku_norm[i] ** 2)
                * ((Ku_norm[i] + (q - 1) / gamma) ** (q - 2))
            )
            nabla_u_x[i, :] = (Ki.T @ (c_rho_i * Ki @ u)).T * (Kx_i @ u)
            nabla_u_y[i, :] = (Ki.T @ (c_rho_i * Ki @ u)).T * (Ky_i @ u)
    nabla_A = np.vstack((nabla_u_x, nabla_u_y))
    return nabla_A
