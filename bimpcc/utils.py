import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

# from scipy.sparse import spdiags, kron, vstack
from typing import Tuple


def gradient_operator_x(W, H):
    """Returns a LinearOperator for the finite difference gradient in the x direction."""

    def matvec(x):
        """Computes forward differences in the x-direction."""
        x = x.reshape(H, W)
        Gx = np.zeros_like(x)
        Gx[:, :-1] = x[:, 1:] - x[:, :-1]  # Forward difference
        Gx[:, -1] = 0  # Zero boundary condition on the last column
        return Gx.ravel()

    def rmatvec(y):
        """Computes adjoint operation (negative divergence in x-direction)."""
        y = y.reshape(H, W)
        div_x = np.zeros_like(y)
        div_x[:, :-1] -= y[:, :-1]
        div_x[:, 1:] += y[:, :-1]
        div_x[:, -1] = 0  # Zero boundary condition
        return div_x.ravel()

    return LinearOperator(
        (H * W, H * W), matvec=matvec, rmatvec=rmatvec, dtype=np.float64
    )


def gradient_operator_y(W, H):
    """Returns a LinearOperator for the finite difference gradient in the y direction."""

    def matvec(x):
        """Computes forward differences in the y-direction."""
        x = x.reshape(H, W)
        Gy = np.zeros_like(x)
        Gy[:-1, :] = x[1:, :] - x[:-1, :]  # Forward difference
        Gy[-1, :] = 0  # Zero boundary condition on the last row
        return Gy.ravel()

    def rmatvec(y):
        """Computes adjoint operation (negative divergence in y-direction)."""
        y = y.reshape(H, W)
        div_y = np.zeros_like(y)
        div_y[:-1, :] -= y[:-1, :]
        div_y[1:, :] += y[:-1, :]
        div_y[-1, :] = 0  # Zero boundary condition
        return div_y.ravel()

    return LinearOperator(
        (H * W, H * W), matvec=matvec, rmatvec=rmatvec, dtype=np.float64
    )


def compute_image_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the finite difference gradient using sparse operators."""
    H, W = image.shape
    image_vector = image.flatten()  # Convert image to 1D vector

    Dx = gradient_operator_x(W, H)
    Dy = gradient_operator_y(W, H)

    Gx = Dx @ image_vector  # Apply x-gradient operator
    Gy = Dy @ image_vector  # Apply y-gradient operator

    return Gx.reshape(H, W), Gy.reshape(H, W)

def generate_2D_gradient_matrices(N) -> tuple:
    '''
    Generate the gradient matrices for a 2D image

    Parameters:
    N: int
        Number of pixels in each dimension

    Returns:
    Kx: np.ndarray
        Gradient matrix in the x-direction
    Ky: np.ndarray
        Gradient matrix in the y-direction
    '''
    Kx_temp = sp.spdiags([-np.ones(N), np.ones(N)], [0, 1], N-1, N, format='csr')
    Kx = sp.kron(sp.spdiags(np.ones(N), [0], N, N), Kx_temp, format='csr')
    Ky_temp = sp.spdiags([-np.ones(N*(N-1))], [0], N*(N-1), N**2, format='csr')
    Ky = Ky_temp + sp.spdiags([np.ones(N*(N-1)+N)], [N],
                           N*(N-1), N**2, format='csr')
    K = sp.vstack((Kx, Ky))

    return Kx, Ky, K