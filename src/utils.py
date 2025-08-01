import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

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

def gaussian_kernel_2d(size=3, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def reflect_index(i, N):
    if i < 0:
        return -i
    elif i >= N:
        return 2*N - i - 2
    return i

def gaussian_blur_sparse_matrix_symmetric(image_shape, kernel_size=3, sigma=1.0):
    H, W = image_shape
    N = H * W
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    offset = kernel_size // 2

    A = lil_matrix((N, N))

    for i in range(H):
        for j in range(W):
            row = i * W + j

            for di in range(-offset, offset + 1):
                for dj in range(-offset, offset + 1):
                    ni = reflect_index(i + di, H)
                    nj = reflect_index(j + dj, W)

                    col = ni * W + nj
                    weight = kernel[di + offset, dj + offset]
                    A[row, col] += weight

    return A.tocsr()


def plot_images_row(images, titles=None, figsize_per_image=4, cmap=None, save_path=None):
    """
    Plots a list of images in a single row.

    Parameters:
    - images: List of numpy arrays representing images (2D or 3D).
    - titles: Optional list of titles (same length as images).
    - figsize_per_image: Width (in inches) for each image. Height is auto-scaled.
    - cmap: Colormap to use for grayscale images (e.g., 'gray').
    - save_path: If provided, saves figure to this path (.png, .pdf, or .pgf supported).
    """
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(figsize_per_image * n, figsize_per_image))

    # Make sure axs is iterable
    if n == 1:
        axs = [axs]

    for i, (img, ax) in enumerate(zip(images, axs)):
        is_gray = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
        ax.imshow(img.squeeze(), cmap='gray' if is_gray and cmap is None else cmap)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()

    # if save_path:
    #     if save_path.endswith(".pgf"):
    #         import tikzplotlib
    #         tikzplotlib.save(save_path)
    #     else:
    #         plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()