import numpy as np
import pandas as pd
from scipy.sparse import spdiags, kron, diags, vstack, coo_matrix
import os
from bimpcc.Dataset import get_dataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Callable

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
    Kx_temp = spdiags([-np.ones(N), np.ones(N)], [0, 1], N-1, N, format='csr')
    Kx = kron(spdiags(np.ones(N), [0], N, N), Kx_temp, format='csr')
    Ky_temp = spdiags([-np.ones(N*(N-1))], [0], N*(N-1), N**2, format='csr')
    Ky = Ky_temp + spdiags([np.ones(N*(N-1)+N)], [N],
                           N*(N-1), N**2, format='csr')

    # CREAR K NO SPARSE
    # Convertir matrices dispersas a arrays densos
    # Kx = Kx.toarray()  # Convertir Kx a array denso
    # Ky = Ky.toarray()  # Convertir Ky a array denso

    # create K matrix
    # K = np.empty((2*N*(N-1), Kx.shape[1]))

    # # Llenar K alternando filas de Kx y Ky
    # K[::2] = Kx  # Fila 1 de Kx, fila 3 de Kx, etc.
    # K[1::2] = Ky  # Fila 2 de Ky, fila 4 de Ky, etc
    # h = 1/(N-1)
    K = vstack((Kx, Ky))

    return Kx, Ky, K


def plot_experiment(experiment_file):
    parent_folder = os.path.abspath(os.path.dirname(experiment_file))
    dataset_name,N = os.path.splitext(os.path.basename(experiment_file))[0].split('_')
    N = int(N)
    M = N*(N-1)
    P = M//2
    results = pd.read_pickle(experiment_file)
    utrue,unoisy = get_dataset(dataset_name,int(N)).get_training_data()
    x = results['x']
    rec = x[:(N**2)]
    rec = rec.reshape(N,N)
    alpha = x[N**2+2*M:]
    # alpha = alpha.reshape(N,N-1)
    print(alpha)
    
    fig,ax = plt.subplots(1,3,figsize=(14,4))
    ax[0].imshow(utrue,cmap='gray')
    ax[0].set_title('True Image')
    ax[0].axis('off')
    ax[1].imshow(unoisy,cmap='gray')
    ax[1].set_title('Noisy Image\nPSNR: {:.4f}'.format(psnr(utrue,unoisy)))
    ax[1].axis('off')
    ax[2].imshow(rec,cmap='gray')
    ax[2].set_title(f'Reconstructed Image\nPSNR: {psnr(utrue,rec):.4f}')
    ax[2].set_xlabel('alpha = {}'.format(N))
    ax[2].axis('off')
    # ax[3].imshow(alpha,cmap='gray')
    # ax[3].set_title('alpha')
    # ax[3].set_xlabel('alpha = {}'.format(N))
    # ax[3].axis('off')
    plt.savefig(f'{parent_folder}/{dataset_name}_{N}.png')
    plt.show()


def create_matrix_from_vector(vector):
    """
    Convert a 2N vector into an Nx2 matrix by splitting it into two halves
    and stacking them as columns.
    
    Args:
        vector (np.ndarray): A 1D NumPy array with size 2N.
    
    Returns:
        np.ndarray: An Nx2 matrix.
    """
    # Ensure the vector has an even length
    if len(vector) % 2 != 0:
        raise ValueError("Vector length must be even.")
    
    # Split the vector into two halves
    mid = len(vector) // 2
    first_half = vector[:mid]
    second_half = vector[mid:]
    
    # Stack the halves as columns to create an Nx2 matrix
    matrix = np.column_stack((first_half, second_half))
    
    return matrix

def create_vector_from_matrix(vector):
    """
    Convert a 2N vector into an Nx2 matrix by splitting it into two halves
    and stacking them as columns.
    
    Args:
        vector (np.ndarray): A 1D NumPy array with size 2N.
    
    Returns:
        np.ndarray: An Nx2 matrix.
    """
    # Ensure the matrix has 2 columns
    if vector.shape[1] != 2:
        raise ValueError("Matrix must have 2 columns.")

    # Stack the halves as columns to create an Nx2 matrix
    matrix = np.concatenate((vector[:, 0], vector[:, 1]))

    return matrix

def apply_function_row_wise(matrix:np.ndarray, function:Callable, *args):
    """
    Apply a function row-wise to a matrix.
    
    Args:
        matrix (np.ndarray): A 2D NumPy array.
        function (callable): A function that takes a 1D NumPy array as input.
        args: Additional arguments to pass to the function.
    
    Returns:
        np.ndarray: A 2D NumPy array with the function applied row-wise.
    """
    # Apply the function row-wise to the matrix
    result = np.apply_along_axis(function, axis=1, arr=matrix, *args)
    
    return result