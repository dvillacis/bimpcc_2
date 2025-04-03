import numpy as np
import scipy.signal
import scipy.sparse as sp


def apply_blur(u, psf):
    """
    Aplica el operador A de deblurring mediante convolución con la PSF.
    
    Args:
        u (numpy.ndarray): Imagen de entrada.
        psf (numpy.ndarray): Función de dispersión del punto (PSF).
    
    Returns:
        numpy.ndarray: Imagen borrosa.
    """
    return scipy.signal.convolve2d(u, psf, mode='same', boundary='symm')

def gradient_f(u, psf, u_true):
    """ Calcula el gradiente de f(u) = (1/2) * ||Au - u_true||^2. """
    Au = apply_blur(u, psf)  # Aplicar A (convolución con PSF)
    residual = Au - u_true  # Diferencia
    psf_adj = np.flip(psf)  # A^T = convolución con PSF rotada 180°
    return scipy.signal.convolve2d(residual, psf_adj, mode='same', boundary='wrap')

def convolution_matrix(psf, u):
    """Construye la matriz de convolución A para imágenes vectorizadas"""
    H, W = u.shape  # Dimensiones de la imagen
    N = H * W  # Tamaño total

    # Crear matriz dispersa de convolución
    A = sp.lil_matrix((N, N))  

    # Crear una imagen identidad para extraer cada fila de A
    for i in range(N):
        impulse = np.zeros((H, W))
        impulse[i // W, i % W] = 1  # Activar un solo píxel
        conv_result = scipy.signal.convolve2d(impulse, psf, mode='same', boundary='wrap')
        A[:, i] = conv_result.flatten()  # Guardar en la matriz A

    return A.tocsr()  # Convertir a formato disperso eficiente

def hessian_matrix(psf, u):
    """Construye la matriz Hessiana H_f = A^T A"""
    A = convolution_matrix(psf, u)  # Obtener A
    H_f = A.T @ A  # Calcular A^T A
    return H_f

def gaussian_psf(size, sigma):
    """Genera un kernel gaussiano 2D de tamaño `size x size` y desviación `sigma`"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalizar