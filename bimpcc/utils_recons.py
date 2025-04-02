import numpy as np
import scipy.signal

def apply_blur(u, psf):
    """
    Aplica el operador A de deblurring mediante convolución con la PSF.
    
    Args:
        u (numpy.ndarray): Imagen de entrada.
        psf (numpy.ndarray): Función de dispersión del punto (PSF).
    
    Returns:
        numpy.ndarray: Imagen borrosa.
    """
    return scipy.signal.convolve2d(u, psf, mode='same', boundary='wrap')

def gradient_f(u, psf, u_true):
    """ Calcula el gradiente de f(u) = (1/2) * ||Au - u_true||^2. """
    Au = apply_blur(u, psf)  # Aplicar A (convolución con PSF)
    residual = Au - u_true  # Diferencia
    psf_adj = np.flip(psf)  # A^T = convolución con PSF rotada 180°
    return scipy.signal.convolve2d(residual, psf_adj, mode='same', boundary='wrap')

def hessian_f(u, psf):
    """ Aplica la Hessiana de f(u), es decir, A^T A u. """
    Au = apply_blur(u, psf)  # Aplicar A
    psf_adj = np.flip(psf)  # A^T
    return apply_blur(Au, psf_adj)  # A^T A u