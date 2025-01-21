import numpy as np


def parse_vars(x):
    pass
    

def objective(u,u_true):
    return 0.5 * np.linalg.norm(u-u_true)**2

def gradient(u,u_true):
    return u-u_true