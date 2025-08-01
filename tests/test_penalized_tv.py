import numpy as np
from mpcc_models.reconstruction_tv import TVDenoising

sz=5
true_img = 0.5* np.ones((sz, sz))
noisy_img = true_img + 0.1 * np.random.randn(sz, sz)

def test_penalized_tv_denoising():
    model = TVDenoising(true_img, noisy_img, inner_penalty_init=0.1, outer_penalty_init=1.0)
    x0 = model.initial_guess()
    res = model.solve(x0, print_level=0, verbose=True)
    print("Penalized TV Denoising Solution:", res["alpha"])
    print("Function value at solution:", res["fn"])
    print("Optimization result:", res)
    