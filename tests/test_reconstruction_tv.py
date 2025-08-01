import numpy as np
from mpcc_models import TVDenoisingCartesian
from dataset import get_dataset

# np.random.seed(0)
# N = 20
# # true_img = np.array([[0.2, 0.1, 0.2], [0.3, 0.4, 0.1], [0.3, 0.8, 0.1]])
# true_img = 0.5*np.ones((N, N)) + 0.1 * np.random.randn(N, N)
# noisy_img = true_img + 0.1 * np.random.randn(N, N)
sz = 5
dataset = get_dataset("cameraman",scale=sz, random_state=100)
true_img, noisy_img = dataset.get_training_data()

def test_tv_denoising():
    # Create a TVDenoising instance
    tv_denoising = TVDenoisingCartesian(true_img, noisy_img, parameter_size=1)

    res = tv_denoising.solve(verbose=True, print_level=1, max_iter=12)

    print("Result:", res)
    print("Noisy:", noisy_img)
    print("u:", res["u"])
    print("alpha:", res["alpha"])

# def test_tv_denoising_fisher():
#     # Create a TVDenoising instance
#     tv_denoising = TVDenoising(true_img, noisy_img, parameter_size=1, relaxation_type="fisher")

#     res = tv_denoising.solve(verbose=True, print_level=1)

#     print("Result:", res)
#     print("Solution:", res["alpha"])
