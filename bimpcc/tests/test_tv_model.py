import numpy as np

# from bimpcc.tv_denoising import TVDenoising, _generate_index
# from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.models.tvdenoising_model import TVDenoisingMPCC, PenalizedTVDenoisingMPCC
from rich import print


def test_tv_denoising():
    """Test the correctness of the TV denoising problem."""
    true_img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    noisy_img = true_img + 0.05 * np.random.randn(*true_img.shape)

    model = TVDenoisingMPCC(true_img, noisy_img, epsilon=1e-8)
    # x0 = np.concatenate([noisy_img.flatten(), 1e-5*np.ones(30), 1e-5*np.ones(1)])
    res, x_opt, fun_opt = model.solve(
        print_level=0, tol=1e-6, t_min=1e-9, max_iter=100, verbose=True
    )
    u, q, r, delta, theta, alpha = model.parse_vars_fn(x_opt)
    print(f"true_img: {true_img.flatten()}")
    print(f"noisy_img: {noisy_img.flatten()}")
    print(f"u: {u}")
    print(f"q: {q}")
    print(f"r: {r}")
    print(f"delta: {delta}")
    print(f"theta: {theta}")
    print(f"alpha: {alpha}")
    print(f"complementarity test: {model.compute_complementarity(x_opt)}")


def test_penalized_tv_denoising():
    """Test the correctness of the TV denoising problem."""
    true_img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    noisy_img = true_img + 0.05 * np.random.randn(*true_img.shape)

    model = PenalizedTVDenoisingMPCC(true_img, noisy_img, epsilon=1e-8)
    # x0 = np.concatenate([noisy_img.flatten(), 1e-5*np.ones(30), 1e-5*np.ones(1)])
    res, x_opt, fun_opt = model.solve(print_level=0, max_iter=100, verbose=True)
    u, q, r, delta, theta, alpha = model.parse_vars_fn(x_opt)
    print(f"true_img: {true_img.flatten()}")
    print(f"noisy_img: {noisy_img.flatten()}")
    print(f"u: {u}")
    print(f"q: {q}")
    print(f"r: {r}")
    print(f"delta: {delta}")
    print(f"theta: {theta}")
    print(f"alpha: {alpha}")
    print(f"complementarity test: {model.compute_complementarity(x_opt)}")
