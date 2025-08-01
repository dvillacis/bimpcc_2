import numpy as np
from bimpcc.models.tvqregularized import TVqRegularized
# from bimpcc.utils import generate_2D_gradient_matrices


def test_tv_denoising():
    """Test the correctness of the TV denoising problem."""
    # true_img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    np.random.seed(0)
    sz = 5
    true_img = 0.5*np.random.randn(sz, sz)    
    # noisy_img = np.array([[0.1, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    noisy_img = true_img + 0.05 * np.random.randn(*true_img.shape)
    model = TVqRegularized(true_img, noisy_img, epsilon=1e-8)
    # x0 = np.concatenate([true_img.flatten(), 1e-5*np.ones(30), 1e-5*np.ones(1)])
    res,x_opt,fun_opt = model.solve(print_level=5)

    u, q, alpha = model.objective_func.parse_vars(x_opt)
    print(f"true_img: {true_img.flatten()}")
    print(f"noisy_img: {noisy_img.flatten()}")
    print(f"u: {u}")
    print(f"q: {q}")
    print(f"alpha: {alpha}")
    print(f"res: {res}")

    # Assert correctness
    # np.testing.assert_allclose(u, expected_u, atol=1e-6, err_msg="u is incorrect")
    # np.testing.assert_allclose(q, expected_q, atol=1e-6, err_msg="q is incorrect")
    # np.testing.assert_allclose(
    #     alpha, expected_alpha, atol=1e-6, err_msg="alpha is incorrect"
    # )
