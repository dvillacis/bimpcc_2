import numpy as np
from bimpcc.nlp_old import DenoisingTVRegNLP

# def test_reg_tv_denoising():
#     """Test the correctness of the TV denoising problem."""
#     u_true = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     u_noisy = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
#     nlp = DenoisingTVRegNLP(u_true, u_noisy)
#     vars, info = nlp.solve()

#     # Expected values
#     expected_u = u_true
#     expected_q = np.zeros(u_true.shape)
#     expected_alpha = np.zeros(1)

#     # Extract the variables
#     u, q, alpha = nlp.getvars(vars)

#     # Assert correctness
#     np.testing.assert_allclose(u, expected_u, atol=1e-6, err_msg="u is incorrect")
#     np.testing.assert_allclose(q, expected_q, atol=1e-6, err_msg="q is incorrect")
#     np.testing.assert_allclose(
#         alpha, expected_alpha, atol=1e-6, err_msg="alpha is incorrect"
#     )