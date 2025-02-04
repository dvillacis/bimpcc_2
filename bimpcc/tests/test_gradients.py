import numpy as np
from bimpcc.utils import compute_image_gradients


def test_image_gradients_x():
    """Test the correctness of finite difference gradient operators."""

    # Simple test image: a linear gradient in the x-direction
    H, W = 5, 5
    test_image = np.tile(np.arange(W), (H, 1))  # Image increases linearly in x

    # Expected gradient values
    expected_Gx = np.ones((H, W))  # Should be 1 everywhere in x-direction
    expected_Gx[:, -1] = 0  # The last column should have zero (boundary effect)

    expected_Gy = np.zeros((H, W))  # No gradient in the y-direction

    # Compute actual gradients
    Gx, Gy = compute_image_gradients(test_image)

    # print(f"test_image: {test_image}")
    # print(f"expected_Gy: {expected_Gy}")
    # print(f"Gy: {Gy}")
    # print(f"Gx: {Gx}")

    # Assert correctness
    np.testing.assert_allclose(
        Gx, expected_Gx, atol=1e-6, err_msg="Gradient in x is incorrect"
    )
    np.testing.assert_allclose(
        Gy, expected_Gy, atol=1e-6, err_msg="Gradient in y is incorrect"
    )


def test_image_gradients_y():
    """Test the correctness of finite difference gradient operators."""

    # Simple test image: a linear gradient in the x-direction
    H, W = 5, 5
    test_image = np.tile(
        np.arange(H).reshape(H, 1), (1, W)
    )  # Image increases linearly in y
    

    # Expected gradient values
    expected_Gy = np.ones((H, W))  # Should be 1 everywhere in y-direction
    expected_Gy[-1, :] = 0  # The last row should have zero (boundary effect)
    

    expected_Gx = np.zeros((H, W))  # No gradient in the x-direction

    # Compute actual gradients
    Gx, Gy = compute_image_gradients(test_image)

    # print(f"test_image: {test_image}")
    # print(f"expected_Gy: {expected_Gy}")
    # print(f"Gy: {Gy}")
    # print(f"Gx: {Gx}")

    # Assert correctness
    np.testing.assert_allclose(
        Gx, expected_Gx, atol=1e-6, err_msg="Gradient in x is incorrect"
    )
    np.testing.assert_allclose(
        Gy, expected_Gy, atol=1e-6, err_msg="Gradient in y is incorrect"
    )


def test_constant_image():
    """Test that the gradient of a constant image is zero."""

    H, W = 10, 10
    constant_image = np.full((H, W), 42)  # Any constant value

    # Compute gradients
    Gx, Gy = compute_image_gradients(constant_image)

    # Gradients should be zero everywhere
    assert np.all(Gx == 0), "Gradient in x should be zero for a constant image"
    assert np.all(Gy == 0), "Gradient in y should be zero for a constant image"
