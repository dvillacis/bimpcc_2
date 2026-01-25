import numpy as np
import matplotlib.pyplot as plt


def two_smooth_squares(size: int = 256, small_size: int = 200):
    # build image
    image = np.reshape(np.array([(x / size) for x in range(size)] * size), (size, size))
    image[28 : small_size + 28, 28 : small_size + 28] = np.reshape(
        np.array([(1 - x / small_size) for x in range(small_size)] * small_size),
        (small_size, small_size),
    )
    image /= np.max(image)

    assert np.all([0 <= np.min(image), np.max(image) == 1])

    return image


image = two_smooth_squares(size=256, small_size=120)

plt.imshow(image, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.title("Two smooth squares")
plt.show()

plt.imsave("square.png", image, cmap="gray", vmin=0, vmax=1)
