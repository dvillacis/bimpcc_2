from PIL import Image
import numpy as np
from abc import ABC


def load_and_scale_image(
    image_path, target_pixels, add_noise=False, random_state=None
) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    resized_image = image.resize((target_pixels, target_pixels))
    grayscale_image = np.array(resized_image) / np.max(resized_image)

    if add_noise:
        rng = np.random.default_rng(random_state)
        # np.random.seed(random_state)
        # noise = 0.05*np.random.randn(target_pixels, target_pixels)
        noise = rng.normal(0, 0.07, grayscale_image.shape)
        grayscale_image += noise

    grayscale_image = np.clip(grayscale_image, 0, 1)
    # Return the grayscale image
    return grayscale_image


class Dataset(ABC):
    def __init__(self, path, scale=256, random_state=None):
        self.scale = scale
        self.img_true = load_and_scale_image(path, scale)
        self.img_noisy = load_and_scale_image(
            path, scale, add_noise=True, random_state=random_state
        )

    def get_training_data(self):
        return self.img_true, self.img_noisy


class Synthetic:
    def __init__(self, scale):
        np.random.seed(0)
        self.scale = scale
        self.img_true = np.tril(0.9 * np.ones((scale, scale)))
        self.img_noisy = self.img_true + 0.2 * np.random.randn(scale, scale).clip(0, 1)

    def get_training_data(self):
        return self.img_true, self.img_noisy


def get_dataset(dataset_name, scale=256, folder="datasets", random_state=None):
    if dataset_name == "cameraman":
        return Dataset("datasets/cameraman/cameraman.png", scale, random_state)
    elif dataset_name == "caballo":
        return Dataset("datasets/caballo/caballo.jpg", scale, random_state)
    elif dataset_name == "guacamayo":
        return Dataset("datasets/guacamayo/guacamayo.png", scale, random_state)
    elif dataset_name == "mariposa":
        return Dataset("datasets/mariposa/mariposa.jpeg", scale, random_state)
    elif dataset_name == "synthetic":
        return Synthetic(scale)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
