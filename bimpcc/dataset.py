from PIL import Image
import numpy as np
from abc import ABC, abstractmethod


def load_and_scale_image(image_path, target_pixels, add_noise=False)->np.ndarray:
    image = Image.open(image_path).convert('L')
    resized_image = image.resize((target_pixels, target_pixels))
    grayscale_image = np.array(resized_image) / np.max(resized_image)
    
    if add_noise:
        mean = 0
        std = 0.2
        np.random.seed(0)
        noise = 0.05*np.random.randn(target_pixels, target_pixels)
        noisy_image = grayscale_image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        grayscale_image = noisy_image

    # Return the grayscale image
    return grayscale_image

class Dataset(ABC):
    def __init__(self, path, scale=256):
        self.scale = scale
        self.img_true = load_and_scale_image(path, scale)
        self.img_noisy = load_and_scale_image(path, scale, add_noise=True)
    
    def get_training_data(self):
        return self.img_true, self.img_noisy
    
class Synthetic:
    def __init__(self, scale):
        np.random.seed(0)
        self.scale = scale
        self.img_true = np.tril(0.9*np.ones((scale, scale)))
        self.img_noisy = self.img_true + 0.2*np.random.randn(scale, scale).clip(0, 1)
    
    def get_training_data(self):
        return self.img_true, self.img_noisy

def get_dataset(dataset_name, scale=256, folder='datasets'):
    if dataset_name == 'cameraman':
        return Dataset(f'../{folder}/cameraman/cameraman.png', scale)
    elif dataset_name == 'wood':
        return Dataset(f'../{folder}/wood/wood.png', scale)
    elif dataset_name == 'circle':
        return Dataset(f'../{folder}/circle/circle.png', scale)
    elif dataset_name == 'synthetic':
        return Synthetic(scale)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

