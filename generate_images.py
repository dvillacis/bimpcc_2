# generate_images_from_h5.py

import os
import h5py
import matplotlib
matplotlib.use('Agg')  # Forzar backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
from bimpcc.dataset_experiments import get_blur_dataset
from skimage.metrics import peak_signal_noise_ratio as psnr

def plot_experiment(true, noisy, u, alpha, save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(true, cmap='gray')
    ax[0].set_title('True Image')
    ax[0].axis('off')
    ax[1].imshow(noisy, cmap='gray')
    ax[1].set_title('Noisy Image\nPSNR: {:.4f}'.format(psnr(true, noisy)))
    ax[1].axis('off')
    ax[2].imshow(u, cmap='gray')
    ax[2].set_title(f'Reconstructed Image\nPSNR: {psnr(true, u):.4f}\n alpha = {alpha}')
    ax[2].axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f" Imagen guardada en {save_path}")
    plt.close(fig)

def generate_images_from_h5(h5_path="outputs/results_mpcc_vs_img_size.h5", output_dir="outputs"):
    if not os.path.exists(h5_path):
        print(f"Archivo HDF5 no encontrado: {h5_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as h5f:
        for group_name in h5f.keys():
            if group_name.startswith("img_size_"):
                img_size = int(group_name.replace("img_size_", ""))
                img_output_path = os.path.join(output_dir, f"reconstruction_img_size_{img_size}.png")

                if os.path.exists(img_output_path):
                    print(f"Imagen PNG para img_size {img_size} ya existe. Se omite.")
                    continue

                try:
                    u_mpcc = h5f[group_name]["u_mpcc"][()]
                    alpha_mpcc = h5f[group_name]["alpha_mpcc"][()]
                    u_mpcc = u_mpcc.reshape((img_size, img_size))

                    dataset = get_blur_dataset("cameraman", scale=img_size)
                    true, noisy = dataset.get_training_data()

                    plot_experiment(true, noisy, u_mpcc, alpha_mpcc, save_path=img_output_path)
                except Exception as e:
                    print(f"Error al procesar img_size {img_size}: {e}")

if __name__ == "__main__":
    generate_images_from_h5()
