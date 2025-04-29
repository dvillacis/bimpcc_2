import yaml
import importlib
import pandas as pd
import h5py
import os
from bimpcc.dataset_experiments import get_blur_dataset
from bimpcc.dataset_experiments import get_dataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def plot_experiment(true,noisy,u, alpha):
    
    fig,ax = plt.subplots(1,3,figsize=(14,4))
    ax[0].imshow(true,cmap='gray')
    ax[0].set_title('True Image')
    ax[0].axis('off')
    ax[1].imshow(noisy,cmap='gray')
    ax[1].set_title('Noisy Image\nPSNR: {:.4f}'.format(psnr(true,noisy)))
    ax[1].axis('off')
    ax[2].imshow(u,cmap='gray')
    ax[2].set_title(f'Reconstructed Image\nPSNR: {psnr(true,u):.4f}\n alpha = {alpha}')
    # ax[2].set_xlabel('alpha = {}'.format(alpha))
    ax[2].axis('off')

    plt.show()

def load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def generate_tables(
    h5_path="outputs_initial_point/results_initial_point.h5",
    latex_output="outputs_initial_point/tabla_initial_point.tex",
    max_iter=100
    ): 
    if not os.path.exists(h5_path):
        print(f"Archivo HDF5 no encontrado: {h5_path}")
        return

    data = []
    with h5py.File(h5_path, "r") as h5f:
        for group_name in h5f.keys():
            if group_name.startswith("img_size_"):
                try:
                    img_size = int(group_name.replace("img_size_", ""))
                    iterations = int(h5f[group_name]["iterations"][()])
                    
                    # Determinar si converge o no
                    converged = "Yes" if iterations < max_iter else "No"
                    
                    data.append({
                        "img_size": img_size,
                        "converged": converged,
                        "iterations": iterations
                    })
                except Exception as e:
                    print(f"Error leyendo {group_name}: {e}")

    if data:
        df = pd.DataFrame(data).sort_values(by="img_size")

        latex_code = df.to_latex(
            index=False,
            caption="Convergence results according to image size",
            label="tab:convergence_vs_imgsize",
            column_format="rcl"  # Formato: right, center, left
        )

        with open(latex_output, "w") as f:
            f.write(latex_code)
        print(f"Código LaTeX guardado en {latex_output}")
    else:
        print("No se encontraron datos de iteraciones.")

config_path = "config/initial_point.yml"
output_file = "results_initial_point.h5"
os.makedirs("outputs_initial_point", exist_ok=True)

if __name__ == "__main__":
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_config = config["experiment"]
    dataset_config = experiment_config["dataset"]
    model_config = experiment_config["model"]
    dataset_name = dataset_config["dataset_name"]
    dataset__img_sizes = dataset_config["dataset__img_size"]
    model_name = model_config["model_name"]

    # Detectar tamaños ya guardados
    h5_existing_sizes = set()
    if os.path.exists(os.path.join("outputs", output_file)):
        with h5py.File(os.path.join("outputs", output_file), "r") as h5f:
            h5_existing_sizes = set(h5f.keys())

    for img_size in dataset__img_sizes:
        group_name = f"img_size_{img_size}"
        if group_name in h5_existing_sizes:
            print(f" img_size {img_size} ya procesado. Se omite.")
            continue

        print(f" Ejecutando experimento con img_size = {img_size}")
        dataset = get_blur_dataset("cameraman", scale=img_size)
        true, noisy = dataset.get_training_data()

        model_mpcc_class = load_class(model_name)
        model_mpcc_instance = model_mpcc_class(true, noisy, x0=None, epsilon=model_config["model_epsilon"])
        res_mpcc, x_opt_mpcc, fun_opt_mpcc = model_mpcc_instance.solve(
            max_iter=model_config["max_iter"],
            tol=float(model_config["tol"]),
            print_level=0,
            verbose=True,
            beta=0.7
        )

        u_mpcc, q_mpcc, r_mpcc, delta_mpcc, theta_mpcc, alpha_mpcc = model_mpcc_instance.objective_func.parse_vars(x_opt_mpcc)

        with h5py.File(os.path.join("outputs_initial_point", output_file), "a") as h5f:
            group = h5f.require_group(group_name)
            group.create_dataset("iterations", data=res_mpcc["iter"])
            group.create_dataset("fun_opt", data=fun_opt_mpcc)
            group.create_dataset("u_mpcc", data=u_mpcc)
            group.create_dataset("alpha_mpcc", data=alpha_mpcc)
            group.create_dataset("theta_mpcc", data=theta_mpcc)
            group.create_dataset("r_mpcc", data=r_mpcc)
            group.create_dataset("delta_mpcc", data=delta_mpcc)

    print("Todos los experimentos han finalizado correctamente.")
    generate_tables(max_iter=model_config["max_iter"])
