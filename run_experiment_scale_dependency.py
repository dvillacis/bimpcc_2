import yaml
import importlib
import pandas as pd
import h5py
import os
from bimpcc.dataset_experiments import get_dataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def plot_experiment(true, noisy, u, alpha):
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
    plt.show()

def calc_theta(Ku, q):
    Kxu = Ku[:n]
    Kyu = Ku[n:]
    qx = q[:n]
    qy = q[n:]
    epsilon = 1e-10
    theta0 = np.zeros(n)

    mask_r_nonzero = r0 > epsilon
    theta0[mask_r_nonzero] = np.arccos(Kxu[mask_r_nonzero] / r0[mask_r_nonzero])
    theta0[mask_r_nonzero & (Kyu < 0)] *= -1

    mask_r_zero_delta_nonzero = (r0 <= epsilon) & (delta0 > epsilon)
    theta0[mask_r_zero_delta_nonzero] = np.arccos(qx[mask_r_zero_delta_nonzero] / delta0[mask_r_zero_delta_nonzero])
    theta0[mask_r_zero_delta_nonzero & (qy < 0)] *= -1

    mask_r_zero_delta_zero = (r0 <= epsilon) & (delta0 <= epsilon)
    theta0[mask_r_zero_delta_zero] = 0
    return theta0

def load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def generate_tables(h5_path="outputs/results_mpcc_vs_img_size.h5",
    plot_output="outputs/iterations_vs_img_size.png",
    latex_output="outputs/tabla_iteraciones.tex"):

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
                    data.append({"img_size": img_size, "iterations": iterations})
                except Exception as e:
                    print(f"Error leyendo {group_name}: {e}")

    if data:
        df = pd.DataFrame(data).sort_values(by="img_size")
        plt.figure(figsize=(8, 5))
        plt.plot(df["img_size"], df["iterations"], marker='o')
        plt.xlabel("Tamaño de imagen (img_size)")
        plt.ylabel("Número de iteraciones")
        plt.title("Iteraciones vs Tamaño de Imagen")
        plt.grid(True)
        plt.savefig(plot_output)
        plt.show()
        print(f"Gráfico guardado en {plot_output}")

        latex_code = df.to_latex(index=False, caption="Número de iteraciones por tamaño de imagen", label="tab:iteraciones")

        with open(latex_output, "w") as f:
            f.write(latex_code)
        print(f"Código LaTeX guardado en {latex_output}")
    else:
        print("No se encontraron datos de iteraciones.")


config_path = "config/scale_dependency.yml"
output_file = "results_mpcc_vs_img_size.h5"
os.makedirs("outputs", exist_ok=True)

if __name__ == "__main__":
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_config = config["experiment"]
    dataset_config = experiment_config["dataset"]
    model_config = experiment_config["model"]
    dataset_name = dataset_config["dataset_name"]
    dataset__img_sizes = dataset_config["dataset__img_size"]
    model_name = model_config["model_name"]
    model_reg_name = model_config["model_reg_name"]

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
        dataset = get_dataset("cameraman", scale=img_size)
        true, noisy = dataset.get_training_data()

        model_reg_class = load_class(model_reg_name)
        model_reg_instance = model_reg_class(true, noisy, model_config["model_epsilon"])
        res, x_opt, fun_opt = model_reg_instance.solve(
            max_iter=model_config["max_iter_reg"],
            tol=float(model_config["tol"]),
            print_level=5
        )

        u, q, alpha = model_reg_instance.objective_func.parse_vars(x_opt)
        u = u.reshape((img_size, img_size))
        m = q.flatten().size
        n = m // 2
        Ku = model_reg_instance.K @ u.flatten()
        V = Ku.reshape(2, -1).T
        normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
        Q = q.reshape(2, -1).T
        normQ = np.apply_along_axis(np.linalg.norm, axis=1, arr=Q)
        r0 = normKu
        delta0 = normQ
        theta0 = calc_theta(Ku, q)
        x0_mpcc = np.concatenate((u.flatten(), q.flatten(), r0, delta0, theta0, alpha))

        model_mpcc_class = load_class(model_name)
        model_mpcc_instance = model_mpcc_class(true, noisy, x0=x0_mpcc, epsilon=model_config["model_epsilon"])
        res_mpcc, x_opt_mpcc, fun_opt_mpcc = model_mpcc_instance.solve(
            max_iter=model_config["max_iter"],
            tol=float(model_config["tol"]),
            print_level=0,
            verbose=True,
            beta=0.7
        )

        u_mpcc, q_mpcc, r_mpcc, delta_mpcc, theta_mpcc, alpha_mpcc = model_mpcc_instance.objective_func.parse_vars(x_opt_mpcc)
        u_mpcc = u_mpcc.reshape((img_size, img_size))

        with h5py.File(os.path.join("outputs", output_file), "a") as h5f:
            group = h5f.require_group(group_name)
            group.create_dataset("iterations", data=res_mpcc["iter"])
            group.create_dataset("fun_opt", data=fun_opt_mpcc)
            group.create_dataset("u_mpcc", data=u_mpcc)
            group.create_dataset("alpha_mpcc", data=alpha_mpcc)
            group.create_dataset("theta_mpcc", data=theta_mpcc)
            group.create_dataset("r_mpcc", data=r_mpcc)
            group.create_dataset("delta_mpcc", data=delta_mpcc)

    print("Todos los experimentos han finalizado correctamente.")
    generate_tables()
