import yaml
import importlib
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

def calc_theta(Ku,q):
    Kxu = Ku[:n]  # Primeras n componentes
    Kyu = Ku[n:]  # Últimas n componentes
    qx = q[:n]  # Primeras n componentes
    qy = q[n:]  # Últimas n componentes
    epsilon = 1e-10

    # Condiciones para calcular theta según los casos dados
    theta0 = np.zeros(n)  # Inicializamos theta0

    # Caso 1: r_i ≠ 0
    mask_r_nonzero = r0 > epsilon
    theta0[mask_r_nonzero] = np.arccos(Kxu[mask_r_nonzero] / r0[mask_r_nonzero])
    theta0[mask_r_nonzero & (Kyu < 0)] *= -1  # Si Ky^i u < 0, negamos theta0

    # Caso 2: r_i = 0 y delta_i ≠ 0
    mask_r_zero_delta_nonzero = (r0 <= epsilon) & (delta0 > epsilon)
    theta0[mask_r_zero_delta_nonzero] = np.arccos(qx[mask_r_zero_delta_nonzero] / delta0[mask_r_zero_delta_nonzero])
    theta0[mask_r_zero_delta_nonzero & (qy < 0)] *= -1  # Si qy^i < 0, negamos theta0

    # Caso 3: r_i = 0 y delta_i = 0 (theta0 indefinido)
    mask_r_zero_delta_zero = (r0 <= epsilon) & (delta0 <= epsilon)
    theta0[mask_r_zero_delta_zero] = 0
    return theta0

config_path = "config/scale_dependency.yml"

def load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def generate_tables():
    pass

def generate_figures():
    pass


if __name__ == "__main__":
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_config = config["experiment"]
    print(experiment_config,type(experiment_config))
    dataset_config = experiment_config["dataset"]
    model_config = experiment_config.get("model")
    dataset_name = dataset_config.get("dataset_name")
    dataset__img_sizes = dataset_config.get("dataset__img_size")
    model_name = model_config.get("model_name")
    model_reg_name = model_config.get("model_reg_name")

    for img_size in dataset__img_sizes:

        print(f"Running experiment with image size: {img_size}")

        dataset = get_dataset("cameraman",scale=img_size)
        true, noisy = dataset.get_training_data()

        model_reg_class = load_class(model_reg_name)
        model_reg_instance = model_reg_class(true, noisy,model_config["model_epsilon"])
        res, x_opt, fun_opt =   model_reg_instance.solve(max_iter = model_config["max_iter_reg"], tol = float(model_config["tol"]), print_level=5)
        u, q, alpha = model_reg_instance.objective_func.parse_vars(x_opt)
        u = u.reshape((img_size,img_size))
        m = q.flatten().size
        n = m // 2
        Ku = model_reg_instance.K @ u.flatten()
        V = Ku.reshape(2, -1).T
        normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
        Q = q.reshape(2,-1).T
        normQ = np.apply_along_axis(np.linalg.norm, axis=1, arr=Q)
        r0 = normKu
        delta0 = normQ
        theta0 = calc_theta(Ku,q)
        x0_mpcc = np.concatenate((u.flatten(),q.flatten(),r0,delta0,theta0,alpha))
        model_mpcc_class = load_class(model_name)
        model_mpcc_instance = model_mpcc_class(true, noisy, x0 = x0_mpcc, epsilon = model_config["model_epsilon"])
        res_mpcc,x_opt_mpcc,fun_opt_mpcc = model_mpcc_instance.solve(max_iter=model_config["max_iter"],tol=float(model_config["tol"]),print_level=0,verbose=True,beta=0.7)
        u_mpcc, q_mpcc, r_mpcc, delta_mpcc, theta_mpcc, alpha_mpcc = model_mpcc_instance.objective_func.parse_vars(x_opt_mpcc)
        u_mpcc = u_mpcc.reshape((img_size,img_size))
        plot_experiment(true,noisy,u_mpcc,alpha_mpcc)



    