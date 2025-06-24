import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import yaml
import pickle
from itertools import product
import importlib
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import root_mean_squared_error as rmse
import time

from bimpcc.dataset import get_dataset, get_blur_dataset


def load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def calc_theta(Ku, q, r0, n, delta0):
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
    theta0[mask_r_zero_delta_nonzero] = np.arccos(
        qx[mask_r_zero_delta_nonzero] / delta0[mask_r_zero_delta_nonzero]
    )
    theta0[mask_r_zero_delta_nonzero & (qy < 0)] *= -1

    mask_r_zero_delta_zero = (r0 <= epsilon) & (delta0 <= epsilon)
    theta0[mask_r_zero_delta_zero] = 0
    return theta0


def generate_combinations(config):
    dataset_fn_name = config["dataset"]["fn_name"]
    dataset_name = config["dataset"]["name"]
    dataset_img_sizes = config["dataset"]["img_sizes"]
    dataset_random_seeds = config["dataset"]["random_seeds"]

    all_combinations = []
    for img_size, random_seed in product(dataset_img_sizes, dataset_random_seeds):
        for gamma in config["model"]["gammas"]:
            combination = {
                "dataset": {
                    "fn_name": dataset_fn_name,
                    "name": dataset_name,
                    "img_size": img_size,
                    "random_seed": random_seed,
                },
                "model": {
                    "name": config["model"]["name"],
                    "reg_name": config["model"]["reg_name"],
                    "tol": config["model"]["tol"],
                    "max_iter_reg": config["model"]["max_iter_reg"],
                    "max_iter": config["model"]["max_iter"],
                    "beta": config["model"]["beta"],
                    "gamma": gamma,
                }
            }
            all_combinations.append(combination)
    return all_combinations


def run_single_experiment(combo):
    fn  = globals()[combo["dataset"]["fn_name"]]
    dataset = fn(
        combo["dataset"]["name"],
        scale=combo["dataset"]["img_size"],
        random_state=combo["dataset"]["random_seed"],
    )
    true, noisy = dataset.get_training_data()
    reg_model_class = load_class(combo["model"]["reg_name"])
    reg_model_instance = reg_model_class(true, noisy)

    start_time = time.perf_counter()
    res, x_opt, fun_opt = reg_model_instance.solve(
        max_iter=int(combo["model"]["max_iter_reg"]),
        tol=float(combo["model"]["tol"]),
        print_level=0,
    )
    reg_elapsed_time = time.perf_counter() - start_time

    u, q, alpha = reg_model_instance.objective_func.parse_vars(x_opt)
    n = q.size // 2
    Ku = reg_model_instance.K @ u
    V = Ku.reshape(2, -1).T
    normKu = np.apply_along_axis(np.linalg.norm, axis=1, arr=V)
    Q = q.reshape(2, -1).T
    normQ = np.apply_along_axis(np.linalg.norm, axis=1, arr=Q)
    r0 = normKu
    delta0 = normQ
    theta0 = calc_theta(Ku, q, r0, n, delta0)
    x0_mpcc = np.concatenate((u, q, r0, delta0, theta0, alpha))

    model_mpcc_class = load_class(combo["model"]["name"])
    model_mpcc_instance = model_mpcc_class(true, noisy, x0=x0_mpcc)

    start_time = time.perf_counter()
    res_mpcc, x_opt_mpcc, fun_opt_mpcc = model_mpcc_instance.solve(
        max_iter=int(combo["model"]["max_iter"]),
        tol=float(combo["model"]["tol"]),
        print_level=0,
        verbose=True,
        beta=float(combo["model"]["beta"]),
    )
    mpcc_elapsed_time = time.perf_counter() - start_time

    u_mpcc, q_mpcc, r_mpcc, delta_mpcc, theta_mpcc, alpha_mpcc = model_mpcc_instance.objective_func.parse_vars(x_opt_mpcc)

    return {
        #"random_seed": combo["dataset"]["random_seed"],
        #"gamma": combo["model"]["gamma"],
        "img size": combo["dataset"]["img_size"],
        "fopt": fun_opt_mpcc,
        "PSNR damage": psnr(noisy, true),
        "PSNR reg": psnr(true.flatten(), u),
        "PSNR MPCC": psnr(true.flatten(), u_mpcc),
        "Comp Viol": model_mpcc_instance.comp,
        "alpha": alpha_mpcc,
        "iter mpcc": res_mpcc["iter"],
        # "iter_reg": res["nit"],
        "reg elapsed time": reg_elapsed_time,
        "mpcc elapsed time": mpcc_elapsed_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment - scale dependency")
    parser.add_argument("--plot", action="store_true", help="Plot the solution")
    parser.add_argument("--table", action="store_true", help="Generate the table")
    parser.add_argument("--run", action="store_true", help="Run the experiment")
    parser.add_argument(
        "--results", type=str, default=None, help="Path to the results file"
    )
    args = parser.parse_args()

    experiment_name = os.path.splitext(os.path.basename(__file__))[0]

    config_file = "config/" + experiment_name + ".yml"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{experiment_name}_{timestamp}.pkl"
    table_path = f"{results_dir}/{experiment_name}.tex"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

        if args.run:
            all_combinations = generate_combinations(config)
            all_results = {}
            for i, combo in enumerate(all_combinations):
                print(f"🔬 Experiment {i + 1}")
                print(combo)
                result = run_single_experiment(combo)
                all_results[i] = result

            with open(results_path, "wb") as f:
                pickle.dump(all_results, f)

        elif args.table:
            if args.results is None:
                raise ValueError("Please provide the path to the results file using --results.")
            with open(args.results, "rb") as f:
                all_results = pickle.load(f)

                df = pd.DataFrame(all_results).T
                #df = df.groupby(["img_size", "gamma"]).mean().reset_index()
                print(df)
                df.to_latex(table_path, index=False, float_format="%.4f", escape=False)
                print(f"Table saved to {table_path}")

        else:
            raise ValueError("Please provide --run argument to execute the experiment.")
