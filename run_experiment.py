import argparse
import yaml
import os
import pickle

import numpy as np


def load_config(config_path):
    """Loads experiment configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def instantiate_class(class_path, params):
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        class_ = getattr(module, class_name)
        return class_(**params) if params else class_()
    except Exception as e:
        raise Exception(f"Error instantiating class {class_path}: {e}")


def prepare_experiment(config):
    experiments = {}
    for experiment_name, experiment_params in config["experiments"].items():
        dataset_type = experiment_params["dataset"]
        dataset_params = experiment_params.get("dataset_params", {})
        model_type = experiment_params["model"]

        algorithm_type = experiment_params["algorithm"]

        dataset_instance = instantiate_class(dataset_type, dataset_params)

        model_params = experiment_params.get("model_params", {}) | {
            "dataset": dataset_instance
        }
        model_instance = instantiate_class(model_type, model_params)

        algorithm_params = experiment_params.get("algorithm_params", {}) | {
            "inner_fn": model_instance.inner_fn,
            "outer_fn": model_instance.outer_fn,
            "solution_map_fn": model_instance.solution_map_fn,
        }
        algorithm_instance = instantiate_class(algorithm_type, algorithm_params)

        outer_size = algorithm_params.get("outer_size", 1)
        outer_var_init_value = experiment_params.get("outer_var_init", 0)
        outer_var_init = outer_var_init_value * np.ones(outer_size)

        experiments[experiment_name] = {
            "dataset": dataset_instance,
            "model": model_instance,
            "algorithm": algorithm_instance,
            "outer_var_init": outer_var_init,
            "result_file": experiment_params.get("result_file", None),
        }

    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with an experiment configuration file."
    )
    parser.add_argument("file_path", type=str, help="Path to the input file")
    args = parser.parse_args()

    if os.path.exists(args.file_path):
        file_path = args.file_path
    else:
        raise FileNotFoundError(f"File {args.file_path} not found.")

    # You can use args.file_path as needed in your code
    print(f"File path provided: {file_path}")

    config = load_config(file_path)
    experiments = prepare_experiment(config)

    # Global configuration
    max_iter = config.get("max_iter", 2000)
    tol = config.get("tol", 1e-8)
    verbose = config.get("verbose", False)

    for experiment_name, experiment in experiments.items():
        print(f"Running experiment {experiment_name}")
        outer_var_init = experiment["outer_var_init"]
        experiment["algorithm"].fit(
            x0=outer_var_init, tol=1e-8, max_iter=2000, verbose=verbose
        )

        result_dict = {
            "experiment_name": experiment_name,
            "dataset": experiment["dataset"].__class__.__name__,
            "dataset_instance": experiment["dataset"],
            "model": experiment["model"].__class__.__name__,
            "algorithm": experiment["algorithm"].__class__.__name__,
            "outer_var_init": outer_var_init,
            "x": experiment["algorithm"].x,
            "y": experiment["algorithm"].y,
            "losses": experiment["algorithm"].losses,
            "num_iterations": experiment["algorithm"].num_iterations,
            "nfev": experiment["algorithm"].nfev,
            "ngev": experiment["algorithm"].ngev,
            "stopping_criteria": experiment["algorithm"].stopping_criteria,
        }

        result_file = experiment["result_file"]

        # Ensure the directory for the results path exists
        if result_file:
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "wb") as f:
            pickle.dump(result_dict, f)
        print(f"Results saved to {result_file}")

    # print(experiments)
