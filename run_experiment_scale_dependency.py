import yaml
import importlib
from bimpcc.dataset import get_dataset

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
        model_reg_instance = model_reg_class(true, noisy,model_config["epsilon"])
        # model_reg_instance.solve(max_iter=model_config["max_iter_reg"], tol=model_config["tol"])


    