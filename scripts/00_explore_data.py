import torch

from cnn_mnist.config.paths import get_paths, get_root_dir
from cnn_mnist.config.read_yaml import read_config_yaml
from cnn_mnist.data.load_data import load_train_data


def main():
    root_dir = get_root_dir()
    paths = get_paths(root_dir)
    path_data_raw = paths["data_raw"]
    # Load the data
    train_data = load_train_data(path_data_raw)
    # Check the shapes and types of the data
    print(f"Train data shape: {train_data.data.shape} and type : {type(train_data)}")

    # Calculate the mean and standard deviation of the training data
    transform = train_data.data.float() / 255.0
    mean = torch.mean(transform)
    std = torch.std(transform)
    print(f"Mean : {mean:.5f} and std : {std:.5f} of the training data")


if __name__ == "__main__":
    main()
