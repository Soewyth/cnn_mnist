from torchvision import datasets


def load_train_data(data_dir):
    """Load the training datasets"""
    return datasets.MNIST(data_dir, train=True, download=True)


def load_test_data(data_dir):
    """Load the test datasets"""
    return datasets.MNIST(data_dir, train=False, download=True)
