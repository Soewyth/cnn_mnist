from torchvision import datasets


def load_train_data(data_dir: str, transform=None):
    """Load the training datasets"""
    return datasets.MNIST(data_dir, train=True, download=True, transform=transform)


def load_test_data(data_dir: str, transform=None):
    """Load the test datasets"""
    return datasets.MNIST(data_dir, train=False, download=True, transform=transform)
