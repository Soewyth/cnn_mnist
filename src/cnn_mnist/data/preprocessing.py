from torchvision import transforms


def get_preprocessing_transforms():
    """Return the preprocessing transforms for the MNIST dataset"""
    return transforms.Compose([
        transforms.ToTensor(), # convert to tensor [0, 255] -> [0.0, 1.0]
        transforms.Normalize((0.1307,),(0.3081,)) # center and reduce variance
    ])
