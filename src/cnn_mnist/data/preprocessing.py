from torchvision import transforms


def get_preprocessing_transforms():
    """Return the preprocessing transforms for the MNIST dataset"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
