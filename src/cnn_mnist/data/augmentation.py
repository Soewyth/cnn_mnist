from torchvision import transforms


def get_augmentation_transforms():
    """Return the augmentation transforms for the MNIST dataset"""
    return transforms.Compose(
        [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))]
    )
