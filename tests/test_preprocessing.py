import torch
from torchvision import transforms

from cnn_mnist.data.augmentation import get_augmentation_transforms
from cnn_mnist.data.preprocessing import get_preprocessing_transforms


def test_preprocessing_returns_compose():
    t = get_preprocessing_transforms()
    assert isinstance(t, transforms.Compose)


def test_preprocessing_has_two_steps():
    t = get_preprocessing_transforms()
    assert len(t.transforms) == 2


def test_preprocessing_output_shape():
    t = get_preprocessing_transforms()
    dummy = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
    pil_img = transforms.ToPILImage()(dummy)
    result = t(pil_img)
    assert result.shape == (1, 28, 28)


def test_preprocessing_output_normalized():
    t = get_preprocessing_transforms()
    dummy = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
    pil_img = transforms.ToPILImage()(dummy)
    result = t(pil_img)
    assert result.min() < 0.0


def test_augmentation_returns_compose():
    t = get_augmentation_transforms()
    assert isinstance(t, transforms.Compose)


def test_augmentation_output_shape():
    t = get_augmentation_transforms()
    dummy = torch.rand(1, 28, 28)
    pil_img = transforms.ToPILImage()(dummy)
    result = t(pil_img)
    assert result.size == (28, 28)
