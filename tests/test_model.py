import torch

from cnn_mnist.models.cnn import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN()
    x = torch.rand(8, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (8, 10)


def test_model_output_is_logits():
    model = SimpleCNN()
    x = torch.rand(1, 1, 28, 28)
    logits = model(x)
    assert not torch.all(logits >= 0)
