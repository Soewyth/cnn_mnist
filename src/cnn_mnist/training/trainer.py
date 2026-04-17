import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    """Train the model for one epoch.
    Args:
        model: The neural network model to be trained.
        dataloader: DataLoader providing the training data.
        optimizer: The optimization algorithm to update the model parameters.
        criterion: The loss function to compute the error.
        device: The device (CPU or GPU) to perform computations on.
    Returns:
        float: The average loss for the epoch."""
    model.train()  # Set the model to training mode
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)  # forward pass
        loss = criterion(logits, labels)  # computing error
        loss.backward()
        optimizer.step()  # update weights
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss
