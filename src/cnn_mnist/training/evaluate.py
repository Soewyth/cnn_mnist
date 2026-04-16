from torch import nn
import torch


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate the model and return the average loss and accuracy.
    Args:
        model: The neural network model to be evaluated.
        dataloader: DataLoader providing the evaluation data.
        criterion: The loss function to compute the error.
        device: The device (CPU or GPU) to perform computations on.
    Returns:
        float: The average loss for the epoch.
        float: The accuracy of the model on the evaluation data.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)  # take the class with max value
            correct += (
                (preds == labels).sum().item()
            )  # item to get the value from the tensor
    average_loss = total_loss / len(dataloader)  # average loss per batch
    accuracy = correct / len(dataloader.dataset) # correct predictions / total samples
    return accuracy, average_loss
