from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    test_accuracies: list[float],
    save_path: Path,
) -> None:
    """Generate and save training curves for loss and accuracy.
    One plots with two subplots :
        - Left : train loss and test loss per epoch
        - Right : test accuracy per epoch"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot : train loss and test loss
    axes[0].plot(train_losses, label="Train Loss", marker="o")
    axes[0].plot(test_losses, label="Test Loss", marker="o")
    axes[0].set_title("Training and Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Right subplot : test accuracy
    axes[1].plot(test_accuracies, label="Test Accuracy", marker="o")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()

    # Save figure
    plt.savefig(save_path)
    plt.close(fig)
