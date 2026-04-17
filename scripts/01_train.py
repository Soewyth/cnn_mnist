import torch
import torch.nn as nn
from torchvision import transforms
import random
import numpy as np
from cnn_mnist.config.paths import get_paths, get_root_dir
from cnn_mnist.config.read_yaml import read_config_yaml
from cnn_mnist.data.augmentation import get_augmentation_transforms
from cnn_mnist.data.load_data import load_test_data, load_train_data
from cnn_mnist.data.preprocessing import get_preprocessing_transforms
from cnn_mnist.evaluation.artifacts import save_metrics, save_model
from cnn_mnist.evaluation.plots import plot_training_curves
from cnn_mnist.models.cnn import SimpleCNN
from cnn_mnist.training.evaluate import evaluate
from cnn_mnist.training.trainer import train_one_epoch
from cnn_mnist.io.run_id import get_run_id


def main():

    # Get paths and config
    root_dir = get_root_dir()
    paths = get_paths(root_dir)
    config = read_config_yaml(root_dir / "config.yaml")
    path_data_raw = paths["data_raw"]
    path_figures = paths["figures"]
    path_model = paths["models"]
    run_id = get_run_id()

    # Set random seeds for reproducibility random && numpy for DataLoader shuffling && augmentation
    seed = config["training"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare data with augmentation and preprocessing & combined for training
    preprocess = get_preprocessing_transforms()
    augmentation = get_augmentation_transforms()
    combined = transforms.Compose([*augmentation.transforms, *preprocess.transforms])

    # Load data
    train_data = load_train_data(data_dir=path_data_raw, transform=combined)
    test_data = load_test_data(data_dir=path_data_raw, transform=preprocess)

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=config["training"]["batch_size"], shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Device setup, model, optimizer, criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device=device)
    optimizer_name = config["training"]["optimizer"]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    # CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Prepare data for loop
    num_epochs = config["training"]["num_epochs"]
    average_train_loss = 0.0
    average_test_loss = 0.0
    average_test_accuracy = 0.0
    train_losses = []
    test_losses = []
    test_accuracies = []

    # Training and Evaluation loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # average train loss for all epochs && append to list for plotting
        average_train_loss += train_loss / num_epochs
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Evaluate on test set
        test_loss, test_accuracy = evaluate(
            model=model, dataloader=test_dataloader, criterion=criterion, device=device
        )
        # average test loss and accuracy for all epochs && append to list for plotting
        average_test_loss += test_loss / num_epochs
        average_test_accuracy += test_accuracy / num_epochs
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%"
        )

    # === SAVE ARTIFACTS ===
    training_plot_dest = path_figures / f"training_curves_{run_id}.png"
    plot_training_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        save_path=training_plot_dest,
    )

    model_dest = path_model / f"model_{run_id}.pth"
    save_model(model=model, save_path=model_dest)
    metrics = {
        "config": {
            "file": "config.yaml",
            "training": config["training"],
            "paths": config["paths"],
        },
        "average_train_loss": average_train_loss,
        "average_test_accuracy": average_test_accuracy,
        "average_test_loss": average_test_loss,
    }
    metrics_dest = path_model / f"metrics_{run_id}.json"
    save_metrics(metrics_dict=metrics, save_path=metrics_dest)

    print(f"\n === Training Values : ===\n")
    print(f"Average Train loss after all epochs: {average_train_loss:.4f}")
    print(f"Average Test loss after all epochs: {average_test_loss:.4f}")
    print(f"Average Test Accuracy after all epochs: {average_test_accuracy*100:.2f}%")
    print(f" === Artifacts saved : ===\n")
    print(f"Model saved at: {model_dest}")
    print(f"Metrics saved at: {metrics_dest}")
    print(f"Training curves saved at: {training_plot_dest}")


if __name__ == "__main__":
    main()
