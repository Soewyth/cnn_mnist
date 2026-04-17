import json
from pathlib import Path

import torch


def save_model(model: torch.nn.Module, save_path: Path) -> None:
    """Save the trained model's state_dict to the specified path."""
    torch.save(model.state_dict(), save_path)


def save_metrics(metrics_dict: dict, save_path: Path) -> None:
    """Save the metrics dictionary to JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
