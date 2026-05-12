# CNN MNIST

A simple Convolutional Neural Network trained on the MNIST handwritten digit dataset.

## What this project demonstrates

- CNN training from scratch on grayscale images (MNIST) with stable convergence
- Data augmentation pipeline integrated into reproducible training loop
- Structured DL project layout (notebooks/, src/, outputs/)

## Architecture

- 2 convolutional layers (Conv → ReLU → MaxPool)
- 2 fully connected layers
- Input: 1×28×28 — Output: 10 logits (one per digit class)

## Project structure

```
.
├── data/
│   ├── raw/                  # MNIST raw data (git-ignored)
│   └── processed/            # Preprocessed tensors (git-ignored)
├── notebooks/
│   └── 01_cnn_mnist.ipynb    # Exploration notebook
├── outputs/
│   ├── figures/              # Training curves (git-ignored)
│   └── models/               # Saved .pth and metrics.json (git-ignored)
├── scripts/
│   ├── 00_explore_data.py    # Data exploration
│   └── 01_train.py           # Training entry point
├── src/cnn_mnist/
│   ├── config/               # Paths and yaml reader
│   ├── data/                 # Load, preprocess, augmentation
│   ├── evaluation/           # Save model, metrics, plots
│   ├── io/                   # Run ID generation
│   ├── models/               # SimpleCNN definition
│   └── training/             # train_one_epoch, evaluate
├── tests/                    # Unit tests
├── config.yaml               # Training hyperparameters
├── Makefile                  # Project commands
├── pyproject.toml
└── requirements.txt
```

## Quickstart

**1. Install dependencies**

```bash
make setup
```

**2. Configure training**

Edit `config.yaml` to adjust hyperparameters:

```yaml
training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 0.001
  random_seed: 42
  optimizer: "adam"
```

**3. Train**

```bash
make train
```

The dataset is downloaded automatically on first run. Artifacts are saved in `outputs/`.

**4. Run tests**

```bash
make test
```

## All Makefile commands

| Command        | Description                                               |
| -------------- | --------------------------------------------------------- |
| `make setup`   | Create `.venv` and install all dependencies               |
| `make explore` | Run the data exploration script                           |
| `make train`   | Run the training script                                   |
| `make test`    | Run the test suite with pytest                            |
| `make lint`    | Check and format code with ruff                           |
| `make clean`   | Remove `__pycache__`, pytest/ruff caches, and all outputs |

## Outputs

Each run generates a unique ID (timestamp-based) and saves:

| File                                           | Description               |
| ---------------------------------------------- | ------------------------- |
| `outputs/models/model_<run_id>.pth`            | Model weights             |
| `outputs/models/metrics_<run_id>.json`         | Average loss and accuracy |
| `outputs/figures/training_curves_<run_id>.png` | Loss and accuracy curves  |
