# CNN MNIST — Report

## Architecture

```
Input (1×28×28)
  → Conv(1→16) + ReLU + MaxPool  →  16×14×14
  → Conv(16→32) + ReLU + MaxPool →  32×7×7
  → Flatten → Linear(1568→128) + ReLU → Linear(128→10)
```

**Why this choice**: MNIST is a relatively simple problem — 28×28 grayscale images, balanced classes, low intra-class variability. Two convolutional layers are sufficient to capture edges and then more complex shapes. MaxPool halves the spatial resolution at each step, reducing the number of parameters while preserving the most relevant features. Augmentation (rotation, translation, scaling) is applied to improve robustness to geometric variations, which is key for handwritten digit recognition.

**Optimizer**: Adam (lr=0.001) — fast convergence without manual learning rate tuning.  
**Loss**: CrossEntropyLoss — standard for multi-class classification.

---

## Results

Configuration: 10 epochs · batch_size=32 · lr=0.001 · seed=42

| Metric                     | Value      |
| -------------------------- | ---------- |
| Train Loss (avg.)          | **0.0880** |
| Test Loss (avg.)           | **0.0360** |
| Test Accuracy (avg.)       | **98.80%** |
| Test Accuracy (last epoch) | **99.11%** |

---

## Overfitting

**Not observed.** Epochs 1 -> 10:
  - Train loss: 0.2889 -> 0.05
  - Test loss: 0.0619 -> 0.025
  - Test accuracy: 98.11% -> 99.11%

---

## Impact of augmentation

Augmentation (`RandomAffine`: rotation ±15°, translation ±10%, scale 0.9–1.1) is applied to the train set only. It has two effects:

1. **Implicit regularization**: training examples are made artificially harder, which explains why train loss stays above test loss — the model is penalized on augmented inputs but evaluated on clean images.
2. **Robustness**: the model learns to recognize digits despite slight geometric variations, improving generalization on real-world data.
