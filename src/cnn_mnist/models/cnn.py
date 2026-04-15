import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # flatten
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(
            F.relu(self.conv1(x))
        )  # conv -> relu -> pool : 1x28x28 -> 16x28x28 -> 16x14x14
        x = self.pool(
            F.relu(self.conv2(x))
        )  # conv -> relu -> pool :  16x14x14 -> 32x14x14 -> 32x7x7
        x = torch.flatten(x, 1)  # flatten : 32x7x7 -> 1568
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
