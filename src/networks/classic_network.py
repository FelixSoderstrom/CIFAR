# This is the network definition of the classical architecture that we will build fro mthe ground up

"""
Classic CNN architecture for CIFAR-10 classification.
"""

from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicCNN(nn.Module):
    """
    Custom CNN architecture for CIFAR-10 classification.

    The architecture consists of several convolutional layers followed by batch normalization,
    max pooling, and dropout, with fully connected layers at the end.
    """

    def __init__(self) -> None:
        """Initialize the CNN architecture."""
        super(ClassicCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(4096, 512)  # 4x4x256 = 4096
        self.fc2 = nn.Linear(512, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits tensor of shape (batch_size, 10)
        """
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Third block
        x = F.relu(self.bn3(self.conv3(x)))

        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the last hidden layer.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Feature embeddings of shape (batch_size, 512)
        """
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Third block
        x = F.relu(self.bn3(self.conv3(x)))

        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Get embeddings from the last hidden layer
        embeddings = F.relu(self.fc1(x))

        return embeddings
