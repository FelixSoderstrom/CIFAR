"""
Transfer learning model using ResNet50 for CIFAR-10 classification.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class TransferResNet50(nn.Module):
    """
    Transfer learning model using pretrained ResNet50 for CIFAR-10 classification.

    This model uses a pretrained ResNet50 as a feature extractor and adds
    a custom classification head on top.
    """

    def __init__(self) -> None:
        """Initialize the transfer learning model."""
        super(TransferResNet50, self).__init__()

        # Load pretrained ResNet50 model with updated API
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all parameters in the base model
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        # ResNet50 has 2048 features in the last layer before classification
        in_features = self.resnet.fc.in_features

        # New classification head for CIFAR-10 (10 classes)
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits tensor of shape (batch_size, 10)
        """
        # CIFAR-10 images are 32x32, but ResNet expects at least 224x224
        # We can either upsample or adjust the model
        # Here we're using the model as-is, but may need to upsample the images
        return self.resnet(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the last hidden layer.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Feature embeddings of shape (batch_size, 512)
        """
        # Get feature vector before the final classification layer
        # Go through all layers except the final fully connected layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Get embeddings from the first layer of the new classification head
        embeddings = self.resnet.fc[0](x)  # 512-dim embedding

        return embeddings
