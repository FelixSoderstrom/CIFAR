"""
Transfer learning model using ResNet20 for CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet20."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet20 model specifically designed for CIFAR-10 classification.

    This model is based on the ResNet architecture from the paper
    "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385)
    but adapted for CIFAR-10 with the 20-layer configuration.
    """

    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        # Create the residual blocks
        # ResNet20 has 3 groups of layers, with 3 BasicBlock each (3×2×3 + 2 = 20 layers)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)

        # Final fully connected layer
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits tensor of shape (batch_size, 10)
        """
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global average pooling and final classification
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def get_embedding(self, x):
        """
        Extract embeddings from the last hidden layer.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Feature embeddings before the final classification layer
        """
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global average pooling
        out = F.avg_pool2d(out, out.size()[2:])
        embeddings = out.view(out.size(0), -1)

        return embeddings


# For backward compatibility with existing code that expects TransferResNet50
class TransferResNet50(ResNet20):
    """
    Alias for ResNet20 to maintain backward compatibility with existing code.
    This class has been renamed but keeps the original class name for compatibility.
    """

    def __init__(self):
        """Initialize the ResNet20 model for CIFAR-10."""
        super(TransferResNet50, self).__init__(num_classes=10)
