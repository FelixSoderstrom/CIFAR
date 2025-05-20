# This is the augmentation file in the data_processing directory.
# Perform all augmentation here

"""
Data augmentation functions for CIFAR-10.
"""

from typing import Callable

import torchvision.transforms as transforms


def get_transform_train() -> Callable:
    """
    Get transformations for training data.

    Returns:
        Composed transforms for training
    """
    return transforms.Compose(
        [
            # Random crop with padding
            transforms.RandomCrop(32, padding=4),
            # Random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Random rotation (±15 degrees)
            transforms.RandomRotation(15),
            # Color jitter
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize with CIFAR-10 mean and std
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )


def get_transform_test() -> Callable:
    """
    Get transformations for validation/test data.

    Returns:
        Composed transforms for validation/test
    """
    return transforms.Compose(
        [
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize with CIFAR-10 mean and std
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )
