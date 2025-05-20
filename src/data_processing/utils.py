# This is the utility file of the data_processing directory
# Place functionality here that is not directly tied to processing data such as transformation or augmentation but still happens during this phase of the pipeline

"""
Utility functions for data processing.
"""

from typing import Tuple, List

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from src.data_processing.augment import (
    get_transform_train,
    get_transform_test,
)


def get_cifar10_dataloaders(
    batch_size: int = 128, num_workers: int = 4, val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset and create dataloaders for train, validation and test sets.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        val_split: Fraction of training data to use for validation

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transformations
    transform_train = get_transform_train()
    transform_test = get_transform_test()

    # Download and load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./src/dataset",
        train=True,
        download=True,
        transform=transform_train,
    )

    # Split into training and validation sets
    train_size = int((1.0 - val_split) * len(trainset))
    val_size = len(trainset) - train_size

    # Use random_split to create train and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainset, [train_size, val_size]
    )

    # Apply test transforms to validation dataset (no augmentation)
    val_dataset.dataset.transform = transform_test

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Download and load test dataset
    testset = torchvision.datasets.CIFAR10(
        root="./src/dataset",
        train=False,
        download=True,
        transform=transform_test,
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_class_names() -> List[str]:
    """
    Get the class names for CIFAR-10 dataset.

    Returns:
        List of class names
    """
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


def get_sample_images(
    dataloader: DataLoader, num_samples: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get sample images and labels from a dataloader.

    Args:
        dataloader: DataLoader to get samples from
        num_samples: Number of samples to get

    Returns:
        Tuple of (images, labels)
    """
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Select a subset of the batch
    images = images[:num_samples]
    labels = labels[:num_samples]

    return images, labels
