# This file is for setting up the trainer

"""
Training functionality for models.
"""

import time
import os
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.utils import save_training_summary, AverageMeter


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    session_dir: str = "output",
    patience: int = 4,  # Patience for early stopping
) -> Dict[str, Any]:
    """
    Train a model on CIFAR-10.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model_name: Name of the model (for saving)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on
        session_dir: Directory to save outputs
        patience: Number of epochs to wait before early stopping

    Returns:
        Dictionary containing training statistics
    """
    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Initialize tracking variables
    best_val_acc = 0.0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping variables
    counter = 0
    early_stopped = False

    # Start timing
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Store statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save checkpoint if it's the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            counter = 0  # Reset early stopping counter
            torch.save(
                model.state_dict(),
                f"{session_dir}/checkpoints/best_{model_name}.ckpt",
            )
        else:
            counter += 1  # Increment early stopping counter

        # Check for early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            early_stopped = True
            break

        # Save regular checkpoint
        torch.save(
            model.state_dict(),
            f"{session_dir}/checkpoints/epoch_{epoch+1}_{model_name}.ckpt",
        )

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Calculate training time
    training_time = time.time() - start_time

    # Determine actual epochs completed
    actual_epochs = epoch + 1 if early_stopped else epochs

    # Prepare training statistics
    training_stats = {
        "model_name": model_name,
        "epochs": epochs,
        "actual_epochs": actual_epochs,
        "early_stopped": early_stopped,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "training_time": training_time,
    }

    # Save training summary
    save_training_summary(training_stats, model_name, session_dir)

    # Load best model
    model.load_state_dict(
        torch.load(f"{session_dir}/checkpoints/best_{model_name}.ckpt")
    )

    return training_stats


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average loss, accuracy)
    """
    # Initialize metrics
    losses = AverageMeter()
    accuracy = AverageMeter()

    # Iterate over batches
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        acc = correct / inputs.size(0)

        # Update metrics
        losses.update(loss.item(), inputs.size(0))
        accuracy.update(acc, inputs.size(0))

    return losses.avg, accuracy.avg


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate the model on validation data.

    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    # Initialize metrics
    losses = AverageMeter()
    accuracy = AverageMeter()

    # No gradient computation for validation
    with torch.no_grad():
        # Iterate over batches
        for inputs, targets in tqdm(
            dataloader, desc="Validating", leave=False
        ):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            acc = correct / inputs.size(0)

            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            accuracy.update(acc, inputs.size(0))

    return losses.avg, accuracy.avg
