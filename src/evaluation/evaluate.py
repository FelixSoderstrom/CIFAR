# This is the evaluation file that handles model evaluation.

"""
Evaluation functionality for models.
"""

from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.data_processing.utils import get_class_names


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        Dictionary containing evaluation results
    """
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Get class names
    class_names = get_class_names()

    # Initialize variables
    all_preds = []
    all_targets = []
    all_probs = []

    # No gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Store predictions and targets
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_targets).mean()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Get classification report
    class_report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )

    # Calculate per-class accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Create evaluation results dictionary
    results = {
        "accuracy": float(accuracy),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "per_class_accuracy": per_class_accuracy,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
    }

    # Print summary
    print(f"Test Accuracy: {accuracy:.4f}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {per_class_accuracy[i]:.4f}")

    return results


def get_misclassified_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Get misclassified samples from a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        num_samples: Number of misclassified samples to return

    Returns:
        Tuple of (images, true labels, predicted labels)
    """
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    misclassified_images = []
    misclassified_true_labels = []
    misclassified_pred_labels = []

    # No gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches
        for inputs, targets in dataloader:
            # If we already have enough samples, break
            if len(misclassified_images) >= num_samples:
                break

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Find misclassified samples
            misclassified_idx = (preds != targets).nonzero(as_tuple=True)[0]

            # Store misclassified samples
            for idx in misclassified_idx:
                if len(misclassified_images) >= num_samples:
                    break

                misclassified_images.append(inputs[idx].cpu())
                misclassified_true_labels.append(targets[idx].item())
                misclassified_pred_labels.append(preds[idx].item())

    # Convert to tensors/lists
    misclassified_images = torch.stack(misclassified_images)

    return (
        misclassified_images,
        misclassified_true_labels,
        misclassified_pred_labels,
    )


def calculate_per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10
) -> np.ndarray:
    """
    Calculate the accuracy for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes

    Returns:
        Array of per-class accuracies
    """
    per_class_acc = np.zeros(n_classes)

    for i in range(n_classes):
        idx = y_true == i
        if np.sum(idx) > 0:
            per_class_acc[i] = np.mean(y_true[idx] == y_pred[idx])

    return per_class_acc
