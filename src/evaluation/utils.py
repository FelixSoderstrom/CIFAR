# This is the utility file for the evaluation process
# Place functions that does not directly generte plots or directly evaluates model performance but still is used within the evaluation step of the process

"""
Utility functions for model evaluation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def get_predictions(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        Tuple of (predictions, targets, probabilities)
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics given true and predicted labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    return metrics


def get_feature_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get feature embeddings from a model for samples in a dataloader.

    Args:
        model: The model to use
        dataloader: DataLoader for data
        device: Device to run on
        num_samples: Number of samples to use (None for all)

    Returns:
        Tuple of (embeddings, labels)
    """
    model.eval()
    embeddings = []
    labels = []
    count = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            batch_embeddings = model.get_embedding(inputs).cpu().numpy()

            embeddings.append(batch_embeddings)
            labels.append(targets.numpy())

            count += inputs.size(0)
            if num_samples is not None and count >= num_samples:
                break

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    if num_samples is not None:
        embeddings = embeddings[:num_samples]
        labels = labels[:num_samples]

    return embeddings, labels


def normalize_confusion_matrix(conf_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a confusion matrix by row to show the percentage of samples.

    Args:
        conf_matrix: Confusion matrix

    Returns:
        Normalized confusion matrix
    """
    row_sums = conf_matrix.sum(axis=1)
    norm_conf_matrix = conf_matrix / row_sums[:, np.newaxis]

    return norm_conf_matrix
