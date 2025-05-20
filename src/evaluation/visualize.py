# This is the visualization file
# In here we create plots that go into the output folder for each learning session

"""
Visualization functions for model evaluation.
"""

import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.data_processing.utils import get_class_names
from src.evaluation.utils import (
    get_feature_embeddings,
    normalize_confusion_matrix,
)
from src.evaluation.evaluate import get_misclassified_samples


def visualize_results(
    model: nn.Module,
    test_loader: DataLoader,
    test_results: Dict[str, Any],
    model_name: str,
    session_dir: str = "output",
) -> None:
    """
    Visualize the results of model evaluation.

    Args:
        model: The trained model
        test_loader: Test data loader
        test_results: Dictionary containing evaluation results
        model_name: Name of the model
        session_dir: Directory to save outputs
    """
    # Plot confusion matrix
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        title=f"Confusion Matrix - {model_name}",
        filename=f"{session_dir}/plots/confusion_matrix_{model_name}.png",
    )

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        normalize=True,
        title=f"Normalized Confusion Matrix - {model_name}",
        filename=f"{session_dir}/plots/norm_confusion_matrix_{model_name}.png",
    )

    # Plot per-class accuracy
    plot_per_class_accuracy(
        test_results["per_class_accuracy"],
        title=f"Per-Class Accuracy - {model_name}",
        filename=f"{session_dir}/plots/per_class_accuracy_{model_name}.png",
    )

    # Get and plot misclassified samples
    device = next(model.parameters()).device
    misclassified_images, true_labels, pred_labels = (
        get_misclassified_samples(model, test_loader, device, num_samples=10)
    )
    plot_misclassified_samples(
        misclassified_images,
        true_labels,
        pred_labels,
        filename=f"{session_dir}/plots/misclassified_{model_name}.png",
    )

    # Plot embeddings
    embeddings, labels = get_feature_embeddings(
        model, test_loader, device, num_samples=1000
    )
    plot_embeddings(
        embeddings,
        labels,
        title=f"Feature Embeddings (t-SNE) - {model_name}",
        filename=f"{session_dir}/plots/embeddings_tsne_{model_name}.png",
        method="tsne",
    )

    # Also plot with PCA
    plot_embeddings(
        embeddings,
        labels,
        title=f"Feature Embeddings (PCA) - {model_name}",
        filename=f"{session_dir}/plots/embeddings_pca_{model_name}.png",
        method="pca",
    )

    # Save test results to file
    save_results_to_file(
        test_results,
        model_name,
        f"{session_dir}/test_results_{model_name}.txt",
    )

    # If there's a training stats file, plot training curves
    stats_path = f"{session_dir}/stats_{model_name}.json"
    if os.path.exists(stats_path):
        import json

        with open(stats_path, "r") as f:
            training_stats = json.load(f)

        plot_training_curves(
            training_stats,
            title=f"Training Curves - {model_name}",
            filename=f"{session_dir}/plots/training_curves_{model_name}.png",
        )


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Plot a confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        filename: Filename to save the plot
    """
    if normalize:
        conf_matrix = normalize_confusion_matrix(conf_matrix)
        fmt = ".2f"
    else:
        fmt = "d"

    class_names = get_class_names()

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_per_class_accuracy(
    per_class_accuracy: np.ndarray,
    title: str = "Per-Class Accuracy",
    filename: str = "per_class_accuracy.png",
) -> None:
    """
    Plot per-class accuracies.

    Args:
        per_class_accuracy: Array of per-class accuracies
        title: Title for the plot
        filename: Filename to save the plot
    """
    class_names = get_class_names()

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, per_class_accuracy)
    plt.title(title, fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.xticks(rotation=45)

    # Add text labels above each bar
    for i, v in enumerate(per_class_accuracy):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_misclassified_samples(
    images: torch.Tensor,
    true_labels: List[int],
    pred_labels: List[int],
    filename: str = "misclassified_samples.png",
) -> None:
    """
    Plot misclassified samples.

    Args:
        images: Tensor of images
        true_labels: List of true labels
        pred_labels: List of predicted labels
        filename: Filename to save the plot
    """
    class_names = get_class_names()

    # De-normalize the images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)

    n_images = min(10, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(n_images):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].set_title(
            f"True: {class_names[true_labels[i]]}\n"
            f"Pred: {class_names[pred_labels[i]]}",
            fontsize=10,
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Feature Embeddings",
    filename: str = "embeddings.png",
    method: str = "tsne",
) -> None:
    """
    Plot high-dimensional embeddings in 2D using dimensionality reduction.

    Args:
        embeddings: Feature embeddings
        labels: Labels for each embedding
        title: Title for the plot
        filename: Filename to save the plot
        method: Dimensionality reduction method ('tsne' or 'pca')
    """
    class_names = get_class_names()

    # Apply dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    # Plot each class with a different color
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
        )

    plt.title(title, fontsize=16)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_results_to_file(
    test_results: Dict[str, Any],
    model_name: str,
    filename: str,
) -> None:
    """
    Save evaluation results to a text file.

    Args:
        test_results: Dictionary containing evaluation results
        model_name: Name of the model
        filename: Path to save the results
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    class_names = get_class_names()

    with open(filename, "w") as f:
        # Write header
        f.write(f"Evaluation Results for {model_name}\n")
        f.write("=" * 50 + "\n\n")

        # Write overall accuracy
        f.write(f"Overall Accuracy: {test_results['accuracy']:.4f}\n\n")

        # Write per-class accuracy
        f.write("Per-Class Accuracy:\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(
                f"{class_name}: {test_results['per_class_accuracy'][i]:.4f}\n"
            )
        f.write("\n")

        # Write classification report
        f.write("Classification Report:\n")
        f.write("-" * 30 + "\n")
        report = test_results["classification_report"]

        # Header row
        f.write(
            f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
        )

        # Per-class metrics
        for class_name in class_names:
            class_data = report[class_name]
            f.write(
                f"{class_name:<15} {class_data['precision']:.4f}{'':<6} {class_data['recall']:.4f}{'':<6} {class_data['f1-score']:.4f}{'':<6} {class_data['support']:<10}\n"
            )

        # Average metrics
        f.write("\n")
        for avg_type in ["micro avg", "macro avg", "weighted avg"]:
            if avg_type in report:
                avg_data = report[avg_type]
                f.write(
                    f"{avg_type:<15} {avg_data['precision']:.4f}{'':<6} {avg_data['recall']:.4f}{'':<6} {avg_data['f1-score']:.4f}{'':<6} {avg_data['support']:<10}\n"
                )

        f.write("\n\n")

        # Write timestamp
        from datetime import datetime

        f.write(
            f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


def plot_training_curves(
    training_stats: Dict[str, Any],
    title: str = "Training Curves",
    filename: str = "training_curves.png",
) -> None:
    """
    Plot training and validation curves.

    Args:
        training_stats: Dictionary with training statistics
        title: Title for the plot
        filename: Filename to save the plot
    """
    epochs = range(1, len(training_stats["train_losses"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(
        epochs, training_stats["train_losses"], "b-", label="Training Loss"
    )
    plt.plot(
        epochs, training_stats["val_losses"], "r-", label="Validation Loss"
    )
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        training_stats["train_accuracies"],
        "b-",
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        training_stats["val_accuracies"],
        "r-",
        label="Validation Accuracy",
    )
    plt.title("Training and Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filename)
    plt.close()
