# This is the utility file for the training process.
# Put functionality that is not directly tied to the trainer but still used during training in this file.

"""
Utility functions for training.
"""

import os
import json
from typing import Dict, Any


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_training_summary(
    training_stats: Dict[str, Any],
    model_name: str,
    session_dir: str = "output",
) -> None:
    """
    Save training summary to a file.

    Args:
        training_stats: Dictionary containing training statistics
        model_name: Name of the model
        session_dir: Directory to save outputs
    """
    # Create summary file
    summary_path = f"{session_dir}/training_summary.txt"

    with open(summary_path, "a") as f:
        f.write(f"=== Training Summary for {model_name} ===\n\n")
        f.write(f"Training epochs: {training_stats['epochs']}\n")

        # Add early stopping information if applicable
        if (
            "early_stopped" in training_stats
            and training_stats["early_stopped"]
        ):
            f.write(
                f"Early stopped: Yes (after {training_stats['actual_epochs']} epochs)\n"
            )
        elif "actual_epochs" in training_stats:
            f.write(
                f"Actual epochs completed: {training_stats['actual_epochs']}\n"
            )

        f.write(f"Learning rate: {training_stats['learning_rate']}\n")
        f.write(f"Weight decay: {training_stats['weight_decay']}\n")
        f.write(
            f"Best validation accuracy: {training_stats['best_val_accuracy']:.4f}\n"
        )
        if "best_val_loss" in training_stats:
            f.write(
                f"Best validation loss: {training_stats['best_val_loss']:.4f}\n"
            )
        f.write(
            f"Training time: {training_stats['training_time'] / 60:.2f} minutes\n\n"
        )

        # Also save the last epoch metrics
        f.write(
            f"Final training loss: {training_stats['train_losses'][-1]:.4f}\n"
        )
        f.write(
            f"Final validation loss: {training_stats['val_losses'][-1]:.4f}\n"
        )
        f.write(
            f"Final training accuracy: {training_stats['train_accuracies'][-1]:.4f}\n"
        )
        f.write(
            f"Final validation accuracy: {training_stats['val_accuracies'][-1]:.4f}\n\n"
        )

    # Save detailed stats to JSON for plotting
    json_path = f"{session_dir}/stats_{model_name}.json"

    # Convert lists to standard Python lists for JSON serialization
    serializable_stats = {
        k: v
        if not isinstance(v, (list, tuple)) or len(v) == 0
        else list(map(float, v))
        for k, v in training_stats.items()
    }

    with open(json_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)
