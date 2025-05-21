#!/usr/bin/env python3
"""
Main entry point for training and evaluating models on CIFAR-10.
"""

import argparse
import os
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate_model
from src.evaluation.visualize import visualize_results
from src.networks.classic_network import ClassicCNN
from src.networks.transfer_network import ResNet20
from src.data_processing.utils import get_cifar10_dataloaders
from src.training.hparam_tuning import tune_hyperparameters


def get_next_session_id(model_type: str) -> int:
    """
    Find the next available session ID.

    Args:
        model_type: The type of model (classic or transfer)

    Returns:
        The next available session ID
    """
    # Set model directory based on model type
    if model_type == "classic":
        model_dir = "output/classic"
    else:  # transfer
        model_dir = "output/pretrained"

    # If the model directory doesn't exist yet, return 1
    if not os.path.exists(model_dir):
        return 1

    # Get all directories in model_dir/ that match the pattern "session_X"
    session_dirs = [
        d
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
        and d.startswith("session_")
    ]

    if not session_dirs:
        return 1

    # Extract session numbers
    session_numbers = [
        int(d.split("_")[1])
        for d in session_dirs
        if d.split("_")[1].isdigit()
    ]

    if not session_numbers:
        return 1

    # Return the maximum session number + 1
    return max(session_numbers) + 1


def setup_directories(session_id: int, model_type: str) -> str:
    """
    Create necessary directories for the project.

    Args:
        session_id: The session ID for this training run
        model_type: The type of model (classic or transfer)

    Returns:
        The session directory path
    """
    # Create main output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Create model type directory
    if model_type == "classic":
        model_dir = "output/classic"
    else:  # transfer
        model_dir = "output/pretrained"

    os.makedirs(model_dir, exist_ok=True)

    # Create session directory with model type prefix
    session_dir = f"{model_dir}/session_{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    # Create subdirectories within the session directory
    os.makedirs(f"{session_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{session_dir}/plots", exist_ok=True)

    # Create dataset directory
    os.makedirs("src/dataset", exist_ok=True)

    return session_dir


def main() -> None:
    """Main function to run the training and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on CIFAR-10"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["classic", "transfer"],
        default="classic",
        help="Model architecture to use",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay"
    )
    parser.add_argument(
        "--patience", type=int, default=4, help="Patience for early stopping"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU if available"
    )
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Only run evaluation"
    )

    # Hyperparameter tuning parameters
    parser.add_argument(
        "--hparam_tuning",
        action="store_true",
        help="Run hyperparameter tuning before training",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter tuning",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Get the next session ID
    session_id = get_next_session_id(args.model)

    # Setup directories and get session directory path
    session_dir = setup_directories(session_id, args.model)
    print(
        f"This is training session {session_id}. Output will be saved to {session_dir}"
    )

    # In main() function, after parsing args:
    if args.gpu and not torch.cuda.is_available():
        print(
            "WARNING: GPU requested but CUDA is not available. Using CPU instead."
        )
        print(
            "To use GPU, install PyTorch with CUDA support: https://pytorch.org/get-started/locally/"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    )
    print(f"Using device: {device}")

    # If using GPU, print additional information
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create model builders - functions that return a new model instance
    if args.model == "classic":
        model_builder = lambda: ClassicCNN()
        model_name = "classic_cnn"
    else:  # transfer
        model_builder = lambda: ResNet20()
        model_name = "resnet20_cifar"

    # Create initial model instance
    model = model_builder()

    # Updated paths with session directory
    model_path = f"{session_dir}/checkpoints/best_{model_name}.ckpt"

    # Get original batch size for data loading
    batch_size = args.batch_size

    # Run hyperparameter tuning if requested
    if args.hparam_tuning and not args.evaluate_only:
        print(
            f"Running hyperparameter tuning for {args.model} model with {args.n_trials} trials"
        )

        # Get dataloaders with a temporary batch size (will be tuned)
        # Use a moderate batch size for initial loaders
        train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=128)

        # Tune hyperparameters
        best_params = tune_hyperparameters(
            model_builder=model_builder,
            train_loader=train_loader,
            val_loader=val_loader,
            model_type=args.model,
            n_trials=args.n_trials,
            device=device,
            session_dir=session_dir,
        )

        # Update arguments with best hyperparameters
        if "lr" in best_params:
            args.lr = best_params["lr"]
        if "weight_decay" in best_params:
            args.weight_decay = best_params["weight_decay"]
        if "batch_size" in best_params:
            batch_size = best_params["batch_size"]

        print(
            f"Using tuned hyperparameters: lr={args.lr}, weight_decay={args.weight_decay}, batch_size={batch_size}"
        )

    # Get dataloaders with potentially updated batch size
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=batch_size
    )

    # Train or load the model
    if not args.evaluate_only:
        print(f"Training {args.model} model for {args.epochs} epochs")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            session_dir=session_dir,  # Pass the session directory
            patience=args.patience,  # Pass the patience parameter
        )
    else:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    print("Evaluating model on test set")
    test_results = evaluate_model(model, test_loader, device)

    # Visualize the results
    print("Generating visualizations")
    visualize_results(
        model, test_loader, test_results, model_name, session_dir
    )  # Pass the session directory

    print(f"Results and visualizations saved to {session_dir}")


if __name__ == "__main__":
    main()
