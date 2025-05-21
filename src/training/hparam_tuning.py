"""
Hyperparameter tuning functionality using Optuna.
"""

import os
import time
from typing import Dict, Any, Callable, Optional

import torch
import optuna
from torch.utils.data import DataLoader
import torch.nn as nn

from src.training.trainer import train_epoch, validate_epoch


def define_model_hyperparameter_space(
    trial: optuna.Trial, model_type: str
) -> Dict[str, Any]:
    """
    Define the hyperparameter search space based on model type.

    Args:
        trial: Optuna trial object
        model_type: Type of model (classic or transfer)

    Returns:
        Dictionary of hyperparameters
    """
    # Common hyperparameters for both model types
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-3, log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256]
        ),
    }

    # Model-specific hyperparameters
    if model_type == "classic":
        # For classic CNN, we might want to tune dropout rate
        params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    elif model_type == "transfer":
        # For transfer learning, we might want different learning rates for different layers
        params["fc_lr_multiplier"] = trial.suggest_float(
            "fc_lr_multiplier", 1.0, 10.0
        )

    return params


def objective(
    trial: optuna.Trial,
    model_builder: Callable[..., nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str,
    device: torch.device,
    num_epochs_per_trial: int = 5,
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        model_builder: Function that builds the model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_type: Type of model (classic or transfer)
        device: Device to train on
        num_epochs_per_trial: Number of epochs to train for each trial

    Returns:
        Validation accuracy
    """
    # Get hyperparameters for this trial
    params = define_model_hyperparameter_space(trial, model_type)

    # Create model (potentially with trial-specific hyperparameters)
    model = model_builder()
    if model_type == "classic" and "dropout" in params:
        # Set dropout rate if the model supports it
        if hasattr(model, "dropout"):
            model.dropout.p = params["dropout"]

    model = model.to(device)

    # Create optimizer with trial hyperparameters
    if model_type == "transfer" and "fc_lr_multiplier" in params:
        # Different learning rates for different parts of the model
        # Handle both nested structure (model.resnet.fc) and direct structure (model.fc)
        if hasattr(model, "resnet"):  # Original TransferResNet50 structure
            fc_params = list(model.resnet.fc.parameters())
        else:  # New ResNet20 structure
            fc_params = list(model.fc.parameters())

        base_params = list(
            filter(
                lambda p: p.requires_grad,
                set(model.parameters()) - set(fc_params),
            )
        )

        optimizer = torch.optim.Adam(
            [
                {"params": base_params},
                {
                    "params": fc_params,
                    "lr": params["lr"] * params["fc_lr_multiplier"],
                },
            ],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    best_val_acc = 0.0

    for epoch in range(num_epochs_per_trial):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Report intermediate results
        trial.report(val_acc, epoch)

        # Update best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


def tune_hyperparameters(
    model_builder: Callable[..., nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str,
    n_trials: int,
    device: torch.device,
    study_name: Optional[str] = None,
    session_dir: str = "output",
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna.

    Args:
        model_builder: Function that builds the model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_type: Type of model (classic or transfer)
        n_trials: Number of trials to run
        device: Device to train on
        study_name: Name for the Optuna study
        session_dir: Directory to save outputs

    Returns:
        Dictionary with best hyperparameters
    """
    if study_name is None:
        study_name = f"{model_type}_optimization"

    # Create the directory for Optuna studies if it doesn't exist
    os.makedirs(f"{session_dir}/optuna", exist_ok=True)
    storage_name = f"sqlite:///{session_dir}/optuna/{study_name}.db"

    # Create or load a study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    # Define lambda function for the objective to include fixed parameters
    objective_func = lambda trial: objective(
        trial, model_builder, train_loader, val_loader, model_type, device
    )

    # Run optimization
    print(
        f"Starting hyperparameter optimization for {model_type} model with {n_trials} trials"
    )
    start_time = time.time()
    study.optimize(objective_func, n_trials=n_trials)
    tuning_time = time.time() - start_time

    # Get best params and trial info
    best_params = study.best_params
    best_value = study.best_value

    # Save the results
    with open(f"{session_dir}/optuna/{study_name}_results.txt", "w") as f:
        f.write(f"=== Hyperparameter Tuning Results for {model_type} ===\n\n")
        f.write(f"Number of trials: {n_trials}\n")
        f.write(f"Best validation accuracy: {best_value:.4f}\n")
        f.write(f"Best hyperparameters:\n")
        for param_name, param_value in best_params.items():
            f.write(f"  {param_name}: {param_value}\n")
        f.write(f"\nTuning time: {tuning_time / 60:.2f} minutes\n")

    print(f"Best hyperparameters for {model_type} model:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")

    return best_params
