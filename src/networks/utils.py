# This is the utility file for the network directory
# Place functionality not directly tied to the network architecture in this file.

"""
Utility functions for neural network models.
"""

from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get a summary of the model architecture and parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing model summary information
    """
    summary = {
        "model_name": model.__class__.__name__,
        "total_parameters": count_parameters(model),
        "layers": [],
    }

    # Get information about each layer
    for name, module in model.named_children():
        layer_info = {
            "name": name,
            "type": module.__class__.__name__,
            "parameters": sum(
                p.numel() for p in module.parameters() if p.requires_grad
            ),
        }
        summary["layers"].append(layer_info)

    return summary


def save_model(model: nn.Module, path: str) -> None:
    """
    Save the model weights to a file.

    Args:
        model: PyTorch model
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str) -> nn.Module:
    """
    Load model weights from a file.

    Args:
        model: PyTorch model
        path: Path to the saved model weights

    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(path))
    return model
