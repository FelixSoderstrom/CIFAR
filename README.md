# CIFAR-10 Image Classification

This project implements deep learning models for image classification on the CIFAR-10 dataset. It provides a complete pipeline for training, evaluating, and visualizing the performance of different CNN architectures.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project offers:

- Training of custom CNN architectures from scratch
- Transfer learning using pre-trained ResNet50
- Comprehensive evaluation metrics and visualizations
- Modular codebase for easy experimentation

## Project Structure

```
CIFAR/
│
├── main.py                   # Entry point for training and evaluation
├── README.md                 # This file
│
└── src/
    ├── data_processing/      # Data loading and augmentation
    │   ├── augment.py        # Data augmentation functions
    │   └── utils.py          # Data utilities
    │
    ├── networks/             # Model architectures
    │   ├── classic_network.py # Custom CNN architecture
    │   ├── transfer_network.py # Transfer learning with ResNet50
    │   └── utils.py          # Network utilities
    │
    ├── training/             # Training functionality
    │   ├── trainer.py        # Training loop implementation
    │   └── utils.py          # Training utilities
    │
    └── evaluation/           # Evaluation functionality
        ├── evaluate.py       # Model evaluation
        ├── utils.py          # Evaluation utilities
        └── visualize.py      # Visualization functions
```

## Usage

### Training a Model

To train a model from scratch:

```bash
python main.py --model classic --epochs 30 --batch_size 128 --lr 0.001 --gpu
```

To train using transfer learning with ResNet50:

```bash
python main.py --model transfer --epochs 20 --batch_size 64 --lr 0.0001 --gpu
```

### Command Line Arguments

- `--model`: Model architecture to use (`classic` or `transfer`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)
- `--seed`: Random seed (default: 42)
- `--gpu`: Use GPU if available (flag)
- `--evaluate_only`: Only run evaluation on a trained model (flag)

### Evaluation Only

To evaluate a trained model without retraining:

```bash
python main.py --model classic --evaluate_only --gpu
```

## Features

### Data Augmentation

The project implements several data augmentation techniques:
- Random cropping
- Random horizontal flips
- Random rotation
- Color jitter

### Model Architectures

1. **ClassicCNN**: A custom CNN architecture with:
   - 4 convolutional blocks with increasing filter sizes
   - Batch normalization
   - Max pooling
   - Dropout for regularization
   - Fully connected layers

2. **TransferResNet50**: A transfer learning approach using:
   - Pre-trained ResNet50 as feature extractor
   - Custom classification head for CIFAR-10

### Evaluation Metrics

- Accuracy (overall and per-class)
- Precision, recall, and F1 score
- Confusion matrix
- Feature embeddings visualization (t-SNE and PCA)
- Visualization of misclassified samples
- Training and validation curves

## Output

The results are saved in an `output/session_X` directory, where `X` is the session number. Each session directory contains:

- `checkpoints/`: Model weights for each epoch and the best model
- `plots/`: Visualization plots (confusion matrices, embeddings, etc.)
- `test_results_*.txt`: Detailed evaluation metrics
- `training_summary.txt`: Summary of the training process
- `stats_*.json`: Training statistics for plotting

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- pytorch-lightning

## License

[MIT License](LICENSE) 