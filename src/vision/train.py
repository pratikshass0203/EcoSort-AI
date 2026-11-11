"""
EcoSort AI — Model Training Pipeline
Bharat AI-SoC Challenge 2026-2027

Provides CPU-compatible training routines for MobileNetV2 waste classification:
- CrossEntropyLoss criterion
- AdamW optimizer with weight decay
- Learning rate scheduling (CosineAnnealingLR)
- Validation after every epoch
- Early stopping mechanism
- Best-checkpoint serialization to models/checkpoints/
- History logging for losses and accuracies
"""

import copy
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.vision.dataset import get_dataloaders
from src.vision.model import MobileNetV2WasteClassifier, build_model

DEFAULT_CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "checkpoints")
)
DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_config.yaml")
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Monitors validation loss and signals when training should terminate."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    epochs: int = 15,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
    patience: int = 3,
    checkpoint_dir: Optional[str] = None,
    checkpoint_name: str = "mobilenetv2_waste_best.pth",
    device_name: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Run full training and validation pipeline with early stopping.

    Args:
        model: MobileNetV2WasteClassifier module.
        dataloaders: Dict containing 'train' and optional 'val' DataLoaders.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularization factor.
        patience: Epochs without improvement before early stopping triggers.
        checkpoint_dir: Directory where the best weights will be saved.
        checkpoint_name: File name for the best checkpoint.
        device_name: Target compute device ('cpu').
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (best_model, training_history_dict).
    """
    set_seed(seed)
    device = torch.device(device_name)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=patience)

    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs_completed": 0,
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
        "checkpoint_path": best_checkpoint_path,
    }

    best_model_weights = copy.deepcopy(model.state_dict())
    has_val = "val" in dataloaders and len(dataloaders["val"]) > 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["epochs_completed"] = epoch

        if has_val:
            val_loss, val_acc = validate_one_epoch(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Check if this is the best epoch
            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                history["best_val_acc"] = val_acc
                history["best_epoch"] = epoch
                best_model_weights = copy.deepcopy(model.state_dict())

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "num_classes": getattr(model, "num_classes", 4),
                    },
                    best_checkpoint_path,
                )

            if early_stopper(val_loss):
                break
        else:
            # When no separate validation set is available, save latest weights as baseline
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "num_classes": getattr(model, "num_classes", 4),
                },
                best_checkpoint_path,
            )

    model.load_state_dict(best_model_weights)
    return model, history

