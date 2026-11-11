"""
EcoSort AI — Model Evaluation Module
Bharat AI-SoC Challenge 2026-2027

Provides comprehensive evaluation metrics for 4-class waste classification:
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- 4x4 Confusion Matrix
- Per-class performance reporting
- Scikit-learn classification report formatting
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.vision.dataset import IDX_TO_CLASS


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the classifier on a test or validation dataloader.

    Args:
        model: Trained MobileNetV2WasteClassifier model.
        dataloader: Test or validation DataLoader.
        device: Torch compute device (default: CPU).
        class_names: List of class labels in index order.

    Returns:
        Structured dictionary containing metrics, per-class stats, and confusion matrix.
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS.keys())]

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    # Core metrics
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix (4x4)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Per-class metrics
    per_class_prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        support_count = int(np.sum(y_true == idx))
        per_class_metrics[name] = {
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": support_count,
        }

    # Text report
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "classification_report": report_text,
        "total_samples": len(y_true),
    }

