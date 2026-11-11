"""
EcoSort AI — Computer Vision Module
Bharat AI-SoC Challenge 2026-2027
"""

from src.vision.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    EcoSortDataset,
    generate_synthetic_dataset,
    get_dataloaders,
    get_transforms,
)
from src.vision.evaluate import evaluate_model
from src.vision.inference import WastePredictor
from src.vision.model import MobileNetV2WasteClassifier, build_model
from src.vision.train import train_model

__all__ = [
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "EcoSortDataset",
    "generate_synthetic_dataset",
    "get_dataloaders",
    "get_transforms",
    "evaluate_model",
    "WastePredictor",
    "MobileNetV2WasteClassifier",
    "build_model",
    "train_model",
]

