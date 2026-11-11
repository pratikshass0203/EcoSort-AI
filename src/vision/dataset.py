"""
EcoSort AI — Dataset Pipeline & Image Transformations
Bharat AI-SoC Challenge 2026-2027

Provides PyTorch Dataset implementations, augmentation pipelines, and
synthetic data generation utilities for 4-class waste segregation:
0: biodegradable, 1: recyclable, 2: electronic, 3: hazardous.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_TO_IDX = {
    "biodegradable": 0,
    "recyclable": 1,
    "electronic": 2,
    "hazardous": 3,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train", input_size: int = 224) -> transforms.Compose:
    """
    Get preprocessing and data augmentation transforms for the given split.

    Args:
        split: One of 'train', 'val', 'test', or 'inference'.
        input_size: Target square image dimension (default: 224).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),  # e.g., 256 for 224
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class EcoSortDataset(Dataset):
    """
    Custom PyTorch Dataset for EcoSort waste classification.
    Supports directory-based loading, RGB verification, and corrupted image handling.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing class subfolders.
            transform: Optional torchvision transform pipeline.
            class_to_idx: Mapping from class name to integer index.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.samples: List[Tuple[str, int]] = []

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """Scan class subdirectories and populate the sample list."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for entry in os.scandir(class_dir):
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_extensions:
                    self.samples.append((entry.path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load or parse image at '{image_path}': {e}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Construct DataLoaders for train, val, and test splits.

    Args:
        data_dir: Base directory containing 'train', 'val', and 'test' folders.
        batch_size: Batch size for loading.
        num_workers: Subprocesses for data loading (0 recommended on Windows).

    Returns:
        Dictionary mapping 'train', 'val', 'test' to their respective DataLoaders.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue

        dataset = EcoSortDataset(root_dir=split_dir, transform=get_transforms(split=split))
        if len(dataset) == 0:
            continue

        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return loaders


def generate_synthetic_dataset(
    output_root: str,
    samples_per_class: int = 10,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Generate synthetic colored sample images across the 4 waste categories.
    Used for offline testing, CI pipelines, and verification without external downloads.

    Args:
        output_root: Root directory to populate ('train', 'val', 'test' will be created).
        samples_per_class: Number of samples per class per split.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary reporting total generated sample count per split.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Distinct color palettes to simulate identifiable visual patterns
    class_palettes = {
        "biodegradable": (34, 139, 34),    # Forest Green
        "recyclable": (30, 144, 255),      # Dodger Blue
        "electronic": (218, 165, 32),      # Goldenrod / Yellow-Orange
        "hazardous": (220, 20, 60),        # Crimson Red
    }

    counts = {}
    splits = {
        "train": samples_per_class,
        "val": max(2, samples_per_class // 4),
        "test": max(2, samples_per_class // 4),
    }

    for split, count in splits.items():
        total_in_split = 0
        for class_name, base_rgb in class_palettes.items():
            target_dir = os.path.join(output_root, split, class_name)
            os.makedirs(target_dir, exist_ok=True)

            for i in range(count):
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                # Introduce subtle noise and variations
                r = int(np.clip(base_rgb[0] + random.randint(-20, 20), 0, 255))
                g = int(np.clip(base_rgb[1] + random.randint(-20, 20), 0, 255))
                b = int(np.clip(base_rgb[2] + random.randint(-20, 20), 0, 255))
                img_array[:, :] = [r, g, b]

                img = Image.fromarray(img_array)
                draw = ImageDraw.Draw(img)
                # Draw geometric shapes to provide feature diversity
                x0, y0 = random.randint(20, 80), random.randint(20, 80)
                x1, y1 = random.randint(120, 200), random.randint(120, 200)
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=3)

                img_path = os.path.join(target_dir, f"sample_{i:03d}.png")
                img.save(img_path)
                total_in_split += 1

        counts[split] = total_in_split

    return counts

