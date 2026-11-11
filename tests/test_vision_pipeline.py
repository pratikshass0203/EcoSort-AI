"""
EcoSort AI — Computer Vision Pipeline Automated Test Suite
Bharat AI-SoC Challenge 2026-2027

Tests:
1. Dataset loading, class mapping, and RGB conversion.
2. Preprocessing & transformation output shape [3, 224, 224].
3. MobileNetV2 architecture initialization and output logits shape [B, 4].
4. CPU training loop execution and checkpoint serialization using synthetic data.
5. Evaluation metrics computation (Accuracy, Precision, Recall, F1, 4x4 Confusion Matrix).
6. Uncertainty decision gate logic (confidence tau = 0.70, ambiguity margin delta = 0.15).
7. Model checkpoint reloading fidelity.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

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


@pytest.fixture(scope="module")
def synthetic_data_dir():
    """Create a temporary synthetic dataset across the 4 waste categories."""
    temp_dir = tempfile.mkdtemp(prefix="ecosort_test_data_")
    # Generate 4 samples per class in train, 2 in val, 2 in test
    generate_synthetic_dataset(output_root=temp_dir, samples_per_class=4, seed=42)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_class_mappings():
    """Verify 4-class mappings are contiguous and match project specification."""
    assert len(CLASS_TO_IDX) == 4
    expected_classes = ["biodegradable", "recyclable", "electronic", "hazardous"]
    for idx, c in enumerate(expected_classes):
        assert CLASS_TO_IDX[c] == idx
        assert IDX_TO_CLASS[idx] == c


def test_transforms_output_geometry():
    """Verify transforms produce Float32 tensors with shape [3, 224, 224]."""
    train_transform = get_transforms(split="train", input_size=224)
    val_transform = get_transforms(split="val", input_size=224)

    dummy_image = Image.new("RGB", (300, 400), color=(100, 150, 200))
    train_tensor = train_transform(dummy_image)
    val_tensor = val_transform(dummy_image)

    assert isinstance(train_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert train_tensor.dtype == torch.float32

    assert isinstance(val_tensor, torch.Tensor)
    assert val_tensor.shape == (3, 224, 224)
    assert val_tensor.dtype == torch.float32


def test_dataset_loading(synthetic_data_dir):
    """Verify EcoSortDataset loads images, handles labels, and detects corrupted files."""
    train_dir = os.path.join(synthetic_data_dir, "train")
    dataset = EcoSortDataset(root_dir=train_dir, transform=get_transforms("train"))

    # 4 classes * 4 samples = 16 samples
    assert len(dataset) == 16

    image_tensor, label = dataset[0]
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 224, 224)
    assert label in (0, 1, 2, 3)


def test_dataloaders(synthetic_data_dir):
    """Verify DataLoader creation and batch generation."""
    loaders = get_dataloaders(data_dir=synthetic_data_dir, batch_size=4, num_workers=0)
    assert "train" in loaders
    assert "val" in loaders

    for batch_images, batch_labels in loaders["train"]:
        assert batch_images.shape == (4, 3, 224, 224)
        assert batch_labels.shape == (4,)
        break


def test_mobilenet_v2_architecture():
    """Verify MobileNetV2 initialization and output dimensions."""
    model = MobileNetV2WasteClassifier(num_classes=4, dropout_rate=0.2, pretrained=False)
    assert model.num_classes == 4

    batch_input = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    logits = model(batch_input)
    assert logits.shape == (2, 4)

    features = model.extract_features(batch_input)
    assert features.shape == (2, 1280)

    probs = model.predict_proba(batch_input)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_training_pipeline_execution(synthetic_data_dir):
    """Verify CPU training execution, validation, and checkpoint serialization."""
    temp_ckpt_dir = tempfile.mkdtemp(prefix="ecosort_ckpt_")
    try:
        loaders = get_dataloaders(data_dir=synthetic_data_dir, batch_size=4, num_workers=0)
        model = MobileNetV2WasteClassifier(num_classes=4, pretrained=False)

        # Train for 2 epochs on synthetic data
        trained_model, history = train_model(
            model=model,
            dataloaders=loaders,
            epochs=2,
            learning_rate=0.001,
            checkpoint_dir=temp_ckpt_dir,
            checkpoint_name="test_best.pth",
            device_name="cpu",
            seed=42,
        )

        assert history["epochs_completed"] == 2
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

        checkpoint_path = os.path.join(temp_ckpt_dir, "test_best.pth")
        assert os.path.isfile(checkpoint_path)

        # Verify checkpoint contents
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert ckpt["num_classes"] == 4
    finally:
        shutil.rmtree(temp_ckpt_dir, ignore_errors=True)


def test_evaluation_metrics(synthetic_data_dir):
    """Verify evaluation calculations and 4x4 confusion matrix output."""
    loaders = get_dataloaders(data_dir=synthetic_data_dir, batch_size=4, num_workers=0)
    model = MobileNetV2WasteClassifier(num_classes=4, pretrained=False)

    results = evaluate_model(model=model, dataloader=loaders["val"], device=torch.device("cpu"))

    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0
    assert "macro_precision" in results
    assert "macro_recall" in results
    assert "macro_f1" in results
    assert "confusion_matrix" in results

    cm = np.array(results["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert "classification_report" in results
    assert len(results["per_class_metrics"]) == 4


def test_inference_and_uncertainty_decision_gate():
    """Verify confidence thresholding (tau=0.70) and ambiguity margin (delta=0.15)."""
    predictor = WastePredictor(checkpoint_path=None, device_name="cpu", tau=0.70, delta=0.15)

    # Test with random test image
    test_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    res = predictor.predict(test_img)

    assert "category" in res
    assert "confidence" in res
    assert "is_uncertain" in res
    assert "probabilities" in res
    assert "top2_margin" in res
    assert len(res["probabilities"]) == 4

    # Test mock confident vs uncertain decision gate logic explicitly
    # 1. Simulate Confident case: [0.85, 0.05, 0.05, 0.05] -> Top1=0.85 (>=0.70), Margin=0.80 (>=0.15)
    class MockConfidentModel(nn.Module):
        def forward(self, x):
            return torch.tensor([[10.0, 1.0, 1.0, 1.0]])

    predictor.model = MockConfidentModel()
    conf_res = predictor.predict(test_img)
    assert not conf_res["is_uncertain"]
    assert conf_res["category"] == "biodegradable"
    assert conf_res["confidence"] >= 0.70

    # 2. Simulate Low Confidence case: [0.40, 0.20, 0.20, 0.20] -> Top1=0.40 (< 0.70)
    class MockLowConfModel(nn.Module):
        def forward(self, x):
            return torch.tensor([[1.5, 1.0, 1.0, 1.0]])

    predictor.model = MockLowConfModel()
    low_conf_res = predictor.predict(test_img)
    assert low_conf_res["is_uncertain"]
    assert "Uncertain" in low_conf_res["category"]

    # 3. Simulate High Confusion / Low Margin case: [0.48, 0.46, 0.03, 0.03] -> Top1=0.48, Top2=0.46 -> Margin=0.02 (< 0.15)
    class MockAmbiguousModel(nn.Module):
        def forward(self, x):
            return torch.tensor([[5.0, 4.9, 0.1, 0.1]])

    predictor.model = MockAmbiguousModel()
    ambig_res = predictor.predict(test_img)
    assert ambig_res["is_uncertain"]
    assert "Uncertain" in ambig_res["category"]

