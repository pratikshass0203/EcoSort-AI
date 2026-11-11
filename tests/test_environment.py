"""
EcoSort AI — Phase 1 Environment & Configuration Verification Tests
Bharat AI-SoC Challenge 2026-2027
"""

import os
import sys
import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_directory_structure():
    """Verify all required project directories exist."""
    required_dirs = [
        "configs",
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        os.path.join("data", "samples"),
        "docs",
        os.path.join("models", "checkpoints"),
        os.path.join("models", "exported"),
        os.path.join("src", "vision"),
        os.path.join("src", "knowledge"),
        os.path.join("src", "llm"),
        os.path.join("src", "edge"),
        os.path.join("src", "utils"),
        "app",
        "tests",
    ]
    for d in required_dirs:
        path = os.path.join(PROJECT_ROOT, d)
        assert os.path.isdir(path), f"Required directory missing: {d}"


def test_required_files_exist():
    """Verify essential project files exist."""
    required_files = [
        "LICENSE",
        "README.md",
        ".gitignore",
        "requirements.txt",
        "pyproject.toml",
        os.path.join("configs", "model_config.yaml"),
        os.path.join("configs", "disposal_rules.yaml"),
        os.path.join("docs", "architecture.md"),
        os.path.join("docs", "hardware_specs.md"),
    ]
    for f in required_files:
        path = os.path.join(PROJECT_ROOT, f)
        assert os.path.isfile(path), f"Required file missing: {f}"


def test_model_config_syntax_and_values():
    """Verify model_config.yaml loads correctly with required schema."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "model" in config, "model block missing in model_config.yaml"
    assert config["model"]["name"] == "mobilenet_v2"
    assert config["model"]["num_classes"] == 4
    assert config["model"]["input_resolution"] == [224, 224]

    assert "classes" in config, "classes block missing in model_config.yaml"
    expected_classes = {0: "biodegradable", 1: "recyclable", 2: "electronic", 3: "hazardous"}
    assert config["classes"] == expected_classes

    assert "decision_gate" in config, "decision_gate block missing in model_config.yaml"
    assert config["decision_gate"]["confidence_threshold_tau"] == 0.70


def test_disposal_rules_syntax_and_safety():
    """Verify disposal_rules.yaml contains all 4 classes and safety alert for hazardous waste."""
    rules_path = os.path.join(PROJECT_ROOT, "configs", "disposal_rules.yaml")
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    assert "categories" in rules, "categories block missing in disposal_rules.yaml"
    expected_categories = ["biodegradable", "recyclable", "electronic", "hazardous"]
    for cat in expected_categories:
        assert cat in rules["categories"], f"Category '{cat}' missing in disposal rules"
        assert "bin_color" in rules["categories"][cat]
        assert "handling_instructions" in rules["categories"][cat]

    # Verify critical safety alert exists for hazardous category
    hazardous = rules["categories"]["hazardous"]
    assert "safety_alert" in hazardous, "Mandatory safety alert missing from hazardous category"
    assert "MANDATORY SAFETY WARNING" in hazardous["safety_alert"]


def test_python_version_compatibility():
    """Verify runtime Python version is Python 3.13.x."""
    version_info = sys.version_info
    assert version_info.major == 3 and version_info.minor == 13, (
        f"Expected Python 3.13.x, but running on Python {version_info.major}.{version_info.minor}.{version_info.micro}."
    )


def test_torch_and_torchvision_imports():
    """Verify torch and torchvision import cleanly and detect CPU device."""
    import torch
    import torchvision

    assert torch.__version__.startswith("2.6.0"), f"Unexpected torch version: {torch.__version__}"
    assert torchvision.__version__.startswith("0.21.0"), f"Unexpected torchvision version: {torchvision.__version__}"
    device = torch.device("cpu")
    assert device.type == "cpu", "Device must be CPU"


def test_mobilenet_v2_instantiation_and_inference():
    """Verify MobileNetV2 can be instantiated with a 4-class head and run CPU dummy inference."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    # Instantiate MobileNetV2 architecture with custom 4-class linear head (weights=None for offline verification)
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 4)
    model.eval()

    # Create dummy float32 tensor [batch_size=1, channels=3, height=224, width=224]
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 4), f"Expected output shape (1, 4), but got {output.shape}"
    probabilities = torch.softmax(output, dim=1)
    assert probabilities.shape == (1, 4)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0)), "Probabilities must sum to 1.0"
