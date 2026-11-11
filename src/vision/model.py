"""
EcoSort AI — MobileNetV2 Architecture Implementation
Bharat AI-SoC Challenge 2026-2027

Implements a MobileNetV2-based 4-class waste classifier:
- Inverted residual feature extractor
- Global average pooling
- Custom dropout + linear classification head (1280 -> 4)
- Compatible with configs/model_config.yaml
"""

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
import yaml

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_config.yaml")
)


class MobileNetV2WasteClassifier(nn.Module):
    """
    MobileNetV2 neural network adapted for 4-class waste categorization.
    Optimized for edge compilation and offline ARM64 inference.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        pretrained: bool = False,
    ):
        """
        Args:
            num_classes: Number of target categories (default: 4).
            dropout_rate: Dropout probability in the classifier head.
            pretrained: Whether to load ImageNet pre-trained weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        # In MobileNetV2, backbone.classifier is a Sequential(Dropout, Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace classification head with our custom 4-class output
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=in_features, out_features=self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass.

        Args:
            x: Input tensor with shape [batch_size, 3, 224, 224].

        Returns:
            Logits tensor with shape [batch_size, num_classes].
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 1280-dimensional feature embedding before classification.

        Args:
            x: Input tensor with shape [batch_size, 3, 224, 224].

        Returns:
            Features tensor with shape [batch_size, 1280].
        """
        features = self.backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(features, 1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized class probabilities via Softmax.

        Args:
            x: Input tensor with shape [batch_size, 3, 224, 224].

        Returns:
            Probability tensor with shape [batch_size, num_classes].
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(
    config: Optional[Union[str, Dict[str, Any]]] = None,
    pretrained: Optional[bool] = None,
) -> MobileNetV2WasteClassifier:
    """
    Factory function to build and configure the classifier using configuration parameters.

    Args:
        config: Path to model_config.yaml or a dictionary of configuration settings.
        pretrained: Explicit override for pretrained weights (True/False). If None, defaults to config.

    Returns:
        Instantiated MobileNetV2WasteClassifier module.
    """
    num_classes = 4
    dropout_rate = 0.2
    use_pretrained = False

    if config is not None:
        if isinstance(config, str) and os.path.isfile(config):
            with open(config, "r", encoding="utf-8") as f:
                loaded_cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            loaded_cfg = config
        else:
            loaded_cfg = {}

        model_cfg = loaded_cfg.get("model", {})
        num_classes = model_cfg.get("num_classes", num_classes)
        dropout_rate = model_cfg.get("dropout_rate", dropout_rate)
        if pretrained is not None:
            use_pretrained = pretrained
        else:
            use_pretrained = model_cfg.get("pretrained", False)
    elif os.path.isfile(DEFAULT_CONFIG_PATH):
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded_cfg = yaml.safe_load(f)
            model_cfg = loaded_cfg.get("model", {})
            num_classes = model_cfg.get("num_classes", num_classes)
            dropout_rate = model_cfg.get("dropout_rate", dropout_rate)
            if pretrained is not None:
                use_pretrained = pretrained
            else:
                use_pretrained = model_cfg.get("pretrained", False)
        except Exception:
            pass
    elif pretrained is not None:
        use_pretrained = pretrained

    return MobileNetV2WasteClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=use_pretrained,
    )
