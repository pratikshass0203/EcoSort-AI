"""
EcoSort AI — Vision Inference & Uncertainty Decision Gate
Bharat AI-SoC Challenge 2026-2027

Implements single-image and batch inference for 4-class waste categorization:
- Model checkpoint loading
- Preprocessing and normalization (224x224 RGB)
- Softmax probability computation
- Confidence thresholding (tau = 0.70)
- Prediction ambiguity margin check (delta = 0.15)
- Structured classification result output (Strictly no disposal instructions)
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image

from src.vision.dataset import IDX_TO_CLASS, get_transforms
from src.vision.model import MobileNetV2WasteClassifier, build_model

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_config.yaml")
)


class WastePredictor:
    """
    Inference engine that executes MobileNetV2 forward pass and applies the
    uncertainty decision gate.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device_name: str = "cpu",
        tau: float = 0.70,
        delta: float = 0.15,
    ):
        """
        Args:
            checkpoint_path: Path to .pth checkpoint file.
            config_path: Path to model_config.yaml.
            device_name: Target compute device ('cpu').
            tau: Minimum top-1 probability threshold for confidence (default: 0.70).
            delta: Minimum probability difference between top-1 and top-2 (default: 0.15).
        """
        self.device = torch.device(device_name)
        self.tau = tau
        self.delta = delta
        self.config_path = config_path or DEFAULT_CONFIG_PATH

        self._load_config()
        self.model = build_model(config=self.config_path, pretrained=False)

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()
        self.transform = get_transforms(split="val", input_size=self.input_resolution[0])

    def _load_config(self) -> None:
        """Load decision gate parameters and resolution from config."""
        self.input_resolution = [224, 224]
        self.uncertain_label = "Uncertain - Manual Inspection Required"

        if os.path.isfile(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                gate = cfg.get("decision_gate", {})
                self.tau = gate.get("confidence_threshold_tau", self.tau)
                self.delta = gate.get("ambiguity_margin", self.delta)
                self.uncertain_label = gate.get("uncertain_label", self.uncertain_label)
                self.input_resolution = cfg.get("model", {}).get("input_resolution", [224, 224])
            except Exception:
                pass

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load state dict from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Invalid checkpoint format in: {checkpoint_path}")

    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Convert input to normalized Float32 tensor [1, 3, 224, 224].

        Args:
            image_input: File path (str), PIL.Image, or numpy array.

        Returns:
            Preprocessed tensor ready for forward pass.
        """
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            with Image.open(image_input) as img:
                pil_image = img.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run inference on a single image and evaluate the uncertainty gate.

        Args:
            image_input: File path, PIL.Image, or numpy array.

        Returns:
            Structured dictionary with predicted category, confidence, and uncertainty flag.
        """
        tensor = self.preprocess_image(image_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])

        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = float(top1_prob - top2_prob)

        # Uncertainty Evaluation:
        # Condition 1: Confidence below threshold tau (0.70)
        # Condition 2: Margin between top-1 and top-2 below delta (0.15)
        is_uncertain = False
        uncertainty_reasons: List[str] = []

        if top1_prob < self.tau:
            is_uncertain = True
            uncertainty_reasons.append(
                f"Confidence {top1_prob:.3f} is below threshold tau ({self.tau:.2f})"
            )

        if margin < self.delta:
            is_uncertain = True
            uncertainty_reasons.append(
                f"Top-2 class margin {margin:.3f} is below threshold delta ({self.delta:.2f})"
            )

        predicted_category = (
            self.uncertain_label if is_uncertain else IDX_TO_CLASS[top1_idx]
        )

        all_probabilities = {
            IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))
        }

        return {
            "category": predicted_category,
            "raw_class_idx": top1_idx,
            "raw_category": IDX_TO_CLASS[top1_idx],
            "confidence": top1_prob,
            "top2_margin": margin,
            "is_uncertain": is_uncertain,
            "uncertainty_reasons": uncertainty_reasons,
            "probabilities": all_probabilities,
            "threshold_tau": self.tau,
            "margin_delta": self.delta,
        }

