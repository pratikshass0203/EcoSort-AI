import io
import os
import json
from typing import Dict

import torch
import torchvision.transforms as T
from PIL import Image

# Simple mapping for demonstration purposes
IMAGENET_TO_WASTE = {
    "plastic bag": "Recyclable",
    "paper": "Biodegradable",
    "metal": "Recyclable",
    "battery": "Hazardous",
    "circuit board": "E-waste",
}

_MODEL = None
_MODEL_DEVICE = torch.device("cpu")

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        # Load a lightweight MobileNetV3 small pretrained on ImageNet (may require download).
        _MODEL = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', pretrained=True)
        _MODEL.eval()
        _MODEL.to(_MODEL_DEVICE)
    except Exception as e:
        print(f"Model load failed: {e}")
        _MODEL = None
    return _MODEL

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_bytes: bytes, language: str = "en") -> Dict:
    """Classify an image and return a structured result.
    If the model cannot be loaded, a deterministic fallback is used.
    """
    model = _load_model()
    if model is None:
        # deterministic fallback – always Unknown
        return {
            "category": "Unknown",
            "material": "Unknown",
            "confidence": 0.0,
            "disposal_instructions": get_disposal_instructions("Unknown"),
            "safety_precautions": get_safety_precautions("Unknown"),
            "environmental_impact": get_environmental_impact("Unknown"),
            "language": language,
        }
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_MODEL_DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_idx = torch.topk(probs, 5)
    confidence = top_prob[0].item()
    # Load ImageNet labels (offline fallback if network blocked)
    labels = []
    labels_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
    if os.path.isfile(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f]
    else:
        # No label file – cannot map classes, fallback to Unknown
        return {
            "category": "Unknown",
            "material": "Unknown",
            "confidence": round(confidence, 3),
            "disposal_instructions": get_disposal_instructions("Unknown"),
            "safety_precautions": get_safety_precautions("Unknown"),
            "environmental_impact": get_environmental_impact("Unknown"),
            "language": language,
        }
    class_names = [labels[idx] if idx < len(labels) else "unknown" for idx in top_idx.tolist()]
    selected_category = "Unknown"
    material = "Unknown"
    for name in class_names:
        lname = name.lower()
        for key, cat in IMAGENET_TO_WASTE.items():
            if key in lname:
                selected_category = cat
                material = key
                break
        if selected_category != "Unknown":
            break
    THRESHOLD = 0.6
    if confidence < THRESHOLD:
        selected_category = "Unknown"
        material = "Unknown"
    return {
        "category": selected_category,
        "material": material,
        "confidence": round(confidence, 3),
        "disposal_instructions": get_disposal_instructions(selected_category),
        "safety_precautions": get_safety_precautions(selected_category),
        "environmental_impact": get_environmental_impact(selected_category),
        "language": language,
    }

def get_disposal_instructions(category: str) -> str:
    mapping = {
        "Biodegradable": "Compost or organic waste bin.",
        "Recyclable": "Place in recycling bin.",
        "E-waste": "Take to e‑waste collection point.",
        "Hazardous": "Handle with protective gloves, take to hazardous waste facility.",
        "Unknown": "Manual review required.",
    }
    return mapping.get(category, "Manual review required.")

def get_safety_precautions(category: str) -> str:
    mapping = {
        "Hazardous": "Wear gloves and mask.",
        "E-waste": "Avoid direct skin contact.",
        "Unknown": "None",
    }
    return mapping.get(category, "None")

def get_environmental_impact(category: str) -> str:
    mapping = {
        "Biodegradable": "Low impact if composted.",
        "Recyclable": "Reduces resource extraction.",
        "E-waste": "Potential heavy metal leakage if not processed.",
        "Hazardous": "High impact if mishandled.",
        "Unknown": "Unknown",
    }
    return mapping.get(category, "Unknown")

# ----- Assistant -----

def assistant_chat(message: str, language: str = "en") -> str:
    lower = message.lower()
    if "dispose" in lower or "waste" in lower:
        return {
            "en": "Place the item in the appropriate bin as per its category.",
            "hi": "आइटम को उसके वर्ग के अनुसार उचित बिन में रखें।",
            "ta": "பொருளை அதன் வகை படி சரியான கொட்டகையில் வைக்கவும்.",
        }.get(language, "Place the item in the appropriate bin.")
    return {
        "en": "I can help with waste classification and disposal guidance.",
        "hi": "मैं कचरा वर्गीकरण और निपटान मार्गदर्शन में मदद करने के लिए यहाँ हूँ।",
        "ta": "நான் கழிவு வகைப்படுத்தல் மற்றும் அகற்றும் வழிகாட்டுதலுக்கு உதவ இங்கு உள்ளேன்.",
    }.get(language, "I can help with waste classification.")

# ----- Device Info -----

def get_device_info() -> dict:
    return {
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

