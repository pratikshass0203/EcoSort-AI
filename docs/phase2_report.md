# EcoSort AI — Phase 2: Computer Vision Model Report

**Bharat AI-SoC Challenge 2026–2027**  
**Problem Statement 2:** Smart Recycling and Waste-Segregation Assistant  
**Lead AI/ML Systems Architect Report**

---

## 1. Executive Summary

Phase 2 implemented the complete end-to-end Computer Vision pipeline for **EcoSort AI**, targeting on-device 4-class waste segregation:
- **`src/vision/dataset.py`**: Custom PyTorch `EcoSortDataset`, image loading with error handling, data augmentations (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter), evaluation transforms, and a reproducible synthetic test generator.
- **`src/vision/model.py`**: MobileNetV2 neural network adapted with a 4-class linear classification head (`Linear(1280, 4)`), feature extraction hook, Softmax probability output, and configuration loader.
- **`src/vision/train.py`**: CPU-optimized training engine with `AdamW`, `CrossEntropyLoss`, `CosineAnnealingLR` scheduler, epoch validation, early stopping, and best-checkpoint saving.
- **`src/vision/evaluate.py`**: Performance evaluation calculating Accuracy, Macro/Weighted Precision, Recall, F1-Score, and a $4 \times 4$ Confusion Matrix.
- **`src/vision/inference.py`**: Production inference engine enforcing the dual-condition uncertainty decision gate (confidence threshold $\tau = 0.70$, ambiguity margin $\delta = 0.15$). Strictly decoupled from disposal reasoning.
- **`tests/test_vision_pipeline.py`**: Comprehensive automated test suite with 8 unit tests covering all components.

---

## 2. Dataset Strategy & Class Mapping

Detailed in [**`docs/dataset_guide.md`**](file:///c:/Users/padma/OneDrive/Desktop/EcoSort-AI/docs/dataset_guide.md):

### 4-Category Taxonomy:
1. **`biodegradable` (0):** Food scraps, fruit peels, vegetable waste, leaves, coffee grounds.
2. **`recyclable` (1):** Clean plastic bottles, paper, cardboard, metal cans, glass containers.
3. **`electronic` (2):** Old smartphones, cables, chargers, circuit boards, consumer batteries.
4. **`hazardous` (3):** Chemical containers, fluorescent bulbs, medicine packs, paint cans.

### Public Dataset Licensing & Mapping:
- **TrashNet (MIT-like):** Maps glass, paper, cardboard, plastic, metal into `recyclable`.
- **Garbage Classification (Kaggle / CC0):** Maps biological to `biodegradable`, containers to `recyclable`, batteries to `electronic`.
- **TACO (MIT):** Maps fine-grained aerosol cans and chemical containers into `hazardous`.
- **Out-of-Distribution (OOD):** Textiles (`clothes`), composite footwear (`shoes`), and unidentifiable debris (`trash`) are formally excluded from training and reserved for testing uncertainty gate rejection.

> [!NOTE]
> **Zero External Downloads Policy:** Per competition engineering rules, no large external datasets have been downloaded yet without explicit user authorization. All unit tests use a deterministic synthetic image generator to ensure 100% offline reproducibility.

---

## 3. MobileNetV2 Architecture & Preprocessing Geometry

- **Input Dimensions:** $[B, 3, 224, 224]$ RGB Float32 tensor.
- **Normalization:** ImageNet mean $\mu = [0.485, 0.456, 0.406]$, std $\sigma = [0.229, 0.224, 0.225]$.
- **Backbone:** MobileNetV2 with inverted residual depthwise separable blocks.
- **Classification Head:**
  ```python
  nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=1280, out_features=4)
  )
  ```
- **Parameter Footprint:** ~2.23 million parameters (~9 MB in FP32), well within edge device RAM constraints.

---

## 4. Uncertainty Decision Gate Mathematics

To prevent hazardous misclassification, `WastePredictor` evaluates two sequential conditions:

1. **Confidence Threshold ($\tau = 0.70$):**
   $$\max_{c \in \{0, 1, 2, 3\}} P(y = c \mid x) \ge 0.70$$
2. **Ambiguity Margin ($\delta = 0.15$):**
   $$P_{\text{top1}} - P_{\text{top2}} \ge 0.15$$

### State Transitions:
- If both conditions pass $\rightarrow$ **Confident Classification** (returns predicted category).
- If either condition fails $\rightarrow$ **`Uncertain - Manual Inspection Required`** (triggers user confirmation).

---

## 5. Automated Test Results (`pytest tests/`)

Executed using the Python 3.13.15 `.venv`:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-8.3.4, pluggy-1.7.0 -- C:\Users\padma\OneDrive\Desktop\EcoSort-AI\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\padma\OneDrive\Desktop\EcoSort-AI
configfile: pyproject.toml
collecting ... collected 15 items

tests/test_environment.py::test_directory_structure PASSED               [  6%]
tests/test_environment.py::test_required_files_exist PASSED              [ 13%]
tests/test_environment.py::test_model_config_syntax_and_values PASSED    [ 20%]
tests/test_environment.py::test_disposal_rules_syntax_and_safety PASSED  [ 26%]
tests/test_environment.py::test_python_version_compatibility PASSED      [ 33%]
tests/test_environment.py::test_torch_and_torchvision_imports PASSED     [ 40%]
tests/test_environment.py::test_mobilenet_v2_instantiation_and_inference PASSED [ 46%]
tests/test_vision_pipeline.py::test_class_mappings PASSED                [ 53%]
tests/test_vision_pipeline.py::test_transforms_output_geometry PASSED    [ 60%]
tests/test_vision_pipeline.py::test_dataset_loading PASSED               [ 66%]
tests/test_vision_pipeline.py::test_dataloaders PASSED                   [ 73%]
tests/test_vision_pipeline.py::test_mobilenet_v2_architecture PASSED     [ 80%]
tests/test_vision_pipeline.py::test_training_pipeline_execution PASSED   [ 86%]
tests/test_vision_pipeline.py::test_evaluation_metrics PASSED            [ 93%]
tests/test_vision_pipeline.py::test_inference_and_uncertainty_decision_gate PASSED [100%]

============================= 15 passed in 58.78s =============================
```

**Overall Test Suite Result:** **15 passed, 0 failed (100% pass rate).**

---

## 6. Scientific Integrity & Experimental Transparency

> [!IMPORTANT]
> **Non-Fabrication Statement:**  
> The training execution tested in Phase 2 ran on **synthetic verification data** to validate pipeline mechanics (loss reduction, backward pass, AdamW optimizer steps, early stopping, and checkpoint saving).  
> **These synthetic test results must not be represented as real-world classification accuracy.**  
> Real-world model training and benchmarked classification accuracy will be measured on actual curated datasets during subsequent training sessions.

---

## 7. Git Working Tree Status

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
	modified:   .gitignore
	modified:   README.md
	modified:   configs/model_config.yaml

Untracked files:
	.github/
	app/
	configs/disposal_rules.yaml
	data/
	docs/
	models/
	pyproject.toml
	requirements.txt
	src/
	tests/
```

- **Safety Checks:** Zero commits or pushes performed. `LICENSE`, Git history, and `.gitkeep` files preserved untouched.

