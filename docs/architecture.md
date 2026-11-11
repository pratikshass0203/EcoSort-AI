# EcoSort AI — Detailed System Architecture

**Bharat AI-SoC Challenge 2026–2027**  
**Problem Statement 2:** Smart Recycling and Waste-Segregation Assistant

---

## 1. Architectural Philosophy: Decoupled Classification & Reasoning

EcoSort AI strictly separates **Computer Vision Pattern Recognition** from **Disposal Protocol Reasoning and Safety Verification**.

### Why Decoupling is Essential for Waste Segregation:
1. **Safety Guarantee for Hazardous Items:** An end-to-end generative model or multimodal LLM can hallucinate disposal instructions (e.g., advising a user to wash a battery or discard pesticide bottles in municipal drains). By restricting the vision model strictly to categorical probability assignment and routing decisions through a deterministic, offline rule-based safety engine, dangerous advice is eliminated.
2. **Computational Efficiency for Edge Devices:** MobileNetV2 uses ~3.5M parameters and ~300M FLOPs, allowing low power consumption and high responsiveness on mobile processors without draining battery or demanding dedicated cloud servers.
3. **Auditability & Regulatory Compliance:** Disposal rules can be directly updated, localized to municipal waste guidelines, and audited without retraining the deep neural network.

---

## 2. End-to-End Pipeline Dataflow

```
[ Input Image ]
     │ (Upload or Camera Capture)
     ▼
[ Preprocessing & Normalization ]
     │ - Resize to 224 x 224 (bicubic/bilinear interpolation)
     │ - Convert to RGB Float32 tensor [1, 3, 224, 224]
     │ - Normalize using ImageNet statistics:
     │     mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
     ▼
[ MobileNetV2 Vision Backbone ]
     │ - Standard Inverted Residual Blocks (Depthwise Separable Convolutions)
     │ - Global Average Pooling -> 1280-dimensional feature vector
     │ - Dropout (p = 0.2)
     │ - Fully-Connected Classification Head: Linear(1280, 4)
     ▼
[ Softmax Probability Computation ]
     │ P(y = c | x) = exp(z_c) / sum_k exp(z_k) for c in {0, 1, 2, 3}
     ▼
[ Decision Gate & Uncertainty Thresholding ]
     │ Check 1: Confidence Score S = max_c P(y = c | x) >= tau (0.70)
     │ Check 2: Margin M = P_top1 - P_top2 >= delta (0.15)
     ├───────────────────────────────┐
     │ PASS                          │ FAIL
     ▼                               ▼
[ Confident Classification ]     [ Uncertain Input Flag ]
     │                               │
     └───────────────┬───────────────┘
                     ▼
[ Deterministic Offline Disposal & Safety Engine ]
     │ - Look up verified category protocol from configs/disposal_rules.yaml
     │ - Apply mandatory hazardous material guardrails
     │ - Return bin assignment, cleaning instructions, and safety warnings
     ▼
[ Application Output (Streamlit / Android UI) ]
```

---

## 3. Uncertainty Mechanism Specification

Real-world waste sorting frequently encounters out-of-distribution (OOD) objects, motion-blurred captures, cluttered backgrounds, or composite items (e.g., an electronic toy with a plastic shell and internal battery).

To handle these reliably, EcoSort AI implements a dual-condition decision gate:

1. **Confidence Threshold ($\tau = 0.70$):**
   $$\max_{c \in \{0, 1, 2, 3\}} P(y = c \mid x) \ge 0.70$$
   If the maximum predicted class probability is below $0.70$, the prediction is rejected as ambiguous.

2. **Prediction Margin Threshold ($\delta = 0.15$):**
   $$P_{\text{top1}} - P_{\text{top2}} \ge 0.15$$
   If the top-two class probabilities are within $15\%$ of each other (indicating strong inter-class confusion, such as between recyclable plastic and electronic casings), the sample is flagged.

3. **Fallback Behavior:**
   - The UI surfaces an **"Uncertain - Manual Inspection Required"** banner.
   - The system displays safe default handling guidelines and prompts the user to re-frame the object under adequate lighting or inspect local municipal guidelines.

---

## 4. Software Subsystems & Module Interfaces

- **`src.vision`**: Encapsulates model definition (`MobileNetV2Classifier`), dataset pipelines, training routines with early stopping, evaluation metrics (accuracy, precision, recall, F1, confusion matrix), and inference pre/post-processing.
- **`src.knowledge`**: Offline taxonomy and safety guardrails. Implements deterministic lookup logic that maps predicted classes to bin standards, safety warnings, and recycling actions.
- **`src.llm`**: On-device conversational assistant wrapper. Grounds conversational responses in the vision classifier's output and knowledge base rules to prevent hallucinations during interactive user Q&A.
- **`src.edge`**: ExecuTorch export pipeline. Converts trained PyTorch model checkpoints into optimized `.pte` packages targeting the XNNPACK runtime for ARM64 Android execution.
- **`app`**: Interactive Streamlit laptop prototype providing real-time UI components, webcam input, and visual feedback.

