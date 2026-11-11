# EcoSort AI: Smart Recycling and Waste-Segregation Assistant

**Bharat AI-SoC Challenge 2026–2027**  
**Problem Statement 2:** Smart Recycling and Waste-Segregation Assistant  
**Target Hardware:** Windows Laptop (Development & Prototyping) | vivo V50e / MediaTek Dimensity 7300 (Target Edge SoC)

---

## 1. Project Overview

**EcoSort AI** is an intelligent on-device waste-classification and disposal assistant designed to address the critical challenge of municipal solid waste segregation at source.

The solution empowers users to identify waste items using camera capture or uploaded imagery, classifies them into four primary waste categories, and delivers actionable, offline disposal protocols while ensuring strict safety guardrails for hazardous materials.

### Core Capabilities:
- **Vision-Based Waste Classification:** Powered by a fine-tuned MobileNetV2 architecture optimized for ARM64 edge inference.
- **Uncertainty-Aware Decision Gate:** Employs a strict confidence threshold ($\tau = 0.70$) and prediction margin checks to prevent dangerous misclassification.
- **Deterministic Offline Disposal Engine:** Guarantees verified, hallucination-free disposal instructions and safety alerts without requiring internet connectivity.
- **Edge Deployment Architecture:** Model compilation target via ExecuTorch utilizing XNNPACK on ARM64 CPU (4× Cortex-A78 + 4× Cortex-A55) for the vivo V50e.
- **Interactive Prototyping:** Streamlit-based web interface for laptop evaluation and camera input.

---

## 2. Waste Classification Categories

EcoSort AI categorizes solid waste into four primary categories in accordance with national municipal waste management rules:

| Category | Bin Standard | Target Items | Handling Protocol |
| :--- | :--- | :--- | :--- |
| **Biodegradable** | **Green** | Food scraps, fruit peels, vegetable waste, garden leaves, eggshells | Segregate at source; direct to compost or anaerobic digestion. |
| **Recyclable** | **Blue** | Plastic bottles (PET/HDPE), paper, cardboard, aluminum cans, glass | Rinse, flatten, keep dry; route to authorized recyclers. |
| **Electronic Waste** | **Yellow / Designated** | Old smartphones, cables, chargers, circuit boards, batteries | Never bin in landfill; tape battery terminals; deposit at e-waste centers. |
| **Hazardous Waste** | **Red / Black** | Chemical bottles, CFLs, mercury items, expired medicine, paint | **Strict Warning:** Keep sealed; wear gloves; drop off at hazardous waste depots. |

---

## 3. System Architecture

EcoSort AI enforces a strict decoupling between **visual classification** and **disposal reasoning**:

```
[ Camera / Upload Image ]
           │
           ▼
[ Aspect-Preserved Preprocessing (224x224 RGB, ImageNet Norm) ]
           │
           ▼
[ MobileNetV2 Backbone + Custom 4-Class Classification Head ]
           │
           ▼
[ Softmax Probabilities [p0, p1, p2, p3] ]
           │
           ▼
     ┌───────────┐
     │ Max(P) >= │
     │ tau (0.70)│
     └─────┬─────┘
      Yes  │  No
     ┌─────┴────────┐
     ▼              ▼
[ Confident ]   [ Flag as Uncertain ]
     │              │
     └──────┬───────┘
            ▼
[ Deterministic Offline Disposal & Safety Engine ]
            │
            ▼
[ Actionable Instructions + Safety Guardrails + Bin Guidelines ]
```

---

## 4. Hardware & Edge SoC Deployment Target

- **Development Platform:** Windows 10/11 Laptop (Intel/AMD x86_64 CPU-prioritized).
- **Edge Device Target:** vivo V50e
  - **SoC:** MediaTek Dimensity 7300 (4nm FinFET)
  - **CPU Core Layout:** 4× ARM Cortex-A78 @ 2.50 GHz (Performance) + 4× ARM Cortex-A55 @ 2.00 GHz (Efficiency)
  - **Target Runtime:** ExecuTorch runtime with **XNNPACK delegate** (ARM64 NEON vectorization).
  - *Note on NPU:* MediaTek NPU 655 is present on the SoC, but open-source ExecuTorch targets the standardized, universally deployable **XNNPACK on ARM64 CPU** backend for offline reliability without proprietary vendor driver locks.

---

## 5. Repository Layout

```
EcoSort-AI/
├── .github/workflows/ci.yml       # CI workflow for linting & tests
├── configs/
│   ├── model_config.yaml          # Model parameters & confidence threshold
│   └── disposal_rules.yaml        # Offline category rules & safety alerts
├── data/
│   ├── raw/                       # Raw waste datasets (git-ignored)
│   ├── processed/                 # 4-class splits (train/val/test)
│   └── samples/                   # Curated sample validation images
├── docs/
│   ├── architecture.md            # Detailed system design & pipeline math
│   ├── hardware_specs.md          # vivo V50e / Dimensity 7300 analysis
│   └── phase1_report.md           # Phase 1 verification report
├── models/
│   ├── checkpoints/               # Trained PyTorch weights (.pt, .pth)
│   └── exported/                  # ExecuTorch edge binaries (.pte)
├── src/
│   ├── vision/                    # MobileNetV2 training & inference
│   ├── knowledge/                 # Offline disposal & safety engine
│   ├── llm/                       # On-device conversational assistant
│   ├── edge/                      # ExecuTorch export & benchmarking
│   └── utils/                     # Logging & measurement utilities
├── app/
│   ├── streamlit_app.py           # Streamlit frontend prototype
│   └── components.py              # UI widgets & confidence gauges
├── tests/
│   └── test_environment.py        # Environment & configuration test
├── requirements.txt               # Pinned dependencies (Python 3.12, CPU)
├── pyproject.toml                 # Package & test metadata
└── LICENSE                        # MIT License
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.12 (x86_64) installed on Windows
- Git

### Environment Setup
```powershell
# 1. Clone the repository
git clone https://github.com/pratikshass0203/EcoSort-AI.git
cd EcoSort-AI

# 2. Create Python 3.12 virtual environment
py -3.12 -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Run Phase 1 verification tests
pytest tests/test_environment.py
```

---

## 7. Challenge Roadmap

- [x] **Phase 1: Project Foundation & Architecture** (Structure, Configs, Documentation, Environment Plan)
- [ ] **Phase 2: Computer Vision Model** (MobileNetV2 dataset preparation, training, evaluation)
- [ ] **Phase 3: Application Frontend** (Streamlit interactive interface with camera capture)
- [ ] **Phase 4: Disposal Knowledge Base** (Offline structured database & safety engine)
- [ ] **Phase 5: Llama Integration** (On-device conversational guidance)
- [ ] **Phase 6: ExecuTorch & Android** (Model export to `.pte` & ARM64 Android verification)
- [ ] **Phase 7: Performance Testing** (Accuracy, latency, and memory benchmarking)
- [ ] **Phase 8: Final Submission** (Submission documentation, video demo, and presentation)

---

## 8. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
