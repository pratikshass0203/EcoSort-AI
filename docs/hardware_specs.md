# EcoSort AI — Target Hardware & Edge SoC Analysis

**Bharat AI-SoC Challenge 2026–2027**  
**Problem Statement 2:** Smart Recycling and Waste-Segregation Assistant

---

## 1. Target Smartphone & SoC Specifications

The deployment target for EcoSort AI is the **vivo V50e** smartphone.

| Component | Technical Specification |
| :--- | :--- |
| **Target Device** | vivo V50e |
| **System-on-Chip (SoC)** | MediaTek Dimensity 7300 (MT6878) |
| **Process Node** | TSMC 4nm FinFET |
| **CPU Architecture** | 64-bit ARMv9-A (ARM64) |
| **CPU Core Cluster** | **Performance:** 4× ARM Cortex-A78 @ 2.50 GHz<br>**Efficiency:** 4× ARM Cortex-A55 @ 2.00 GHz |
| **GPU** | ARM Mali-G615 MC2 |
| **AI Accelerator (NPU)** | MediaTek NPU 655 (Hardware AI Processing Unit) |
| **Memory** | 8 GB / 12 GB LPDDR4X / LPDDR5 |
| **Operating System** | Android 14 / 15 (Funtouch OS) |
| **Connectivity Requirement** | **Fully Offline Execution** (No cloud dependence) |

---

## 2. ExecuTorch Backend Analysis & Technical Rationale

### A. Evaluation of Hardware Acceleration Options
When targeting the MediaTek Dimensity 7300 for on-device PyTorch inference via ExecuTorch, two primary deployment pathways exist:

1. **MediaTek NPU 655 via NeuroPilot:**
   - *Technical Constraint:* MediaTek's NeuroPilot runtime requires proprietary vendor toolchains, closed-source HAL libraries, and OEM hardware signing keys. These are not bundled in open-source ExecuTorch distributions and cannot be deployed to commercial, un-rooted retail smartphones such as the vivo V50e.
   - *Engineering Decision:* We **do not assume** MediaTek NPU 655 acceleration is available for open deployment.

2. **Standardized XNNPACK on ARM64 CPU (Selected Target):**
   - *Technical Characteristics:* ExecuTorch provides first-class, fully open-source support for the **XNNPACK backend**, which generates highly optimized ARM NEON vector instructions for ARMv8/ARMv9 architectures.
   - *Multi-threading:* Utilizes the 4× Cortex-A78 high-performance cores.
   - *Precision:* Supports FP32 and INT8 post-training quantization.
   - *Portability:* Guaranteed to run offline across all modern ARM64 Android devices without proprietary vendor locks.

---

## 3. Theoretical Design Targets (Non-Measured Projections)

> [!NOTE]
> The performance metrics listed below represent **theoretical design targets** derived from the architectural complexity of MobileNetV2 (~300M FLOPs) on modern ARM Cortex-A78 microarchitectures.  
> **These are not measured results.** Actual experimental benchmarks will be measured and recorded on the physical target device during Phase 7.

| Metric | Target Design Objective (Theoretical) | Target Feasibility Analysis |
| :--- | :--- | :--- |
| **Vision Model Size** | < 15 MB (FP32) / < 4 MB (INT8) | Fits easily within mobile application package constraints. |
| **Inference Latency** | < 50 ms per frame | Sufficient for interactive camera feedback. |
| **RAM Utilization** | < 150 MB peak heap | Minimal footprint on 8 GB device. |
| **Offline Capability** | 100% On-Device | Zero network requests during classification or disposal lookup. |

---

## 4. Workstation Development Environment

The development machine used for data preparation, model training, and prototype frontend execution is:
- **Operating System:** Windows 11 (64-bit)
- **Execution Target:** CPU-prioritized execution (no external accelerators assumed)
- **Primary Toolchain:** Python 3.12, PyTorch 2.5.1 (CPU), Streamlit
- **Edge Compilation:** PyTorch Ahead-of-Time (AOT) export to ExecuTorch `.pte` binary

