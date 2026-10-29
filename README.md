# Unified Noise-Robust Traffic Sign Classification Framework

A modular, production-ready deep learning system designed for noise-resilient traffic sign classification across multiple benchmarks (**GTSRB**, **BTSD**, and **CTSD**) with spatial rectification, dual-stream attention, and test-time adaptation.

---

## 🌟 Key Architecture & Innovations

1. **Spatial Transformer Network (STN)**: Dynamically aligns and rectifies rotations, scale variations, and perspective distortions.
2. **Dual-Stream Adaptive Attention Gate**: Dynamically weights chromatic representations (**HSV**) against structural edge maps (**Sobel gradient magnitude**).
3. **Multi-Task ResNet-18 Backbone**: Shared feature extraction with task-specific projection heads and an intermediate **early-exit classifier** for low-latency inference.
4. **Test-Time Adaptation (TENT)**: Online entropy minimization of BatchNorm affine parameters during severe corruptions without test-leakage.
5. **Real-world Robustness Suite**: Standardized benchmark across 14 environmental corruptions (weather, blurs, digital noise) over 5 severity levels.

---

## 📁 Directory Structure

```text
├── config.py                 # Global configurations, hyperparameters, and device setup
├── train.py                  # Multi-task / individual training pipeline with AMP & telemetry
├── evaluate.py               # Robustness benchmarking across 14 corruptions and baseline
├── data/                     # Localized dataset directory
│   ├── GTSRB/                # Train/, Test/, Train.csv, Test.csv (43 classes)
│   ├── BTSD/                 # BelgiumTSC_Training & BelgiumTSC_Testing (62 classes)
│   └── CTSD/                 # images/ & annotations.csv (58 classes)
└── src/
    ├── __init__.py           # Package interface
    ├── models.py             # STN, DualStreamAttentionGate, MultiTaskRobustSignNet
    ├── datasets.py           # Real dataset parsers & UnifiedTrafficSignDataset (RGB, HSV, Sobel)
    ├── corruptions.py        # 14 environmental corruptions & convolution filter kernels
    ├── adaptation.py         # TENT test-time adaptation and weight snapshot/restore
    └── utils.py              # Reproducibility seed, metrics calculation & heartbeat tracker
```

---

## 🚀 Quickstart

### 1. Training

- **Train on all datasets (Multi-Task)**:
  ```powershell
  .venv\Scripts\python.exe train.py --epochs 10 --batch-size 64
  ```

- **Train on an individual dataset**:
  ```powershell
  .venv\Scripts\python.exe train.py --datasets GTSRB --epochs 10
  .venv\Scripts\python.exe train.py --datasets BTSD --epochs 10
  .venv\Scripts\python.exe train.py --datasets CTSD --epochs 10
  ```

### 2. Robustness Evaluation & Benchmarking

- **Benchmark all datasets**:
  ```powershell
  .venv\Scripts\python.exe evaluate.py --checkpoint-path checkpoint_multitask.pth --output-csv novel_framework_corruption_results.csv
  ```

- **Benchmark an individual dataset**:
  ```powershell
  .venv\Scripts\python.exe evaluate.py --datasets GTSRB --checkpoint-path checkpoint_multitask.pth --output-csv gtsrb_results.csv
  .venv\Scripts\python.exe evaluate.py --datasets BTSD --checkpoint-path checkpoint_multitask.pth --output-csv btsd_results.csv
  .venv\Scripts\python.exe evaluate.py --datasets CTSD --checkpoint-path checkpoint_multitask.pth --output-csv ctsd_results.csv
  ```

---

## 📊 Evaluation Outputs

The evaluation engine computes and logs:
- **Clean Baseline Accuracy & F1-Scores** (Macro and Weighted)
- **Mean Corrupted Accuracy** across 14 environmental distortions
- **Per-frame Inference Latency** (ms/frame) with CUDA synchronization
- Results exported cleanly to CSV for comparative analysis

