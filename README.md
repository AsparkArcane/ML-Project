# SG-NeRF-CTR: Spatially-Gated Neural Radiance Field Invariance with Continuous-Topology Routing

A modular, production-ready deep learning system designed for noise-resilient traffic sign classification across multiple benchmarks (**GTSRB**, **BTSD**, and **CTSD**) implementing the novel **SG-NeRF-CTR** architecture.

---

## 🌟 Key Architecture & Innovations

1. **Implicit Neural Distortion Field (INDF)**:
   - Evaluates random Fourier frequency encodings of 2D pixel coordinates $(x, y) \in [-1, 1]^2$.
   - A coordinate-based MLP estimates continuous atmospheric scattering, motion vectors, and high-frequency sensor noise.
   - Decouples and subtracts localized degradation prior to deep feature extraction with a learnable residual gate.

2. **Coordinate-Guided Continuous Topological Router (CTR)**:
   - Replaces rigid sequential depth and binary early exits with a soft, differentiable spatial routing gate ($\alpha \in [0, 1]$).
   - Dynamically routes localized feature patches: degraded patches undergo deep shape reconstruction via bottleneck micro-blocks with squeeze-and-excitation recalibration, while clean patches pass via lightweight identity bridges.

3. **Spatially-Gated Multi-Frequency HyperNetwork (SMFH)**:
   - A spatial frequency profiler analyzes low-frequency chromatic gradients versus high-frequency Sobel boundary textures.
   - Generates passband convolution filter weights on-the-fly, adapting frequency selectivity in real-time according to measured corruption profiles.

4. **Multi-Task ResNet-18 Feature Backbone**:
   - Smooth gradient propagation via Mish non-linearities.
   - Dedicated projection heads for **GTSRB** (43 classes), **BTSD** (62 classes), and **CTSD** (58 classes).

5. **Test-Time Adaptation (TENT)**:
   - Online entropy minimization over BatchNorm and LayerNorm parameters on corrupted test distributions without test-leakage.

6. **Standardized Corruption Benchmark Suite**:
   - 14 real-world environmental corruptions across 5 severity levels (weather, blurs, digital noise, geometric distortions).

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
    ├── models.py             # INDF, CTR, SMFH, and SGNeRFCTR unified architecture
    ├── datasets.py           # Real dataset parsers & UnifiedTrafficSignDataset (RGB, HSV, Sobel)
    ├── corruptions.py        # 14 environmental corruptions & convolution filter kernels
    ├── adaptation.py         # TENT test-time adaptation and weight snapshot/restore
    └── utils.py              # Reproducibility seed, metrics calculation & heartbeat tracker
```

---

## 🚀 Quickstart

### 1. Multi-Task & Individual Training

- **Train concurrently on all datasets**:
  ```powershell
  python train.py --epochs 10 --batch-size 64
  ```

- **Train on an individual dataset**:
  ```powershell
  python train.py --datasets GTSRB --epochs 10
  python train.py --datasets BTSD --epochs 10
  python train.py --datasets CTSD --epochs 10
  ```

### 2. Robustness Evaluation & Benchmarking

- **Benchmark all datasets**:
  ```powershell
  python evaluate.py --checkpoint-path checkpoint_multitask.pth --output-csv novel_framework_corruption_results.csv
  ```

- **Benchmark an individual dataset**:
  ```powershell
  python evaluate.py --datasets GTSRB --checkpoint-path checkpoint_multitask.pth --output-csv gtsrb_results.csv
  python evaluate.py --datasets BTSD --checkpoint-path checkpoint_multitask.pth --output-csv btsd_results.csv
  python evaluate.py --datasets CTSD --checkpoint-path checkpoint_multitask.pth --output-csv ctsd_results.csv
  ```

---

## 📊 Evaluation Outputs

The evaluation engine computes and logs:
- **Clean Baseline Accuracy & F1-Scores** (Macro and Weighted)
- **Mean Corrupted Accuracy** across 14 environmental distortions
- **Per-frame Inference Latency** (ms/frame) with CUDA synchronization
- Results exported cleanly to CSV for comparative analysis
