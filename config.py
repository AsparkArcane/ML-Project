"""
Configuration module for the SG-NeRF-CTR Noise-Robust Traffic Sign Classification Framework.
Optimized for NVIDIA CUDA acceleration with multi-task training across GTSRB, BTSD, and CTSD datasets.
"""

import os
import torch

# ==============================================================================
# 1. REPRODUCIBILITY & DEVICE CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42

# Optimal compute device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cuDNN optimizations for RTX Tensor Cores
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Optimize CPU worker threading when dispatching to GPU
    torch.set_num_threads(min(8, os.cpu_count() or 4))
else:
    torch.set_num_threads(os.cpu_count() or 4)

def print_device_info():
    """Logs compute device, GPU model, and memory statistics."""
    print(f"[INFO] Active Compute Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU Model             : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] VRAM Available        : {total_vram:.2f} GB")
        print(f"[INFO] cuDNN Benchmark Mode  : {torch.backends.cudnn.benchmark}")
    else:
        print(f"[INFO] CPU Thread Count      : {torch.get_num_threads()}")

# ==============================================================================
# 2. PATHS & DATASET SPECIFICATIONS
# ==============================================================================
DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", "./data"))

GTSRB_DIR = os.path.join(DATA_ROOT, "GTSRB")
BTSD_DIR = os.path.join(DATA_ROOT, "BTSD")
CTSD_DIR = os.path.join(DATA_ROOT, "CTSD")

# Dataset specifications: Class counts
DATASET_SPECS = {
    "GTSRB": 43,
    "BTSD": 62,
    "CTSD": 58
}

DATASETS_TO_RUN = ["GTSRB", "BTSD", "CTSD"]

# Model checkpoint & evaluation output artifacts
CHECKPOINT_PATH = os.path.abspath(os.environ.get("CHECKPOINT_PATH", "checkpoint_multitask.pth"))
RESULTS_CSV_PATH = os.path.abspath(os.environ.get("RESULTS_CSV_PATH", "novel_framework_corruption_results.csv"))

# ==============================================================================
# 3. IMAGE PREPROCESSING & FEATURE EXTRACTION
# ==============================================================================
IMAGE_SIZE = (64, 64)
CHANNELS_RGB = 3
CHANNELS_HSV = 3
CHANNELS_SOBEL = 1

# ==============================================================================
# 4. SG-NeRF-CTR NOVEL ARCHITECTURE HYPERPARAMETERS
# ==============================================================================
# Implicit Neural Distortion Field (INDF)
INDF_NUM_FREQS = 16        # Fourier positional encoding frequency bands
INDF_HIDDEN_DIM = 64       # Width of INDF coordinate MLP

# Continuous Topological Router (CTR)
CTR_CHANNELS = 64          # Channel width at router insertion point (after layer1)

# Spatially-Gated Multi-Frequency HyperNetwork (SMFH)
SMFH_FREQ_DESC_DIM = 128   # Frequency descriptor vector length
SMFH_HYPER_OUT_CH = 64     # Passband dynamic convolution channel depth

# ==============================================================================
# 5. TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 64 if DEVICE.type == "cuda" else 32
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 2
PIN_MEMORY = (DEVICE.type == "cuda")
PERSISTENT_WORKERS = (NUM_WORKERS > 0)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_EPOCHS = 10
EARLY_EXIT_CONFIDENCE = 0.95

# ==============================================================================
# 6. TEST-TIME ADAPTATION (TENT) HYPERPARAMETERS
# ==============================================================================
TENT_LR = 1e-3
TENT_STEPS = 1
TENT_SEVERITY_THRESHOLD = 4  # Apply TENT adaptation on severe corruptions (severity >= 4)

# ==============================================================================
# 7. LOGGING & HEARTBEAT SETTINGS
# ==============================================================================
HEARTBEAT_INTERVAL_SEC = 300  # 5-minute periodic status logs
LOG_INTERVAL_BATCHES = 200

# ==============================================================================
# 8. NOISE CORRUPTION BENCHMARK SUITE (14 CORRUPTIONS)
# ==============================================================================
NOISE_TYPES = [
    "Gaussian",
    "Impulse",
    "Shot",
    "Snow",
    "Fog",
    "Frost",
    "Motion Blur",
    "Zoom Blur",
    "Defocus Blur",
    "Brightness",
    "Contrast",
    "Pixelate",
    "Elastic Transform",
    "JPEG Compression"
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]
