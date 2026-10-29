"""
Utilities module for the Unified Traffic Sign Classification System.
Provides seed management, heartbeat logging, metrics computation, and checkpoint handling.
"""

import os
import time
import random
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def set_seed(seed: int = 42):
    """
    Sets global deterministic seeds across Python, NumPy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class HeartbeatTracker:
    """
    Manages long-running task heartbeats and periodic ETA / progress updates.
    Default heartbeat interval is 300 seconds (5 minutes).
    """

    def __init__(self, interval_sec: int = 300):
        self.interval_sec = interval_sec
        self.start_time = time.time()
        self.last_heartbeat = self.start_time

    def check_heartbeat(self, context_msg: str = "") -> bool:
        """
        Emits a heartbeat message if 5 minutes have elapsed since the last check.
        Returns True if a heartbeat was emitted.
        """
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.interval_sec:
            elapsed_min = (current_time - self.start_time) / 60.0
            print(f"[HEARTBEAT] Still executing... ({elapsed_min:.1f}m elapsed) {context_msg}")
            self.last_heartbeat = current_time
            return True
        return False

    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60.0


def evaluate_framework(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_name: str = "GTSRB",
    use_early_exit: bool = False
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Computes performance metrics: accuracy, macro-F1, weighted-F1, and per-sample latency.
    Employs Automatic Mixed Precision (AMP) and CUDA synchronization for accurate timing.

    Returns:
        acc: Classification accuracy in [0.0, 1.0]
        f1_macro: Macro-averaged F1 score
        f1_weighted: Weighted-averaged F1 score
        avg_latency_ms: Average inference latency in milliseconds per image
        all_preds: 1D numpy array of predicted class indices
        all_labels: 1D numpy array of ground truth labels
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    latencies: List[float] = []

    use_cuda = (device.type == "cuda")

    with torch.no_grad():
        for rgb, hsv, sobel, labels in dataloader:
            rgb = rgb.to(device, non_blocking=True)
            hsv = hsv.to(device, non_blocking=True)
            sobel = sobel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_cuda:
                torch.cuda.synchronize()
            start_t = time.perf_counter()

            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                logits, _, _ = model(
                    rgb, hsv, sobel,
                    task_name=task_name,
                    eval_early_exit=use_early_exit
                )

            if use_cuda:
                torch.cuda.synchronize()
            end_t = time.perf_counter()

            batch_size = max(1, rgb.size(0))
            batch_latency_ms = (end_t - start_t) * 1000.0 / batch_size
            latencies.append(batch_latency_ms)

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    acc = float(accuracy_score(all_labels, all_preds))
    f1_m = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    f1_w = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return acc, f1_m, f1_w, avg_latency, np.array(all_preds), np.array(all_labels)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None
):
    """Saves model weights, optimizer state, and training metadata to disk."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "num_classes_dict": getattr(model, "num_classes_dict", None),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics or {}
    }
    torch.save(checkpoint, path)
    print(f"[CHECKPOINT] Model saved successfully to '{path}'.")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Loads weights and training state from checkpoint file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: '{path}'")

    map_location = device if device else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[CHECKPOINT] Loaded checkpoint from '{path}' (Epoch {checkpoint.get('epoch', 0)}).")
    return checkpoint

