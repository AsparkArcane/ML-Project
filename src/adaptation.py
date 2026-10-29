"""
Test-Time Adaptation (TENT) module for the Unified Traffic Sign Classification System.
Minimizes model entropy on corrupted test distributions by adapting BatchNorm affine parameters online.
Includes strict state checkpointing and restoration utilities to prevent weight leakage across evaluation loops.
"""

import copy
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def snapshot_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Creates an independent deep-copied snapshot of the model's state_dict
    on CPU memory to prevent test-time weight leakage across corruption trials.
    """
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def restore_model_state(model: nn.Module, state_dict_snapshot: Dict[str, torch.Tensor], device: torch.device):
    """
    Restores the model parameters and buffers exactly from a pristine snapshot.
    """
    device_state = {k: v.to(device) for k, v in state_dict_snapshot.items()}
    model.load_state_dict(device_state)


def apply_tent_adaptation(
    model: nn.Module,
    corrupted_loader: DataLoader,
    device: torch.device,
    task_name: str = "GTSRB",
    steps: int = 1,
    lr: float = 1e-3
):
    """
    Applies Test-Time Entropy Minimization (TENT) to the model.
    Optimizes only BatchNorm affine parameters (weight & bias) while keeping
    the rest of the backbone frozen.

    Args:
        model: PyTorch model to adapt
        corrupted_loader: DataLoader containing the target out-of-distribution / corrupted samples
        device: Active torch compute device
        task_name: Key identifying the target dataset / task head
        steps: Number of test-time optimization passes over the loader
        lr: Learning rate for test-time Adam optimizer
    """
    model.eval()

    # Collect only BatchNorm parameters (weights and biases)
    bn_params = []
    for name, param in model.named_parameters():
        if "bn" in name or "batchnorm" in name:
            param.requires_grad = True
            bn_params.append(param)
        else:
            param.requires_grad = False

    if not bn_params:
        return

    optimizer_tent = optim.Adam(bn_params, lr=lr)
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    for _ in range(steps):
        for rgb, hsv, sobel, _ in corrupted_loader:
            rgb = rgb.to(device, non_blocking=True)
            hsv = hsv.to(device, non_blocking=True)
            sobel = sobel.to(device, non_blocking=True)

            optimizer_tent.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                logits, _, _ = model(rgb, hsv, sobel, task_name=task_name)
                probs = F.softmax(logits, dim=1)
                # Entropy minimization objective
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()

            scaler.scale(entropy_loss).backward()
            scaler.step(optimizer_tent)
            scaler.update()

    # Reset parameter requires_grad state
    for param in model.parameters():
        param.requires_grad = True

