"""
Test-Time Adaptation (TENT) module for the SG-NeRF-CTR Traffic Sign Classification System.
Minimises model entropy on corrupted test distributions by adapting BatchNorm and LayerNorm
affine parameters online.  Includes strict state checkpointing and restoration utilities to
prevent weight leakage across evaluation loops.

Compatible with the SG-NeRF-CTR forward signature:
    model(rgb, hsv, sobel, task_name=...) -> (logits, routing_map, blend_weight)
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
    Applies Test-Time Entropy Minimization (TENT) to the SG-NeRF-CTR model.

    Optimises only the affine parameters of BatchNorm and LayerNorm layers
    (weight & bias) while keeping the rest of the backbone frozen.  These
    normalisation layers are the lightest adaptation target and carry most of
    the distribution-shift signal.

    INDF residual_scale and SMFH blend_gate are also unlocked so the model
    can dynamically re-tune its corruption-correction and frequency-selectivity
    responses to the new test distribution.

    Args:
        model:            SG-NeRF-CTR (or any nn.Module) to adapt.
        corrupted_loader: DataLoader with out-of-distribution / corrupted samples.
        device:           Active torch compute device.
        task_name:        Key identifying the target dataset / task head.
        steps:            Number of test-time optimisation passes over the loader.
        lr:               Learning rate for the test-time Adam optimiser.
    """
    model.eval()

    # Collect adaptable parameters: norm layer affines + lightweight scalar gates
    adapt_params = []
    _norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
    for name, module in model.named_modules():
        if isinstance(module, _norm_types):
            for param in module.parameters():
                param.requires_grad = True
                adapt_params.append(param)
        else:
            for param in module.parameters(recurse=False):
                param.requires_grad = False

    # Also adapt the INDF residual scale and SMFH blend gate (small scalars)
    for attr in ("indf", "smfh"):
        sub = getattr(model, attr, None)
        if sub is not None:
            for pname in ("residual_scale", "blend_gate"):
                p = getattr(sub, pname, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad = True
                    adapt_params.append(p)

    if not adapt_params:
        return

    optimizer_tent = optim.Adam(adapt_params, lr=lr)
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    for _ in range(steps):
        for rgb, hsv, sobel, _ in corrupted_loader:
            rgb   = rgb.to(device, non_blocking=True)
            hsv   = hsv.to(device, non_blocking=True)
            sobel = sobel.to(device, non_blocking=True)

            optimizer_tent.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                logits, _, _ = model(rgb, hsv, sobel, task_name=task_name)
                probs = F.softmax(logits, dim=1)
                # Shanon entropy minimisation objective
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()

            scaler.scale(entropy_loss).backward()
            scaler.step(optimizer_tent)
            scaler.update()

    # Restore full trainability after adaptation so evaluation runs normally
    for param in model.parameters():
        param.requires_grad = True
