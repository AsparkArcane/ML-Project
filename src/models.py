"""
Neural network architectures for robust multi-task traffic sign classification.
Includes:
  1. Spatial Transformer Network (STN) for geometric rectification
  2. Dual-Stream Adaptive Attention Gate (HSV color vs. Sobel shape)
  3. MultiTaskRobustSignNet with ResNet-18 backbone, early-exit classifier, and task-specific projection heads.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network (STN) to dynamically align and rectify
    geometric distortions, rotation, scaling, and perspective skews.
    """

    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2, 2),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Mish()
        )
        # Regressor for the 2x3 affine transformation matrix theta
        # Input size: 64 channels * 16 * 16 (for 64x64 inputs) = 16384
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.Mish(),
            nn.Linear(128, 6)
        )
        # Initialize the affine matrix to the identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)


class DualStreamAttentionGate(nn.Module):
    """
    Dual-stream attention gating mechanism that adaptively weights
    photometric chromatic information (HSV stream) against geometric boundary information (Sobel edge stream).
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super(DualStreamAttentionGate, self).__init__()
        # Chromatic stream (HSV)
        self.color_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.Mish()
        )
        # Structural stream (Sobel gradient edge magnitude)
        self.shape_branch = nn.Sequential(
            nn.Conv2d(1, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.Mish()
        )
        # Adaptive dynamic gating MLP
        self.gating_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, 64),
            nn.Mish(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        x_aligned_rgb: torch.Tensor,
        x_hsv: torch.Tensor,
        x_sobel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f_color = self.color_branch(x_hsv)
        f_shape = self.shape_branch(x_sobel)
        combined_feat = torch.cat([f_color, f_shape], dim=1)

        weights = self.gating_network(combined_feat)
        w_color = weights[:, 0].view(-1, 1, 1, 1)
        w_shape = weights[:, 1].view(-1, 1, 1, 1)

        fused_features = w_color * f_color + w_shape * f_shape
        return fused_features, weights


class MultiTaskRobustSignNet(nn.Module):
    """
    Unified multi-task architecture featuring:
      1. Spatial Transformer Network (STN) frontend
      2. Dual-Stream Attention Gating
      3. ResNet-18 feature extraction backbone with Mish activations
      4. Early-exit classifier for accelerated low-complexity inference
      5. Task-specific classification heads for GTSRB, BTSD, and CTSD.
    """

    def __init__(self, num_classes_dict: Optional[Dict[str, int]] = None):
        super(MultiTaskRobustSignNet, self).__init__()

        if num_classes_dict is None:
            num_classes_dict = {"GTSRB": 43, "BTSD": 62, "CTSD": 58}
        self.num_classes_dict = num_classes_dict

        self.stn = SpatialTransformerNetwork()
        self.attention_gate = DualStreamAttentionGate(in_channels=3, feature_dim=64)

        # ResNet-18 backbone initialization with safe fallback for offline execution
        try:
            base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            base_resnet = models.resnet18(weights=None)

        self.conv1_adapter = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = base_resnet.bn1
        self.relu = nn.Mish()
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2

        # Early exit classifier head attached after layer2 (feature channels: 128)
        max_classes = max(num_classes_dict.values())
        self.early_exit_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, max_classes)
        )

        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

        # Task-specific sandwich projection heads
        in_features = base_resnet.fc.in_features  # 512 for ResNet-18
        self.heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.Mish(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_hsv: torch.Tensor,
        x_sobel: torch.Tensor,
        task_name: str = "GTSRB",
        confidence_threshold: float = 0.95,
        eval_early_exit: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Forward computation with optional early-exit decision.
        Returns:
            logits: Output classification logits
            gate_weights: Attention weights for [color, shape]
            exited_early: Boolean indicating whether early exit triggered
        """
        x_stn = self.stn(x_rgb)
        fused_feats, gate_weights = self.attention_gate(x_stn, x_hsv, x_sobel)

        x = self.conv1_adapter(fused_feats)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        if eval_early_exit:
            early_logits = self.early_exit_fc(x)
            # Slice early logits to task class count if needed
            task_num_classes = self.num_classes_dict.get(task_name, early_logits.size(1))
            task_early_logits = early_logits[:, :task_num_classes]

            early_probs = F.softmax(task_early_logits, dim=1)
            max_conf, _ = torch.max(early_probs, dim=1)
            if max_conf.mean().item() >= confidence_threshold:
                return task_early_logits, gate_weights, True

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        task_head_key = task_name if task_name in self.heads else "GTSRB"
        logits = self.heads[task_head_key](x)

        return logits, gate_weights, False

