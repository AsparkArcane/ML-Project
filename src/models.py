"""
SG-NeRF-CTR: Spatially-Gated Neural Radiance Field Invariance with
Continuous-Topology Routing for robust multi-task traffic sign classification.

Architecture components:
  1. Implicit Neural Distortion Field (INDF)
       - Fourier-encoded coordinate MLP that maps (x, y) pixel positions to a
         continuous 3-D degradation field.  The field predicts per-pixel
         atmospheric scattering, motion-vector offsets, and high-frequency
         sensor noise, which are *subtracted* from the raw input so that the
         downstream backbone sees a pre-cleaned feature map.

  2. Coordinate-Guided Continuous Topological Router (CTR)
       - Replaces the hard early-exit gate with a soft, differentiable patch
         router.  Each spatial patch is assigned a routing weight in [0, 1]:
         low-quality patches flow through deep shape-reconstruction micro-blocks
         (residual dark-channel bottlenecks); high-quality patches skip directly
         via a lightweight identity bridge.  Gradients flow through both paths,
         so routing improves jointly with the rest of the network.

  3. Spatially-Gated Multi-Frequency HyperNetwork (SMFH)
       - A HyperNetwork analyses the local spatial-frequency profile of each
         sample (low-freq colour gradients vs. high-freq Sobel textures) and
         *generates* the weight tensors for a small "passband" convolution on
         the fly.  This lets the network dynamically tune its frequency
         selectivity based on the real-time corruption level rather than relying
         on fixed convolutional filters.

  4. ResNet-18 backbone (Mish) + task-specific projection heads
       - Shared feature extractor; separate fully-connected classifiers for
         GTSRB (43 classes), BTSD (62 classes), and CTSD (58 classes).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Helper: Mish activation (used project-wide)
# ---------------------------------------------------------------------------
class Mish(nn.Module):
    """Mish = x * tanh(softplus(x)).  Smooth, non-monotonic activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# ===========================================================================
# 1.  Implicit Neural Distortion Field (INDF)
# ===========================================================================

class FourierFeatureEncoder(nn.Module):
    """
    Random Fourier feature encoder for 2-D pixel coordinates.

    Maps a (B, 2, H, W) coordinate grid to a (B, 2*num_freqs, H, W) tensor
    of sinusoidal features, providing the MLP with a rich positional basis
    that captures fine-grained spatial structure (NeRF-style).

    Args:
        num_freqs: Number of frequency bands.
        sigma:     Standard deviation of the random frequency matrix (controls
                   the range of spatial frequencies sampled).
    """

    def __init__(self, num_freqs: int = 16, sigma: float = 10.0):
        super().__init__()
        self.num_freqs = num_freqs
        # Fixed random projection matrix (not trained); shape (2, num_freqs)
        B_mat = torch.randn(2, num_freqs) * sigma
        self.register_buffer("B_mat", B_mat)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2, H, W) normalised coordinates in [-1, 1].
        Returns:
            (B, 2*num_freqs, H, W) Fourier features.
        """
        B_batch, _, H, W = coords.shape
        # (B, 2, H*W) -> (B, H*W, 2)
        c_flat = coords.view(B_batch, 2, -1).permute(0, 2, 1)
        # (B, H*W, num_freqs)
        proj = 2.0 * math.pi * (c_flat @ self.B_mat)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, H*W, 2F)
        return feats.permute(0, 2, 1).view(B_batch, 2 * self.num_freqs, H, W)


class ImplicitNeuralDistortionField(nn.Module):
    """
    Implicit Neural Distortion Field (INDF).

    Given an input RGB image, the module:
      1. Builds a normalised (x, y) pixel coordinate grid.
      2. Encodes coordinates with random Fourier features.
      3. Fuses the Fourier features with the image via a pixel-wise MLP
         (implemented as 1x1 convolutions for efficient parallelisation).
      4. Outputs a 3-channel *distortion residual* that is subtracted from
         the raw RGB to yield a pre-cleaned, geometry-aware input for the
         backbone.  The residual models atmospheric scattering (haze), motion
         ghosting, and high-frequency sensor noise jointly.

    Args:
        num_freqs:   Fourier encoding frequency bands.
        hidden_dim:  Width of the MLP hidden layers.
    """

    def __init__(self, num_freqs: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.fourier = FourierFeatureEncoder(num_freqs=num_freqs, sigma=10.0)

        # Input channels: 2*num_freqs (Fourier coords) + 3 (RGB)
        in_ch = 2 * num_freqs + 3

        # Pixel-wise MLP via 1x1 convolutions
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
            Mish(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            Mish(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Tanh(),          # Distortion residual in [-1, 1]
        )

        # Learnable scale so INDF starts near identity (residual ≈ 0)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def _build_coord_grid(
        self, batch: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Returns a (B, 2, H, W) grid of (x, y) coords normalised to [-1, 1]."""
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each
        grid = torch.stack([grid_x, grid_y], dim=0)               # (2, H, W)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)        # (B, 2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) raw RGB image, values in [0, 1].
        Returns:
            x_cleaned: (B, 3, H, W) distortion-corrected image, clipped to [0, 1].
        """
        B, _, H, W = x.shape
        coords = self._build_coord_grid(B, H, W, x.device)   # (B, 2, H, W)
        fourier_feats = self.fourier(coords)                   # (B, 2F, H, W)

        # Concatenate spatial frequency embeddings with the raw pixel values
        combined = torch.cat([fourier_feats, x], dim=1)       # (B, 2F+3, H, W)

        # Predict degradation residual; scaled near zero at initialisation
        distortion = self.mlp(combined) * self.residual_scale  # (B, 3, H, W)

        return torch.clamp(x - distortion, 0.0, 1.0)


# ===========================================================================
# 2.  Coordinate-Guided Continuous Topological Router (CTR)
# ===========================================================================

class ShapeReconstructionMicroBlock(nn.Module):
    """
    Deep shape-reconstruction micro-block for severely degraded patches.

    A bottleneck residual block with extra depth that specialises in recovering
    structural information from low-SNR feature patches.  Two 3x3 convolutions
    plus a squeeze-excitation gate focus attention on salient edge information.

    Args:
        channels: Number of feature channels (input = output).
    """

    def __init__(self, channels: int):
        super().__init__()
        bottleneck = max(channels // 4, 16)

        self.body = nn.Sequential(
            # Compress to bottleneck
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            Mish(),
            # Depthwise spatial convolution (cheap, preserves structure)
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1,
                      groups=bottleneck, bias=False),
            nn.BatchNorm2d(bottleneck),
            Mish(),
            # Standard 3x3 to mix channels
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            Mish(),
            # Expand back
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Squeeze-and-Excitation gate to re-weight recovered channels
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 8, 4)),
            Mish(),
            nn.Linear(max(channels // 8, 4), channels),
            nn.Sigmoid(),
        )
        self.act = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        se_weight = self.se(residual).view(x.size(0), -1, 1, 1)
        return self.act(x + residual * se_weight)


class ContinuousTopologicalRouter(nn.Module):
    """
    Continuous Topological Router (CTR).

    Instead of hard early-exit branching, CTR computes a per-spatial-location
    routing weight alpha in [0, 1] from local image statistics:

        out = alpha * deep_path(x)  +  (1 - alpha) * skip_path(x)

    where:
      • alpha ≈ 1  → the patch is severely degraded; route through the deep
                      ShapeReconstructionMicroBlock.
      • alpha ≈ 0  → the patch is clean; skip directly via a lightweight 1x1
                      residual bridge (near-identity).

    Gradients flow through *both* paths and through the alpha gate, so the
    routing policy co-evolves with the feature extractor during training.

    Args:
        channels: Number of input/output feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Gate network: maps local feature statistics to a spatial routing map
        # Outputs a single-channel sigmoid map (B, 1, H, W)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            Mish(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Deep path: shape reconstruction for degraded patches
        self.deep_path = ShapeReconstructionMicroBlock(channels)

        # Skip / clean path: lightweight residual bridge
        self.skip_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input feature map.
        Returns:
            out:   (B, C, H, W) routed output.
            alpha: (B, 1, H, W) routing weight map (high = deep path used).
        """
        # Compute soft routing weights per spatial location
        alpha = self.gate_conv(x)           # (B, 1, H, W) in [0, 1]

        deep_out = self.deep_path(x)        # (B, C, H, W)
        skip_out = self.skip_path(x)        # (B, C, H, W)

        # Soft continuous blend — both paths contribute with learned weights
        out = alpha * deep_out + (1.0 - alpha) * skip_out
        return out, alpha


# ===========================================================================
# 3.  Spatially-Gated Multi-Frequency HyperNetwork (SMFH)
# ===========================================================================

class SpatialFrequencyProfiler(nn.Module):
    """
    Extracts a compact spatial-frequency profile from an RGB + Sobel pair.

    Low-frequency colour gradients (smooth regions) and high-frequency edge
    textures (fine structure) are pooled separately and concatenated into a
    single descriptor vector that summarises the frequency content of the
    current sample.

    Args:
        rgb_channels:   Input RGB channels (3).
        sobel_channels: Input Sobel channels (1).
        out_dim:        Length of the output frequency descriptor.
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        sobel_channels: int = 1,
        out_dim: int = 128,
    ):
        super().__init__()
        # Low-frequency branch: heavily pooled colour gradients
        self.lf_branch = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.AdaptiveAvgPool2d(4),      # Aggressive spatial averaging → smooth info
            nn.Flatten(),
            nn.Linear(32 * 16, out_dim // 2),
            Mish(),
        )
        # High-frequency branch: minimal pooling on Sobel edge maps
        self.hf_branch = nn.Sequential(
            nn.Conv2d(sobel_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.AdaptiveMaxPool2d(4),      # Max pooling preserves sharp edges
            nn.Flatten(),
            nn.Linear(32 * 16, out_dim // 2),
            Mish(),
        )

    def forward(
        self, x_rgb: torch.Tensor, x_sobel: torch.Tensor
    ) -> torch.Tensor:
        """Returns a (B, out_dim) frequency descriptor."""
        lf = self.lf_branch(x_rgb)
        hf = self.hf_branch(x_sobel)
        return torch.cat([lf, hf], dim=1)


class HyperConvGenerator(nn.Module):
    """
    HyperNetwork that generates the weight tensor of a target 3x3 convolution
    on the fly from a frequency descriptor.

    Generated weights replace the fixed filters of a downstream 'passband'
    convolution so that the network's frequency selectivity adapts at runtime
    to the measured corruption profile of each sample.

    Args:
        freq_dim:      Dimension of input frequency descriptor.
        in_channels:   In-channels of the target convolution.
        out_channels:  Out-channels of the target convolution.
        kernel_size:   Spatial extent of the generated kernel (default 3).
    """

    def __init__(
        self,
        freq_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight_dim = out_channels * in_channels * kernel_size * kernel_size

        self.gen = nn.Sequential(
            nn.Linear(freq_dim, 256),
            Mish(),
            nn.LayerNorm(256),
            nn.Linear(256, self.weight_dim),
        )

    def forward(self, freq_desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_desc: (B, freq_dim) frequency descriptor.
        Returns:
            weights: (B, out_ch, in_ch, K, K) generated kernel weights.
        """
        w = self.gen(freq_desc)     # (B, weight_dim)
        return w.view(
            -1,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )


class SpatiallyGatedMultiFrequencyHyperNetwork(nn.Module):
    """
    Spatially-Gated Multi-Frequency HyperNetwork (SMFH).

    Workflow per forward call:
      1. SpatialFrequencyProfiler analyses the current sample's frequency content.
      2. HyperConvGenerator produces a custom 3x3 kernel batch from the profile.
      3. A sample-specific 'passband' convolution is applied to the input
         feature map using the generated kernels (via grouped batch convolution).
      4. The passband output is fused (residual add) with the original features.

    This lets the model act as a learnable bandpass filter whose cutoff adapts
    to the severity and type of corruption without any hand-crafted frequency
    engineering.

    Args:
        feature_channels: Number of channels of the input feature map.
        freq_desc_dim:    Dimension of the SpatialFrequencyProfiler output.
        hyper_out_ch:     Number of output channels for the generated passband conv.
    """

    def __init__(
        self,
        feature_channels: int = 64,
        freq_desc_dim: int = 128,
        hyper_out_ch: int = 64,
    ):
        super().__init__()
        self.profiler = SpatialFrequencyProfiler(
            rgb_channels=3,
            sobel_channels=1,
            out_dim=freq_desc_dim,
        )
        self.hyper_gen = HyperConvGenerator(
            freq_dim=freq_desc_dim,
            in_channels=feature_channels,
            out_channels=hyper_out_ch,
            kernel_size=3,
        )
        self.feature_channels = feature_channels
        self.hyper_out_ch = hyper_out_ch

        # Projection back to feature_channels so residual add is dimension-safe
        self.project_back = nn.Sequential(
            nn.Conv2d(hyper_out_ch, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            Mish(),
        )
        # Learnable gate controlling how strongly the HyperNet output is blended
        self.blend_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        x_rgb: torch.Tensor,
        x_sobel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, feature_channels, H, W) input feature map.
            x_rgb:   (B, 3, H, W) raw RGB (for frequency profiling).
            x_sobel: (B, 1, H, W) Sobel edge map (for frequency profiling).
        Returns:
            (B, feature_channels, H, W) frequency-adapted feature map.
        """
        B, C, H, W = x.shape

        # 1. Obtain a per-sample frequency descriptor
        freq_desc = self.profiler(x_rgb, x_sobel)   # (B, freq_desc_dim)

        # 2. Generate sample-specific convolution weights
        hyper_weights = self.hyper_gen(freq_desc)    # (B, out_ch, C, 3, 3)

        # 3. Apply generated kernels sample-by-sample using grouped batch trick
        #    Treat the batch dimension as independent groups so each sample
        #    gets its own unique filter while avoiding Python-level loops.
        x_grouped = x.view(1, B * C, H, W)          # (1, B*C, H, W)
        w_grouped = hyper_weights.view(
            B * self.hyper_out_ch, C, 3, 3
        )                                            # (B*out_ch, C, 3, 3)

        # Grouped convolution: group i processes sample i's channels
        passband = F.conv2d(
            x_grouped,
            w_grouped,
            padding=1,
            groups=B,
        )                                            # (1, B*out_ch, H, W)
        passband = passband.view(B, self.hyper_out_ch, H, W)

        # 4. Project back and blend with the residual connection
        passband = self.project_back(passband)       # (B, C, H, W)
        blend = torch.sigmoid(self.blend_gate)       # scalar in [0, 1]
        return x + blend * passband


# ===========================================================================
# 4.  SG-NeRF-CTR  —  Unified Model
# ===========================================================================

class SGNeRFCTR(nn.Module):
    """
    Spatially-Gated Neural Radiance Field Invariance with
    Continuous-Topology Routing (SG-NeRF-CTR).

    Full forward dataflow:
      RGB ──► INDF (distortion correction)
              ├──► SMFH profiler sees raw RGB + Sobel
              └──► Corrected features ──► ResNet-18 stem
                                           ├──► CTR (after layer1)
                                           │      soft-routes degraded patches
                                           ├──► SMFH passband filter
                                           ├──► layer2 → layer3 → layer4
                                           └──► Task-specific head

    Args:
        num_classes_dict: Mapping from dataset name to number of output classes.
                          Defaults to {"GTSRB": 43, "BTSD": 62, "CTSD": 58}.
        num_freqs:        Fourier frequency bands for INDF.
        indf_hidden:      Hidden width of the INDF coordinate MLP.
        freq_desc_dim:    Dimension of SMFH's SpatialFrequencyProfiler output.
    """

    def __init__(
        self,
        num_classes_dict: Optional[Dict[str, int]] = None,
        num_freqs: int = 16,
        indf_hidden: int = 64,
        freq_desc_dim: int = 128,
    ):
        super().__init__()

        if num_classes_dict is None:
            num_classes_dict = {"GTSRB": 43, "BTSD": 62, "CTSD": 58}
        self.num_classes_dict = num_classes_dict

        # ------------------------------------------------------------------
        # Block 1: Implicit Neural Distortion Field
        # ------------------------------------------------------------------
        self.indf = ImplicitNeuralDistortionField(
            num_freqs=num_freqs, hidden_dim=indf_hidden
        )

        # ------------------------------------------------------------------
        # ResNet-18 backbone (Mish activations replace ReLU)
        # ------------------------------------------------------------------
        try:
            _rn = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        except Exception:
            _rn = tv_models.resnet18(weights=None)

        # Replace all ReLU activations in the backbone with Mish
        self._replace_relu_with_mish(_rn)

        # Keep stem components
        self.stem_conv   = _rn.conv1      # (B, 3, H, W) → (B, 64, H/2, W/2)
        self.stem_bn     = _rn.bn1
        self.stem_act    = Mish()
        self.stem_pool   = _rn.maxpool    # → (B, 64, H/4, W/4)

        self.layer1 = _rn.layer1          # → (B, 64,  H/4,  W/4)
        self.layer2 = _rn.layer2          # → (B, 128, H/8,  W/8)
        self.layer3 = _rn.layer3          # → (B, 256, H/16, W/16)
        self.layer4 = _rn.layer4          # → (B, 512, H/32, W/32)
        self.avgpool = _rn.avgpool        # → (B, 512, 1, 1)

        # ------------------------------------------------------------------
        # Block 2: Continuous Topological Router  (after layer1, ch=64)
        # ------------------------------------------------------------------
        self.ctr = ContinuousTopologicalRouter(channels=64)

        # ------------------------------------------------------------------
        # Block 3: Spatially-Gated Multi-Frequency HyperNetwork (ch=64)
        # ------------------------------------------------------------------
        self.smfh = SpatiallyGatedMultiFrequencyHyperNetwork(
            feature_channels=64,
            freq_desc_dim=freq_desc_dim,
            hyper_out_ch=64,
        )

        # ------------------------------------------------------------------
        # Block 4: Task-specific projection heads
        # ------------------------------------------------------------------
        in_features = 512   # ResNet-18 final feature dimension
        self.heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                Mish(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

    # -----------------------------------------------------------------------
    # Utility: replace ReLU → Mish throughout a module tree
    # -----------------------------------------------------------------------
    @staticmethod
    def _replace_relu_with_mish(module: nn.Module) -> None:
        """Recursively replaces every ReLU in *module* with a Mish activation."""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, Mish())
            else:
                SGNeRFCTR._replace_relu_with_mish(child)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    def forward(
        self,
        x_rgb: torch.Tensor,
        x_hsv: torch.Tensor,           # kept for API compatibility (not consumed directly)
        x_sobel: torch.Tensor,
        task_name: str = "GTSRB",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_rgb:    (B, 3, H, W)  Raw RGB image normalised to [0, 1].
            x_hsv:    (B, 3, H, W)  HSV image (passed through for API compat).
            x_sobel:  (B, 1, H, W)  Sobel edge magnitude map normalised to [0, 1].
            task_name: Key selecting which classification head to use.
            **kwargs: Absorbs legacy keyword arguments (e.g., confidence_threshold,
                      eval_early_exit) to maintain call-site compatibility.

        Returns:
            logits:       (B, num_classes)  Classification logits.
            routing_map:  (B, 1, H', W')    CTR alpha map (useful for analysis).
            blend_weight: scalar tensor, SMFH blend gate value (diagnostic).
        """
        # ── Stage 1: Implicit Neural Distortion Field ────────────────────
        x_clean = self.indf(x_rgb)          # (B, 3, H, W) — degradation-corrected

        # ── Stage 2: ResNet-18 stem ──────────────────────────────────────
        x = self.stem_conv(x_clean)
        x = self.stem_bn(x)
        x = self.stem_act(x)
        x = self.stem_pool(x)               # (B, 64, H/4, W/4)
        x = self.layer1(x)                  # (B, 64, H/4, W/4)

        # ── Stage 3: Continuous Topological Router ───────────────────────
        x, routing_map = self.ctr(x)        # x: (B, 64, …), routing_map: (B,1,…)

        # ── Stage 4: SMFH Passband Filter ───────────────────────────────
        # Resize raw inputs to match current spatial resolution for profiling
        _, _, H_feat, W_feat = x.shape
        rgb_down  = F.interpolate(x_clean, size=(H_feat, W_feat), mode="bilinear",
                                  align_corners=False)
        sobel_down = F.interpolate(x_sobel, size=(H_feat, W_feat), mode="bilinear",
                                   align_corners=False)
        x = self.smfh(x, rgb_down, sobel_down)  # (B, 64, H/4, W/4)

        # ── Stage 5: Deep backbone ───────────────────────────────────────
        x = self.layer2(x)                  # (B, 128, H/8,  W/8)
        x = self.layer3(x)                  # (B, 256, H/16, W/16)
        x = self.layer4(x)                  # (B, 512, H/32, W/32)
        x = self.avgpool(x)                 # (B, 512, 1, 1)
        x = torch.flatten(x, 1)             # (B, 512)

        # ── Stage 6: Task-specific projection head ───────────────────────
        head_key  = task_name if task_name in self.heads else next(iter(self.heads))
        logits    = self.heads[head_key](x)

        blend_weight = torch.sigmoid(self.smfh.blend_gate).detach()

        return logits, routing_map, blend_weight


# ---------------------------------------------------------------------------
# Backward-compatibility aliases so existing call sites don't break
# ---------------------------------------------------------------------------

# train.py and evaluate.py import `MultiTaskRobustSignNet`.
# Alias it to the new architecture transparently.
MultiTaskRobustSignNet = SGNeRFCTR



# ===========================================================================
# 5. Legacy & Standalone Modules for Full Backward Compatibility
# ===========================================================================

class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network (STN) to dynamically align and rectify
    geometric distortions, rotation, scaling, and perspective skews.
    Provided for backward compatibility with baseline scripts and checkpoints.
    """

    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2, 2),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            Mish()
        )
        # Regressor for the 2x3 affine transformation matrix theta
        # Input size: 64 channels * 16 * 16 (for 64x64 inputs) = 16384
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            Mish(),
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
    Provided for backward compatibility with baseline scripts and checkpoints.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super(DualStreamAttentionGate, self).__init__()
        # Chromatic stream (HSV)
        self.color_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            Mish()
        )
        # Structural stream (Sobel gradient edge magnitude)
        self.shape_branch = nn.Sequential(
            nn.Conv2d(1, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            Mish()
        )
        # Adaptive dynamic gating MLP
        self.gating_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, 64),
            Mish(),
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
