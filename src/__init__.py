"""
Unified Traffic Sign Classification System package.
Modular architectures, corruptions, datasets, adaptation, and evaluation utilities.
Supports both the novel SG-NeRF-CTR architecture and legacy baseline modules.
"""

from .corruptions import EnvironmentalCorruptions
from .datasets import UnifiedTrafficSignDataset, load_real_dataset_splits
from .models import (
    SGNeRFCTR,
    MultiTaskRobustSignNet,
    ImplicitNeuralDistortionField,
    ContinuousTopologicalRouter,
    SpatiallyGatedMultiFrequencyHyperNetwork,
    SpatialTransformerNetwork,
    DualStreamAttentionGate,
)
from .adaptation import apply_tent_adaptation, snapshot_model_state, restore_model_state
from .utils import set_seed, evaluate_framework, HeartbeatTracker, save_checkpoint, load_checkpoint

__all__ = [
    "EnvironmentalCorruptions",
    "UnifiedTrafficSignDataset",
    "load_real_dataset_splits",
    "SGNeRFCTR",
    "MultiTaskRobustSignNet",
    "ImplicitNeuralDistortionField",
    "ContinuousTopologicalRouter",
    "SpatiallyGatedMultiFrequencyHyperNetwork",
    "SpatialTransformerNetwork",
    "DualStreamAttentionGate",
    "apply_tent_adaptation",
    "snapshot_model_state",
    "restore_model_state",
    "set_seed",
    "evaluate_framework",
    "HeartbeatTracker",
    "save_checkpoint",
    "load_checkpoint",
]
