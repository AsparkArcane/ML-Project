"""
Independent multi-task training script for the SG-NeRF-CTR architecture.

SG-NeRF-CTR = Spatially-Gated Neural Radiance Field Invariance with
              Continuous-Topology Routing.

Trains concurrently across GTSRB, BTSD, and CTSD datasets using:
  - Implicit Neural Distortion Field (INDF) pre-processing
  - Coordinate-Guided Continuous Topological Router (CTR)
  - Spatially-Gated Multi-Frequency HyperNetwork (SMFH)
  - PyTorch Automatic Mixed Precision (AMP) via GradScaler
  - AugMix-style environmental noise regularisation
  - Cosine Annealing LR schedule
  - Periodic 5-minute heartbeat telemetry

The model forward signature is:
    model(rgb, hsv, sobel, task_name=...) -> (logits, routing_map, blend_weight)
Only `logits` is used for loss computation during training.
"""

import os
import sys
import time
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.datasets import load_real_dataset_splits, UnifiedTrafficSignDataset
from src.models import SGNeRFCTR as MultiTaskRobustSignNet   # canonical name + compat alias
from src.utils import set_seed, HeartbeatTracker, save_checkpoint, evaluate_framework


def parse_args():
    parser = argparse.ArgumentParser(description="SG-NeRF-CTR Multi-Task Traffic Sign Training Engine")
    parser.add_argument("--data-root", type=str, default=config.DATA_ROOT, help="Root folder containing dataset subdirectories")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size per training worker")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY, help="AdamW weight decay regularizer")
    parser.add_argument("--checkpoint-path", type=str, default=config.CHECKPOINT_PATH, help="Destination weights checkpoint path")
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS_TO_RUN,
                        help="One or more datasets to train on (e.g., --datasets GTSRB or --datasets GTSRB BTSD). Options: GTSRB, BTSD, CTSD")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader CPU worker thread count")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    heartbeat: HeartbeatTracker
) -> float:
    """
    Executes one multi-task training epoch across all active dataset loaders.

    The SG-NeRF-CTR forward returns (logits, routing_map, blend_weight).
    routing_map and blend_weight are auxiliary outputs used for analysis;
    only logits contribute to the cross-entropy training loss here.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_cuda = (device.type == "cuda")

    for task_name, loader in dataloaders.items():
        total_batches = len(loader)
        for batch_idx, (rgb, hsv, sobel, labels) in enumerate(loader):
            rgb   = rgb.to(device, non_blocking=True)
            hsv   = hsv.to(device, non_blocking=True)
            sobel = sobel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed-precision forward pass
            # logits: (B, num_classes)
            # routing_map: (B, 1, H', W') — CTR alpha map (not used in loss)
            # blend_weight: scalar — SMFH gate value (not used in loss)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                logits, _routing_map, _blend = model(rgb, hsv, sobel, task_name=task_name)
                loss = criterion(logits, labels)

            # Mixed-precision scaled backward step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_samples += batch_count

            # 5-minute heartbeat logging check
            heartbeat.check_heartbeat(
                context_msg=f"| Epoch {epoch}/{total_epochs} | Task: {task_name:<5} | Batch {batch_idx + 1}/{total_batches}"
            )

            # Periodic batch progress logging
            if (batch_idx + 1) % config.LOG_INTERVAL_BATCHES == 0:
                print(
                    f"   -> Epoch {epoch:02d}/{total_epochs:02d} | Task: {task_name:<5} | "
                    f"Batch {batch_idx + 1:04d}/{total_batches:04d} | Batch Loss: {loss.item():.4f}"
                )

    return total_loss / max(1, total_samples)


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("STARTING SG-NeRF-CTR MULTI-TASK TRAFFIC SIGN TRAINING ENGINE")
    print("=" * 80)

    # Initialize deterministic seeds
    set_seed(config.RANDOM_SEED)
    config.print_device_info()

    device = config.DEVICE
    pin_mem = config.PIN_MEMORY
    num_workers = args.num_workers

    dataloaders_train: Dict[str, DataLoader] = {}
    dataloaders_val: Dict[str, DataLoader] = {}
    num_classes_dict: Dict[str, int] = {}

    # Normalize and validate selected datasets
    selected_datasets = [d.upper() for d in args.datasets]
    for d in selected_datasets:
        if d not in config.DATASET_SPECS:
            raise ValueError(f"Unknown dataset '{d}'. Supported options: {list(config.DATASET_SPECS.keys())}")

    print(f"\n[INFO] Initializing dataset splits from root: '{args.data_root}' for datasets: {selected_datasets}...")

    for dataset_name in selected_datasets:
        tr_p, tr_l, te_p, te_l, n_cls = load_real_dataset_splits(
            dataset_name=dataset_name,
            data_root=args.data_root
        )
        num_classes_dict[dataset_name] = n_cls

        train_ds = UnifiedTrafficSignDataset(
            tr_p, tr_l,
            image_size=config.IMAGE_SIZE,
            is_train=True
        )
        val_ds = UnifiedTrafficSignDataset(
            te_p, te_l,
            image_size=config.IMAGE_SIZE,
            is_train=False
        )

        dataloaders_train[dataset_name] = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=(num_workers > 0)
        )
        dataloaders_val[dataset_name] = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=(num_workers > 0)
        )

        print(
            f"  • {dataset_name:<5}: Train Samples={len(tr_p):<5} | "
            f"Test Samples={len(te_p):<5} | Classes={n_cls}"
        )

    # Instantiate SG-NeRF-CTR model
    model = MultiTaskRobustSignNet(num_classes_dict=num_classes_dict).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    heartbeat = HeartbeatTracker(interval_sec=config.HEARTBEAT_INTERVAL_SEC)
    start_time = time.time()

    print(f"\n[INFO] Training SG-NeRF-CTR across {args.epochs} epochs with AMP & Cosine Annealing...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model=model,
            dataloaders=dataloaders_train,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            heartbeat=heartbeat
        )

        scheduler.step()
        epoch_sec = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[EPOCH COMPLETE] Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Multi-Task Loss: {train_loss:.4f} | LR: {current_lr:.6f} | "
            f"Duration: {epoch_sec:.1f}s"
        )

        # Periodic checkpoint save at every epoch
        save_checkpoint(
            path=args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss}
        )

    total_duration_min = (time.time() - start_time) / 60.0
    print("\n" + "=" * 80)
    print(f"[SUCCESS] SG-NeRF-CTR Multi-Task Training Finished in {total_duration_min:.1f} minutes.")
    print(f"[INFO] Weights saved to: '{args.checkpoint_path}'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
