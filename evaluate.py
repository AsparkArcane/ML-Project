"""
Comprehensive robustness benchmarking script for the MultiTaskRobustSignNet system.
Evaluates Clean Baseline performance and 14 environmental corruptions across 5 severity levels for
GTSRB, BTSD, and CTSD datasets with Test-Time Adaptation (TENT) for severe corruptions.
Strictly restores pristine model weights between adaptation trials to prevent weight leakage.
Exports detailed benchmark metrics to novel_framework_corruption_results.csv.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path for local imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.datasets import load_real_dataset_splits, UnifiedTrafficSignDataset
from src.models import MultiTaskRobustSignNet
from src.adaptation import apply_tent_adaptation, snapshot_model_state, restore_model_state
from src.utils import set_seed, evaluate_framework, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task Traffic Sign Robustness Benchmark Suite")
    parser.add_argument("--data-root", type=str, default=config.DATA_ROOT, help="Dataset root directory")
    parser.add_argument("--checkpoint-path", type=str, default=config.CHECKPOINT_PATH, help="Path to trained checkpoint")
    parser.add_argument("--output-csv", type=str, default=config.RESULTS_CSV_PATH, help="CSV destination for benchmark results")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS_TO_RUN,
                        help="One or more datasets to evaluate (e.g., --datasets GTSRB or --datasets BTSD CTSD). Options: GTSRB, BTSD, CTSD")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader CPU worker thread count")
    parser.add_argument("--use-early-exit", action="store_true", default=True, help="Enable early exit head evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE TRAFFIC SIGN ROBUSTNESS BENCHMARK")
    print("=" * 80)

    set_seed(config.RANDOM_SEED)
    config.print_device_info()

    device = config.DEVICE
    pin_mem = config.PIN_MEMORY
    num_workers = args.num_workers

    test_splits: Dict[str, Tuple[List[str], List[int]]] = {}
    num_classes_dict: Dict[str, int] = {}

    selected_datasets = [d.upper() for d in args.datasets]
    for d in selected_datasets:
        if d not in config.DATASET_SPECS:
            raise ValueError(f"Unknown dataset '{d}'. Supported options: {list(config.DATASET_SPECS.keys())}")

    print(f"\n[INFO] Loading evaluation splits from: '{args.data_root}' for datasets: {selected_datasets}...")
    for dataset_name in selected_datasets:
        _, _, te_paths, te_labels, num_classes = load_real_dataset_splits(
            dataset_name=dataset_name,
            data_root=args.data_root
        )
        test_splits[dataset_name] = (te_paths, te_labels)
        num_classes_dict[dataset_name] = num_classes
        print(f"  • {dataset_name:<5}: Test Samples = {len(te_paths)} | Classes = {num_classes}")

    # Initialize model
    model = MultiTaskRobustSignNet(num_classes_dict=num_classes_dict).to(device)

    # Restore checkpoint
    if os.path.isfile(args.checkpoint_path):
        load_checkpoint(args.checkpoint_path, model, device=device)
    else:
        print(f"[WARNING] Checkpoint '{args.checkpoint_path}' not found. Benchmarking initialized model.")

    # SNAPSHOT PRISTINE WEIGHTS: Used to reset parameters before/after each TENT trial
    clean_state_snapshot = snapshot_model_state(model)
    print("[INFO] Captured clean model weights snapshot to prevent test-time parameter leakage.")

    master_results: List[Dict[str, Any]] = []
    total_evals = len(selected_datasets) * (1 + len(config.NOISE_TYPES) * len(config.SEVERITY_LEVELS))
    completed_evals = 0

    benchmark_start_time = time.time()
    last_log_time = benchmark_start_time

    for task_name in selected_datasets:
        print("\n" + "-" * 75)
        print(f"EVALUATING BENCHMARK FOR DATASET: {task_name}")
        print("-" * 75)

        te_paths, te_labels = test_splits[task_name]

        # ----------------------------------------------------------------------
        # 1. Clean Baseline Evaluation
        # ----------------------------------------------------------------------
        restore_model_state(model, clean_state_snapshot, device)
        clean_ds = UnifiedTrafficSignDataset(
            file_paths=te_paths,
            labels=te_labels,
            noise_type=None,
            image_size=config.IMAGE_SIZE,
            is_train=False
        )
        clean_loader = DataLoader(
            clean_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem
        )

        acc, f1_m, f1_w, lat, _, _ = evaluate_framework(
            model=model,
            dataloader=clean_loader,
            device=device,
            task_name=task_name,
            use_early_exit=args.use_early_exit
        )

        print(
            f"[{task_name}] Clean Baseline -> Acc: {acc * 100:.2f}% | "
            f"Macro-F1: {f1_m:.4f} | Weighted-F1: {f1_w:.4f} | Latency: {lat:.2f} ms/frame"
        )

        master_results.append({
            "Dataset": task_name,
            "Noise_Type": "Baseline (Clean)",
            "Severity": 0,
            "Testing_Accuracy": round(acc, 4),
            "F1_Macro": round(f1_m, 4),
            "F1_Weighted": round(f1_w, 4),
            "Latency_ms": round(lat, 2),
            "Adapted_TENT": False
        })
        completed_evals += 1

        # ----------------------------------------------------------------------
        # 2. Corrupted Benchmark Suite (14 Corruptions x 5 Severities)
        # ----------------------------------------------------------------------
        for noise in config.NOISE_TYPES:
            for sev in config.SEVERITY_LEVELS:
                # IMPORTANT: Reset to clean snapshot before every evaluation trial to prevent parameter leakage
                restore_model_state(model, clean_state_snapshot, device)

                corr_ds = UnifiedTrafficSignDataset(
                    file_paths=te_paths,
                    labels=te_labels,
                    noise_type=noise,
                    severity=sev,
                    image_size=config.IMAGE_SIZE,
                    is_train=False
                )
                corr_loader = DataLoader(
                    corr_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_mem
                )

                adapted_tent = False
                if sev >= config.TENT_SEVERITY_THRESHOLD:
                    apply_tent_adaptation(
                        model=model,
                        corrupted_loader=corr_loader,
                        device=device,
                        task_name=task_name,
                        steps=config.TENT_STEPS,
                        lr=config.TENT_LR
                    )
                    adapted_tent = True

                acc, f1_m, f1_w, lat, _, _ = evaluate_framework(
                    model=model,
                    dataloader=corr_loader,
                    device=device,
                    task_name=task_name,
                    use_early_exit=args.use_early_exit
                )

                master_results.append({
                    "Dataset": task_name,
                    "Noise_Type": noise,
                    "Severity": sev,
                    "Testing_Accuracy": round(acc, 4),
                    "F1_Macro": round(f1_m, 4),
                    "F1_Weighted": round(f1_w, 4),
                    "Latency_ms": round(lat, 2),
                    "Adapted_TENT": adapted_tent
                })

                # Always restore model state immediately after evaluation
                restore_model_state(model, clean_state_snapshot, device)

                completed_evals += 1
                curr_time = time.time()

                if curr_time - last_log_time >= 15 or completed_evals == total_evals:
                    elapsed = curr_time - benchmark_start_time
                    progress_pct = (completed_evals / total_evals) * 100.0
                    eta_sec = (elapsed / completed_evals) * (total_evals - completed_evals)
                    print(
                        f"[PROGRESS] {completed_evals:03d}/{total_evals:03d} ({progress_pct:.1f}%) | "
                        f"Task: {task_name:<5} | Noise: {noise:<17} | Sev: {sev} | "
                        f"Acc: {acc * 100:.1f}% | Elapsed: {elapsed / 60:.1f}m | ETA: {eta_sec / 60:.1f}m"
                    )
                    last_log_time = curr_time

    # Export results to CSV
    results_df = pd.DataFrame(master_results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n[SUCCESS] Robustness evaluation results exported to: '{args.output_csv}'.")

    # Final Solution Quality Verdict
    clean_rows = results_df[results_df["Severity"] == 0]
    noisy_rows = results_df[results_df["Severity"] > 0]

    mean_clean_acc = clean_rows["Testing_Accuracy"].mean()
    mean_noisy_acc = noisy_rows["Testing_Accuracy"].mean()
    mean_latency = results_df["Latency_ms"].mean()

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK PERFORMANCE METRICS & SOLUTION QUALITY VERDICT")
    print("=" * 80)
    print(f"• Datasets Evaluated  : {', '.join(config.DATASETS_TO_RUN)}")
    print(f"• Total Test Trials   : {len(results_df)}")
    print(f"• Mean Clean Accuracy : {mean_clean_acc * 100:.2f}%")
    print(f"• Mean Noisy Accuracy : {mean_noisy_acc * 100:.2f}%")
    print(f"• Mean Inference Time : {mean_latency:.2f} ms/frame")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
