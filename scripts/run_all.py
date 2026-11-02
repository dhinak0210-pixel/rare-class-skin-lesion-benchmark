#!/usr/bin/env python3
"""
scripts/run_all.py
==================
Autonomous Benchmark Runner & Orchestrator:
  - Supports fine-tuned CNN baselines (B0–B4)
  - Supports pretrained foundation models (P1–P5)
  - Supports generative synthetic augmentation (S1–S2)
  - Resumption-safe: Checks results/logs/runs.csv and skips completed runs.
  - Device-adaptive: Auto-detects CUDA / MPS / CPU and tunes batch sizes and workers.
"""

import os
import sys
import argparse
import pandas as pd
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_device, seed_everything, load_yaml_config
from src.dataset import get_dataloaders
from src.models import build_model
from src.losses import build_loss
from src.train import run_training_fold
from src.embed import extract_and_cache_embeddings, train_cached_embedding_arm
from src.stats import aggregate_benchmark_runs

ARM_CONFIGS = {
    "B0": "configs/b0_plain_ce.yaml",
    "B1": "configs/b1_classical_aug.yaml",
    "B2": "configs/b2_balanced_sampler.yaml",
    "B3": "configs/b3_class_weighted.yaml",
    "B4": "configs/b4_focal_loss.yaml",
    "P1": "configs/p1_dinov2_logreg.yaml",
    "P2": "configs/p2_dinov2_mlp_weighted.yaml",
    "P3": "configs/p3_biomedclip_linear.yaml",
    "P4": "configs/p4_dinov2_finetune.yaml",
    "P5": "configs/p5_best_pretrained_focal.yaml",
    "S1": "configs/s1_mixup_minority.yaml",
    "S2_50": "configs/s2_sd15_lora_50.yaml",
    "S2_100": "configs/s2_sd15_lora_100.yaml",
    "S2_200": "configs/s2_sd15_lora_200.yaml",
    "FINAL_COMBINED": "configs/final_combined.yaml"
}

def is_run_completed(arm_id: str, seed: int, fold: int, log_file: str = "results/logs/runs.csv") -> bool:
    """Checks whether the specific experiment fold already completed and is logged to disk."""
    if not os.path.exists(log_file):
        return False
    try:
        df = pd.read_csv(log_file)
        if "arm_id" not in df.columns or "seed" not in df.columns or "fold" not in df.columns:
            return False
        matches = df[(df["arm_id"] == arm_id) & (df["seed"] == seed) & (df["fold"] == fold)]
        return len(matches) > 0
    except Exception:
        return False


def run_benchmark(
    arms: list,
    seeds: list = [42, 1337, 2024],
    folds: list = [0, 1, 2, 3, 4],
    epochs_override: int = None,
    debug_mode: bool = False
):
    print("=" * 70)
    print("RARE-CLASS SKIN LESION BENCHMARK ORCHESTRATOR")
    print(f"Target Arms: {arms}")
    print(f"Seeds: {seeds} | Folds: {folds} | Epochs Override: {epochs_override} | Debug Mode: {debug_mode}")
    print("=" * 70)

    device = get_device()
    print(f"Active Compute Device: {device.type.upper()}")

    # Foundation embedding caches
    dinov2_cache = None
    biomedclip_cache = None

    for arm in arms:
        if arm not in ARM_CONFIGS:
            print(f"[ERROR] Unknown arm ID: {arm}. Skipping...")
            continue

        config_path = ARM_CONFIGS[arm]
        cfg = load_yaml_config(config_path)
        if epochs_override is not None:
            cfg["training"]["epochs"] = epochs_override

        model_family = cfg.get("model", {}).get("family", "cnn")

        # Handle pre-extracted foundation embeddings (P1, P2, P3, P5)
        if model_family == "foundation_embed":
            backbone = cfg["model"]["backbone"]
            print(f"\n>>> Preparing Embeddings for Arm {arm} ({backbone}) <<<")

            if "dino" in backbone.lower():
                if dinov2_cache is None:
                    embeddings, img_ids, labels = extract_and_cache_embeddings(
                        metadata_path=cfg["protocol"]["metadata_file"],
                        images_dir=cfg["protocol"]["images_dir"],
                        backbone_name=backbone
                    )
                    id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}
                    dinov2_cache = (embeddings, id_to_idx, labels)
                emb_mat, id_map, lbl_arr = dinov2_cache

            elif "biomedclip" in backbone.lower():
                if biomedclip_cache is None:
                    embeddings, img_ids, labels = extract_and_cache_embeddings(
                        metadata_path=cfg["protocol"]["metadata_file"],
                        images_dir=cfg["protocol"]["images_dir"],
                        backbone_name=backbone
                    )
                    id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}
                    biomedclip_cache = (embeddings, id_to_idx, labels)
                emb_mat, id_map, lbl_arr = biomedclip_cache

            for seed in seeds:
                for fold in folds:
                    if is_run_completed(arm, seed, fold):
                        print(f"[SKIP] Arm {arm} | Seed {seed} | Fold {fold} already completed.")
                        continue

                    train_cached_embedding_arm(
                        config=cfg,
                        arm_id=arm,
                        seed=seed,
                        fold=fold,
                        embeddings_matrix=emb_mat,
                        image_id_to_idx=id_map,
                        labels_array=lbl_arr
                    )
            continue

        # Standard end-to-end model training (CNN / P4 Fine-Tuner / S1-S3)
        for seed in seeds:
            for fold in folds:
                if is_run_completed(arm, seed, fold):
                    print(f"[SKIP] Arm {arm} | Seed {seed} | Fold {fold} already completed.")
                    continue

                seed_everything(seed)

                # Get DataLoaders
                train_loader, val_loader, test_loader, class_counts = get_dataloaders(
                    cfg, seed=seed, fold=fold, debug_mode=debug_mode
                )

                # Build Model & Loss
                model = build_model(cfg["model"], num_classes=cfg["protocol"]["num_classes"])
                criterion = build_loss(cfg["loss"], class_counts=class_counts, device=device)

                # Execute Fold
                run_training_fold(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    config=cfg,
                    arm_id=arm,
                    seed=seed,
                    fold=fold
                )

    # Post-run aggregation
    runs_log = "results/logs/runs.csv"
    if os.path.exists(runs_log):
        print("\n>>> Aggregating Benchmark Results Across Completed Evaluations <<<")
        aggregate_benchmark_runs(runs_log)


def main():
    parser = argparse.ArgumentParser(description="Rare-Class Benchmark Orchestrator")
    parser.add_argument("--arms", nargs="+", default=["B0", "B1", "B2", "B3", "B4", "P1", "P2", "P3", "P4", "P5", "S1", "S2_50", "S2_100", "S2_200", "FINAL_COMBINED"], help="Experiment arm IDs to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024], help="Random seeds")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold indices")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch count override")
    parser.add_argument("--debug", action="store_true", help="Run in DermaMNIST CPU debug mode")
    args = parser.parse_args()

    run_benchmark(
        arms=args.arms,
        seeds=args.seeds,
        folds=args.folds,
        epochs_override=args.epochs,
        debug_mode=args.debug
    )


if __name__ == "__main__":
    main()
