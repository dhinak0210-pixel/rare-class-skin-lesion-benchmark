"""Orchestrator for the scarcity-aware synthetic augmentation experimental benchmark.

Resume-safe, idempotent, and disconnect-proof. Flushes results/runs.csv to disk
after every single evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import ExperimentConfig, cfg, ensure_dirs
from src.gan import train_and_sample_gan
from src.scarcity import (
    HybridAugmentedDataset,
    SplitDataset,
    generate_classical_k,
    get_scarcity_subset,
)
from src.signals import SignalFeatureExtractor, compute_all_signals
from src.split import prepare_dataset_splits
from src.train_clf import get_synthetic_confidence, train_classifier


def load_or_init_runs_df(csv_path: str) -> pd.DataFrame:
    """Load existing runs.csv or initialize with proper schema."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure all columns exist
        for col in cfg.runs_csv_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df
    return pd.DataFrame(columns=list(cfg.runs_csv_columns))


def is_run_completed(
    df: pd.DataFrame,
    dataset: str,
    rare_class_id: int,
    N: int,
    method: str,
    K: int,
    seed: int,
) -> bool:
    """Check if the experiment run has already been recorded in runs.csv."""
    if len(df) == 0:
        return False

    matches = df[
        (df["dataset"] == dataset)
        & (df["rare_class_id"] == rare_class_id)
        & (df["N"] == N)
        & (df["method"] == method)
        & (df["K"] == K)
        & (df["seed"] == seed)
    ]
    return len(matches) > 0


def append_and_flush_run(
    df: pd.DataFrame,
    row_dict: Dict[str, object],
    csv_path: str,
) -> pd.DataFrame:
    """Append a single experiment record and immediately flush to disk."""
    new_row_df = pd.DataFrame([row_dict])
    df = pd.concat([df, new_row_df], ignore_index=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def run_benchmark(
    dataset_name: str = "dermamnist",
    k_generated: int = 200,
    seeds: Optional[Tuple[int, ...]] = None,
    scarcity_levels: Optional[Tuple[int, ...]] = None,
    device_name: Optional[str] = None,
    fast_dev_run: bool = False,
) -> None:
    """Run complete experimental grid with idempotent resume safety."""
    ensure_dirs()
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"============================================================")
    print(f"Starting Benchmark: {dataset_name.upper()} on {device.type.upper()}")
    print(f"Fast Dev Run: {fast_dev_run} | K: {k_generated}")
    print(f"============================================================")

    # 1. Ensure splits exist
    split_dir = Path(cfg.splits_dir) / dataset_name
    train_csv = split_dir / "train_split.csv"
    val_csv = split_dir / "val_split.csv"
    test_csv = split_dir / "test_split.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        print(f"Splits not found for {dataset_name}. Generating splits...")
        train_df, val_df, test_df, rare_class_id = prepare_dataset_splits(
            dataset_name=dataset_name,
            data_dir=cfg.data_dir,
            output_dir=cfg.splits_dir,
        )
    else:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        train_counts = train_df["label"].value_counts()
        rare_class_id = int(train_counts.idxmin())

    num_classes = len(train_df["label"].unique())
    print(f"Dataset Loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    print(f"Classes: {num_classes} | Rare Class ID: {rare_class_id}")

    val_dataset = SplitDataset(val_df)
    test_dataset = SplitDataset(test_df)

    # Pre-instantiate frozen feature extractor for fast signal computation
    extractor = SignalFeatureExtractor(device)

    active_seeds = seeds if seeds is not None else cfg.seeds
    active_scarcity = scarcity_levels if scarcity_levels is not None else cfg.scarcity_levels

    runs_df = load_or_init_runs_df(cfg.results_csv_path)

    # -------------------------------------------------------------
    # Phase A: Full Data Baseline (N = -1, method = 'full_data', K = 0)
    # -------------------------------------------------------------
    for seed in active_seeds:
        if is_run_completed(runs_df, dataset_name, rare_class_id, -1, "full_data", 0, seed):
            print(f"[SKIP] Completed: {dataset_name} | full_data | seed {seed}")
            continue

        print(f"\n[RUN] {dataset_name} | method: full_data | N: FULL | seed: {seed}")
        train_ds = SplitDataset(train_df)
        val_metrics, test_metrics, _ = train_classifier(
            train_dataset=train_ds,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            rare_class_id=rare_class_id,
            num_classes=num_classes,
            seed=seed,
            epochs=cfg.clf_epochs,
            lr=cfg.clf_lr,
            batch_size=cfg.batch_size,
            device=device,
            fast_dev_run=fast_dev_run,
        )

        row = {
            "dataset": dataset_name,
            "rare_class_id": rare_class_id,
            "N": -1,
            "method": "full_data",
            "K": 0,
            "seed": seed,
            "val_rare_recall": val_metrics["rare_recall"],
            "val_rare_f1": val_metrics["rare_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_bal_acc": val_metrics["bal_acc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "test_rare_f1": test_metrics["rare_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "signal_fid": np.nan,
            "signal_kid": np.nan,
            "signal_diversity": np.nan,
            "signal_nn_copy_dist": np.nan,
            "signal_clf_conf": np.nan,
        }
        runs_df = append_and_flush_run(runs_df, row, cfg.results_csv_path)
        print(f"  -> val_rare_f1: {val_metrics['rare_f1']:.4f} | val_macro_f1: {val_metrics['macro_f1']:.4f}")

    # -------------------------------------------------------------
    # Phase B: Scarcity Grid across N in (10, 20, 50, 100) and Seeds
    # -------------------------------------------------------------
    actual_k = min(k_generated, 8) if fast_dev_run else k_generated

    for N in active_scarcity:
        for seed in active_seeds:
            sampled_rare_df, maj_df = get_scarcity_subset(
                train_df=train_df,
                rare_class_id=rare_class_id,
                N=N,
                seed=seed,
            )
            real_rare_paths = sampled_rare_df["image_path"].tolist()

            # 1. Method: none (Baseline without augmentation)
            if is_run_completed(runs_df, dataset_name, rare_class_id, N, "none", 0, seed):
                print(f"[SKIP] Completed: {dataset_name} | N={N} | none | seed {seed}")
            else:
                print(f"\n[RUN] {dataset_name} | N={N} | method: none | seed: {seed}")
                train_ds = HybridAugmentedDataset(
                    majority_df=maj_df,
                    rare_df=sampled_rare_df,
                    augmented_paths=None,
                    rare_class_id=rare_class_id,
                )
                val_metrics, test_metrics, _ = train_classifier(
                    train_dataset=train_ds,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    rare_class_id=rare_class_id,
                    num_classes=num_classes,
                    seed=seed,
                    epochs=cfg.clf_epochs,
                    lr=cfg.clf_lr,
                    batch_size=cfg.batch_size,
                    device=device,
                    fast_dev_run=fast_dev_run,
                )
                row = {
                    "dataset": dataset_name,
                    "rare_class_id": rare_class_id,
                    "N": N,
                    "method": "none",
                    "K": 0,
                    "seed": seed,
                    "val_rare_recall": val_metrics["rare_recall"],
                    "val_rare_f1": val_metrics["rare_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_bal_acc": val_metrics["bal_acc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "test_rare_f1": test_metrics["rare_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "signal_fid": np.nan,
                    "signal_kid": np.nan,
                    "signal_diversity": np.nan,
                    "signal_nn_copy_dist": np.nan,
                    "signal_clf_conf": np.nan,
                }
                runs_df = append_and_flush_run(runs_df, row, cfg.results_csv_path)
                print(f"  -> val_rare_f1: {val_metrics['rare_f1']:.4f} | val_pr_auc: {val_metrics['pr_auc']:.4f}")

            # 2. Method: classical_k (Count-matched classical control)
            if is_run_completed(runs_df, dataset_name, rare_class_id, N, "classical_k", actual_k, seed):
                print(f"[SKIP] Completed: {dataset_name} | N={N} | classical_k | K={actual_k} | seed {seed}")
            else:
                print(f"\n[RUN] {dataset_name} | N={N} | method: classical_k | K={actual_k} | seed: {seed}")
                save_dir = os.path.join(
                    cfg.generated_dir, dataset_name, f"classical_N{N}_K{actual_k}_seed{seed}"
                )
                aug_paths = generate_classical_k(
                    sampled_rare_df=sampled_rare_df,
                    K=actual_k,
                    seed=seed,
                    save_dir=save_dir,
                )

                # Compute quality / memorization signals
                signals = compute_all_signals(
                    real_image_paths=real_rare_paths,
                    synthetic_image_paths=aug_paths,
                    device=device,
                    extractor=extractor,
                )

                train_ds = HybridAugmentedDataset(
                    majority_df=maj_df,
                    rare_df=sampled_rare_df,
                    augmented_paths=aug_paths,
                    rare_class_id=rare_class_id,
                )
                val_metrics, test_metrics, clf_model = train_classifier(
                    train_dataset=train_ds,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    rare_class_id=rare_class_id,
                    num_classes=num_classes,
                    seed=seed,
                    epochs=cfg.clf_epochs,
                    lr=cfg.clf_lr,
                    batch_size=cfg.batch_size,
                    device=device,
                    fast_dev_run=fast_dev_run,
                )
                clf_conf = get_synthetic_confidence(clf_model, save_dir, rare_class_id, device)

                row = {
                    "dataset": dataset_name,
                    "rare_class_id": rare_class_id,
                    "N": N,
                    "method": "classical_k",
                    "K": actual_k,
                    "seed": seed,
                    "val_rare_recall": val_metrics["rare_recall"],
                    "val_rare_f1": val_metrics["rare_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_bal_acc": val_metrics["bal_acc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "test_rare_f1": test_metrics["rare_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "signal_fid": signals["signal_fid"],
                    "signal_kid": signals["signal_kid"],
                    "signal_diversity": signals["signal_diversity"],
                    "signal_nn_copy_dist": signals["signal_nn_copy_dist"],
                    "signal_clf_conf": clf_conf,
                }
                runs_df = append_and_flush_run(runs_df, row, cfg.results_csv_path)
                print(f"  -> val_rare_f1: {val_metrics['rare_f1']:.4f} | KID: {signals['signal_kid']:.6f} | Div: {signals['signal_diversity']:.4f}")

            # 3. Method: gan_k (Few-Shot DCGAN + DiffAugment)
            if is_run_completed(runs_df, dataset_name, rare_class_id, N, "gan_k", actual_k, seed):
                print(f"[SKIP] Completed: {dataset_name} | N={N} | gan_k | K={actual_k} | seed {seed}")
            else:
                print(f"\n[RUN] {dataset_name} | N={N} | method: gan_k | K={actual_k} | seed: {seed}")
                save_dir = os.path.join(
                    cfg.generated_dir, dataset_name, f"gan_N{N}_K{actual_k}_seed{seed}"
                )
                gan_paths = train_and_sample_gan(
                    rare_df=sampled_rare_df,
                    dataset_name=dataset_name,
                    N=N,
                    K=actual_k,
                    seed=seed,
                    output_dir=save_dir,
                    figures_dir=cfg.figures_dir,
                    epochs=cfg.gan_epochs,
                    lr=cfg.gan_lr,
                    betas=cfg.gan_betas,
                    latent_dim=cfg.gan_latent_dim,
                    batch_size=cfg.batch_size,
                    device=device,
                    fast_dev_run=fast_dev_run,
                )

                # Compute quality / memorization signals
                signals = compute_all_signals(
                    real_image_paths=real_rare_paths,
                    synthetic_image_paths=gan_paths,
                    device=device,
                    extractor=extractor,
                )

                train_ds = HybridAugmentedDataset(
                    majority_df=maj_df,
                    rare_df=sampled_rare_df,
                    augmented_paths=gan_paths,
                    rare_class_id=rare_class_id,
                )
                val_metrics, test_metrics, clf_model = train_classifier(
                    train_dataset=train_ds,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    rare_class_id=rare_class_id,
                    num_classes=num_classes,
                    seed=seed,
                    epochs=cfg.clf_epochs,
                    lr=cfg.clf_lr,
                    batch_size=cfg.batch_size,
                    device=device,
                    fast_dev_run=fast_dev_run,
                )
                clf_conf = get_synthetic_confidence(clf_model, save_dir, rare_class_id, device)

                row = {
                    "dataset": dataset_name,
                    "rare_class_id": rare_class_id,
                    "N": N,
                    "method": "gan_k",
                    "K": actual_k,
                    "seed": seed,
                    "val_rare_recall": val_metrics["rare_recall"],
                    "val_rare_f1": val_metrics["rare_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_bal_acc": val_metrics["bal_acc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "test_rare_f1": test_metrics["rare_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "signal_fid": signals["signal_fid"],
                    "signal_kid": signals["signal_kid"],
                    "signal_diversity": signals["signal_diversity"],
                    "signal_nn_copy_dist": signals["signal_nn_copy_dist"],
                    "signal_clf_conf": clf_conf,
                }
                runs_df = append_and_flush_run(runs_df, row, cfg.results_csv_path)
                print(f"  -> val_rare_f1: {val_metrics['rare_f1']:.4f} | KID: {signals['signal_kid']:.6f} | Div: {signals['signal_diversity']:.4f}")

    print(f"\n[DONE] Finished benchmark runs for {dataset_name}. Total rows in runs.csv: {len(runs_df)}")


def main():
    parser = argparse.ArgumentParser(description="Run Scarcity-Aware Synthetic Augmentation Experiments.")
    parser.add_argument("--dataset", type=str, default="dermamnist", help="Dataset: dermamnist or bloodmnist")
    parser.add_argument("--k", type=int, default=200, help="Number of synthetic/classical images")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run rapid CPU smoke test across all configurations")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
        k_generated=args.k,
        device_name=args.device,
        fast_dev_run=args.fast_dev_run,
    )


if __name__ == "__main__":
    main()
