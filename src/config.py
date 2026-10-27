"""Configuration file for Scarcity-Aware Synthetic Augmentation experiments.

This module serves as the single source of truth for all hyperparameters,
dataset configurations, training parameters, and metrics logged throughout
the benchmark.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration dataclass."""

    # Dataset & image properties
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 2

    # Reproducibility seeds and experimental grid
    seeds: Tuple[int, ...] = (42, 123, 999)
    scarcity_levels: Tuple[int, ...] = (10, 20, 50, 100)
    k_generated: Tuple[int, ...] = (200, 500)
    methods: Tuple[str, ...] = ("none", "classical_k", "gan_k")

    # Downstream classifier hyperparameters (ResNet18)
    clf_lr: float = 1e-4
    clf_epochs: int = 20
    clf_weight_decay: float = 1e-4

    # Generative model hyperparameters (Few-shot DCGAN + DiffAugment)
    gan_lr: float = 2e-4
    gan_betas: Tuple[float, float] = (0.5, 0.999)
    gan_latent_dim: int = 128
    gan_epochs: int = 300

    # Paths
    data_dir: str = "data"
    splits_dir: str = "data/splits"
    generated_dir: str = "data/generated"
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"
    results_csv_path: str = "results/runs.csv"

    # CSV result schema
    runs_csv_columns: Tuple[str, ...] = (
        "dataset",
        "rare_class_id",
        "N",
        "method",
        "K",
        "seed",
        "val_rare_recall",
        "val_rare_f1",
        "val_macro_f1",
        "val_bal_acc",
        "val_pr_auc",
        "test_rare_f1",
        "test_macro_f1",
        "test_pr_auc",
        "signal_fid",
        "signal_kid",
        "signal_diversity",
        "signal_nn_copy_dist",
        "signal_clf_conf",
    )


# Default global configuration instance
cfg = ExperimentConfig()


def ensure_dirs(config: ExperimentConfig = cfg) -> None:
    """Create all necessary output directories if they do not exist."""
    for path_str in [
        config.data_dir,
        config.splits_dir,
        config.generated_dir,
        config.results_dir,
        config.figures_dir,
        config.tables_dir,
    ]:
        Path(path_str).mkdir(parents=True, exist_ok=True)
