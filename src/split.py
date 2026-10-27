"""Data splitting and leakage prevention module.

Supports MedMNIST (DermaMNIST, BloodMNIST) and raw HAM10000 metadata CSV.
Guarantees zero data leakage across patient lesion IDs (when present) or
deterministic stratified splitting. Generates class distribution visualizations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from src.config import ExperimentConfig, cfg, ensure_dirs


def load_medmnist_data(
    dataset_name: str = "dermamnist",
    data_dir: str = "data",
    size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Load or download MedMNIST dataset and return concatenated images and labels."""
    import medmnist
    from medmnist import INFO

    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset: {dataset_name}. Available: {list(INFO.keys())}")

    info = INFO[dataset_name_lower]
    DataClass = getattr(medmnist, info["python_class"])

    label_names = {int(k): v for k, v in info["label"].items()}

    root = os.path.join(data_dir, "medmnist_raw")
    os.makedirs(root, exist_ok=True)

    # Load official train, val, test subsets
    train_ds = DataClass(split="train", download=True, root=root, size=size)
    val_ds = DataClass(split="val", download=True, root=root, size=size)
    test_ds = DataClass(split="test", download=True, root=root, size=size)

    imgs = np.concatenate([train_ds.imgs, val_ds.imgs, test_ds.imgs], axis=0)
    labels = np.concatenate([train_ds.labels, val_ds.labels, test_ds.labels], axis=0).flatten()

    return imgs, labels, label_names


def group_aware_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform group-aware stratified splitting using StratifiedGroupKFold.

    Ensures zero group leakage across train, val, and test partitions.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    groups = df[group_col].values
    y = df[label_col].values

    # Step 1: Split into Train (approx 70%) and Temp (approx 30%)
    folds = list(sgkf.split(df, y, groups))
    # Combine first 3 or 4 folds for train (~70-80%) or use 2 folds for temp
    # To get ~70/15/15: split 5 folds into 3 folds train (60%), or use 7-fold:
    # More precise approach:
    sgkf_10 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds_10 = list(sgkf_10.split(df, y, groups))
    # 7 folds train (70%), 1.5 folds val (15%), 1.5 folds test (15%) -> folds 0-6 train (70%), 7-8 val (20%), 9 test (10%)
    # Let's do a 2-stage split:
    test_val_ratio = val_ratio + test_ratio  # 0.30
    n_splits_stage1 = round(1.0 / test_val_ratio)  # ~3 or 4

    train_indices = []
    temp_indices = []
    for train_idx, temp_idx in sgkf.split(df, y, groups):
        # Pick the fold where temp is closest to 30%
        if abs(len(temp_idx) / len(df) - 0.30) < 0.1:
            train_indices = train_idx
            temp_indices = temp_idx
            break
    if len(train_indices) == 0:
        train_indices, temp_indices = folds[0]

    train_df = df.iloc[train_indices].copy()
    temp_df = df.iloc[temp_indices].copy()

    # Step 2: Split Temp 50/50 into Val and Test
    sgkf_temp = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    temp_groups = temp_df[group_col].values
    temp_y = temp_df[label_col].values
    val_idx_rel, test_idx_rel = next(sgkf_temp.split(temp_df, temp_y, temp_groups))

    val_df = temp_df.iloc[val_idx_rel].copy()
    test_df = temp_df.iloc[test_idx_rel].copy()

    # Leakage assertions
    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])

    assert len(train_groups & val_groups) == 0, "Data leakage detected: train & val groups overlap!"
    assert len(train_groups & test_groups) == 0, "Data leakage detected: train & test groups overlap!"
    assert len(val_groups & test_groups) == 0, "Data leakage detected: val & test groups overlap!"

    return train_df, val_df, test_df


def stratified_split_standard(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standard stratified train/val/test split when group labels are absent."""
    sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(sss_1.split(df, df[label_col]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Split temp evenly into val and test
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
    val_idx, test_idx = next(sss_2.split(temp_df, temp_df[label_col]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    label_names: Optional[Dict[int, str]],
    output_path: str,
) -> None:
    """Generate and save publication-quality class distribution bar plot."""
    classes = sorted(train_df["label"].unique())
    class_labels = [label_names.get(c, f"Class {c}") if label_names else f"Class {c}" for c in classes]

    train_counts = [int((train_df["label"] == c).sum()) for c in classes]
    val_counts = [int((val_df["label"] == c).sum()) for c in classes]
    test_counts = [int((test_df["label"] == c).sum()) for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.bar(x - width, train_counts, width, label=f"Train ({len(train_df)})", color="#1f77b4", alpha=0.9)
    ax.bar(x, val_counts, width, label=f"Val ({len(val_df)})", color="#ff7f0e", alpha=0.9)
    ax.bar(x + width, test_counts, width, label=f"Test ({len(test_df)})", color="#2ca02c", alpha=0.9)

    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title(f"Class Distribution Across Splits — {dataset_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=25, ha="right", fontsize=10)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def prepare_dataset_splits(
    dataset_name: str = "dermamnist",
    data_dir: str = "data",
    output_dir: str = "data/splits",
    split_seed: int = 42,
    rare_class_id: Optional[int] = None,
    ham10000_csv: Optional[str] = None,
    ham10000_img_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Build leak-free stratified splits and save CSV files."""
    ensure_dirs()
    target_split_dir = Path(output_dir) / dataset_name
    target_split_dir.mkdir(parents=True, exist_ok=True)

    img_storage_dir = Path(data_dir) / "images" / dataset_name
    img_storage_dir.mkdir(parents=True, exist_ok=True)

    label_names: Dict[int, str] = {}

    if ham10000_csv and os.path.exists(ham10000_csv):
        df = pd.read_csv(ham10000_csv)
        required_cols = {"image_id", "dx"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"ham10000_csv must contain {required_cols}")

        # Map string diagnosis to int
        dx_unique = sorted(df["dx"].unique())
        dx_map = {dx: i for i, dx in enumerate(dx_unique)}
        df["label"] = df["dx"].map(dx_map)
        label_names = {i: dx for dx, i in dx_map.items()}

        if "lesion_id" in df.columns:
            train_df, val_df, test_df = group_aware_stratified_split(
                df, group_col="lesion_id", label_col="label", seed=split_seed
            )
        else:
            train_df, val_df, test_df = stratified_split_standard(
                df, label_col="label", seed=split_seed
            )

        # Build image path
        if ham10000_img_dir:
            for d in [train_df, val_df, test_df]:
                d["image_path"] = d["image_id"].apply(
                    lambda x: os.path.join(ham10000_img_dir, f"{x}.jpg")
                    if not str(x).endswith((".jpg", ".png"))
                    else os.path.join(ham10000_img_dir, str(x))
                )
    else:
        # Load MedMNIST (DermaMNIST, BloodMNIST, etc.)
        imgs, labels, label_names = load_medmnist_data(
            dataset_name=dataset_name, data_dir=data_dir, size=cfg.image_size
        )

        # Save images to disk for uniform dataset loading
        records = []
        for idx in range(len(imgs)):
            img_arr = imgs[idx]
            lbl = int(labels[idx])
            img_filename = f"{dataset_name}_{idx:06d}.png"
            img_path = str(img_storage_dir / img_filename)

            # Only write image if not already written
            if not os.path.exists(img_path):
                if img_arr.ndim == 2:  # Grayscale
                    pil_img = Image.fromarray(img_arr, mode="L")
                elif img_arr.shape[-1] == 1:
                    pil_img = Image.fromarray(img_arr.squeeze(-1), mode="L")
                else:  # RGB
                    pil_img = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
                pil_img.save(img_path)

            records.append({
                "image_id": f"{dataset_name}_{idx:06d}",
                "image_path": img_path,
                "label": lbl,
            })

        df = pd.DataFrame(records)

        train_df, val_df, test_df = stratified_split_standard(
            df, label_col="label", seed=split_seed
        )

    # Determine rarest class if not specified
    if rare_class_id is None:
        train_counts = train_df["label"].value_counts()
        rare_class_id = int(train_counts.idxmin())

    # Save split CSVs
    train_csv = target_split_dir / "train_split.csv"
    val_csv = target_split_dir / "val_split.csv"
    test_csv = target_split_dir / "test_split.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save class distribution figure
    fig_path = Path(cfg.figures_dir) / f"{dataset_name}_class_dist.png"
    plot_class_distribution(train_df, val_df, test_df, dataset_name, label_names, str(fig_path))

    print(f"[{dataset_name}] Splits generated successfully:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Rarest Class ID: {rare_class_id} ('{label_names.get(rare_class_id, rare_class_id)}')")
    print(f"  Saved to: {target_split_dir}")
    print(f"  Distribution plot: {fig_path}")

    return train_df, val_df, test_df, rare_class_id


def main():
    parser = argparse.ArgumentParser(description="Create deterministic leak-free data splits.")
    parser.add_argument("--dataset", type=str, default="dermamnist", help="Dataset name: dermamnist or bloodmnist")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Output directory for split CSVs")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for reproducible splitting")
    parser.add_argument("--rare_class_id", type=int, default=None, help="Optional explicit rare class id")
    parser.add_argument("--ham10000_csv", type=str, default=None, help="Optional raw HAM10000 metadata CSV")
    parser.add_argument("--ham10000_img_dir", type=str, default=None, help="Optional raw HAM10000 image directory")
    args = parser.parse_args()

    prepare_dataset_splits(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        rare_class_id=args.rare_class_id,
        ham10000_csv=args.ham10000_csv,
        ham10000_img_dir=args.ham10000_img_dir,
    )


if __name__ == "__main__":
    main()
