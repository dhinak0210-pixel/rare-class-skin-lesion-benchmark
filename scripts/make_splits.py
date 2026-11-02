#!/usr/bin/env python3
"""
scripts/make_splits.py
======================
Generates deterministic, leak-free Stratified Group K-Fold splits for HAM10000.
Non-negotiable research rules enforced:
  1. Splitting by `lesion_id` exclusively (zero image-level patient leakage).
  2. 5 folds repeated across 3 seeds (42, 1337, 2024) = 15 test evaluations.
  3. Strict inner validation carve-out within the training fold (also grouped by lesion_id).
  4. Saves persistent output to `data/splits.csv`.
  5. Generates publication-grade EDA figures in `results/figures/`.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold

SEEDS = [42, 1337, 2024]
N_FOLDS = 5
TARGET_MINORITY = "df"

def make_splits(
    metadata_path: str = "data/metadata.csv",
    output_splits_path: str = "data/splits.csv",
    figures_dir: str = "results/figures"
):
    print("=" * 70)
    print("LEAK-FREE GROUPED STRATIFIED SPLIT GENERATION (HAM10000)")
    print("=" * 70)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} images across {df['lesion_id'].nunique()} unique lesions.")
    print(f"Minority class '{TARGET_MINORITY}' total count: {(df['dx'] == TARGET_MINORITY).sum()}")

    os.makedirs(os.path.dirname(output_splits_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    records = []

    for seed in SEEDS:
        print(f"\n--- Generating 5-Fold StratifiedGroupKFold (Seed {seed}) ---")
        outer_sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (train_val_idx, test_idx) in enumerate(outer_sgkf.split(df, df["dx"], groups=df["lesion_id"])):
            test_subset = df.iloc[test_idx].copy()
            train_val_subset = df.iloc[train_val_idx].copy()

            # Inner carve-out: 20% of train_val for validation, strictly grouped by lesion_id
            inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for inner_train_idx, inner_val_idx in inner_sgkf.split(
                train_val_subset, train_val_subset["dx"], groups=train_val_subset["lesion_id"]
            ):
                train_subset = train_val_subset.iloc[inner_train_idx].copy()
                val_subset = train_val_subset.iloc[inner_val_idx].copy()
                break  # First fold chosen as deterministic validation split

            # Leakage verification assertions
            train_lesions = set(train_subset["lesion_id"])
            val_lesions = set(val_subset["lesion_id"])
            test_lesions = set(test_subset["lesion_id"])

            assert len(train_lesions & test_lesions) == 0, f"LEAKAGE DETECTED between Train and Test in seed {seed} fold {fold}!"
            assert len(train_lesions & val_lesions) == 0, f"LEAKAGE DETECTED between Train and Val in seed {seed} fold {fold}!"
            assert len(val_lesions & test_lesions) == 0, f"LEAKAGE DETECTED between Val and Test in seed {seed} fold {fold}!"

            # Record splits
            for idx, row in train_subset.iterrows():
                records.append({
                    "seed": seed,
                    "fold": fold,
                    "image_id": row["image_id"],
                    "lesion_id": row["lesion_id"],
                    "dx": row["dx"],
                    "split": "train"
                })
            for idx, row in val_subset.iterrows():
                records.append({
                    "seed": seed,
                    "fold": fold,
                    "image_id": row["image_id"],
                    "lesion_id": row["lesion_id"],
                    "dx": row["dx"],
                    "split": "val"
                })
            for idx, row in test_subset.iterrows():
                records.append({
                    "seed": seed,
                    "fold": fold,
                    "image_id": row["image_id"],
                    "lesion_id": row["lesion_id"],
                    "dx": row["dx"],
                    "split": "test"
                })

            df_train_count = (train_subset["dx"] == TARGET_MINORITY).sum()
            df_val_count = (val_subset["dx"] == TARGET_MINORITY).sum()
            df_test_count = (test_subset["dx"] == TARGET_MINORITY).sum()

            print(
                f"  Fold {fold}: Train={len(train_subset)} ({TARGET_MINORITY}={df_train_count}), "
                f"Val={len(val_subset)} ({TARGET_MINORITY}={df_val_count}), "
                f"Test={len(test_subset)} ({TARGET_MINORITY}={df_test_count}) [OK - Disjoint]"
            )

    splits_df = pd.DataFrame(records)
    splits_df.to_csv(output_splits_path, index=False)
    print(f"\n[SUCCESS] Deterministic splits written to: {output_splits_path}")
    print(f"Total split-indexed records: {len(splits_df)} (3 seeds x 5 folds x 10,015 images)")

    # Generate EDA Figures
    generate_eda_figures(df, splits_df, figures_dir)


def generate_eda_figures(df: pd.DataFrame, splits_df: pd.DataFrame, figures_dir: str):
    print("\n--- Generating Exploratory Data Analysis (EDA) Figures ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Overall Class Distribution
    plt.figure(figsize=(9, 5))
    class_order = df["dx"].value_counts().index
    palette = sns.color_palette("muted", len(class_order))
    ax = sns.countplot(data=df, x="dx", order=class_order, palette=palette)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, xytext=(0, 4),
                    textcoords='offset points')
    plt.title("HAM10000 Class Frequency Imbalance (df = minority target, N=115)")
    plt.xlabel("Diagnostic Category (dx)")
    plt.ylabel("Number of Dermoscopic Images")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "eda_class_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Images Per Lesion Distribution
    images_per_lesion = df.groupby("lesion_id").size()
    plt.figure(figsize=(8, 4.5))
    sns.histplot(images_per_lesion, discrete=True, color="#2b5c8f", shrink=0.8)
    plt.title("Distribution of Images per Patient Lesion (Demonstrating Leakage Risk)")
    plt.xlabel("Number of Images Associated with Same Lesion ID")
    plt.ylabel("Number of Unique Lesions")
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "eda_lesion_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Stratified Split Balance Across Folds (Seed 42)
    s42_splits = splits_df[splits_df["seed"] == 42]
    fold_balance = s42_splits.groupby(["fold", "split", "dx"]).size().reset_index(name="count")
    df_minority = fold_balance[fold_balance["dx"] == TARGET_MINORITY]

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_minority, x="fold", y="count", hue="split", palette="Set2")
    plt.title(f"Target Minority ('{TARGET_MINORITY}') Balance Across Folds (Seed 42)")
    plt.xlabel("Fold Index (0 to 4)")
    plt.ylabel(f"Number of '{TARGET_MINORITY}' Samples")
    plt.legend(title="Split")
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "eda_splits_balance.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    print("[SUCCESS] All EDA figures successfully generated.")


if __name__ == "__main__":
    make_splits()
