"""
tests/test_splits.py
====================
Verifies strict isolation, absence of patient lesion leakage, and determinism
across all 15 folds (5 folds x 3 seeds) in data/splits.csv.
"""

import os
import pytest
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLITS_PATH = "data/splits.csv"
METADATA_PATH = "data/metadata.csv"

def test_splits_file_exists_and_complete():
    assert os.path.exists(SPLITS_PATH), f"Splits file not found at {SPLITS_PATH}"
    df_splits = pd.read_csv(SPLITS_PATH)
    # 3 seeds * 5 folds * 10,015 images = 150,225
    expected_rows = 3 * 5 * 10015
    assert len(df_splits) == expected_rows, f"Expected {expected_rows} records, got {len(df_splits)}"

def test_no_lesion_leakage_across_any_fold():
    df_splits = pd.read_csv(SPLITS_PATH)
    seeds = df_splits["seed"].unique()
    folds = df_splits["fold"].unique()

    assert set(seeds) == {42, 1337, 2024}
    assert set(folds) == {0, 1, 2, 3, 4}

    for seed in seeds:
        for fold in folds:
            fold_df = df_splits[(df_splits["seed"] == seed) & (df_splits["fold"] == fold)]
            
            train_lesions = set(fold_df[fold_df["split"] == "train"]["lesion_id"])
            val_lesions = set(fold_df[fold_df["split"] == "val"]["lesion_id"])
            test_lesions = set(fold_df[fold_df["split"] == "test"]["lesion_id"])

            # Critical rule: No lesion in train should appear in test
            train_test_overlap = train_lesions.intersection(test_lesions)
            assert len(train_test_overlap) == 0, (
                f"LEAKAGE DETECTED: Seed {seed}, Fold {fold} has {len(train_test_overlap)} "
                f"overlapping lesions between Train and Test! Examples: {list(train_test_overlap)[:3]}"
            )

            # No lesion in train should appear in val
            train_val_overlap = train_lesions.intersection(val_lesions)
            assert len(train_val_overlap) == 0, (
                f"LEAKAGE DETECTED: Seed {seed}, Fold {fold} has {len(train_val_overlap)} "
                f"overlapping lesions between Train and Val! Examples: {list(train_val_overlap)[:3]}"
            )

            # No lesion in val should appear in test
            val_test_overlap = val_lesions.intersection(test_lesions)
            assert len(val_test_overlap) == 0, (
                f"LEAKAGE DETECTED: Seed {seed}, Fold {fold} has {len(val_test_overlap)} "
                f"overlapping lesions between Val and Test! Examples: {list(val_test_overlap)[:3]}"
            )

def test_minority_representation_in_all_splits():
    df_splits = pd.read_csv(SPLITS_PATH)
    for (seed, fold, split), group in df_splits.groupby(["seed", "fold", "split"]):
        minority_count = (group["dx"] == "df").sum()
        assert minority_count > 0, (
            f"Minority class 'df' is missing in Seed {seed}, Fold {fold}, Split {split}!"
        )

def test_split_determinism():
    # Re-running StratifiedGroupKFold on seed 42 must match data/splits.csv exactly
    df_meta = pd.read_csv(METADATA_PATH)
    df_splits = pd.read_csv(SPLITS_PATH)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    s42_splits = df_splits[df_splits["seed"] == 42]

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(df_meta, df_meta["dx"], groups=df_meta["lesion_id"])):
        expected_test_images = set(df_meta.iloc[test_idx]["image_id"])
        actual_test_images = set(s42_splits[(s42_splits["fold"] == fold) & (s42_splits["split"] == "test")]["image_id"])
        assert expected_test_images == actual_test_images, f"Determinism failure in fold {fold}!"
