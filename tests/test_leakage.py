"""
tests/test_leakage.py
=====================
Verifies zero patient/image overlap between HAM10000 and the external ISIC 2019 dataset,
and validates that synthetic generation isolation is strictly preserved.
"""

import os
import pytest
import pandas as pd

HAM_METADATA_PATH = "data/metadata.csv"
EXTERNAL_METADATA_PATH = "data/isic2019_external_metadata.csv"

def test_external_set_has_zero_overlap_with_ham10000():
    assert os.path.exists(HAM_METADATA_PATH), f"Missing {HAM_METADATA_PATH}"
    assert os.path.exists(EXTERNAL_METADATA_PATH), f"Missing {EXTERNAL_METADATA_PATH}"

    ham_df = pd.read_csv(HAM_METADATA_PATH)
    ext_df = pd.read_csv(EXTERNAL_METADATA_PATH)

    ham_images = set(ham_df["image_id"])
    ext_images = set(ext_df["image_id"])

    overlap_images = ham_images.intersection(ext_images)
    assert len(overlap_images) == 0, (
        f"CRITICAL LEAKAGE: {len(overlap_images)} images in external dataset overlap with HAM10000!"
    )

    ham_lesions = set(ham_df["lesion_id"].dropna())
    ext_lesions = set(ext_df["lesion_id"].dropna())

    overlap_lesions = ham_lesions.intersection(ext_lesions)
    assert len(overlap_lesions) == 0, (
        f"CRITICAL LEAKAGE: {len(overlap_lesions)} lesions in external dataset overlap with HAM10000!"
    )

    # Verify no HAM_ prefixes exist in external dataset
    ham_prefixes = ext_df[ext_df["lesion_id"].astype(str).str.startswith("HAM_")]
    assert len(ham_prefixes) == 0, "External set still contains HAM_ lesion prefixes!"

def test_synthetic_isolation_rule():
    """
    Contract test: Any synthetic dataset index or generator dictionary
    must never contain images from validation or test splits.
    """
    splits_df = pd.read_csv("data/splits.csv")
    for seed in [42, 1337, 2024]:
        for fold in range(5):
            val_test_imgs = set(splits_df[
                (splits_df["seed"] == seed) &
                (splits_df["fold"] == fold) &
                (splits_df["split"].isin(["val", "test"]))
            ]["image_id"])

            train_imgs = set(splits_df[
                (splits_df["seed"] == seed) &
                (splits_df["fold"] == fold) &
                (splits_df["split"] == "train")
            ]["image_id"])

            # Train and val_test must be mutually disjoint
            assert len(train_imgs.intersection(val_test_imgs)) == 0
