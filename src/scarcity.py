"""Scarcity sampling, count-matched classical augmentation, and hybrid datasets.

Extracts exactly N rare-class training samples without replacement and generates
count-matched classical augmentations (K images) using deterministic transforms.
Provides the unified HybridAugmentedDataset for classifier training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import ExperimentConfig, cfg


def get_scarcity_subset(
    train_df: pd.DataFrame,
    rare_class_id: int,
    N: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample exactly N rare-class training instances while retaining majority classes.

    Args:
        train_df: Full training dataframe with columns ['image_path', 'label', ...].
        rare_class_id: Class integer index designating the scarce class.
        N: Number of rare-class instances to sample. If N == -1, returns all rare samples.
        seed: Random seed for deterministic subsampling.

    Returns:
        (sampled_rare_df, majority_df)
    """
    rare_df = train_df[train_df["label"] == rare_class_id].copy().reset_index(drop=True)
    majority_df = train_df[train_df["label"] != rare_class_id].copy().reset_index(drop=True)

    if N == -1 or N >= len(rare_df):
        return rare_df, majority_df

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(rare_df), size=N, replace=False)
    sampled_rare_df = rare_df.iloc[chosen_indices].copy().reset_index(drop=True)

    return sampled_rare_df, majority_df


def get_classical_transform(seed: int) -> transforms.Compose:
    """Return standard torchvision classical augmentation pipeline for medical images."""
    torch.manual_seed(seed)
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    ])


def generate_classical_k(
    sampled_rare_df: pd.DataFrame,
    K: int,
    seed: int,
    save_dir: str,
) -> List[str]:
    """Generate exactly K classical augmentations from N sampled rare training images.

    Args:
        sampled_rare_df: Dataframe containing the N real rare samples.
        K: Number of augmented samples to generate.
        seed: Random seed for deterministic augmentation.
        save_dir: Directory where generated PNG images will be saved.

    Returns:
        List of absolute file paths to the K generated PNG images.
    """
    os.makedirs(save_dir, exist_ok=True)
    N = len(sampled_rare_df)
    if N == 0:
        raise ValueError("Cannot generate augmentations from an empty rare-class subset.")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    aug_pipeline = get_classical_transform(seed)

    # Preload the N source images into memory as PIL images
    source_images: List[Image.Image] = []
    for _, row in sampled_rare_df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        if img.size != (cfg.image_size, cfg.image_size):
            img = img.resize((cfg.image_size, cfg.image_size), Image.Resampling.BILINEAR)
        source_images.append(img)

    generated_paths: List[str] = []
    for k_idx in range(K):
        # Deterministically select one of the N source images
        src_idx = int(rng.integers(0, N))
        src_img = source_images[src_idx]

        # Apply augmentation transform
        aug_img = aug_pipeline(src_img)

        # Save to disk
        out_filename = f"classical_aug_{k_idx:05d}.png"
        out_path = os.path.join(save_dir, out_filename)
        aug_img.save(out_path)
        generated_paths.append(out_path)

    return generated_paths


# Standard normalization for ResNet models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

clf_eval_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class HybridAugmentedDataset(Dataset):
    """Combines real majority-class images, N scarce rare-class images, and optional K synthetic/classical images."""

    def __init__(
        self,
        majority_df: pd.DataFrame,
        rare_df: pd.DataFrame,
        augmented_paths: Optional[List[str]] = None,
        rare_class_id: int = 0,
        transform: Optional[transforms.Compose] = None,
    ):
        self.transform = transform if transform is not None else clf_eval_transform
        self.samples: List[Tuple[str, int]] = []

        # Add majority samples
        for _, row in majority_df.iterrows():
            self.samples.append((str(row["image_path"]), int(row["label"])))

        # Add sampled rare samples
        for _, row in rare_df.iterrows():
            self.samples.append((str(row["image_path"]), int(row["label"])))

        # Add augmented / synthetic rare samples
        if augmented_paths is not None:
            for p in augmented_paths:
                self.samples.append((str(p), int(rare_class_id)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


class SplitDataset(Dataset):
    """Dataset for val/test splits with standardized ResNet normalization."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ):
        self.transform = transform if transform is not None else clf_eval_transform
        self.samples: List[Tuple[str, int]] = [
            (str(row["image_path"]), int(row["label"])) for _, row in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


if __name__ == "__main__":
    # Smoke test
    print("Testing src/scarcity.py smoke test...")
    dummy_records = [
        {"image_path": f"/tmp/dummy_{i}.png", "label": 3 if i < 15 else (i % 3)}
        for i in range(50)
    ]
    df = pd.DataFrame(dummy_records)
    rare_df, maj_df = get_scarcity_subset(df, rare_class_id=3, N=10, seed=42)
    assert len(rare_df) == 10, f"Expected 10, got {len(rare_df)}"
    assert len(maj_df) == 35, f"Expected 35, got {len(maj_df)}"
    print("Smoke test passed: get_scarcity_subset works correctly!")
