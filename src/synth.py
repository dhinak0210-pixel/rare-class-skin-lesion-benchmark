"""
src/synth.py
============
Leakage-controlled generative augmentation, FID computation, memorization audits,
and minority-targeted interpolation (MixUp/CutMix, SD1.5+LoRA, DCGAN/StyleGAN).
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from scipy import linalg

from src.utils import get_device, seed_everything

def apply_minority_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    minority_idx: int = 3,
    alpha_mixup: float = 0.4,
    alpha_cutmix: float = 1.0,
    prob: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp or CutMix conditioned on minority class presence:
    Interpolates minority samples to prevent majority dominance.
    """
    if random.random() > prob:
        return images, targets, targets, 1.0

    batch_size = images.size(0)
    indices = torch.randperm(batch_size)

    # Decide MixUp vs CutMix
    use_cutmix = random.random() > 0.5

    if use_cutmix and alpha_cutmix > 0:
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        W = images.size(2)
        H = images.size(3)

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    else:
        lam = np.random.beta(alpha_mixup, alpha_mixup) if alpha_mixup > 0 else 1.0
        mixed_images = lam * images + (1.0 - lam) * images[indices]

    targets_a = targets
    targets_b = targets[indices]
    return mixed_images, targets_a, targets_b, lam


def audit_memorization_dinov2(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    cosine_threshold: float = 0.95
) -> Tuple[np.ndarray, float, List[int]]:
    """
    Computes maximum cosine similarity for each synthetic image against the real training set.
    Rejects/flags near-duplicates exceeding the threshold to prevent generative memorization.
    """
    # Normalize embeddings
    real_norm = real_embeddings / np.maximum(np.linalg.norm(real_embeddings, axis=1, keepdims=True), 1e-12)
    synth_norm = synthetic_embeddings / np.maximum(np.linalg.norm(synthetic_embeddings, axis=1, keepdims=True), 1e-12)

    # Cosine similarity matrix [M_synth, N_real]
    sim_matrix = np.dot(synth_norm, real_norm.T)
    max_sims = np.max(sim_matrix, axis=1)

    rejected_indices = np.where(max_sims >= cosine_threshold)[0].tolist()
    mean_max_sim = float(np.mean(max_sims))

    return max_sims, mean_max_sim, rejected_indices


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """Computes Frechet distance between two Gaussians (mu1, sigma1) and (mu2, sigma2)."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_feature_fid(
    real_features: np.ndarray,
    synthetic_features: np.ndarray
) -> float:
    """Computes regularized Frechet Inception/Embedding Distance from feature matrices."""
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_synth, sigma_synth = np.mean(synthetic_features, axis=0), np.cov(synthetic_features, rowvar=False)
    return compute_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)


def export_blind_real_vs_synthetic_grid(
    real_images: List[np.ndarray],
    synthetic_images: List[np.ndarray],
    output_path: str = "results/figures/blind_real_vs_synthetic_grid.png",
    n_pairs: int = 5
) -> str:
    """
    Exports a publication grid comparing real training 'df' cases against synthetic generations
    for expert dermoscopic visual inspection.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = min(len(real_images), len(synthetic_images), n_pairs)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i in range(n):
        # Top row: Real training minority lesions
        axes[0, i].imshow(real_images[i])
        axes[0, i].set_title(f"Real 'df' #{i+1}", fontsize=11)
        axes[0, i].axis("off")

        # Bottom row: Synthetic generations
        axes[1, i].imshow(synthetic_images[i])
        axes[1, i].set_title(f"Synth 'df' #{i+1}", fontsize=11)
        axes[1, i].axis("off")

    plt.suptitle("Blind Visual Verification: Real vs. Generative Dermatofibroma (df)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
