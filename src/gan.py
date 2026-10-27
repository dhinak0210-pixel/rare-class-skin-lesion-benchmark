"""Few-Shot DCGAN with Spectral Normalization and DiffAugment for 64x64 medical image synthesis.

Designed specifically to prevent discriminator overfitting and mode collapse
when training on as few as N=10 rare-class training images.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.config import ExperimentConfig, cfg, ensure_dirs


# ---------------------------------------------------------------------------
# DiffAugment (Zhao et al., Differentiable Augmentation for GAN Training)
# ---------------------------------------------------------------------------
def diff_color(x: torch.Tensor) -> torch.Tensor:
    """Differentiable brightness, saturation, and contrast adjustment."""
    b, c, h, w = x.shape
    # Brightness
    factor_b = (torch.rand(b, 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 0.5
    x = x + factor_b
    # Contrast
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor_c = torch.rand(b, 1, 1, 1, dtype=x.dtype, device=x.device) * 0.5 + 0.75
    x = (x - mean) * factor_c + mean
    return torch.clamp(x, -1.0, 1.0)


def diff_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Differentiable random translation up to 12.5% image size."""
    b, c, h, w = x.shape
    max_delta_x = int(w * ratio)
    max_delta_y = int(h * ratio)

    if max_delta_x == 0 and max_delta_y == 0:
        return x

    delta_x = torch.randint(-max_delta_x, max_delta_x + 1, size=[b], device=x.device)
    delta_y = torch.randint(-max_delta_y, max_delta_y + 1, size=[b], device=x.device)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

    grid[..., 0] = grid[..., 0] + (delta_x.view(b, 1, 1) * (2.0 / w))
    grid[..., 1] = grid[..., 1] + (delta_y.view(b, 1, 1) * (2.0 / h))

    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Apply differentiable translation and color augmentations."""
    x = diff_translation(x)
    x = diff_color(x)
    return x


# ---------------------------------------------------------------------------
# Generator and Discriminator Architectures (64x64)
# ---------------------------------------------------------------------------
class Generator64(nn.Module):
    """DCGAN Generator for 64x64 RGB images from latent vectors."""

    def __init__(self, latent_dim: int = 128, channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Input: (latent_dim, 1, 1) -> (feature_dim * 8, 4, 4)
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
        )
        # 4x4 -> 8x8
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
        )
        # 8x8 -> 16x16
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
        )
        # 16x16 -> 32x32
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
        )
        # 32x32 -> 64x64
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        x = self.init_block(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final_block(x)


class Discriminator64(nn.Module):
    """Spectral-Normalized Discriminator without BatchNorm to preserve Lipschitz continuity."""

    def __init__(self, channels: int = 3, feature_dim: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm

        self.net = nn.Sequential(
            # 64x64 -> 32x32
            sn(nn.Conv2d(channels, feature_dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            sn(nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            sn(nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            sn(nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1x1 scalar logit
            sn(nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(-1, 1)


# ---------------------------------------------------------------------------
# Training Dataset for Rare-Class Images
# ---------------------------------------------------------------------------
class RareClassDataset(Dataset):
    """Loads only the N rare-class real images, normalized to [-1, 1]."""

    def __init__(self, df: pd.DataFrame, image_size: int = 64):
        self.image_paths = df["image_path"].tolist()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Training & Sampling Loop
# ---------------------------------------------------------------------------
def train_and_sample_gan(
    rare_df: pd.DataFrame,
    dataset_name: str,
    N: int,
    K: int,
    seed: int,
    output_dir: str,
    figures_dir: str = "results/figures",
    epochs: int = 300,
    lr: float = 2e-4,
    betas: Tuple[float, float] = (0.5, 0.999),
    latent_dim: int = 128,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    fast_dev_run: bool = False,
) -> List[str]:
    """Train DCGAN+DiffAugment on N rare images and sample K synthetic images."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    actual_epochs = 2 if fast_dev_run else epochs
    actual_k = min(K, 8) if fast_dev_run else K

    ds = RareClassDataset(rare_df, image_size=cfg.image_size)
    actual_batch_size = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=actual_batch_size, shuffle=True, drop_last=False)

    gen = Generator64(latent_dim=latent_dim).to(device)
    disc = Discriminator64().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    # Fixed noise for tracking visual progression
    fixed_noise = torch.randn(min(16, actual_k), latent_dim, device=device)

    for epoch in range(actual_epochs):
        gen.train()
        disc.train()
        for real_imgs in loader:
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = gen(z)

            # DiffAugment on both real and fake
            d_real_aug = diff_augment(real_imgs)
            d_fake_aug = diff_augment(fake_imgs.detach())

            d_real = disc(d_real_aug)
            d_fake = disc(d_fake_aug)

            # Hinge loss for Discriminator
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_loss.backward()
            opt_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad()
            # DiffAugment fake for generator
            g_fake_aug = diff_augment(fake_imgs)
            g_pred = disc(g_fake_aug)
            g_loss = -g_pred.mean()
            g_loss.backward()
            opt_g.step()

    # Generate visual comparison grid (Real vs Synthetic)
    gen.eval()
    with torch.no_grad():
        syn_sample = gen(fixed_noise)
        # Denormalize [-1, 1] -> [0, 1]
        syn_sample = (syn_sample + 1.0) / 2.0

    # Pick up to 16 real images
    real_sample_tensors = []
    for i in range(min(16, len(ds))):
        real_sample_tensors.append((ds[i] + 1.0) / 2.0)
    real_sample = torch.stack(real_sample_tensors).to(device)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    comparison_fig_path = os.path.join(
        figures_dir, f"gan_samples_{dataset_name}_N{N}_seed{seed}.png"
    )

    n_display = min(16, syn_sample.size(0), real_sample.size(0))
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200)
    real_grid = make_grid(real_sample[:n_display], nrow=n_display, padding=2, normalize=False)
    syn_grid = make_grid(syn_sample[:n_display], nrow=n_display, padding=2, normalize=False)

    axes[0].imshow(real_grid.cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title(f"Real Rare Samples (N={N})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(syn_grid.cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1].set_title(f"DCGAN+DiffAugment Synthetic Samples (Seed={seed})", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(comparison_fig_path)
    plt.close()

    # Generate K synthetic images and save to disk as uint8 PNGs
    os.makedirs(output_dir, exist_ok=True)
    generated_paths: List[str] = []

    batch_gen = 64
    count = 0
    with torch.no_grad():
        while count < actual_k:
            cur_bs = min(batch_gen, actual_k - count)
            z = torch.randn(cur_bs, latent_dim, device=device)
            fake_batch = gen(z)
            fake_batch = ((fake_batch + 1.0) / 2.0).clamp(0.0, 1.0)

            for i in range(cur_bs):
                img_tensor = fake_batch[i]
                img_arr = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_arr)
                out_path = os.path.join(output_dir, f"gan_syn_{count:05d}.png")
                pil_img.save(out_path)
                generated_paths.append(out_path)
                count += 1

    return generated_paths
