"""
src/generative.py
=================
Generative Augmentation Modules (Few-Shot DCGAN and DDPM Diffusion)
and Unified Strategy Dataset Builder for the 5-Strategy Benchmark:
  1. Original data
  2. Conventional aug.
  3. Class-weighted
  4. GAN-based aug.
  5. Diffusion-based aug.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.gan import Generator64, Discriminator64, diff_augment


# ==============================================================================
# 1. Lightweight DDPM (Denoising Diffusion Probabilistic Model)
# ==============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DiffusionBlock(nn.Module):
    """Residual convolutional block with timestep injection."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class LightweightUNet(nn.Module):
    """Compact UNet for 64x64 or 128x128 few-shot medical diffusion."""
    def __init__(self, in_channels: int = 3, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.inc = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = DiffusionBlock(base_channels, base_channels * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DiffusionBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = DiffusionBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_block1 = DiffusionBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_block2 = DiffusionBlock(base_channels * 2, base_channels, time_dim)

        self.outc = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep)
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(self.pool1(x2), t_emb)
        m = self.mid(self.pool2(x3), t_emb)

        u1 = self.up1(m)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.up_block1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up_block2(u2, t_emb)

        return self.outc(u2)


class SimpleDiffusion:
    """Linear beta scheduler and sampler for DDPM."""
    def __init__(self, timesteps: int = 300, beta_start: float = 1e-4, beta_end: float = 0.02, device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...]) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0.0

            x = (1.0 / torch.sqrt(alpha)) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)) * pred_noise) + torch.sqrt(beta) * noise
        return torch.clamp(x, -1.0, 1.0)


# ==============================================================================
# 2. Generative Synthesizers: GAN and Diffusion
# ==============================================================================
def train_quick_dcgan(
    rare_images: torch.Tensor,
    num_epochs: int = 25,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu")
) -> Generator64:
    """Trains a few-shot DCGAN on scarce minority class samples."""
    netG = Generator64(latent_dim=128, channels=3, feature_dim=32).to(device)
    netD = Discriminator64(channels=3, feature_dim=32).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    dataset = torch.utils.data.TensorDataset(rare_images)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(rare_images)), shuffle=True, drop_last=False)

    netG.train()
    netD.train()

    for epoch in range(num_epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            b = batch_x.size(0)

            # Update D
            netD.zero_grad()
            real_aug = diff_augment(batch_x)
            d_real = netD(real_aug)
            errD_real = F.relu(1.0 - d_real).mean()

            z = torch.randn(b, 128, device=device)
            fake = netG(z)
            fake_aug = diff_augment(fake.detach())
            d_fake = netD(fake_aug)
            errD_fake = F.relu(1.0 + d_fake).mean()

            errD = errD_real + errD_fake
            errD.backward()
            optD.step()

            # Update G
            netG.zero_grad()
            d_fake_g = netD(diff_augment(fake))
            errG = -d_fake_g.mean()
            errG.backward()
            optG.step()

    return netG


def sample_dcgan(netG: Generator64, num_samples: int, device: torch.device) -> torch.Tensor:
    """Generates synthetic samples from trained DCGAN."""
    netG.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 128, device=device)
        fakes = netG(z)
        # Scale [-1, 1] to [0, 1]
        fakes = (fakes + 1.0) / 2.0
        return torch.clamp(fakes, 0.0, 1.0)


def train_quick_diffusion(
    rare_images: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu")
) -> Tuple[LightweightUNet, SimpleDiffusion]:
    """Trains a fast lightweight DDPM on scarce minority class samples."""
    diff_system = SimpleDiffusion(timesteps=100, device=device)
    model = LightweightUNet(in_channels=3, base_channels=24, time_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scale [0, 1] to [-1, 1]
    norm_images = (rare_images * 2.0) - 1.0
    dataset = torch.utils.data.TensorDataset(norm_images)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(rare_images)), shuffle=True, drop_last=False)

    model.train()
    for epoch in range(num_epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            b = batch_x.size(0)
            t = torch.randint(0, diff_system.timesteps, (b,), device=device).long()
            x_noisy, noise = diff_system.q_sample(batch_x, t)

            optimizer.zero_grad()
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()

    return model, diff_system


def sample_diffusion(model: LightweightUNet, diff_system: SimpleDiffusion, num_samples: int, img_size: int = 64) -> torch.Tensor:
    """Generates synthetic samples from trained DDPM."""
    shape = (num_samples, 3, img_size, img_size)
    fakes = diff_system.p_sample_loop(model, shape)
    # Scale [-1, 1] to [0, 1]
    fakes = (fakes + 1.0) / 2.0
    return torch.clamp(fakes, 0.0, 1.0)


# ==============================================================================
# 3. Hybrid Strategy Dataset
# ==============================================================================
class StrategyAugmentedDataset(Dataset):
    """
    Unified dataset capable of applying:
      - 'Original data': Raw images, NO augmentation (Resize + Norm only)
      - 'Conventional aug.': Flips, rotations, color jitter
      - 'Class-weighted': Conventional aug (loss handles weighting)
      - 'GAN-based aug.': Conventional aug + K synthetic GAN samples
      - 'Diffusion-based aug.': Conventional aug + K synthetic Diffusion samples
    """
    def __init__(
        self,
        real_images: List[np.ndarray],
        real_labels: List[int],
        strategy: str,
        synthetic_tensors: Optional[torch.Tensor] = None,
        synthetic_label: int = 3,
        img_size: int = 128,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        self.strategy = strategy
        self.img_size = img_size
        self.mean = mean
        self.std = std

        # Standard transforms
        self.norm = T.Normalize(mean=mean, std=std)
        self.clean_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            self.norm
        ])

        self.aug_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            self.norm
        ])

        # Base samples
        self.images: List[Union[Image.Image, torch.Tensor]] = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in real_images]
        self.labels: List[int] = list(real_labels)

        # Append synthetic samples if strategy is GAN or Diffusion
        if strategy in ["GAN-based aug.", "Diffusion-based aug."] and synthetic_tensors is not None:
            resize_fn = T.Resize((img_size, img_size))
            for i in range(len(synthetic_tensors)):
                syn_img = synthetic_tensors[i].cpu()
                syn_resized = resize_fn(syn_img)
                syn_norm = self.norm(syn_resized)
                self.images.append(syn_norm)
                self.labels.append(synthetic_label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.images[idx]
        label = self.labels[idx]

        if isinstance(item, torch.Tensor):
            return item, label

        # For PIL image:
        if self.strategy == "Original data":
            return self.clean_transform(item), label
        else:
            return self.aug_transform(item), label

