"""
src/losses.py
=============
Loss functions for class-imbalanced skin lesion classification:
  1. Plain Cross-Entropy
  2. Balanced Cost-Sensitive Class-Weighted Cross-Entropy
  3. Multiclass Focal Loss (Lin et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import numpy as np

def get_class_weights(
    class_counts: Union[List[int], np.ndarray, torch.Tensor],
    scheme: str = "balanced",
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Computes class weighting vectors:
      'balanced': w_c = N / (C * N_c)
      'inverse': w_c = 1 / N_c (normalized)
      'effective_samples': Cui et al. (2019) Class-Balanced Loss
    """
    counts = np.asarray(class_counts, dtype=np.float32)
    n_samples = np.sum(counts)
    n_classes = len(counts)

    if scheme == "balanced":
        weights = n_samples / (n_classes * np.maximum(counts, 1.0))
    elif scheme == "inverse":
        weights = 1.0 / np.maximum(counts, 1.0)
        weights = weights / np.sum(weights) * n_classes
    elif scheme == "effective_samples":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * n_classes
    else:
        weights = np.ones(n_classes, dtype=np.float32)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return weight_tensor


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss:
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights easy well-classified majority samples to concentrate gradient updates
    on rare and hard lesions (e.g. dermatofibroma).
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, List[float]]] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C] unnormalized scores
        targets: [B] class integer labels in [0, C-1]
        """
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Gather target log_p and p
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_config: dict,
    class_counts: Optional[List[int]] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Builds appropriate loss module based on configuration dictionary."""
    loss_type = loss_config.get("type", "cross_entropy")

    if loss_type == "cross_entropy":
        label_smoothing = loss_config.get("label_smoothing", 0.0)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type in ["weighted_ce", "class_weighted"]:
        if class_counts is None:
            raise ValueError("class_counts must be provided for class-weighted loss.")
        scheme = loss_config.get("weight_scheme", "balanced")
        weights = get_class_weights(class_counts, scheme=scheme, device=device)
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == "focal":
        gamma = loss_config.get("focal_gamma", 2.0)
        alpha = None
        if loss_config.get("focal_alpha") is not None:
            alpha = loss_config["focal_alpha"]
        elif loss_config.get("alpha_scheme") == "balanced" and class_counts is not None:
            alpha = get_class_weights(class_counts, scheme="balanced", device=device)
        return FocalLoss(gamma=gamma, alpha=alpha)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
