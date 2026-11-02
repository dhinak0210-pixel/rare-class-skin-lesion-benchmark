"""
src/models.py
=============
Model architectures for the Rare-Class Skin Lesion Benchmark:
  1. EfficientNet-B0 (ImageNet-pretrained via timm)
  2. Foundation model linear probes & class-weighted MLP heads
  3. DINOv2 ViT-S/14 partial fine-tuning wrapper
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import timm

class EfficientNetClassifier(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 fine-tuned for 7-class dermatoscopy."""
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 7,
        dropout: float = 0.2
    ):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate feature representation before classifier head."""
        return self.model.forward_features(x)


class MLPHead(nn.Module):
    """Non-linear 2-layer MLP probe for foundation model embeddings."""
    def __init__(
        self,
        in_features: int = 384,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProbeHead(nn.Module):
    """Single linear layer for linear evaluation protocol."""
    def __init__(self, in_features: int = 384, num_classes: int = 7):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DINOv2FineTuner(nn.Module):
    """
    DINOv2 ViT-S/14 partial fine-tuning module:
    Freezes patch embed and earlier blocks, keeping only the last k blocks + head trainable.
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        num_classes: int = 7,
        unfreeze_last_blocks: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        # Load from transformers or torch.hub fallback
        try:
            from transformers import Dinov2Model
            self.backbone = Dinov2Model.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
            self.use_hf = True
        except Exception:
            # Fallback to torch.hub or timm
            self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
            hidden_size = 384
            self.use_hf = False

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last k blocks
        if self.use_hf and hasattr(self.backbone, "encoder"):
            blocks = self.backbone.encoder.layer
            for block in blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
            for block in blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_hf:
            outputs = self.backbone(x)
            cls_token = outputs.last_hidden_state[:, 0]
        else:
            cls_token = self.backbone(x)
        return self.classifier(cls_token)


def build_model(model_config: dict, num_classes: int = 7) -> nn.Module:
    """Factory builder for model instantiations."""
    family = model_config.get("family", "cnn")

    if family == "cnn":
        backbone = model_config.get("backbone", "efficientnet_b0")
        pretrained = model_config.get("pretrained", True)
        dropout = model_config.get("dropout", 0.2)
        return EfficientNetClassifier(backbone=backbone, pretrained=pretrained, num_classes=num_classes, dropout=dropout)

    elif family == "foundation_embed":
        head_type = model_config.get("head_type", "mlp")
        in_dim = model_config.get("embedding_dim", 384)
        if head_type == "mlp":
            hidden_dim = model_config.get("hidden_dim", 128)
            dropout = model_config.get("dropout", 0.3)
            return MLPHead(in_features=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        else:
            return LinearProbeHead(in_features=in_dim, num_classes=num_classes)

    elif family == "foundation_finetune":
        backbone = model_config.get("backbone", "facebook/dinov2-small")
        unfreeze = model_config.get("freeze_blocks_except_last", 3)
        dropout = model_config.get("dropout", 0.2)
        return DINOv2FineTuner(model_name=backbone, num_classes=num_classes, unfreeze_last_blocks=unfreeze, dropout=dropout)

    else:
        raise ValueError(f"Unknown model family: {family}")
