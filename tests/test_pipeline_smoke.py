"""
tests/test_pipeline_smoke.py
============================
End-to-end CPU smoke tests for dataset loaders, models, losses, Grad-CAM,
and the full single-epoch training-to-evaluation pipeline.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils import get_device, seed_everything, load_yaml_config
from src.models import EfficientNetClassifier, MLPHead, LinearProbeHead, build_model
from src.losses import FocalLoss, build_loss, get_class_weights
from src.dataset import get_dataloaders, get_transforms, HAM10000Dataset
from src.train import train_epoch, evaluate_epoch
from src.eval import compute_metrics, TemperatureScaler
from src.gradcam import GradCAM, overlay_heatmap_on_image
from src.synth import apply_minority_mixup_cutmix, audit_memorization_dinov2

def test_model_forward_passes():
    seed_everything(42)
    device = torch.device("cpu")

    # 1. EfficientNet-B0 (pretrained=False for fast offline test)
    effnet = EfficientNetClassifier(backbone="efficientnet_b0", pretrained=False, num_classes=7).to(device)
    dummy_img = torch.randn(2, 3, 224, 224, device=device)
    out_eff = effnet(dummy_img)
    assert out_eff.shape == (2, 7)

    # 2. MLP Head
    mlp = MLPHead(in_features=384, hidden_dim=64, num_classes=7).to(device)
    dummy_emb = torch.randn(4, 384, device=device)
    out_mlp = mlp(dummy_emb)
    assert out_mlp.shape == (4, 7)

    # 3. Linear Probe Head
    probe = LinearProbeHead(in_features=512, num_classes=7).to(device)
    dummy_clip = torch.randn(4, 512, device=device)
    out_probe = probe(dummy_clip)
    assert out_probe.shape == (4, 7)

def test_losses_gradient_flow():
    seed_everything(42)
    logits = torch.randn(8, 7, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 3])

    # 1. Plain CE
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    ce_loss.backward(retain_graph=True)
    assert torch.isfinite(ce_loss).item()

    # 2. Weighted CE
    weights = get_class_weights([10, 20, 15, 2, 30, 100, 5])
    w_loss = nn.CrossEntropyLoss(weight=weights)(logits, targets)
    w_loss.backward(retain_graph=True)
    assert torch.isfinite(w_loss).item()

    # 3. Focal Loss
    focal = FocalLoss(gamma=2.0)
    f_loss = focal(logits, targets)
    f_loss.backward()
    assert torch.isfinite(f_loss).item()

def test_gradcam_cpu_execution():
    model = EfficientNetClassifier(backbone="efficientnet_b0", pretrained=False, num_classes=7)
    cam = GradCAM(model)
    dummy_img = torch.randn(1, 3, 224, 224)
    heatmap = cam.generate(dummy_img, target_class=3)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap)
    assert np.max(heatmap) <= 1.0

    pil_img = Image.new("RGB", (224, 224), (128, 128, 128))
    overlay = overlay_heatmap_on_image(pil_img, heatmap)
    assert overlay.size == (224, 224)

def test_minority_mixup_and_memorization_audit():
    seed_everything(42)
    imgs = torch.randn(4, 3, 224, 224)
    targets = torch.tensor([3, 0, 1, 3])
    mixed, ta, tb, lam = apply_minority_mixup_cutmix(imgs, targets, minority_idx=3, prob=1.0)
    assert mixed.shape == (4, 3, 224, 224)
    assert 0.0 <= lam <= 1.0

    # Memorization test
    real_feats = np.random.randn(10, 384)
    synth_feats = np.random.randn(5, 384)
    # Inject exact copy at index 0
    synth_feats[0] = real_feats[0].copy()
    max_sims, mean_sim, rejs = audit_memorization_dinov2(real_feats, synth_feats, cosine_threshold=0.95)
    assert 0 in rejs, "Failed to catch memorized exact duplicate!"

def test_cpu_end_to_end_training_smoke():
    seed_everything(42)
    device = torch.device("cpu")

    config = load_yaml_config("configs/b0_plain_ce.yaml")
    config["training"]["batch_size"] = 16
    config["compute"]["num_workers"] = 0

    train_loader, val_loader, test_loader, class_counts = get_dataloaders(
        config, seed=42, fold=0, debug_mode=True
    )

    model = EfficientNetClassifier(backbone="efficientnet_b0", pretrained=False, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run 1 train epoch
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    assert tr_loss > 0
    assert 0.0 <= tr_acc <= 1.0

    # Run 1 val epoch
    val_loss, val_targets, val_preds, val_probs, val_logits = evaluate_epoch(model, val_loader, criterion, device)
    assert val_loss > 0

    # Test calibration
    scaler = TemperatureScaler().to(device)
    temp = scaler.fit(val_logits, torch.tensor(val_targets, dtype=torch.long))
    assert temp > 0

    # Run test epoch & compute metrics
    te_loss, te_targets, te_preds, te_probs, _ = evaluate_epoch(model, test_loader, criterion, device)
    metrics = compute_metrics(te_targets, te_preds, te_probs)
    assert "minority_recall" in metrics
    assert "balanced_accuracy" in metrics
    assert 0.0 <= metrics["minority_recall"] <= 1.0
