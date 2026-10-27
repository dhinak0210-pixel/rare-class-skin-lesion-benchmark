"""Classifier training and evaluation module with strict experimental controls.

Uses ResNet18 backbone, evaluating macro F1, rare class recall, rare class F1,
balanced accuracy, and rare class PR-AUC.
"""

from __future__ import annotations

import copy
import glob
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.config import ExperimentConfig, cfg
from src.scarcity import HybridAugmentedDataset, SplitDataset, clf_eval_transform


def set_seed(seed: int) -> None:
    """Set all random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate ResNet18 with modified linear classification head."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception:
        # Fallback if offline or weights cannot be fetched
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    rare_class_id: int,
    device: torch.device,
) -> Dict[str, float]:
    """Compute rare recall, rare F1, macro F1, balanced accuracy, and rare PR-AUC."""
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_matrix = np.concatenate(all_probs, axis=0)

    # 1. Rare recall
    rare_recall = float(recall_score(y_true, y_pred, labels=[rare_class_id], average="micro", zero_division=0.0))

    # 2. Rare F1
    rare_f1 = float(f1_score(y_true, y_pred, labels=[rare_class_id], average="micro", zero_division=0.0))

    # 3. Macro F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))

    # 4. Balanced accuracy
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # 5. Precision-Recall AUC for rare class (One-vs-Rest)
    y_true_binary = (y_true == rare_class_id).astype(int)
    rare_probs = probs_matrix[:, rare_class_id]

    if np.sum(y_true_binary) > 0:
        pr_auc = float(average_precision_score(y_true_binary, rare_probs))
    else:
        pr_auc = 0.0

    return {
        "rare_recall": rare_recall,
        "rare_f1": rare_f1,
        "macro_f1": macro_f1,
        "bal_acc": bal_acc,
        "pr_auc": pr_auc,
    }


@torch.no_grad()
def get_synthetic_confidence(
    model: nn.Module,
    syn_image_dir: str,
    rare_class_id: int,
    device: torch.device,
) -> float:
    """Compute the mean classifier confidence for rare_class_id on the generated images."""
    if not os.path.exists(syn_image_dir):
        return float("nan")

    img_files = sorted(glob.glob(os.path.join(syn_image_dir, "*.png")))
    if len(img_files) == 0:
        return float("nan")

    model.eval()
    confidences: List[float] = []

    batch_tensors = []
    for p in img_files:
        pil_img = Image.open(p).convert("RGB")
        t = clf_eval_transform(pil_img)
        batch_tensors.append(t)
        if len(batch_tensors) >= 32:
            x = torch.stack(batch_tensors).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, rare_class_id]
            confidences.extend(probs.cpu().tolist())
            batch_tensors = []

    if len(batch_tensors) > 0:
        x = torch.stack(batch_tensors).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, rare_class_id]
        confidences.extend(probs.cpu().tolist())

    return float(np.mean(confidences))


def train_classifier(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    rare_class_id: int,
    num_classes: int,
    seed: int,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    fast_dev_run: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], nn.Module]:
    """Train downstream ResNet18 with early checkpointing on val_macro_f1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    actual_epochs = 1 if fast_dev_run else epochs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,  # robust across platforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = create_classifier(num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_val_metrics: Dict[str, float] = {}
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(actual_epochs):
        model.train()
        batch_count = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_count += 1
            if fast_dev_run and batch_count >= 2:
                break

        # Validate after each epoch
        val_metrics = evaluate_classifier(model, val_loader, rare_class_id, device)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(best_model_weights)
    test_metrics = evaluate_classifier(model, test_loader, rare_class_id, device)

    return best_val_metrics, test_metrics, model
