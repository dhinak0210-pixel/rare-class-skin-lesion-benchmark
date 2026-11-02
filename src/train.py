"""
src/train.py
============
Robust, resumable training loop with early stopping on validation minority recall,
mixed precision support, checkpointing, post-hoc calibration, and disk logging.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple

from src.utils import get_device, save_checkpoint, load_checkpoint
from src.eval import compute_metrics, TemperatureScaler, evaluate_uncertainty_thresholds

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += len(targets)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total = 0

    all_logits = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * len(targets)
        total += len(targets)

        all_logits.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    targets_np = all_targets.numpy()

    return total_loss / max(total, 1), targets_np, preds, probs, all_logits


def run_training_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    config: dict,
    arm_id: str,
    seed: int,
    fold: int,
    checkpoint_dir: Optional[str] = None,
    log_dir: str = "results/logs"
) -> Dict[str, Any]:
    """
    Executes full training for a single fold:
      1. Train & Early stop on validation minority-recall
      2. Checkpoint best weights to disk
      3. Fit temperature scaling on validation set
      4. Single, final evaluation on test set
      5. Uncertainty threshold sweep
      6. Writes row to results/logs/runs.csv
    """
    device = get_device()
    model = model.to(device)

    t_cfg = config.get("training", {})
    epochs = t_cfg.get("epochs", 25)
    lr = t_cfg.get("lr", 3.0e-4)
    weight_decay = t_cfg.get("weight_decay", 1.0e-4)
    patience = t_cfg.get("early_stopping_patience", 7)
    use_amp = config.get("compute", {}).get("mixed_precision", True) and (device.type == "cuda")

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            config.get("compute", {}).get("checkpoint_dir", "results/checkpoints"),
            f"{arm_id}_s{seed}_f{fold}"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    best_metric = -1.0
    patience_counter = 0
    best_epoch = 0

    print(f"\n[START] Arm {arm_id} | Seed {seed} | Fold {fold} | Device: {device.type} | Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp, scaler=scaler
        )
        val_loss, val_targets, val_preds, val_probs, val_logits = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        val_metrics = compute_metrics(val_targets, val_preds, val_probs)
        current_metric = val_metrics["minority_recall"]

        elapsed = time.time() - t0

        print(
            f"  Ep {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Tr Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} MinRecall: {current_metric:.4f} BalAcc: {val_metrics['balanced_accuracy']:.4f}"
        )

        # Resumable checkpoint
        save_checkpoint({
            "epoch": epoch,
            "arm_id": arm_id,
            "seed": seed,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        }, checkpoint_dir, filename="checkpoint.pt")

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch,
                "arm_id": arm_id,
                "seed": seed,
                "fold": fold,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics
            }, checkpoint_dir, filename="best_model.pt", is_best=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Triggered at epoch {epoch}. Best epoch was {best_epoch} (MinRecall: {best_metric:.4f})")
                break

    # 1. Load best model for final evaluation
    load_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"), model)

    # 2. Extract final validation logits and fit Temperature Scaler
    _, val_targets, _, val_probs_raw, val_logits = evaluate_epoch(model, val_loader, criterion, device)
    temp_scaler = TemperatureScaler().to(device)
    val_labels_tensor = torch.tensor(val_targets, dtype=torch.long, device=device)
    best_temp = temp_scaler.fit(val_logits.to(device), val_labels_tensor)

    # 3. Final evaluation on untouched test set
    test_loss, test_targets, test_preds, test_probs, test_logits = evaluate_epoch(
        model, test_loader, criterion, device
    )

    # Calibrate test probabilities
    with torch.no_grad():
        calibrated_test_logits = temp_scaler(test_logits.to(device))
        calibrated_test_probs = torch.softmax(calibrated_test_logits, dim=1).cpu().numpy()
        calibrated_test_preds = np.argmax(calibrated_test_probs, axis=1)

    raw_test_metrics = compute_metrics(test_targets, test_preds, test_probs)
    calibrated_test_metrics = compute_metrics(test_targets, calibrated_test_preds, calibrated_test_probs)

    # 4. Uncertainty Threshold Evaluation
    uncertainty_results = evaluate_uncertainty_thresholds(test_targets, calibrated_test_probs)

    # 5. Build summary result record
    run_record = {
        "arm_id": arm_id,
        "seed": seed,
        "fold": fold,
        "best_epoch": best_epoch,
        "optimal_temperature": best_temp,
        # Headline metrics
        "test_minority_recall": raw_test_metrics["minority_recall"],
        "test_minority_f1": raw_test_metrics["minority_f1"],
        "test_balanced_accuracy": raw_test_metrics["balanced_accuracy"],
        "test_macro_f1": raw_test_metrics["macro_f1"],
        "test_auroc_macro": raw_test_metrics["auroc_macro"],
        "test_minority_auroc": raw_test_metrics["minority_auroc"],
        "test_auprc_macro": raw_test_metrics["auprc_macro"],
        "test_minority_auprc": raw_test_metrics["minority_auprc"],
        "test_accuracy": raw_test_metrics["accuracy"],
        "test_ece_uncalibrated": raw_test_metrics["expected_calibration_error"],
        "test_ece_calibrated": calibrated_test_metrics["expected_calibration_error"],
        # Validation headline
        "val_minority_recall": val_metrics["minority_recall"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "confusion_matrix": json.dumps(raw_test_metrics["confusion_matrix"]),
        "uncertainty_results": json.dumps(uncertainty_results)
    }

    # Immediately flush result to CSV
    log_csv = os.path.join(log_dir, "runs.csv")
    df_row = pd.DataFrame([run_record])
    if not os.path.exists(log_csv):
        df_row.to_csv(log_csv, index=False)
    else:
        df_row.to_csv(log_csv, mode="a", header=False, index=False)

    print(f"\n[COMPLETED] Arm {arm_id} | Seed {seed} | Fold {fold}")
    print(f"  Test Minority Recall: {run_record['test_minority_recall']:.4f} | Balanced Acc: {run_record['test_balanced_accuracy']:.4f} | ECE: {run_record['test_ece_calibrated']:.4f}")

    return run_record
