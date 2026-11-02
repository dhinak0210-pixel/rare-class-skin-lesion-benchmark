"""
src/eval.py
===========
Rigorous metric computation, temperature scaling calibration, and selective prediction
uncertainty analysis for the Rare-Class Skin Lesion Benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MINORITY_CLASS_IDX = 3  # 'df' (dermatofibroma)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 7,
    minority_idx: int = MINORITY_CLASS_IDX
) -> Dict[str, Any]:
    """
    Computes all benchmark metrics following leak-free evaluation protocol.
    Minority recall is the primary headline metric. Accuracy is strictly secondary.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob)

    # 1. Minority Recall & F1
    minority_recall = float(recall_score(
        y_true, y_pred, labels=[minority_idx], average="micro", zero_division=0
    ))
    minority_f1 = float(f1_score(
        y_true, y_pred, labels=[minority_idx], average="micro", zero_division=0
    ))

    # 2. Balanced Accuracy & Macro F1
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    overall_acc = float(accuracy_score(y_true, y_pred))

    # 3. AUROC and AUPRC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Multi-class macro AUROC
    try:
        if len(np.unique(y_true)) > 1:
            auroc_macro = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
        else:
            auroc_macro = 0.5
    except Exception:
        auroc_macro = float("nan")

    # Minority AUROC (One-vs-Rest)
    try:
        y_minority_binary = (y_true == minority_idx).astype(int)
        if len(np.unique(y_minority_binary)) > 1:
            minority_auroc = float(roc_auc_score(y_minority_binary, y_prob[:, minority_idx]))
            minority_auprc = float(average_precision_score(y_minority_binary, y_prob[:, minority_idx]))
        else:
            minority_auroc = 0.5
            minority_auprc = 0.0
    except Exception:
        minority_auroc = float("nan")
        minority_auprc = float("nan")

    # Multi-class macro AUPRC
    try:
        auprc_macro = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    except Exception:
        auprc_macro = float("nan")

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 5. Expected Calibration Error (ECE)
    ece = compute_ece(y_true, y_prob, n_bins=10)

    return {
        "minority_recall": minority_recall,
        "minority_f1": minority_f1,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "auroc_macro": auroc_macro,
        "minority_auroc": minority_auroc,
        "auprc_macro": auprc_macro,
        "minority_auprc": minority_auprc,
        "accuracy": overall_acc,
        "expected_calibration_error": ece,
        "confusion_matrix": cm.tolist()
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across prediction confidence bins."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration learned on validation logits.
    Optimizes a single positive scalar T using Cross-Entropy loss via L-BFGS.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Finds optimal temperature parameter on validation set."""
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def evaluate_uncertainty_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    minority_idx: int = MINORITY_CLASS_IDX
) -> List[Dict[str, Any]]:
    """
    'Uncertain, please review' referral rule evaluation:
    If max confidence < threshold, sample is flagged as uncertain and referred.
    Reports referral rate and minority recall on retained vs referred sets.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    n_samples = len(y_true)

    results = []
    for thresh in thresholds:
        referred_mask = confidences < thresh
        retained_mask = ~referred_mask

        referral_rate = float(np.mean(referred_mask))
        n_retained = int(np.sum(retained_mask))

        if n_retained > 0:
            retained_recall = float(recall_score(
                y_true[retained_mask],
                predictions[retained_mask],
                labels=[minority_idx],
                average="micro",
                zero_division=0
            ))
            retained_acc = float(accuracy_score(
                y_true[retained_mask],
                predictions[retained_mask]
            ))
        else:
            retained_recall = 0.0
            retained_acc = 0.0

        minority_referred = int(np.sum((y_true == minority_idx) & referred_mask))
        total_minority = int(np.sum(y_true == minority_idx))

        results.append({
            "threshold": thresh,
            "referral_rate": referral_rate,
            "retained_samples": n_retained,
            "retained_minority_recall": retained_recall,
            "retained_accuracy": retained_acc,
            "minority_referred_count": minority_referred,
            "total_minority_count": total_minority,
            "minority_referral_fraction": minority_referred / max(total_minority, 1)
        })

    return results
