"""
tests/test_metrics.py
=====================
Validates that src/eval.py metric computations exactly match scikit-learn references
on controlled toy examples.
"""

import numpy as np
import pytest
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
from src.eval import compute_metrics, compute_ece, evaluate_uncertainty_thresholds

def test_metrics_match_sklearn_exact():
    np.random.seed(42)
    n_samples = 100
    n_classes = 7
    minority_idx = 3

    # Create synthetic ground truth with severe imbalance
    # Class 3 ('df') has only 5 samples
    y_true = np.random.choice([0, 1, 2, 4, 5, 6], size=n_samples, p=[0.05, 0.05, 0.1, 0.1, 0.65, 0.05])
    minority_positions = [10, 25, 40, 70, 85]
    for pos in minority_positions:
        y_true[pos] = minority_idx

    # Generate synthetic probabilistic predictions
    raw_logits = np.random.randn(n_samples, n_classes)
    # Slightly boost minority logits on actual minority positions
    raw_logits[minority_positions, minority_idx] += 2.0
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
    y_prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)

    # Compute via src.eval
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=n_classes, minority_idx=minority_idx)

    # Expected reference values via sklearn
    exp_minority_recall = recall_score(y_true, y_pred, labels=[minority_idx], average="micro", zero_division=0)
    exp_minority_f1 = f1_score(y_true, y_pred, labels=[minority_idx], average="micro", zero_division=0)
    exp_balanced_acc = balanced_accuracy_score(y_true, y_pred)
    exp_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    exp_acc = accuracy_score(y_true, y_pred)
    exp_cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    exp_auroc_macro = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    exp_minority_auroc = roc_auc_score((y_true == minority_idx).astype(int), y_prob[:, minority_idx])
    exp_minority_auprc = average_precision_score((y_true == minority_idx).astype(int), y_prob[:, minority_idx])

    # Assertions within floating-point tolerance
    assert np.isclose(metrics["minority_recall"], exp_minority_recall), "Minority recall mismatch!"
    assert np.isclose(metrics["minority_f1"], exp_minority_f1), "Minority F1 mismatch!"
    assert np.isclose(metrics["balanced_accuracy"], exp_balanced_acc), "Balanced accuracy mismatch!"
    assert np.isclose(metrics["macro_f1"], exp_macro_f1), "Macro F1 mismatch!"
    assert np.isclose(metrics["accuracy"], exp_acc), "Accuracy mismatch!"
    assert np.isclose(metrics["auroc_macro"], exp_auroc_macro), "Macro AUROC mismatch!"
    assert np.isclose(metrics["minority_auroc"], exp_minority_auroc), "Minority AUROC mismatch!"
    assert np.isclose(metrics["minority_auprc"], exp_minority_auprc), "Minority AUPRC mismatch!"
    assert metrics["confusion_matrix"] == exp_cm, "Confusion matrix mismatch!"

def test_ece_computation_bounded():
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1])
    # Perfectly calibrated case
    y_prob = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert np.isclose(ece, 0.0), f"Expected 0.0 for perfect confidence, got {ece}"

def test_uncertainty_thresholds_logic():
    y_true = np.array([3, 3, 0, 1]) # 2 minority samples
    y_prob = np.array([
        [0.1, 0.1, 0.1, 0.55, 0.05, 0.05, 0.05], # max=0.55
        [0.05, 0.05, 0.05, 0.75, 0.05, 0.02, 0.03], # max=0.75
        [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01], # max=0.9
        [0.1, 0.85, 0.01, 0.01, 0.01, 0.01, 0.01]  # max=0.85
    ])
    results = evaluate_uncertainty_thresholds(y_true, y_prob, thresholds=[0.6, 0.8], minority_idx=3)
    assert len(results) == 2

    # At threshold 0.6: only sample 0 (max=0.55) is referred (1/4 = 0.25 referral rate)
    res_06 = results[0]
    assert res_06["threshold"] == 0.6
    assert np.isclose(res_06["referral_rate"], 0.25)
    assert res_06["retained_samples"] == 3
    assert res_06["minority_referred_count"] == 1
