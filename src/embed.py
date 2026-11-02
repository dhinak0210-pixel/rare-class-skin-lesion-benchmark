"""
src/embed.py
============
Foundation model embedding extraction and cached head training:
  - DINOv2 ViT-S/14 (facebook/dinov2-small) -> 384-d embeddings
  - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) -> 512-d embeddings
  - Head trainers: L-BFGS Logistic Regression & Class-Weighted / Focal MLP
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression

from src.utils import get_device, seed_everything
from src.dataset import CLASS_MAP, BACKBONE_NORMS, get_transforms
from src.models import MLPHead
from src.losses import build_loss
from src.eval import compute_metrics, TemperatureScaler, evaluate_uncertainty_thresholds

EMBEDDINGS_DIR = "data/embeddings"

def load_foundation_backbone(backbone_name: str, device: torch.device):
    """Loads foundation feature extractor in eval mode."""
    if "dino" in backbone_name.lower():
        try:
            from transformers import AutoImageProcessor, AutoModel
            processor = AutoImageProcessor.from_pretrained(backbone_name)
            model = AutoModel.from_pretrained(backbone_name).to(device).eval()
            return model, processor, "dino_hf"
        except Exception:
            import timm
            model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(device).eval()
            return model, None, "dino_timm"

    elif "biomedclip" in backbone_name.lower():
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model = model.to(device).eval()
        return model, preprocess, "biomedclip"

    else:
        raise ValueError(f"Unsupported foundation backbone: {backbone_name}")


@torch.no_grad()
def extract_single_embedding(
    image: Image.Image,
    model: nn.Module,
    processor: Any,
    model_type: str,
    device: torch.device
) -> np.ndarray:
    """Extracts a normalized embedding vector from a PIL image."""
    if model_type == "dino_hf":
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0].cpu().numpy().squeeze(0)
    elif model_type == "dino_timm":
        t = get_transforms("val", img_size=224, norm_preset="dinov2")
        tensor = t(image).unsqueeze(0).to(device)
        feat = model(tensor).cpu().numpy().squeeze(0)
    elif model_type == "biomedclip":
        tensor = processor(image).unsqueeze(0).to(device)
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.cpu().numpy().squeeze(0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return feat.astype(np.float32)


def extract_and_cache_embeddings(
    metadata_path: str = "data/metadata.csv",
    images_dir: str = "data/images",
    backbone_name: str = "facebook/dinov2-small",
    cache_filename: Optional[str] = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Extracts embeddings for all images in metadata.csv and caches to disk.
    Returns: (embeddings_matrix [N, D], image_ids [N], labels [N])
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    device = get_device()

    sanitized_name = backbone_name.replace("/", "_").replace(":", "_")
    if cache_filename is None:
        cache_filename = f"{sanitized_name}_embeddings.npz"
    cache_path = os.path.join(EMBEDDINGS_DIR, cache_filename)

    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["image_ids"].tolist(), data["labels"]

    print(f"[INFO] Extracting embeddings using {backbone_name} on {device.type}...")
    df = pd.read_csv(metadata_path)
    model, processor, model_type = load_foundation_backbone(backbone_name, device)

    embeddings = []
    image_ids = []
    labels = []

    for idx, row in df.iterrows():
        img_id = row["image_id"]
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            # Deterministic fallback for local smoke-test without raw images
            img = Image.new("RGB", (224, 224), color=(idx % 255, (idx * 3) % 255, (idx * 5) % 255))

        emb = extract_single_embedding(img, model, processor, model_type, device)
        embeddings.append(emb)
        image_ids.append(img_id)
        labels.append(CLASS_MAP[row["dx"]])

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} images...")

    embeddings_mat = np.array(embeddings, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings_mat,
        image_ids=np.array(image_ids),
        labels=labels_arr
    )
    print(f"[SUCCESS] Saved cached embeddings to {cache_path} (Shape: {embeddings_mat.shape})")
    return embeddings_mat, image_ids, labels_arr


def train_cached_embedding_arm(
    config: dict,
    arm_id: str,
    seed: int,
    fold: int,
    embeddings_matrix: np.ndarray,
    image_id_to_idx: Dict[str, int],
    labels_array: np.ndarray,
    log_dir: str = "results/logs"
) -> Dict[str, Any]:
    """
    Trains classification heads (Logistic Regression or MLP) on pre-extracted embeddings.
    Runs in seconds and provides leak-free evaluation on the test fold.
    """
    seed_everything(seed)
    device = get_device()
    splits_df = pd.read_csv(config["protocol"]["split_file"])

    fold_df = splits_df[(splits_df["seed"] == seed) & (splits_df["fold"] == fold)]
    train_ids = fold_df[fold_df["split"] == "train"]["image_id"].values
    val_ids = fold_df[fold_df["split"] == "val"]["image_id"].values
    test_ids = fold_df[fold_df["split"] == "test"]["image_id"].values

    train_idx = [image_id_to_idx[img_id] for img_id in train_ids]
    val_idx = [image_id_to_idx[img_id] for img_id in val_ids]
    test_idx = [image_id_to_idx[img_id] for img_id in test_ids]

    X_train, y_train = embeddings_matrix[train_idx], labels_array[train_idx]
    X_val, y_val = embeddings_matrix[val_idx], labels_array[val_idx]
    X_test, y_test = embeddings_matrix[test_idx], labels_array[test_idx]

    head_type = config.get("model", {}).get("head_type", "logistic_regression")

    if head_type == "logistic_regression":
        class_weight = config.get("head_training", {}).get("class_weight", None)
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=config.get("head_training", {}).get("c_param", 1.0),
            class_weight=class_weight,
            random_state=seed
        )
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_metrics(y_val, val_preds, val_probs)

        test_probs = clf.predict_proba(X_test)
        test_preds = np.argmax(test_probs, axis=1)

        # Calibrate via temperature scaling on val logits
        val_logits = torch.tensor(clf.decision_function(X_val), dtype=torch.float32, device=device)
        test_logits = torch.tensor(clf.decision_function(X_test), dtype=torch.float32, device=device)

        scaler = TemperatureScaler().to(device)
        best_temp = scaler.fit(val_logits, torch.tensor(y_val, dtype=torch.long, device=device))

        with torch.no_grad():
            cal_test_logits = scaler(test_logits)
            cal_test_probs = torch.softmax(cal_test_logits, dim=1).cpu().numpy()
            cal_test_preds = np.argmax(cal_test_probs, axis=1)

        raw_test_metrics = compute_metrics(y_test, test_preds, test_probs)
        cal_test_metrics = compute_metrics(y_test, cal_test_preds, cal_test_probs)

    elif head_type == "mlp":
        in_dim = X_train.shape[1]
        hidden_dim = config.get("model", {}).get("hidden_dim", 128)
        dropout = config.get("model", {}).get("dropout", 0.3)
        num_classes = config.get("protocol", {}).get("num_classes", 7)

        mlp = MLPHead(in_features=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)

        # Count classes for loss
        class_counts = [int((y_train == c).sum()) for c in range(num_classes)]
        loss_cfg = config.get("loss", {"type": "weighted_ce", "weight_scheme": "balanced"})
        criterion = build_loss(loss_cfg, class_counts=class_counts, device=device)

        epochs = config.get("head_training", {}).get("epochs", 40)
        lr = config.get("head_training", {}).get("lr", 0.001)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-3)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

        best_val_recall = -1.0
        best_state = None

        for epoch in range(epochs):
            mlp.train()
            optimizer.zero_grad()
            logits = mlp(X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()

            mlp.eval()
            with torch.no_grad():
                val_l = mlp(X_val_t)
                val_p = torch.softmax(val_l, dim=1).cpu().numpy()
                val_pr = np.argmax(val_p, axis=1)
                vm = compute_metrics(y_val, val_pr, val_p)
                if vm["minority_recall"] > best_val_recall:
                    best_val_recall = vm["minority_recall"]
                    best_state = mlp.state_dict().copy()

        if best_state is not None:
            mlp.load_state_dict(best_state)

        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(X_val_t)
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_preds = np.argmax(val_probs, axis=1)
            val_metrics = compute_metrics(y_val, val_preds, val_probs)

            X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            test_logits = mlp(X_te_t)
            test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
            test_preds = np.argmax(test_probs, axis=1)

            scaler = TemperatureScaler().to(device)
            best_temp = scaler.fit(val_logits, y_val_t)

            cal_test_logits = scaler(test_logits)
            cal_test_probs = torch.softmax(cal_test_logits, dim=1).cpu().numpy()
            cal_test_preds = np.argmax(cal_test_probs, axis=1)

        raw_test_metrics = compute_metrics(y_test, test_preds, test_probs)
        cal_test_metrics = compute_metrics(y_test, cal_test_preds, cal_test_probs)

    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    uncertainty_results = evaluate_uncertainty_thresholds(y_test, cal_test_probs)

    record = {
        "arm_id": arm_id,
        "seed": seed,
        "fold": fold,
        "optimal_temperature": best_temp,
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
        "test_ece_calibrated": cal_test_metrics["expected_calibration_error"],
        "val_minority_recall": val_metrics["minority_recall"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "confusion_matrix": json.dumps(raw_test_metrics["confusion_matrix"]),
        "uncertainty_results": json.dumps(uncertainty_results)
    }

    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, "runs.csv")
    df_row = pd.DataFrame([record])
    if not os.path.exists(log_csv):
        df_row.to_csv(log_csv, index=False)
    else:
        df_row.to_csv(log_csv, mode="a", header=False, index=False)

    print(
        f"[COMPLETED] Arm {arm_id} | Seed {seed} | Fold {fold} -> "
        f"Test Minority Recall: {record['test_minority_recall']:.4f} | "
        f"Balanced Acc: {record['test_balanced_accuracy']:.4f} | "
        f"Macro F1: {record['test_macro_f1']:.4f}"
    )
    return record
