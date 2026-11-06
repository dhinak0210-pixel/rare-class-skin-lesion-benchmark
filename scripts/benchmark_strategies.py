#!/usr/bin/env python3
"""
scripts/benchmark_strategies.py
===============================
Benchmarks the 5 Data Augmentation & Imbalance Strategies:
  1. Original data
  2. Conventional aug.
  3. Class-weighted
  4. GAN-based aug.
  5. Diffusion-based aug.

Calculates the exact 5 metrics:
  - Acc. (Accuracy)
  - Prec. (Macro Precision)
  - Rec. (Macro Recall)
  - F1 (Macro F1)
  - Bal. Acc. (Balanced Accuracy)

Outputs:
  - Formatted benchmark table matching publication styling
  - Markdown table (results/tables/strategy_comparison.md)
  - CSV table (results/tables/strategy_comparison.csv)
  - LaTeX code (results/tables/strategy_comparison.tex)
  - Comparison plot (results/figures/strategy_comparison.png)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm.auto import tqdm

from src.generative import (
    StrategyAugmentedDataset,
    train_quick_dcgan,
    sample_dcgan,
    train_quick_diffusion,
    sample_diffusion
)
from src.models import EfficientNetClassifier


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model_strategy(
    strategy_name: str,
    train_images: list[np.ndarray],
    train_labels: list[int],
    val_loader: DataLoader,
    test_loader: DataLoader,
    synthetic_tensors: torch.Tensor | None = None,
    num_classes: int = 7,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> dict:
    """Trains an EfficientNet-B0 classifier under a designated augmentation/imbalance strategy."""
    print(f"\n=======================================================")
    print(f"  Training Strategy: {strategy_name.upper()}")
    print(f"=======================================================")

    set_seed(42)

    # Build dataset
    train_ds = StrategyAugmentedDataset(
        real_images=train_images,
        real_labels=train_labels,
        strategy=strategy_name,
        synthetic_tensors=synthetic_tensors,
        synthetic_label=3,  # rare class: df
        img_size=128
    )

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = EfficientNetClassifier(backbone="efficientnet_b0", pretrained=True, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Loss criterion
    if strategy_name == "Class-weighted":
        counts = np.bincount(train_labels, minlength=num_classes)
        weights = len(train_labels) / (num_classes * np.maximum(counts, 1.0))
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("  Using Cost-Sensitive Balanced Class Weights.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Using Standard Cross-Entropy Loss.")

    # Training Loop
    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lbls)

        # Validation Check
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(lbls.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {total_loss / len(train_ds):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Load best weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Evaluate on held-out test set
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(lbls.numpy())

    # Compute exact 5 metrics matching the paper
    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, average="macro", zero_division=0)
    rec = recall_score(test_targets, test_preds, average="macro", zero_division=0)
    f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(test_targets, test_preds)

    print(f"\n  Final Test Metrics for '{strategy_name}':")
    print(f"    Acc:      {acc:.2f}")
    print(f"    Prec:     {prec:.2f}")
    print(f"    Rec:      {rec:.2f}")
    print(f"    F1:       {f1:.2f}")
    print(f"    Bal. Acc: {bal_acc:.2f}")

    return {
        "Strategy": strategy_name,
        "Acc.": round(float(acc), 2),
        "Prec.": round(float(prec), 2),
        "Rec.": round(float(rec), 2),
        "F1": round(float(f1), 2),
        "Bal. Acc.": round(float(bal_acc), 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Run 5-Strategy Augmentation & Imbalance Benchmark.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per strategy.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--k-synthetic", type=int, default=100, help="Number of synthetic samples to generate for rare class.")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode on subset.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Benchmark device: {device.type.upper()}")

    # 1. Load Data
    import medmnist
    from medmnist import DermaMNIST

    print("[INFO] Loading DermaMNIST dataset...")
    train_raw = DermaMNIST(split="train", download=True, size=64)
    val_raw = DermaMNIST(split="val", download=True, size=64)
    test_raw = DermaMNIST(split="test", download=True, size=64)

    train_images = [train_raw[i][0] for i in range(len(train_raw))]
    train_labels = [int(train_raw[i][1][0]) for i in range(len(train_raw))]

    if args.fast:
        train_images = train_images[:1000]
        train_labels = train_labels[:1000]
        print(f"[FAST MODE] Subsampled training set to {len(train_images)} samples.")

    # Validation & Test DataLoaders
    eval_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class SimpleDataset(Dataset):
        def __init__(self, raw_ds, transform):
            self.raw_ds = raw_ds
            self.transform = transform

        def __len__(self):
            return len(self.raw_ds)

        def __getitem__(self, idx):
            img, target = self.raw_ds[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            return self.transform(img), int(target[0])

    val_ds = SimpleDataset(val_raw, eval_transform)
    test_ds = SimpleDataset(test_raw, eval_transform)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Extract Rare Class ('df', class 3) Images for Generative Training
    rare_indices = [i for i, lbl in enumerate(train_labels) if lbl == 3]
    print(f"[INFO] Identified {len(rare_indices)} rare 'df' (Dermatofibroma) training samples.")

    rare_tensors = []
    to_tensor = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    for idx in rare_indices:
        img = train_images[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        rare_tensors.append(to_tensor(img))
    rare_tensor_batch = torch.stack(rare_tensors)

    # 3. Train Few-Shot GAN & Generate Samples
    print("\n[INFO] Training Few-Shot DCGAN on rare class samples...")
    netG = train_quick_dcgan(rare_tensor_batch, num_epochs=15, device=device)
    gan_synthetic = sample_dcgan(netG, num_samples=args.k_synthetic, device=device)
    print(f"[INFO] Generated {len(gan_synthetic)} GAN synthetic images.")

    # 4. Train Few-Shot Diffusion & Generate Samples
    print("\n[INFO] Training Few-Shot DDPM Diffusion on rare class samples...")
    diff_unet, diff_sys = train_quick_diffusion(rare_tensor_batch, num_epochs=12, device=device)
    diffusion_synthetic = sample_diffusion(diff_unet, diff_sys, num_samples=args.k_synthetic, img_size=64)
    print(f"[INFO] Generated {len(diffusion_synthetic)} Diffusion synthetic images.")

    # 5. Benchmark All 5 Strategies
    strategies = [
        ("Original data", None),
        ("Conventional aug.", None),
        ("Class-weighted", None),
        ("GAN-based aug.", gan_synthetic),
        ("Diffusion-based aug.", diffusion_synthetic)
    ]

    results = []
    for strat_name, syn_tensors in strategies:
        res = train_model_strategy(
            strategy_name=strat_name,
            train_images=train_images,
            train_labels=train_labels,
            val_loader=val_loader,
            test_loader=test_loader,
            synthetic_tensors=syn_tensors,
            num_classes=7,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        results.append(res)

    # 6. Format and Display Results
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 68)
    print("           DATA AUGMENTATION & IMBALANCE BENCHMARK")
    print("=" * 68)
    print(df_results.to_string(index=False))
    print("=" * 68)

    # Save CSV and Markdown
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "strategy_comparison.csv"
    md_path = out_dir / "strategy_comparison.md"
    tex_path = out_dir / "strategy_comparison.tex"

    df_results.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Data Augmentation Strategy Benchmark\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n")

    # Generate LaTeX code matching the user's paper table exactly
    latex_code = "\\begin{tabular}{lccccc}\n\\hline\n"
    latex_code += "\\textbf{Strategy} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{Bal. Acc.} \\\\\n\\hline\n"
    for r in results:
        latex_code += f"{r['Strategy']:<22} & {r['Acc.']:.2f} & {r['Prec.']:.2f} & {r['Rec.']:.2f} & {r['F1']:.2f} & {r['Bal. Acc.']:.2f} \\\\\n"
    latex_code += "\\hline\n\\end{tabular}\n"

    with open(tex_path, "w") as f:
        f.write(latex_code)

    print(f"\n[INFO] Saved results to:")
    print(f"  - CSV:      {csv_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - LaTeX:    {tex_path}")

    # 7. Generate Comparison Plot
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "strategy_comparison.png"

    metrics = ["Acc.", "Prec.", "Rec.", "F1", "Bal. Acc."]
    df_melt = pd.melt(df_results, id_vars=["Strategy"], value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("viridis", len(strategies))
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Strategy", palette=palette)
    plt.title("Comparative Performance Across 5 Augmentation & Imbalance Strategies", fontsize=14, fontweight="bold")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"  - Plot:     {fig_path}")
    print(f"\n🎉 Benchmark complete!")


if __name__ == "__main__":
    main()

