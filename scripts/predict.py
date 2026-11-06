#!/usr/bin/env python3
"""
scripts/predict.py
==================
Production-ready single-image inference and explainability CLI.
Reads architecture and normalization parameters from model_config.json,
loads the best checkpoint, outputs predicted class probabilities with
uncertainty referral flags, and generates a Grad-CAM saliency heatmap overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.models import EfficientNetClassifier
from src.gradcam import GradCAM, overlay_heatmap_on_image

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma (Precancerous)",
    "bcc": "Basal Cell Carcinoma (Common malignant keratinocytic cancer)",
    "bkl": "Benign Keratosis (Seborrheic keratosis / solar lentigo)",
    "df": "Dermatofibroma (Benign fibrohistiocytic lesion - Rare class)",
    "mel": "Melanoma (Malignant neoplasm of melanocytes)",
    "nv": "Melanocytic Nevus (Common benign mole)",
    "vasc": "Vascular Lesion (Angioma / pyogenic granuloma)"
}


def load_config(config_path: str = "model_config.json") -> dict:
    """Loads configuration metadata from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint(config: dict, cli_checkpoint: str | None = None) -> tuple[str, str]:
    """
    Resolves the best available checkpoint:
    1. CLI explicit argument (--checkpoint)
    2. Local file specified in model_config.json (e.g. ./best_skin_lesion_model.pth)
    3. TorchScript binary (./model_torchscript.pt)
    4. Fallback baseline checkpoint (results/checkpoints/B0_s42_f0/best_model.pt)
    """
    if cli_checkpoint and os.path.exists(cli_checkpoint):
        return cli_checkpoint, "cli_argument"

    cfg_ckpt = config.get("checkpoint_file", "./best_skin_lesion_model.pth")
    if os.path.exists(cfg_ckpt):
        return cfg_ckpt, "model_config_checkpoint"

    if os.path.exists("./model_torchscript.pt"):
        return "./model_torchscript.pt", "torchscript"

    fallback_ckpt = "results/checkpoints/B0_s42_f0/best_model.pt"
    if os.path.exists(fallback_ckpt):
        return fallback_ckpt, "baseline_fallback"

    raise FileNotFoundError(
        f"No model checkpoint could be located! Please ensure '{cfg_ckpt}' "
        "is downloaded into the repository root, or specify --checkpoint <path>."
    )


def load_model_and_weights(
    config: dict,
    checkpoint_path: str,
    device: torch.device
) -> tuple[torch.nn.Module, bool]:
    """Initializes the model architecture and loads weights."""
    is_torchscript = checkpoint_path.endswith(".pt") and "torchscript" in checkpoint_path.lower()

    if is_torchscript:
        print(f"[INFO] Loading TorchScript model: {checkpoint_path}")
        model = torch.jit.load(checkpoint_path, map_location=device)
        model.eval()
        return model, True

    backbone = config.get("backbone", "efficientnet_b0")
    num_classes = config.get("num_classes", 7)
    model = EfficientNetClassifier(backbone=backbone, pretrained=False, num_classes=num_classes)

    print(f"[INFO] Loading checkpoint weights: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, False


def run_prediction(
    image_path: str,
    config_path: str = "model_config.json",
    checkpoint_path_override: str | None = None,
    uncertainty_threshold: float = 0.70,
    save_gradcam: bool = True,
    gradcam_output_path: str = "results/figures/prediction_gradcam.png"
) -> dict:
    """Executes single-image inference, uncertainty assessment, and Grad-CAM."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device.type.upper()}")

    ckpt_path, ckpt_source = resolve_checkpoint(config, checkpoint_path_override)
    print(f"[INFO] Active Checkpoint: {ckpt_path} (source: {ckpt_source})")

    model, is_torchscript = load_model_and_weights(config, ckpt_path, device)

    # Prepare Transforms
    img_size = config.get("image_size", 128)
    norm = config.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    mean = norm["mean"]
    std = norm["std"]

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    pil_img = Image.open(image_path).convert("RGB")
    tensor_input = transform(pil_img).unsqueeze(0).to(device)

    # Forward Pass
    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    class_names = config.get("class_names", ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_confidence = float(probs[pred_idx])

    # Uncertainty Flag
    is_uncertain = pred_confidence < uncertainty_threshold

    # Probabilities dictionary
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # Optional Grad-CAM explainability
    cam_saved_path = None
    if save_gradcam and not is_torchscript:
        try:
            gradcam_generator = GradCAM(model)
            cam = gradcam_generator.generate(tensor_input, target_class=pred_idx)
            overlay = overlay_heatmap_on_image(pil_img, cam, alpha=0.45, colormap_name="jet")

            Path(gradcam_output_path).parent.mkdir(parents=True, exist_ok=True)
            overlay.save(gradcam_output_path)
            cam_saved_path = gradcam_output_path
            print(f"[INFO] Grad-CAM saliency map saved to: {gradcam_output_path}")
        except Exception as e:
            print(f"[WARNING] Grad-CAM generation skipped: {e}")

    # Display Results
    print("\n" + "=" * 65)
    print("      DERMATOLOGICAL LESION DIAGNOSTIC PREDICTION")
    print("=" * 65)
    print(f"Target Image:      {image_path}")
    print(f"Predicted Class:   {pred_label.upper()} ({CLASS_DESCRIPTIONS.get(pred_label, 'Unknown')})")
    print(f"Confidence:        {pred_confidence * 100:.2f}%")
    if is_uncertain:
        print(f"Clinical Triage:   ⚠️  UNCERTAIN (Confidence < {uncertainty_threshold*100:.0f}%) -> REFER FOR BIOPSY/EXPERT REVIEW")
    else:
        print(f"Clinical Triage:   ✅ HIGH CONFIDENCE (>= {uncertainty_threshold*100:.0f}%)")
    print("-" * 65)
    print("Class Probability Distribution:")
    for c_name in class_names:
        p = prob_dict[c_name]
        bar = "█" * int(p * 30)
        marker = " <== PREDICTION" if c_name == pred_label else ""
        print(f"  {c_name.upper():5s} : {p*100:6.2f}% | {bar:<30s}{marker}")
    print("=" * 65)

    return {
        "image_path": image_path,
        "predicted_class": pred_label,
        "confidence": pred_confidence,
        "is_uncertain": is_uncertain,
        "probabilities": prob_dict,
        "gradcam_path": cam_saved_path,
        "checkpoint_used": ckpt_path
    }


def main():
    parser = argparse.ArgumentParser(description="Skin lesion inference and Grad-CAM explainability tool.")
    parser.add_argument("--image", type=str, default="data/samples/sample_df.png", help="Path to input lesion image.")
    parser.add_argument("--config", type=str, default="model_config.json", help="Path to model_config.json metadata.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to model weights checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold for clinical uncertainty referral.")
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM heatmap visualization.")
    parser.add_argument("--out-gradcam", type=str, default="results/figures/prediction_gradcam.png", help="Path to save Grad-CAM overlay.")

    args = parser.parse_args()

    run_prediction(
        image_path=args.image,
        config_path=args.config,
        checkpoint_path_override=args.checkpoint,
        uncertainty_threshold=args.threshold,
        save_gradcam=not args.no_gradcam,
        gradcam_output_path=args.out_gradcam
    )


if __name__ == "__main__":
    main()
