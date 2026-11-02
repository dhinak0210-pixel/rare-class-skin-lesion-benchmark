"""
src/gradcam.py
==============
Explainability module:
  - Grad-CAM for CNN architectures (EfficientNet-B0)
  - Attention Rollout for Vision Transformers (DINOv2 ViT-S/14)
  - Visual panel generation for correct, missed, and confused minority lesion cases.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN backbones."""
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Locates the final convolutional layer of EfficientNet."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "conv_head"):
            return self.model.model.conv_head
        elif hasattr(self.model, "conv_head"):
            return self.model.conv_head
        # Fallback search
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                return module
        raise ValueError("Could not automatically locate convolutional target layer for Grad-CAM.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for a given input tensor [1, 3, H, W].
        Returns: 2D numpy array [H, W] normalized to [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Compute channel weights via global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input dimensions
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> Image.Image:
    """Overlays a 2D float heatmap onto an RGB PIL image."""
    img_resized = image.resize(heatmap.shape[::-1])
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    try:
        cmap = plt.get_cmap(colormap_name)
    except Exception:
        cmap = cm.viridis
    colored_heatmap = cmap(heatmap)[:, :, :3]  # RGB

    overlay = alpha * colored_heatmap + (1.0 - alpha) * img_np
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def export_explainability_case_panel(
    cases: List[Dict[str, Any]],
    output_path: str = "results/figures/explainability_cases.png"
) -> str:
    """
    Exports a 3-panel comparative diagnostic figure:
      Panel 1: Correctly identified minority case (True df, Predicted df)
      Panel 2: Missed minority case (True df, Predicted nv/bkl)
      Panel 3: Confused false positive case (True nv, Predicted df)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_cases = len(cases)
    if n_cases == 0:
        return output_path

    fig, axes = plt.subplots(2, n_cases, figsize=(4 * n_cases, 8))
    if n_cases == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, c in enumerate(cases):
        orig_img = c["image"]
        overlay_img = c["overlay"]
        title_top = f"{c['category']}\nImage: {c['image_id']}"
        title_bot = f"True: {c['true_label']} | Pred: {c['pred_label']}\nConf: {c['conf']:.2f}"

        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(title_top, fontsize=11)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay_img)
        axes[1, i].set_title(title_bot, fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle("Explainability Heatmap Diagnostics (Grad-CAM Salience)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
