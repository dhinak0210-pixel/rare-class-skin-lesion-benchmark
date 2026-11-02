"""
app/app.py
==========
Interactive Gradio Web Application & Clinician Decision Support Demo
Features:
  - CPU-optimized real-time inference
  - Diagnostic category prediction with calibrated confidence
  - 'Uncertain, Please Review' selective classification referral flag
  - Live Grad-CAM explainability heatmap overlay
  - Clear medical disclaimers ("Research prototype. Not for clinical use.")
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import gradio as gr

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import EfficientNetClassifier
from src.gradcam import GradCAM, overlay_heatmap_on_image

CLASS_DESCRIPTIONS = {
    "df": "Dermatofibroma (Benign fibrohistiocytic lesion - Rare target class)",
    "mel": "Melanoma (Malignant neoplasm of melanocytes)",
    "nv": "Melanocytic Nevus (Common benign mole)",
    "bcc": "Basal Cell Carcinoma (Common malignant keratinocytic cancer)",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma (Precancerous / SCC in situ)",
    "bkl": "Benign Keratosis (Seborrheic keratosis / solar lentigo)",
    "vasc": "Vascular Lesion (Angioma / pyogenic granuloma)"
}

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Initialize Model
device = torch.device("cpu")
model = EfficientNetClassifier(backbone="efficientnet_b0", pretrained=False, num_classes=7).to(device)

# Load checkpoint if available, or initialize default weights
checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "results/checkpoints/B4_s42_f0/best_model.pt")
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] Loaded calibrated checkpoint into Gradio app.")
else:
    print("[NOTICE] Running with base feature extractor in demo mode.")

model.eval()
grad_cam = GradCAM(model)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_lesion(image: Image.Image, uncertainty_threshold: float = 0.70):
    if image is None:
        return None, {}, "No image uploaded.", None

    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Forward Pass
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

    # Probability dictionary
    prob_dict = {
        f"{CLASS_NAMES[i].upper()} - {CLASS_DESCRIPTIONS[CLASS_NAMES[i]]}": float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    # Selective Prediction Rule
    is_uncertain = pred_conf < uncertainty_threshold
    if is_uncertain:
        status_msg = (
            f"⚠️ **STATUS: UNCERTAIN (REFERRED FOR EXPERT DERMATOLOGY REVIEW)**\n\n"
            f"Top Prediction: **{pred_class.upper()}** (Confidence: {pred_conf*100:.1f}%)\n"
            f"The model confidence is below the safety threshold of {uncertainty_threshold*100:.1f}%. "
            f"Autonomous triage is inhibited; immediate in-person dermatoscopic biopsy/clinical review is recommended."
        )
    else:
        status_msg = (
            f"✅ **STATUS: CONFIDENT TRIAGE**\n\n"
            f"Primary Diagnosis: **{pred_class.upper()}** ({CLASS_DESCRIPTIONS[pred_class]})\n"
            f"Confidence: **{pred_conf*100:.1f}%** (exceeds safety referral threshold of {uncertainty_threshold*100:.1f}%)."
        )

    # Explainability Heatmap via Grad-CAM
    tensor.requires_grad = True
    heatmap = grad_cam.generate(tensor, target_class=pred_idx)
    cam_overlay = overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.55)

    return prob_dict, status_msg, cam_overlay


# Build Interface
with gr.Blocks(title="Rare-Class Skin Lesion Benchmark") as demo:
    gr.Markdown(
        """
        # 🔬 Rare-Class Skin Lesion Benchmark
        ### Autonomous Dermatoscopy Triage & Explainability Demo
        **Target Minority:** Dermatofibroma (`df`, N=115 in HAM10000)  
        **Safety Feature:** Selective Prediction Referral ("Uncertain, please review") & Grad-CAM Saliency.

        > **DISCLAIMER:** Research prototype. Not for clinical use or autonomous medical diagnosis.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Dermoscopic Lesion Image")
            thresh_slider = gr.Slider(
                minimum=0.50, maximum=0.95, value=0.70, step=0.05,
                label="Safety Uncertainty Referral Threshold"
            )
            btn = gr.Button("Analyze Lesion", variant="primary")

        with gr.Column(scale=1):
            status_box = gr.Markdown(label="Clinical Triage Decision")
            cam_output = gr.Image(type="pil", label="Grad-CAM Saliency Map Overlay")
            prob_label = gr.Label(num_top_classes=7, label="Class Probabilities")

    btn.click(
        fn=predict_lesion,
        inputs=[input_img, thresh_slider],
        outputs=[prob_label, status_box, cam_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
