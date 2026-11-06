---
pipeline_tag: image-classification
tags:
  - dermatology
  - skin-lesion
  - medical-imaging
  - efficientnet
  - grad-cam
  - class-imbalance
  - synthetic-augmentation
license: mit
datasets:
  - kmader/skin-cancer-mnist-ham10000
  - medmnist/dermamnist
metrics:
  - accuracy
  - precision
  - recall
  - f1
---

# 🔬 Rare-Class Skin Lesion Classification & Decision Support Benchmark
### *A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses, and Generative Augmentation*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MANDATORY CLINICAL DISCLAIMER:**  
> This model and accompanying code constitute an **academic research prototype**. They are **not certified medical devices** and are **not intended for clinical use, autonomous diagnosis, patient triage, or treatment planning**.

---

## 1. Overview
In dermatological computer vision, severe class imbalance is ubiquitous. In canonical datasets such as HAM10000, common melanocytic nevi (`nv`) account for 67% of cases, while rare categories like **dermatofibroma (`df`)** represent only **1.15% (115 images)**.

This repository provides the calibrated **EfficientNet-B0** classifier fine-tuned across 7 dermatoscopic lesion categories with strict patient-level isolation (`StratifiedGroupKFold`) and cost-sensitive loss functions.

---

## 2. Benchmark Performance Across 5 Strategies

Evaluated under identical leak-free evaluation protocol across all 5 data augmentation and imbalance strategies:

| Strategy | Acc. | Prec. | Rec. | F1 | Bal. Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original data** | 0.82 | 0.72 | 0.65 | 0.67 | 0.65 |
| **Conventional aug.** | 0.84 | 0.74 | 0.69 | 0.71 | 0.69 |
| **Class-weighted** | 0.81 | 0.70 | 0.72 | 0.70 | 0.72 |
| **GAN-based aug.** | 0.83 | 0.73 | 0.71 | 0.71 | 0.71 |
| **Diffusion-based aug.** | 0.83 | 0.74 | 0.72 | 0.72 | 0.72 |

![Benchmark Comparison](strategy_comparison.png)

---

## 3. Diagnostic Target Classes

| Class Code | Full Diagnostic Name | Clinical Category |
| :--- | :--- | :--- |
| `akiec` | Actinic keratoses / Intraepithelial carcinoma | Precancerous / SCC in situ |
| `bcc` | Basal cell carcinoma | Malignant keratinocytic cancer |
| `bkl` | Benign keratosis | Seborrheic keratosis / solar lentigo |
| `df` | **Dermatofibroma** | **Rare target class (1.15% prevalence)** |
| `mel` | Melanoma | Malignant neoplasm of melanocytes |
| `nv` | Melanocytic nevi | Common benign mole |
| `vasc` | Vascular lesions | Angioma / pyogenic granuloma |

---

## 4. How to Use in PyTorch

```python
import torch
import timm
from PIL import Image
import torchvision.transforms as T
from huggingface_hub import hf_hub_download

# 1. Download model weights
model_path = hf_hub_download(repo_id="dhina0210/rare-class-skin-lesion-benchmark", filename="best_model.pt")

# 2. Build model architecture
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# 3. Preprocess Image
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("sample_lesion.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0)

# 4. Predict
with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]

classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
top_idx = torch.argmax(probs).item()
print(f"Prediction: {classes[top_idx]} ({probs[top_idx]*100:.2f}%)")
```

---

## 5. Trust, Calibration & Safety Referral
The model incorporates an **Uncertainty Referral Gate** ($T = 0.70$):
- Predictions with confidence $\ge 70\%$ are classified with high confidence.
- Predictions with confidence $< 70\%$ trigger an explicit clinical review flag: `⚠️ UNCERTAIN -> REFER FOR BIOPSY/EXPERT REVIEW`.

