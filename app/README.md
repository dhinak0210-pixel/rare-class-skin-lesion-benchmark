---
title: Rare-Class Skin Lesion Benchmark Demo
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Does Synthetic Data Really Help? Rare-Class Skin Lesion Benchmark

Official demonstration application for the research study:
*"Does Synthetic Data Really Help? A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses and Synthetic Augmentation for Minority-Class Skin Lesion Classification"*

### Key Capabilities:
- **7-Class Dermatoscopy Classification:** Targeting rare dermatofibroma (`df`, N=115).
- **Selective Classification Safety Rule:** Flags low-confidence samples (< threshold) for clinical referral.
- **Explainability:** Real-time Grad-CAM saliency heatmaps highlighting morphological lesion cues.

> **DISCLAIMER:** Research prototype. Not for clinical use or autonomous medical diagnosis.
