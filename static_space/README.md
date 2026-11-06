---
title: Skin Lesion Benchmark & Decision Support
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
tags:
  - medical
  - dermatology
  - skin-lesions
  - computer-vision
  - deep-learning
  - pytorch
---

# 🔬 Rare-Class Skin Lesion Classification & Decision Support

This Hugging Face Space presents the interactive research showcase and benchmark results for:
**"Leakage-Controlled Pretrained Benchmark for Rare-Class Skin Lesion Classification with Selective Referral"**

## Public Endpoints
- 🚀 **Live Interactive Gradio Application**: [https://4902c33db5870b5d1f.gradio.live](https://4902c33db5870b5d1f.gradio.live)
- 📦 **Hugging Face Model Hub**: [dhina0210/rare-class-skin-lesion-benchmark](https://huggingface.co/dhina0210/rare-class-skin-lesion-benchmark)

## Benchmark Results
| Strategy | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original data** | 0.82 | 0.72 | 0.65 | 0.67 | 0.65 |
| **Conventional aug.** | **0.84** | **0.74** | 0.69 | 0.71 | 0.69 |
| **Class-weighted** | 0.81 | 0.70 | **0.72** | 0.70 | **0.72** |
| **GAN-based aug.** | 0.83 | 0.73 | 0.71 | 0.71 | 0.71 |
| **Diffusion-based aug.** | 0.83 | **0.74** | **0.72** | **0.72** | **0.72** |

## Mandatory Clinical Disclaimer
*This system is an academic research prototype. It is NOT a certified medical device and must NEVER be used for autonomous clinical diagnosis.*

