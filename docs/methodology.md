# Methodology & Scientific Protocol

## 1. Problem Formulation & Motivation

In dermatological computer vision, benign and malignant conditions exhibit extreme class imbalances. In the canonical HAM10000 dataset, melanocytic nevi (`nv`) account for 6,705 of the 10,015 images (67.0%), while dermatofibroma (`df`) contains only 115 images (1.15%). 

Standard empirical risk minimization (ERM) with plain cross-entropy loss naturally optimizes for majority classes, frequently resulting in near-zero recall for clinically critical or scarce minority conditions. While generative augmentations (e.g., Diffusion Models, GANs) are widely proposed in medical imaging literature, their reported benefits are frequently compromised by:
1. **Patient Leakage:** Splitting by image rather than patient lesion ID, inflating cross-validation performance.
2. **Confounded Augmentation Budgets:** Comparing scarce real sets ($N$) directly against generously augmented sets ($N + K$) without count-matched classical controls.
3. **Generative Memorization:** Few-shot generators memorizing training examples and replaying near-duplicates into test evaluations.

This benchmark establishes a strictly controlled, reproducible comparison of classical imbalance remedies, modern self-supervised/biomedical foundation models, and parameter-efficient generative augmentation.

---

## 2. Non-Negotiable Leakage-Control Protocol

### 2.1 Grouped Stratified Partitioning
- **Grouping Key:** `lesion_id`. Multiple dermoscopic images corresponding to the same patient lesion are strictly quarantined within the same partition.
- **Cross-Validation Scheme:** 5-Fold Stratified Group K-Fold (`StratifiedGroupKFold`), repeated across 3 independent random seeds: `42`, `1337`, `2024` (yielding 15 distinct test fold evaluations per method).
- **Nested Validation Carve-Out:** Within each training fold (80% of data), an inner 5-fold Stratified Group K-Fold split isolates a disjoint validation fold (20% of training set, 16% of total data). The remaining 64% is used for training.
- **Untouched Test Fold:** The 20% test fold is strictly held out and evaluated only once per trained model.

### 2.2 Strict Synthesis Isolation
- Generative models (e.g., Stable Diffusion 1.5 + LoRA, DCGAN) are trained **exclusively** on the training partition of the specific fold.
- Synthetic images are injected **exclusively into the training split**. Under no circumstances are synthetic images introduced into validation or test sets.

### 2.3 External Cohort Curation (Cleaned ISIC 2019)
- ISIC 2019 contains 25,331 images, of which 10,015 are re-hosted from HAM10000.
- All 10,015 HAM10000 images and matching `lesion_id` entries were purged.
- The resulting 15,316 images constitute an independent, multi-center (Hospital Clínic de Barcelona & Memorial Sloan Kettering) out-of-distribution evaluation cohort.

---

## 3. Evaluated Model Families & Arms

| Arm ID | Paradigm | Backbone / Head | Loss / Sampling Remedy |
| :--- | :--- | :--- | :--- |
| **B0** | CNN Baseline | EfficientNet-B0 (ImageNet-pretrained) | Plain Cross-Entropy, Standard Sampler |
| **B1** | CNN Baseline | EfficientNet-B0 | Classical Augmentation (Flips, Rotation, Jitter) |
| **B2** | CNN Baseline | EfficientNet-B0 | Classical Aug + Balanced `WeightedRandomSampler` |
| **B3** | CNN Baseline | EfficientNet-B0 | Classical Aug + Cost-Sensitive Class-Weighted CE |
| **B4** | CNN Baseline | EfficientNet-B0 | Classical Aug + Focal Loss ($\gamma=2.0$) |
| **P1** | Foundation Embed | DINOv2 ViT-S/14 (384-d frozen) | Multinomial Logistic Regression (L-BFGS) |
| **P2** | Foundation Embed | DINOv2 ViT-S/14 (384-d frozen) | 2-Layer MLP (Hidden=128) + Class-Weighted CE |
| **P3** | Foundation Embed | BiomedCLIP ViT-B/16 (512-d frozen) | Linear Probe Head |
| **P4** | Foundation Finetune | DINOv2 ViT-S/14 (Unfrozen last 3 blocks) | Low LR ($2\times 10^{-5}$) + Cross-Entropy |
| **P5** | Foundation Embed | DINOv2 ViT-S/14 (384-d frozen) | 2-Layer MLP + Focal Loss ($\gamma=2.0$) |
| **S1** | Generative / Interp | EfficientNet-B0 | Minority-Targeted MixUp / CutMix ($\alpha=0.4$) |
| **S2_50** | Generative Diffusion | EfficientNet-B0 | SD1.5 + LoRA (50% synthetic 'df' ratio) |
| **S2_100** | Generative Diffusion | EfficientNet-B0 | SD1.5 + LoRA (100% synthetic 'df' ratio) |
| **S2_200** | Generative Diffusion | EfficientNet-B0 | SD1.5 + LoRA (200% synthetic 'df' ratio) |
| **COMBINED** | Best Multi-Factor | Best Foundation + Imbalance Loss + Synthetic | Integrated optimal configuration |

---

## 4. Evaluation Metrics & Statistical Testing

- **Primary Headline Metric:** Minority-Class Recall ($R_{\text{df}} = \frac{\text{TP}_{\text{df}}}{\text{TP}_{\text{df}} + \text{FN}_{\text{df}}}$).
- **Secondary Multi-Class Metrics:**
  - Minority F1-Score
  - Macro F1-Score
  - Balanced Accuracy (unweighted mean of recalls)
  - Area Under the ROC Curve (One-vs-Rest Macro & Minority)
  - Area Under the Precision-Recall Curve (Macro & Minority AUPRC)
  - Expected Calibration Error (ECE)
- **Statistical Significance Protocol:**
  - 95% Non-parametric Bootstrap Confidence Intervals ($B=2,000$ resamples).
  - Paired Wilcoxon Signed-Rank Tests across the 15 matched evaluation folds.
  - Claims of superiority require both $p < 0.05$ and a 95% paired difference confidence interval strictly excluding zero.
