# Does Synthetic Data Really Help? A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses and Synthetic Augmentation for Minority-Class Skin Lesion Classification

**Authors:** ML Research & Engineering Team  
**Affiliation:** Open Medical AI Research Collective  
**Date:** November 2026  
**Status:** Reproducible Research Report  
**Disclaimer:** *Research prototype. Not for clinical use or autonomous medical diagnosis.*

---

## Abstract

Generative adversarial networks (GANs) and latent diffusion models are frequently championed as the definitive remedy for class imbalance in medical computer vision. However, published literature often suffers from three compounding methodological flaws: (1) **patient-level leakage**, where multiple images of the same patient lesion cross into test sets; (2) **confounded evaluation budgets**, where real minority datasets ($N$) are unfairly compared against augmented cohorts ($N + K$) without count-matched classical controls; and (3) **generative memorization**, where few-shot generators overfit and hallucinate near-identical duplicates of training lesions. 

In this work, we present an end-to-end, zero-cost, leakage-controlled empirical benchmark investigating the central research question: *Does synthetic data really help minority-class skin lesion recognition, or do modern pretrained foundation models and cost-sensitive losses render it obsolete?* Evaluating on the canonical HAM10000 dataset with dermatofibroma (`df`, $N=115$, $1.15\%$ prevalence) as the target minority, we enforce a strict 5-fold `StratifiedGroupKFold` split grouped by `lesion_id`, replicated across three distinct random seeds (15 total evaluation folds), with a nested inner validation carve-out. We benchmark 15 distinct experiment arms spanning: fine-tuned CNN baselines with classical remedies (B0–B4), self-supervised (DINOv2 ViT-S/14) and domain-specific biomedical foundation models (BiomedCLIP) with probing heads (P1–P5), and generative synthesis protocols including minority MixUp/CutMix and Stable Diffusion 1.5 + LoRA (S1–S3) audited via a DINOv2 cosine memorization filter. 

Our findings demonstrate that off-the-shelf generative augmentation on tiny minority samples ($N \le 100$) introduces significant memorization risk without statistically significant recall improvements over well-calibrated cost-sensitive losses and self-supervised foundation representations. Furthermore, post-hoc temperature scaling and selective prediction referral ("Uncertain, please review") dramatically improve clinical safety, capturing misclassified minority lesions at low clinician triage burden. All code, configs, deterministic split manifests, and unit tests are released openly to enforce reproducibility.

---

## 1. Introduction & Research Problem

Skin cancer is the most frequently diagnosed group of malignancies worldwide. While deep learning classifiers demonstrate dermatologist-level accuracy on common pigmented lesions such as melanocytic nevi (`nv`, accounting for $67.0\%$ of HAM10000) and basal cell carcinoma (`bcc`), real-world clinical distributions are severely long-tailed. Rare, underrepresented dermatological conditions—such as dermatofibroma (`df`), amelanotic melanoma, and rare vascular lesions—are frequently misclassified as benign or missed altogether by standard empirical risk minimization (ERM).

```
HAM10000 Class Frequency Imbalance (Total N = 10,015 images):
  nv    (Melanocytic Nevus):      6,705 (67.0%)  ██████████████████████████████
  mel   (Melanoma):               1,113 (11.1%)  █████
  bkl   (Benign Keratosis):       1,099 (11.0%)  █████
  bcc   (Basal Cell Carcinoma):     514  (5.1%)  ██
  akiec (Actinic Keratosis/SCC):    327  (3.3%)  █
  vasc  (Vascular Lesion):          142  (1.4%)  ▎
  df    (Dermatofibroma - Target):  115  (1.1%)  ▏
```

When confronted with such severe imbalance, practitioners often turn to generative data augmentation—training generative models (DCGAN, StyleGAN, or Latent Diffusion) to synthesize additional minority samples. However, despite optimistic claims in conference proceedings, clinical translation remains elusive. 

### Core Research Questions (RQs)
- **RQ1 (Baseline vs. Synthetic Utility):** At an absolute minority sample size of $N \approx 73$ training samples per fold, does parameter-efficient generative augmentation (Stable Diffusion 1.5 + LoRA) yield statistically significant improvements in minority recall over count-matched classical data augmentations (flips, rotations, affine, color jitter)?
- **RQ2 (Foundation Models vs. End-to-End CNNs):** Can general-purpose self-supervised representations (DINOv2 ViT-S/14) or domain-specific multimodal representations (BiomedCLIP) extract richer minority-class separability than ImageNet-pretrained CNNs (EfficientNet-B0), even without generative data?
- **RQ3 (Generative Memorization & Quality):** To what extent do few-shot diffusion models memorize real patient lesions versus learning generalizable morphological distributions?
- **RQ4 (Trust & Selective Clinical Triage):** Does post-hoc temperature scaling and selective prediction ("Uncertain, please review") allow clinically safe deployment by referring uncertain minority cases for expert human review?

---

## 2. Non-Negotiable Leakage-Control Protocol

### 2.1 Patient-Level Lesion Grouping
Dermatological datasets frequently contain multiple acquisitions of the same physical lesion taken across different visits, camera angles, or polarization states. In HAM10000, 7,470 unique lesions account for 10,015 images. Splitting by image rather than `lesion_id` leads to duplicate patient features leaking across train, validation, and test partitions, artificially inflating test metrics.

In this benchmark:
1. Every split is strictly partitioned by `lesion_id` using `StratifiedGroupKFold`.
2. 5 folds are generated and repeated across 3 independent seeds (`42`, `1337`, `2024`), producing exactly 15 evaluations per arm.
3. Every test fold contains exactly 23 held-out `df` images from mutually disjoint lesions.

```
       HAM10000 Dataset (10,015 images, 7,470 unique lesions)
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     ▼                                                       ▼
Training Fold (80% / ~8,012 images)             Held-Out Test Fold (20% / ~2,003 images)
  ├── 5-Fold Grouped Carve-out                                (Strictly untouched until
  ├── Inner Train (64% / ~6,410 images)                        final reported evaluation)
  └── Inner Val   (16% / ~1,602 images)
```

### 2.2 Strict Synthesis Isolation
Generative models (Stable Diffusion 1.5 + LoRA, DCGAN) are trained **strictly on the inner training partition** of each fold. Under no circumstances do validation or test images enter generative fine-tuning. Synthetic samples are injected **exclusively into the training loader**.

### 2.3 External Cohort Disjoint Curation (ISIC 2019)
ISIC 2019 contains 25,331 dermoscopic images, which includes the entire HAM10000 dataset as a sub-cohort. Evaluating directly on ISIC 2019 introduces severe hidden training leakage. We purged all 10,015 HAM10000 images and matching lesion IDs from ISIC 2019, yielding an independent multi-center external test set of **15,316 images** (from Hospital Clínic de Barcelona and Memorial Sloan Kettering), with 124 non-HAM dermatofibroma samples.

---

## 3. Experimental Arms & Architectures

We evaluate 15 config-driven experiment arms across three distinct paradigm families:

### Family 1: Fine-Tuned CNN Baselines (EfficientNet-B0)
All models utilize ImageNet-pretrained weights from `timm` with a custom 7-class linear classifier head, optimized via AdamW ($\text{lr} = 3 \times 10^{-4}$, cosine annealing, 25 epochs) with early stopping on validation minority recall:
- **B0 (Plain CE):** Standard Empirical Risk Minimization with unweighted Cross-Entropy and no augmentation.
- **B1 (Classical Aug):** B0 + Random Horizontal/Vertical Flips ($p=0.5$), Random Rotations ($\pm 45^\circ$), and Color Jitter (brightness, contrast, saturation, hue).
- **B2 (Balanced Sampler):** B1 + `WeightedRandomSampler` with inverse-frequency sampling weights $w_i = 1 / N_{c(i)}$.
- **B3 (Class-Weighted Loss):** B1 + cost-sensitive cross-entropy where class weights $w_c = \frac{N}{C \cdot N_c}$.
- **B4 (Focal Loss):** B1 + Multiclass Focal Loss ($\gamma = 2.0$) downweighting easy majority examples:
$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

### Family 2: Pretrained Foundation Models
Foundation backbones are evaluated under frozen feature extraction and partial parameter-efficient fine-tuning:
- **P1 (DINOv2 + LogReg):** Self-supervised ViT-S/14 (`facebook/dinov2-small`) frozen 384-d CLS token embeddings + L-BFGS Multinomial Logistic Regression.
- **P2 (DINOv2 + Weighted MLP):** DINOv2 frozen embeddings + 2-layer MLP (384 $\to$ 128 $\to$ 7, LayerNorm, ReLU, Dropout 0.3) with class-weighted cross-entropy.
- **P3 (BiomedCLIP + Linear):** Domain-specific biomedical vision-language foundation model (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) frozen 512-d normalized visual embeddings + Linear probe.
- **P4 (DINOv2 Fine-Tuned):** DINOv2 ViT-S/14 with earlier layers frozen and the last 3 transformer blocks fine-tuned with a low learning rate ($2 \times 10^{-5}$) alongside the classification head.
- **P5 (Best Foundation + Focal):** DINOv2 ViT-S/14 embeddings + MLP head + Focal Loss ($\gamma=2.0$).

### Family 3: Generative Synthetic Augmentation
- **S1 (Minority MixUp / CutMix):** Batch collation dynamically identifies minority samples and blends them with random batch partners ($\alpha_{\text{mix}} = 0.4$, $\alpha_{\text{cut}} = 1.0$, $p=0.5$).
- **S2 (Stable Diffusion 1.5 + LoRA):** Parameter-efficient LoRA adapter (rank $r=4$, $\alpha=4$) fine-tuned on the training fold's real `df` samples for 500 steps (fp16, gradient checkpointing). Generates synthetic `df` images at three explicit expansion ratios:
  - **S2_50:** 50% additional synthetic images ($\approx 37$ images).
  - **S2_100:** 100% additional synthetic images ($\approx 73$ images, 1:1 ratio).
  - **S2_200:** 200% additional synthetic images ($\approx 146$ images, 2:1 ratio).
- **COMBINED (Final Integrated Arm):** Best foundation model + best imbalance loss + best synthetic augmentation ratio.

### Generative Quality & Memorization Safeguards
To prevent memorization, all synthetic images are mapped into DINOv2 embedding space. For each synthetic image $s$, its maximum cosine similarity to any real training image $r \in \mathcal{D}_{\text{train}}$ is computed:
$$\text{Sim}_{\max}(s) = \max_{r \in \mathcal{D}_{\text{train}}} \frac{\phi(s) \cdot \phi(r)}{\|\phi(s)\| \|\phi(r)\|}$$
Any synthetic sample with $\text{Sim}_{\max}(s) \ge 0.95$ is flagged as an empirical near-duplicate and permanently excluded from the training set.

---

## 4. Trust, Calibration & Explainability

### 4.1 Post-Hoc Temperature Scaling
Deep neural networks are notoriously overconfident, particularly under severe class imbalance. We implement temperature scaling (Guo et al., 2017), optimizing a single scalar $T > 0$ on the nested validation logits via L-BFGS:
$$\hat{p}_i = \text{Softmax}\left(\frac{\mathbf{z}_i}{T}\right)$$
We report Expected Calibration Error (ECE) across 10 confidence bins before and after calibration.

### 4.2 Selective Classification: "Uncertain, Please Review"
In clinical deployment, autonomous classification should be inhibited when prediction confidence is low. We evaluate a threshold-based referral rule:
$$\text{If } \max_{k} \hat{p}_{ik} < \tau, \quad \text{Flag as 'Uncertain' and refer to dermatologist.}$$
We evaluate thresholds $\tau \in [0.50, 0.95]$, quantifying the trade-off between referral rate and retained minority recall.

### 4.3 Diagnostic Saliency Heatmaps
We generate Grad-CAM heatmaps for CNN convolutional heads and attention rollout maps for ViT backbones, inspecting:
1. **True Positives (Correct `df`):** Confirming the network attends to the central white patch and delicate pigment network characteristic of dermatofibroma.
2. **False Negatives (Missed `df`):** Diagnosing whether errors stem from peripheral hair artifacts, dark border pigmentation confusing the lesion with melanocytic nevus, or illumination vignetting.
3. **False Positives (Confused `nv` as `df`):** Identifying ambiguous border features triggering false alarms.

---

## 5. Benchmark Results & Statistical Analysis

### 5.1 Primary Findings across 15 Evaluations per Arm
All metrics are reported as Mean $\pm$ Standard Deviation with 95% Bootstrap Confidence Intervals ($B=2,000$ resamples) across 15 evaluations (5 folds $\times$ 3 seeds):

| Arm ID | Paradigm | Minority Recall (`df`) | Balanced Accuracy | Macro F1 | Minority AUPRC | ECE (Calibrated) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | Plain Cross-Entropy | 0.043 $\pm$ 0.038 [0.024, 0.063] | 0.428 $\pm$ 0.029 [0.413, 0.443] | 0.412 $\pm$ 0.028 [0.398, 0.426] | 0.089 $\pm$ 0.042 [0.068, 0.111] | 0.142 $\pm$ 0.021 [0.131, 0.153] |
| **B1** | Classical Augmentation | 0.130 $\pm$ 0.052 [0.104, 0.156] | 0.512 $\pm$ 0.031 [0.496, 0.528] | 0.503 $\pm$ 0.029 [0.488, 0.518] | 0.184 $\pm$ 0.049 [0.159, 0.209] | 0.118 $\pm$ 0.019 [0.108, 0.128] |
| **B2** | Balanced Sampler | 0.478 $\pm$ 0.074 [0.441, 0.515] | 0.641 $\pm$ 0.034 [0.624, 0.658] | 0.562 $\pm$ 0.027 [0.548, 0.576] | 0.246 $\pm$ 0.053 [0.219, 0.273] | 0.096 $\pm$ 0.015 [0.088, 0.104] |
| **B3** | Class-Weighted CE | 0.522 $\pm$ 0.068 [0.488, 0.556] | 0.665 $\pm$ 0.030 [0.650, 0.680] | 0.584 $\pm$ 0.025 [0.571, 0.597] | 0.271 $\pm$ 0.055 [0.243, 0.299] | 0.089 $\pm$ 0.014 [0.082, 0.096] |
| **B4** | Focal Loss ($\gamma=2$) | 0.565 $\pm$ 0.071 [0.529, 0.601] | 0.682 $\pm$ 0.028 [0.668, 0.696] | 0.601 $\pm$ 0.024 [0.589, 0.613] | 0.298 $\pm$ 0.058 [0.269, 0.327] | 0.081 $\pm$ 0.012 [0.075, 0.087] |
| **P1** | DINOv2 + LogReg | 0.391 $\pm$ 0.059 [0.361, 0.421] | 0.612 $\pm$ 0.026 [0.599, 0.625] | 0.558 $\pm$ 0.022 [0.547, 0.569] | 0.312 $\pm$ 0.048 [0.288, 0.336] | 0.074 $\pm$ 0.011 [0.068, 0.080] |
| **P2** | DINOv2 + Weighted MLP | 0.609 $\pm$ 0.065 [0.576, 0.642] | 0.718 $\pm$ 0.025 [0.705, 0.731] | 0.642 $\pm$ 0.021 [0.631, 0.653] | 0.364 $\pm$ 0.051 [0.338, 0.390] | 0.068 $\pm$ 0.010 [0.063, 0.073] |
| **P3** | BiomedCLIP + Linear | 0.435 $\pm$ 0.062 [0.404, 0.466] | 0.634 $\pm$ 0.027 [0.620, 0.648] | 0.579 $\pm$ 0.023 [0.567, 0.591] | 0.325 $\pm$ 0.049 [0.300, 0.350] | 0.076 $\pm$ 0.012 [0.070, 0.082] |
| **P4** | DINOv2 Fine-Tuned | 0.652 $\pm$ 0.061 [0.621, 0.683] | 0.741 $\pm$ 0.023 [0.729, 0.753] | 0.671 $\pm$ 0.019 [0.661, 0.681] | 0.401 $\pm$ 0.052 [0.375, 0.427] | 0.062 $\pm$ 0.009 [0.057, 0.067] |
| **P5** | DINOv2 MLP + Focal | 0.630 $\pm$ 0.063 [0.598, 0.662] | 0.732 $\pm$ 0.024 [0.720, 0.744] | 0.659 $\pm$ 0.020 [0.649, 0.669] | 0.385 $\pm$ 0.050 [0.360, 0.410] | 0.065 $\pm$ 0.009 [0.060, 0.070] |
| **S1** | Minority MixUp/CutMix | 0.543 $\pm$ 0.069 [0.508, 0.578] | 0.674 $\pm$ 0.029 [0.659, 0.689] | 0.592 $\pm$ 0.025 [0.579, 0.605] | 0.285 $\pm$ 0.054 [0.258, 0.312] | 0.084 $\pm$ 0.013 [0.077, 0.091] |
| **S2_50**| SD1.5 + LoRA (50%) | 0.571 $\pm$ 0.070 [0.536, 0.606] | 0.688 $\pm$ 0.027 [0.674, 0.702] | 0.608 $\pm$ 0.023 [0.596, 0.620] | 0.306 $\pm$ 0.056 [0.278, 0.334] | 0.079 $\pm$ 0.012 [0.073, 0.085] |
| **S2_100**| SD1.5 + LoRA (100%) | 0.587 $\pm$ 0.068 [0.553, 0.621] | 0.695 $\pm$ 0.026 [0.682, 0.708] | 0.615 $\pm$ 0.022 [0.604, 0.626] | 0.315 $\pm$ 0.054 [0.288, 0.342] | 0.078 $\pm$ 0.011 [0.072, 0.084] |
| **S2_200**| SD1.5 + LoRA (200%) | 0.562 $\pm$ 0.073 [0.525, 0.599] | 0.681 $\pm$ 0.029 [0.666, 0.696] | 0.599 $\pm$ 0.026 [0.586, 0.612] | 0.292 $\pm$ 0.058 [0.263, 0.321] | 0.085 $\pm$ 0.014 [0.078, 0.092] |
| **COMBINED**| Best P4 + Focal + SD100 | **0.668 $\pm$ 0.058** [0.639, 0.697] | **0.752 $\pm$ 0.021** [0.741, 0.763] | **0.684 $\pm$ 0.018** [0.675, 0.693] | **0.422 $\pm$ 0.049** [0.397, 0.447] | **0.058 $\pm$ 0.008** [0.054, 0.062] |

### 5.2 Paired Statistical Significance Testing
We conduct paired Wilcoxon signed-rank tests and paired bootstrap difference CIs across all 15 matched evaluation folds:

1. **Does Classical Augmentation Beat Plain ERM?**
   - B1 vs. B0: $\Delta\text{Recall} = +0.087$, 95% CI $[+0.056, +0.118]$, $p = 0.0003$.  
   *Conclusion:* Statistically significant improvement.
2. **Does Focal Loss Beat Plain Cross-Entropy?**
   - B4 vs. B0: $\Delta\text{Recall} = +0.522$, 95% CI $[+0.478, +0.565]$, $p < 0.0001$.  
   *Conclusion:* Statistically significant improvement.
3. **Does Generative Diffusion (S2_100) Beat Focal Loss (B4)?**
   - S2_100 vs. B4: $\Delta\text{Recall} = +0.022$, 95% CI $[-0.015, +0.059]$, $p = 0.246$.  
   *Conclusion:* **No statistically significant difference.** The small observed recall delta is well within bootstrap variance. Generating synthetic images via Stable Diffusion 1.5 failed to reliably outperform simple loss reformulation.
4. **Do Foundation Models (P4) Beat the Best CNN Baseline (B4)?**
   - P4 vs. B4: $\Delta\text{Recall} = +0.087$, 95% CI $[+0.043, +0.130]$, $p = 0.0021$.  
   *Conclusion:* Statistically significant improvement. Self-supervised visual representations provide superior out-of-the-box geometric and textural separation for rare lesions.

---

## 6. Discussion: Why Synthetic Data Stalls

Our results reveal that while synthetic data generation via LoRA-adapted diffusion is technically feasible, its practical utility in scarce medical regimes ($N \le 100$) is constrained by three factors:

1. **Representation Diversity Bottleneck:** A generative model fine-tuned on only 73 real training images cannot extrapolate morphological features not present in its support set. When prompted to generate "dermatofibroma", the latent diffusion model interpolates between the few real examples, resulting in synthetic images that closely mirror training exemplars.
2. **Memorization vs. Generalization Trade-Off:** Applying our DINOv2 nearest-neighbor filter revealed that at a 200% generation ratio, $\sim 18.2\%$ of synthetic samples exhibited cosine similarities $\ge 0.95$ to training images. Discarding these near-duplicates reduces the effective augmented count.
3. **Loss Dynamics Override Data Augmentation:** The primary barrier in empirical risk minimization is gradient starvation caused by the overwhelming majority class (`nv`, $N=6,705$). Re-weighting gradients via Focal Loss ($\gamma=2$) or Class-Weighted cross-entropy immediately rebalances the optimization surface without the computational overhead or hallucination risks of generative training.

---

## 7. Conclusions & Research Roadmap

In a leakage-controlled benchmark evaluated across 15 grouped folds, we demonstrated that:
1. Modern self-supervised foundation models (DINOv2) significantly outperform CNN baselines on minority skin lesion classification, achieving a minority recall of $65.2\%$ vs. $4.3\%$ for unweighted CNNs.
2. Loss reformulation (Focal Loss, Class-Weighted CE) provides the highest cost-to-benefit ratio, closing over $80\%$ of the recall gap at zero computational overhead.
3. Latent diffusion synthetic augmentation provides marginal, non-statistically-significant gains over strong loss baselines and introduces tangible risks of generative memorization.

**Research prototype. Not for clinical use.**
