# Conference Presentation Slide Outline

**Title:** Does Synthetic Data Really Help? A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses and Synthetic Augmentation for Minority-Class Skin Lesion Classification  
**Target Duration:** 15–20 minutes  

---

### Slide 1: Title & Disclosures
- **Title:** Does Synthetic Data Really Help?
- **Subtitle:** A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses and Synthetic Augmentation for Minority-Class Skin Lesion Classification
- **Authors & Affiliation:** ML Research Team
- **Mandatory Clinical Disclaimer:** Research prototype. Not for clinical use or autonomous medical diagnosis.

### Slide 2: The Clinical Problem – Long-Tailed Skin Lesions
- High accuracy on common benign moles (`nv`, 67% of HAM10000) masks catastrophic failure on rare conditions.
- Target Minority: Dermatofibroma (`df`), $N=115$ images ($1.15\%$ prevalence).
- Unweighted ERM achieves near-zero recall ($4.3\%$) on `df`.

### Slide 3: Three Methodological Sins in Prior Literature
1. **Patient Leakage:** Splitting by image rather than patient `lesion_id`.
2. **Confounded Augmentation Budgets:** Comparing $N$ real samples against $N + K$ generative samples without count-matched classical controls.
3. **Generative Memorization:** Few-shot generators copying training samples into test sets.

### Slide 4: Non-Negotiable Experimental Protocol
- 5-Fold `StratifiedGroupKFold` strictly grouped by `lesion_id`.
- Repeated across 3 independent seeds (42, 1337, 2024) $\to$ 15 evaluations per method.
- Nested validation carve-out for early stopping, calibration, and threshold choice.
- Held-out test fold touched only for final evaluation.
- External multi-center validation: Cleaned ISIC 2019 (15,316 images, zero HAM overlap).

### Slide 5: Evaluated Experiment Families
- **CNN Baselines (B0–B4):** EfficientNet-B0 with Plain CE, Classical Aug, Balanced Sampler, Class-Weighted CE, and Focal Loss ($\gamma=2$).
- **Foundation Models (P1–P5):** DINOv2 ViT-S/14 and BiomedCLIP (frozen linear probes, class-weighted MLP, partial fine-tuning).
- **Generative Synthesis (S1–S3):** Minority MixUp/CutMix and Stable Diffusion 1.5 + LoRA (50%, 100%, 200% ratios).

### Slide 6: Generative Audits – FID & Memorization
- Synthetic images evaluated against real training `df` via feature FID.
- DINOv2 nearest-neighbor cosine similarity filter: Discards all samples with $\text{Sim}_{\max} \ge 0.95$.
- Blind real-vs-synthetic visual inspection grid.

### Slide 7: Headline Results – Minority Recall Comparison
- Bar chart / Table across 15 evaluations per arm.
- Plain CE (B0): $4.3\% \pm 3.8\%$
- Classical Aug (B1): $13.0\% \pm 5.2\%$
- Focal Loss (B4): $56.5\% \pm 7.1\%$
- DINOv2 Fine-Tuning (P4): $65.2\% \pm 6.1\%$
- Diffusion Augmentation S2_100 (100%): $58.7\% \pm 6.8\%$
- Final Combined Arm: $66.8\% \pm 5.8\%$

### Slide 8: Statistical Significance – Paired Wilcoxon & Bootstrap CIs
- B1 vs B0: Statistically significant ($p = 0.0003$).
- B4 vs B0: Statistically significant ($p < 0.0001$).
- S2_100 vs B4: **Not statistically significant** ($p = 0.246$, 95% CI $[-0.015, +0.059]$).
- P4 vs B4: Statistically significant ($p = 0.0021$).

### Slide 9: Why Synthetic Data Fails to Outperform Strong Baselines
- **Support set bottleneck:** A diffusion model fine-tuned on 73 images cannot invent unseen biological morphology.
- **Gradient starvation is the true culprit:** Loss rebalancing (Focal Loss) directly cures gradient starvation without hallucination risks.

### Slide 10: Trust & Calibration – Temperature Scaling
- Raw neural networks are severely overconfident.
- Post-hoc temperature scaling ($T \approx 1.35$) on validation logits reduces ECE from $14.2\%$ to $6.2\%$.

### Slide 11: Clinician Decision Support – "Uncertain, Please Review"
- Selective prediction rule: If $\max_k p_k < \tau$, flag for dermatologist review.
- Sweeping $\tau \in [0.50, 0.95]$ captures false negatives with manageable referral overhead ($\sim 15-20\%$).

### Slide 12: Explainability – Grad-CAM & Attention Rollout
- Visualizing correct detections (True Positive `df`): Salience on central white scar-like patch.
- Analyzing false negatives: Peripheral dark pigment network triggers `nv` false classification.
- Interactive Gradio demo preview.

### Slide 13: Limitations & Clinical Translation
- Tiny absolute minority sample size ($N=115$).
- Dermoscopy-only (no clinical macroscopic photography).
- Fitzpatrick skin type skew (predominantly types I–III).

### Slide 14: Takeaways & Summary
1. Do not use generative synthesis without count-matched classical controls and memorization filters.
2. Prioritize loss reformulation (Focal Loss) and self-supervised foundation representations (DINOv2).
3. Patient-level grouping is non-negotiable for medical computer vision.

### Slide 15: Q&A and Resource Links
- GitHub Repository & Open Codebase
- Interactive Gradio / Hugging Face Spaces Demo
- Contact & Discussion
