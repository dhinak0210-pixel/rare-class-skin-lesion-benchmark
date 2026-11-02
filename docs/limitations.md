# Limitations & Risk Analysis

> **MANDATORY CLINICAL DISCLAIMER:**  
> This software, benchmark results, and accompanying models constitute a **research prototype**. They are **not certified medical devices** and are **not intended for clinical use, patient diagnosis, triage, or autonomous treatment planning**.

---

## 1. Data-Level & Domain Limitations

1. **Severe Absolute Minority Count:**
   - In HAM10000, dermatofibroma (`df`) is represented by only 115 dermoscopic images from 82 unique lesions. While cross-validation is carefully stratified and grouped, a test set of 23 minority images per fold means that a single misclassification changes the measured minority recall by $\sim 4.35\%$. Confidence intervals appropriately capture this variance, but findings should be validated on larger international registries.
2. **Single Imaging Modality:**
   - The primary dataset consists exclusively of dermoscopic polarized/contact imaging. The models have not been evaluated on clinical macro-photographs (e.g. smartphone images), which exhibit divergent lighting, scale, viewing angles, and background skin artifacts.
3. **Demographic and Phenotypic Skew:**
   - The HAM10000 cohort was collected primarily from fair-skinned European and Australian populations (Fitzpatrick skin types I–III). Performance on richly pigmented skin (Fitzpatrick types IV–VI) is unverified and likely impaired due to underrepresentation in the training distribution.

---

## 2. Model & Algorithmic Limitations

1. **Generative Memorization & Hallucination:**
   - Few-shot diffusion and GAN generators trained on small cohorts risk memorizing real lesion morphology. While our DINOv2 cosine filter discards samples with $\ge 0.95$ similarity, generative artifacts (e.g., unnatural vascular borders or blended pigment structures) may provide misleading signals to downstream discriminative classifiers.
2. **Computational Constraints & Low-Cost Trade-offs:**
   - To adhere to the strict zero-cost constraint, parameter-efficient LoRA adapters and frozen embedding feature extractors were prioritized over multi-node pre-training or trillion-parameter medical vision backbones.
3. **Threshold Calibration Sensitivity:**
   - Selective prediction ("Uncertain, please review") thresholds chosen on validation folds can experience distribution shift on out-of-distribution clinical populations, causing higher-than-expected referral rates in practice.

---

## 3. Translational & Ethical Considerations

- **Biopsy Decision-Making:** In clinical dermatology, misclassifying an early-stage invasive melanoma or squamous cell carcinoma carries catastrophic consequences, whereas misclassifying a benign dermatofibroma primarily introduces unnecessary biopsy burden. Algorithmic trade-offs must always prioritize patient safety and clinician-in-the-loop oversight.
