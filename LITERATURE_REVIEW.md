# Literature Review & Novelty Matrix

**Paper Title:** *Scarcity-Aware Synthetic Augmentation: Predicting When Generative Data Helps Rare Disease Classification*

---

## 1. Structured Related Work Matrix

| Citation (Author, Year) | Dataset & Modality | Generative Method | Scarcity Levels Tested ($N$) | Count-Matched Classical Control? | Predictive Signal Analysis? (FID/KID/Div $\to$ $\Delta\text{F1}$) | Key Finding & Critical Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frid-Adar et al. (2018)** *Neurocomputing* | Liver CT (182 lesions: cysts, metastases, hemangiomas) | DCGAN / ACGAN (64×64) | None ($N = \text{full}$, ~50–60/class) | **No** (Combined classical + GAN without isolating $K$) | **No** (Visual inspection only) | Reported sensitivity boost (78.6% $\to$ 85.7%), but confounded classical augmentations with GAN samples without fixing budget $K$. |
| **Waheed et al. (2020)** *IEEE Access* | COVID-19 Chest X-ray (403 images) | CovidGAN (Auxiliary Classifier GAN) | None ($N = \text{full}$, 403 total) | **No** (Compared GAN vs unaugmented only) | **No** (Reported only downstream accuracy) | Synthetic data improved accuracy from 85% to 95%, but failed to control for affine transformations (rotations/crops) of the identical budget. |
| **Bissoto et al. (2021)** *MICCAI* | HAM10000 Dermoscopy (10,015 images) | StyleGAN2 / Pix2PixHD | None ($N = \text{full}$, ~115 dermatofibroma) | **Partial** (Included classical baseline, but unequal sample counts) | **No** (Only evaluated FID without correlating to downstream gain) | Showed GAN artifacts leak into classifier decisions; highlighted GAN memorization risks but did not study few-shot regimes ($N < 50$). |
| **Mittal et al. (2023)** *MedIA* | ISIC 2018 Skin Lesions & CheXpert | Stable Diffusion + LoRA fine-tuning | $N \in \{25, 50, 100, \text{all}\}$ | **No** (Tested zero-shot/few-shot diffusion vs unaugmented) | **Partial** (Examined FID, but observed FID did not rank downstream classifiers) | Diffusion outperforms GANs at $N \ge 50$, but high computational cost prevents local utility checks; lacks count-matched affine controls. |
| **AlAmir & Bialowas (2022)** *Diagnostics* | HAM10000 Dermoscopy | CycleGAN / StarGAN multi-domain | None ($N = \text{full}$) | **No** (Evaluated synthetic additions on top of existing sets) | **No** (No memorization or diversity checks) | Balanced class imbalance, but did not investigate mode collapse at small $N \le 20$. |
| **Skandarani et al. (2023)** *Computer Vision & Image Understanding* | MedMNIST (PathMNIST, DermaMNIST) | DCGAN, WGAN-GP, VAE, DDPM | None ($N = \text{full}$) | **No** (Systematic generative model comparison, no count controls) | **Partial** (Measured FID and Precision/Recall, but no regression to $\Delta\text{F1}$) | Extensive benchmarking showing generative models often fail to beat strong classical baselines; called for utility predictors. |
| **Karras et al. (2020)** *NeurIPS* | CIFAR-10, FFHQ | DiffAugment + StyleGAN2 | $N \in \{100, 1000\}$ | **No** (Vision benchmark, not rare disease classification) | **No** (Generative image fidelity focus) | Differentiable augmentation prevents discriminator memorization on small data; establishes training foundation for few-shot GANs. |
| **Our Paper (2026)** | **HAM10000/DermaMNIST & BloodMNIST** (2 distinct modalities) | **Few-Shot DCGAN + DiffAugment** | **$N \in \{10, 20, 50, 100\}$** | **YES (Strict count-matched classical control $K=200$)** | **YES (FID, KID, Diversity, NN-Copying $\to$ downstream $\Delta\text{F1}$)** | **First framework isolating generative value vs volume, proving classical beats GANs at $N \le 20$, and providing a depth-2 decision rule that zero-shot transfers across modalities.** |

---

## 2. Research Gap & Novelty Synthesis

Despite the proliferation of generative adversarial networks (GANs) and diffusion models for medical image augmentation, existing literature suffers from two pervasive methodological flaws:
1. **Confounded Sample Budgets:** The vast majority of published studies compare a baseline trained on $N$ scarce samples against a hybrid dataset containing $N + K$ samples (where $K$ synthetic images are injected), without evaluating a count-matched control where the same $K$ budget is generated purely via classical affine/photometric transformations. Consequently, reported performance boosts are frequently attributable to increased sample volume rather than generative novelty.
2. **The "Train-and-Hope" Paradigm:** Practitioners currently lack cheap, pre-classifier diagnostic signals to determine *a priori* whether training a generative model will yield positive downstream utility ($\Delta\text{F1} > 0$) or induce catastrophic performance drops via mode collapse and training set memorization.

**Our Contribution:**
Our work directly resolves both gaps. By instituting a rigorous count-matched experimental protocol across four fine-grained scarcity levels ($N \in \{10, 20, 50, 100\}$) over multiple random seeds, we demonstrate that **at extreme scarcity ($N \le 20$), count-matched classical augmentation decisively outperforms few-shot GANs** due to severe discriminator memorization (confirmed by near-zero nearest-neighbor copying distances). Furthermore, we formulate four cheap pre-training diagnostic signals—most notably unbiased polynomial **Kernel Inception Distance (KID)** and **Nearest-Neighbor Copying Distance**—that strongly correlate with downstream rare-class gain. We distill these signals into an interpretable depth-2 decision tree that achieves high decision accuracy and successfully zero-shot transfers from dermatoscopy to peripheral blood cytology, providing clinicians and researchers with an actionable rubric before committing valuable GPU compute.
