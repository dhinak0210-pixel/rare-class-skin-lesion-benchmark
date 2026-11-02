# Benchmark Progress & Roadmap Checklist

**Project Title:** Does Synthetic Data Really Help? A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses and Synthetic Augmentation for Minority-Class Skin Lesion Classification  
**Target Minority Class:** Dermatofibroma (`df`, 115 images in HAM10000)  
**Primary Metric:** Minority-class recall (with mean ± std, 95% bootstrap CIs, paired tests)  
**Protocol:** 5-fold StratifiedGroupKFold grouped by `lesion_id`, repeated across 3 seeds (15 folds total), strict train-only synthesis, zero patient overlap.

---

## Phase Breakdown & Tracking

- [x] **Phase 0: Project Setup, Environment & Progress Tracking**
  - [x] Canonical directory structure (`configs/`, `data/`, `src/`, `notebooks/`, `scripts/`, `results/`, `docs/`, `app/`, `report/`, `tests/`)
  - [x] Production & research dependencies defined in `requirements.txt`
  - [x] Environment configuration notes (`environment.md`)
  - [x] MIT License added
  - [x] Master progress checklist initialized (`docs/PROGRESS.md`)

- [x] **Phase 1: Data Acquisition, Leak-Free Splitting & Leakage Verification**
  - [x] HAM10000 download script (`scripts/download_data.sh`) & metadata validator
  - [x] Grouped stratified splitting (`scripts/make_splits.py`) by `lesion_id` across 5 folds and 3 seeds (`data/splits.csv`)
  - [x] Exploratory Data Analysis (EDA) generation with class/lesion distributions saved to `results/figures/`
  - [x] Automated leakage tests (`tests/test_splits.py`, `tests/test_leakage.py`) confirming zero lesion overlap between train/val/test
  - [x] External test curation (`scripts/prepare_external_set.py`, `data/isic2019_external_metadata.csv`, `docs/external_set.md`) with 15,316 disjoint non-HAM images

- [x] **Phase 2: CPU End-to-End Pipeline Smoke Test**
  - [x] Modular components in `src/` (`dataset.py`, `models.py`, `losses.py`, `train.py`, `eval.py`, `utils.py`, `synth.py`, `embed.py`, `stats.py`, `gradcam.py`)
  - [x] DermaMNIST / synthetic toy verification on CPU (`tests/test_pipeline_smoke.py`)
  - [x] Validation that metrics calculations match `scikit-learn` exactly (`tests/test_metrics.py`)
  - [x] Full test suite running and passing (14/14 tests)

- [x] **Phase 3: CNN Baselines (B0–B4) on HAM10000**
  - [x] B0: Plain cross-entropy EfficientNet-B0
  - [x] B1: B0 + Classical augmentations (flips, affine, color jitter)
  - [x] B2: B1 + Weighted Random Sampler (balanced oversampling)
  - [x] B3: B1 + Class-weighted cross-entropy loss
  - [x] B4: B1 + Focal loss ($\gamma=2$)
  - [x] Checkpoint-resumable Kaggle GPU runner notebook (`notebooks/kaggle_runner_baselines.ipynb`)

- [x] **Phase 4: Pretrained Foundation Models (P1–P5)**
  - [x] Embeddings extractor (`src/embed.py`) for DINOv2 (`facebook/dinov2-small`) and BiomedCLIP
  - [x] P1: DINOv2 ViT-S/14 frozen embeddings + Logistic Regression
  - [x] P2: DINOv2 frozen embeddings + Class-Weighted MLP
  - [x] P3: BiomedCLIP frozen embeddings + Linear Head
  - [x] P4: DINOv2 partial fine-tuning (last 3 blocks, low learning rate)
  - [x] P5: Best foundation setup + Focal Loss
  - [x] Kaggle GPU runner notebook (`notebooks/kaggle_runner_pretrained.ipynb`)

- [x] **Phase 5: Generative Synthetic Augmentation (S1–S3)**
  - [x] Strict isolation guard: generative models train ONLY on training fold `df` images
  - [x] S1: Minority-targeted MixUp and CutMix
  - [x] S2: Stable Diffusion 1.5 + LoRA fine-tuning on training `df` at 50%, 100%, 200% ratios
  - [x] S3: StyleGAN2-ADA / DCGAN minority generator configuration
  - [x] Synthetic evaluation: FID score against real fold `df`, DINOv2 nearest-neighbor memorization filter (< 0.95)
  - [x] Blind real-vs-synthetic diagnostic grid generation function
  - [x] Few-shot curve configuration and evaluation pipeline
  - [x] Final combined arm: Best Pretrained + Imbalance Loss + Synthetic Augmentation
  - [x] Kaggle GPU runner notebook (`notebooks/kaggle_runner_synthetic.ipynb`)

- [x] **Phase 6: Trust, Calibration & Explainability**
  - [x] Post-hoc temperature scaling on validation fold & Expected Calibration Error (ECE)
  - [x] Selective prediction ("Uncertain, please review") referral curves and threshold tuning
  - [x] Explainability maps: Grad-CAM for EfficientNet-B0 and Attention Rollout for DINOv2
  - [x] Export illustrative panels: correctly classified, false negatives, and confused minority cases

- [x] **Phase 7: External Out-of-Distribution Validation (Cleaned ISIC 2019)**
  - [x] Disjoint ISIC 2019 subset extraction (filtering out all 10,015 HAM10000 images and matching lesions)
  - [x] Documented dataset reconciliation in `docs/external_set.md` (15,316 images, 124 non-HAM `df` samples)
  - [x] Automated test verifying zero patient/lesion leakage in `tests/test_leakage.py`

- [x] **Phase 8: Rigorous Statistical Aggregation & Publication Figures**
  - [x] Metric aggregation across 15 evaluations per arm (5 folds × 3 seeds)
  - [x] Paired bootstrap confidence intervals (95%, $B=2,000$) and Wilcoxon signed-rank tests
  - [x] Publication-grade EDA plots (`results/figures/`)
  - [x] Automated summary table generation (`results/tables/benchmark_table.md`, `results/tables/aggregated_benchmark_summary.csv`)

- [x] **Phase 9: Interactive Gradio Demo & Deployment Packaging**
  - [x] Gradio web application (`app/app.py`) with image upload, classification, uncertainty referral flag, and Grad-CAM overlay
  - [x] CPU-optimized inference and Hugging Face Spaces deployment manifests (`app/README.md`, `app/requirements.txt`)
  - [x] Prominent clinical disclaimer: "Research prototype. Not for clinical use."

- [x] **Phase 10: Academic Report, Methodology & Limitations**
  - [x] Comprehensive manuscript / technical report (`report/report.md`, 6–8 pages)
  - [x] Complete presentation slides outline (`report/slides_outline.md`, 15 slides)
  - [x] Detailed methodology document (`docs/methodology.md`) and honest limitations (`docs/limitations.md`)
  - [x] Benchmark master README with reproducibility instructions

---

## Log of Completed Actions
- **Phase 0 Completed:**
  - Standardized repository layout created.
  - Core requirements.txt created with strict version constraints.
  - Progression log and verification plan outlined.
- **Phase 1 Completed:**
  - HAM10000 metadata and split pipeline constructed (`scripts/make_splits.py`).
  - 5-fold StratifiedGroupKFold splits generated across 3 seeds (`data/splits.csv`), 100% leak-free.
  - Exploratory Data Analysis figures generated in `results/figures/`.
  - ISIC 2019 purged of all HAM10000 images and lesions (`scripts/prepare_external_set.py`), producing `data/isic2019_external_metadata.csv` (15,316 images).
  - Cleaned external set documented in `docs/external_set.md`.
  - Unit tests in `tests/test_splits.py` and `tests/test_leakage.py` passing (6/6).
- **Phase 2 Completed:**
  - Core modular codebase implemented in `src/` (`utils.py`, `eval.py`, `losses.py`, `dataset.py`, `models.py`, `train.py`, `embed.py`, `synth.py`, `stats.py`, `gradcam.py`).
  - End-to-end CPU smoke tests and metric parity verified against scikit-learn (`tests/test_pipeline_smoke.py`, `tests/test_metrics.py`).
  - Full automated test suite passing cleanly (14/14 tests).
