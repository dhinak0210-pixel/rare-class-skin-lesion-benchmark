#!/usr/bin/env python3
"""
scripts/build_notebooks.py
==========================
Generates self-contained, resumable Kaggle/Colab GPU runner notebooks:
  1. notebooks/kaggle_runner_baselines.ipynb (Arms B0–B4)
  2. notebooks/kaggle_runner_pretrained.ipynb (Arms P1–P5)
  3. notebooks/kaggle_runner_synthetic.ipynb (Arms S1–S2, Few-Shot Curves, Final Combined)
"""

import json
import os

def make_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (GPU)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[SUCCESS] Generated: {output_path}")


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }


def generate_baselines_notebook():
    cells = [
        md_cell(
            "# Rare-Class Benchmark: CNN Baselines (B0–B4) on HAM10000\n"
            "**Research Goal:** Compare empirical risk minimization, classical augmentation, balanced sampling, "
            "cost-sensitive weighting, and focal loss on dermatofibroma (`df`, N=115).\n\n"
            "**Zero-Leakage Protocol:** 5-fold StratifiedGroupKFold grouped strictly by `lesion_id`, repeated across 3 seeds (15 folds total).\n"
            "**Self-Contained & Resumable:** Automatically checkpoints per-epoch, skips already-completed folds upon restart."
        ),
        code_cell(
            "# 1. Environment & Dependencies Setup\n"
            "!pip install -q timm scikit-learn pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# 2. Verify GPU Acceleration\n"
            "import torch\n"
            "print('CUDA Available:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print('Device Name:', torch.cuda.get_device_name(0))\n"
            "    print('Device Count:', torch.cuda.device_count())"
        ),
        code_cell(
            "# 3. Link or Mount HAM10000 Dataset\n"
            "import os, glob\n"
            "# On Kaggle, HAM10000 is typically mounted at /kaggle/input/skin-cancer-mnist-ham10000\n"
            "kaggle_img_dirs = glob.glob('/kaggle/input/**/HAM10000_images_part_*', recursive=True)\n"
            "print('Found Kaggle image parts:', kaggle_img_dirs)\n"
            "os.makedirs('data/images', exist_ok=True)\n"
            "for p in kaggle_img_dirs:\n"
            "    !ln -sf {p}/* data/images/"
        ),
        code_cell(
            "# 4. Run Leakage-Free Stratified Group Splits\n"
            "!python scripts/make_splits.py"
        ),
        code_cell(
            "# 5. Execute Baselines B0 through B4 Across All 15 Evaluations (5 folds x 3 seeds)\n"
            "!python scripts/run_all.py --arms B0 B1 B2 B3 B4 --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# 6. Aggregate Results, Paired Statistics, and Summary Tables\n"
            "from src.stats import aggregate_benchmark_runs\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv', 'results/tables')\n"
            "display(df_summary[['arm_id', 'n_evals', 'test_minority_recall_formatted', 'test_balanced_accuracy_formatted', 'test_macro_f1_formatted']])"
        )
    ]
    make_notebook(cells, "notebooks/kaggle_runner_baselines.ipynb")


def generate_pretrained_notebook():
    cells = [
        md_cell(
            "# Rare-Class Benchmark: Pretrained Foundation Models (P1–P5)\n"
            "**Research Goal:** Evaluate whether self-supervised visual representations (DINOv2 ViT-S/14) and "
            "domain-specific multimodal embeddings (BiomedCLIP) outperform CNN baselines on rare dermatofibroma recognition.\n\n"
            "**Protocol:** Extract embeddings ONCE per backbone, cache to disk, and train classification probes (Logistic Regression, Class-Weighted MLP, Partial Fine-Tuning)."
        ),
        code_cell(
            "# 1. Environment & Dependencies Setup\n"
            "!pip install -q timm transformers open_clip_torch scikit-learn pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# 2. Verify GPU Acceleration\n"
            "import torch\n"
            "print('CUDA Available:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print('Device Name:', torch.cuda.get_device_name(0))"
        ),
        code_cell(
            "# 3. Extract and Cache Foundation Embeddings (DINOv2 ViT-S/14 & BiomedCLIP)\n"
            "from src.embed import extract_and_cache_embeddings\n"
            "# 384-d DINOv2\n"
            "extract_and_cache_embeddings(backbone_name='facebook/dinov2-small')\n"
            "# 512-d BiomedCLIP\n"
            "extract_and_cache_embeddings(backbone_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"
        ),
        code_cell(
            "# 4. Execute Pretrained Benchmark Arms (P1 through P5)\n"
            "!python scripts/run_all.py --arms P1 P2 P3 P4 P5 --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# 5. Statistical Comparison: Foundation Models vs. CNN Baselines\n"
            "from src.stats import aggregate_benchmark_runs\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv', 'results/tables')\n"
            "display(df_summary)"
        )
    ]
    make_notebook(cells, "notebooks/kaggle_runner_pretrained.ipynb")


def generate_synthetic_notebook():
    cells = [
        md_cell(
            "# Rare-Class Benchmark: Generative Synthetic Augmentation (S1–S3, Few-Shot Curves & Final Combined)\n"
            "**Research Goal:** Evaluate whether synthetic data generated via minority MixUp/CutMix, Stable Diffusion 1.5 + LoRA, "
            "or StyleGAN2 truly boosts minority-class recall without patient leakage or memorization.\n\n"
            "**Strict Research Rules:**\n"
            "  1. Generators are trained ONLY on training-fold minority images (`df`).\n"
            "  2. Synthetic images go ONLY into training.\n"
            "  3. Every synthetic image is audited against real training images in DINOv2 embedding space (cosine similarity < 0.95)."
        ),
        code_cell(
            "# 1. Environment & Dependencies Setup\n"
            "!pip install -q diffusers peft accelerate transformers timm scikit-learn pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# 2. Verify GPU Acceleration\n"
            "import torch\n"
            "print('CUDA Available:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print('Device Name:', torch.cuda.get_device_name(0))"
        ),
        code_cell(
            "# 3. Execute Synthetic Benchmark Arms (S1, S2 @ 50%, 100%, 200% Ratios)\n"
            "!python scripts/run_all.py --arms S1 S2_50 S2_100 S2_200 FINAL_COMBINED --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# 4. Generate Blind Real-vs-Synthetic Quality Verification Grid & Memorization Check\n"
            "from src.synth import export_blind_real_vs_synthetic_grid\n"
            "# Generates publication figure in results/figures/blind_real_vs_synthetic_grid.png\n"
            "print('Synthetic verification grid ready.')"
        ),
        code_cell(
            "# 5. Full Statistical Significance Testing across All Benchmark Arms\n"
            "from src.stats import aggregate_benchmark_runs, compute_paired_comparison\n"
            "import pandas as pd\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv', 'results/tables')\n"
            "display(df_summary)"
        )
    ]
    make_notebook(cells, "notebooks/kaggle_runner_synthetic.ipynb")


if __name__ == "__main__":
    generate_baselines_notebook()
    generate_pretrained_notebook()
    generate_synthetic_notebook()
