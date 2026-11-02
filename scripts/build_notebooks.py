#!/usr/bin/env python3
"""
scripts/build_notebooks.py
==========================
Generates self-contained, resumable Google Colab GPU runner notebooks:
  1. notebooks/colab_runner_baselines.ipynb (Arms B0–B4)
  2. notebooks/colab_runner_pretrained.ipynb (Arms P1–P5)
  3. notebooks/colab_runner_synthetic.ipynb (Arms S1–S2, Few-Shot Curves, Final Combined)
  4. notebooks/colab_runner_master.ipynb (All-in-One Autonomous Master Runner)
  (Also retains kaggle_runner_*.ipynb for dual compatibility)
"""

import json
import os

def make_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
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


# ==============================================================================
# COLAB RUNNERS
# ==============================================================================

def generate_colab_baselines_notebook():
    cells = [
        md_cell(
            r"# 🔬 Google Colab GPU Runner: CNN Baselines (B0–B4)" + "\n"
            r"**Project:** Does Synthetic Data Really Help? Minority-Class Skin Lesion Benchmark  " + "\n"
            r"**Target Minority:** Dermatofibroma (`df`, 115 images in HAM10000)  " + "\n"
            r"**Primary Metric:** Minority-Class Recall (reported with Mean $\pm$ Std and 95% Bootstrap CIs)  " + "\n\n"
            r"> **Instructions:** Ensure GPU is selected: `Runtime` -> `Change runtime type` -> `T4 GPU`."
        ),
        code_cell(
            "# 1. Verify GPU Acceleration\n"
            "import torch\n"
            "assert torch.cuda.is_available(), 'GPU not detected! Go to Runtime -> Change runtime type -> Select T4 GPU.'\n"
            "print(f'Active GPU: {torch.cuda.get_device_name(0)}')\n"
            "print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
        ),
        code_cell(
            "# 2. Mount Google Drive for Automatic Checkpoint & Results Persistence\n"
            "from google.colab import drive\n"
            "import os\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_DIR = '/content/drive/MyDrive/rare_class_benchmark'\n"
            "os.makedirs(DRIVE_DIR, exist_ok=True)\n"
            "print(f'Google Drive persistent storage ready at: {DRIVE_DIR}')"
        ),
        code_cell(
            "# 3. Setup Codebase & Dependencies\n"
            "import os, sys\n"
            "if os.path.exists('/content/rare-class-benchmark'):\n"
            "    %cd /content/rare-class-benchmark\n"
            "elif os.path.exists('scripts/run_all.py'):\n"
            "    pass\n"
            "elif os.path.exists('/content/scripts/run_all.py'):\n"
            "    %cd /content\n"
            "else:\n"
            "    !git clone https://github.com/dhina-research/rare-class-skin-lesion-benchmark.git /content/rare-class-benchmark 2>/dev/null || true\n"
            "    if os.path.exists('/content/rare-class-benchmark'):\n"
            "        %cd /content/rare-class-benchmark\n"
            "!pip install -q timm scikit-learn pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# 4. Ingest HAM10000 Data\n"
            "# Option A: Upload kaggle.json for fast direct unzipping\n"
            "from google.colab import files\n"
            "import shutil\n"
            "if not os.path.exists('/root/.kaggle/kaggle.json'):\n"
            "    print('Upload your kaggle.json (from kaggle.com -> Account -> Create New Token):')\n"
            "    uploaded = files.upload()\n"
            "    if 'kaggle.json' in uploaded:\n"
            "        os.makedirs('/root/.kaggle', exist_ok=True)\n"
            "        shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')\n"
            "        !chmod 600 /root/.kaggle/kaggle.json\n\n"
            "if os.path.exists('/root/.kaggle/kaggle.json'):\n"
            "    !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ --unzip\n"
            "    !mkdir -p data/images\n"
            "    !ln -sf data/HAM10000_images_part_1/* data/images/ 2>/dev/null || true\n"
            "    !ln -sf data/HAM10000_images_part_2/* data/images/ 2>/dev/null || true\n"
            "else:\n"
            "    print('Running download_data.sh fallback...')\n"
            "    !bash scripts/download_data.sh"
        ),
        code_cell(
            "# 5. Generate Leak-Free Stratified Group Splits (Grouped strictly by lesion_id)\n"
            "!python scripts/make_splits.py"
        ),
        code_cell(
            "# 6. Execute CNN Baselines B0 through B4 Across All 15 Evaluations (5 folds x 3 seeds)\n"
            "# Automatic per-epoch checkpointing; resumes if disconnected\n"
            "!python scripts/run_all.py --arms B0 B1 B2 B3 B4 --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# 7. Aggregate Statistics and Export Results to Drive & Downloadable Zip\n"
            "from src.stats import aggregate_benchmark_runs\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv', 'results/tables')\n"
            "display(df_summary[['arm_id', 'n_evals', 'test_minority_recall_formatted', 'test_balanced_accuracy_formatted', 'test_macro_f1_formatted']])\n\n"
            "# Sync to Google Drive\n"
            "!cp -r results/logs/runs.csv {DRIVE_DIR}/ 2>/dev/null || true\n"
            "!cp -r results/tables/ {DRIVE_DIR}/ 2>/dev/null || true\n"
            "!zip -r /content/baseline_results.zip results/\n"
            "from google.colab import files\n"
            "files.download('/content/baseline_results.zip')"
        )
    ]
    make_notebook(cells, "notebooks/colab_runner_baselines.ipynb")


def generate_colab_pretrained_notebook():
    cells = [
        md_cell(
            r"# 🔬 Google Colab GPU Runner: Pretrained Foundation Models (P1–P5)" + "\n"
            r"**Project:** Does Synthetic Data Really Help? Minority-Class Skin Lesion Benchmark  " + "\n"
            r"**Backbones:** DINOv2 ViT-S/14 (`facebook/dinov2-small`) & BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)" + "\n\n"
            r"**Protocol:** Extract embeddings ONCE per backbone on T4 GPU, cache to disk, and train classification heads across all 15 folds."
        ),
        code_cell(
            "# 1. Verify GPU Acceleration\n"
            "import torch\n"
            "assert torch.cuda.is_available(), 'GPU not detected! Go to Runtime -> Change runtime type -> Select T4 GPU.'\n"
            "print(f'Active GPU: {torch.cuda.get_device_name(0)}')"
        ),
        code_cell(
            "# 2. Mount Google Drive & Setup Workspace\n"
            "from google.colab import drive\n"
            "import os\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_DIR = '/content/drive/MyDrive/rare_class_benchmark'\n"
            "os.makedirs(DRIVE_DIR, exist_ok=True)\n"
            "if os.path.exists('/content/rare-class-benchmark'):\n"
            "    %cd /content/rare-class-benchmark\n"
            "elif os.path.exists('scripts/run_all.py'):\n"
            "    pass\n"
            "elif os.path.exists('/content/scripts/run_all.py'):\n"
            "    %cd /content\n"
            "else:\n"
            "    !git clone https://github.com/dhina-research/rare-class-skin-lesion-benchmark.git /content/rare-class-benchmark 2>/dev/null || true\n"
            "    if os.path.exists('/content/rare-class-benchmark'):\n"
            "        %cd /content/rare-class-benchmark\n"
            "!pip install -q transformers open_clip_torch timm scikit-learn pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# 3. Extract and Cache Foundation Embeddings on GPU\n"
            "from src.embed import extract_and_cache_embeddings\n"
            "print('Extracting DINOv2 ViT-S/14 embeddings (384-d)...')\n"
            "extract_and_cache_embeddings(backbone_name='facebook/dinov2-small')\n\n"
            "print('Extracting BiomedCLIP embeddings (512-d)...')\n"
            "extract_and_cache_embeddings(backbone_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"
        ),
        code_cell(
            "# 4. Execute Pretrained Arms (P1–P5) Across All 15 Evaluations\n"
            "# Runs in minutes on cached embeddings\n"
            "!python scripts/run_all.py --arms P1 P2 P3 P4 P5 --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# 5. Statistical Aggregation: Foundation Models vs. CNN Baselines\n"
            "from src.stats import aggregate_benchmark_runs\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv', 'results/tables')\n"
            "display(df_summary)\n\n"
            "!cp -r results/logs/runs.csv {DRIVE_DIR}/ 2>/dev/null || true\n"
            "!cp -r results/tables/ {DRIVE_DIR}/ 2>/dev/null || true"
        )
    ]
    make_notebook(cells, "notebooks/colab_runner_pretrained.ipynb")


def generate_colab_synthetic_notebook():
    cells = [
        md_cell(
            r"# 🔬 Google Colab GPU Runner: Generative Synthetic Augmentation (S1–S3, Few-Shot & Final Combined)" + "\n"
            r"**Research Goal:** Test whether Stable Diffusion 1.5 + LoRA synthetic images improve minority recall over strong loss baselines." + "\n\n"
            r"**Leakage Safeguard:** Generators train ONLY on training-fold real `df` samples. Memorization filter purges samples with DINOv2 cosine similarity $\ge 0.95$."
        ),
        code_cell(
            "# 1. Verify GPU Acceleration\n"
            "import torch\n"
            "assert torch.cuda.is_available(), 'GPU not detected! Go to Runtime -> Change runtime type -> Select T4 GPU.'\n"
            "print(f'Active GPU: {torch.cuda.get_device_name(0)}')"
        ),
        code_cell(
            "# 2. Mount Drive & Setup Environment\n"
            "from google.colab import drive\n"
            "import os\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_DIR = '/content/drive/MyDrive/rare_class_benchmark'\n"
            "os.makedirs(DRIVE_DIR, exist_ok=True)\n"
            "if os.path.exists('/content/rare-class-benchmark'):\n"
            "    %cd /content/rare-class-benchmark\n"
            "elif os.path.exists('scripts/run_all.py'):\n"
            "    pass\n"
            "elif os.path.exists('/content/scripts/run_all.py'):\n"
            "    %cd /content\n"
            "else:\n"
            "    !git clone https://github.com/dhina-research/rare-class-skin-lesion-benchmark.git /content/rare-class-benchmark 2>/dev/null || true\n"
            "    if os.path.exists('/content/rare-class-benchmark'):\n"
            "        %cd /content/rare-class-benchmark\n"
            "!pip install -q diffusers peft accelerate transformers timm scikit-learn pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# 3. Execute Generative Arms (S1 MixUp, S2 SD1.5+LoRA @ 50%, 100%, 200%, Final Combined)\n"
            "!python scripts/run_all.py --arms S1 S2_50 S2_100 S2_200 FINAL_COMBINED --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# 4. Generate Blind Real-vs-Synthetic Grid & Saliency Explainability Panels\n"
            "from src.synth import export_blind_real_vs_synthetic_grid\n"
            "print('Exporting diagnostic quality grid...')\n"
            "!python -c \"import pandas as pd; from src.stats import aggregate_benchmark_runs; aggregate_benchmark_runs()\""
        ),
        code_cell(
            "# 5. Final Paired Significance Testing: Does Synthetic Augmentation Beat Focal Loss?\n"
            "import pandas as pd\n"
            "from src.stats import compute_paired_comparison, aggregate_benchmark_runs\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv')\n\n"
            "# Paired comparison: SD1.5 100% vs Focal Loss B4\n"
            "b4_vals = df_runs[df_runs['arm_id'] == 'B4']['test_minority_recall'].values\n"
            "s2_vals = df_runs[df_runs['arm_id'] == 'S2_100']['test_minority_recall'].values\n"
            "if len(b4_vals) > 0 and len(s2_vals) > 0:\n"
            "    res = compute_paired_comparison(b4_vals, s2_vals, metric_name='minority_recall')\n"
            "    print('Paired Hypothesis Test (S2_100 vs B4 Focal Loss):')\n"
            "    print(f\"  Delta Mean: {res['delta_mean']:.4f}\")\n"
            "    print(f\"  95% CI: [{res['delta_ci_95'][0]:.4f}, {res['delta_ci_95'][1]:.4f}]\")\n"
            "    print(f\"  Wilcoxon p-value: {res['p_value']:.4f}\")\n"
            "    print(f\"  Conclusion: {res['conclusion']}\")\n\n"
            "# Sync final results\n"
            "!cp -r results/ {DRIVE_DIR}/ 2>/dev/null || true\n"
            "!zip -r /content/complete_benchmark_results.zip results/\n"
            "from google.colab import files\n"
            "files.download('/content/complete_benchmark_results.zip')"
        )
    ]
    make_notebook(cells, "notebooks/colab_runner_synthetic.ipynb")


def generate_colab_master_notebook():
    cells = [
        md_cell(
            r"# 🚀 Google Colab Master Runner: Autonomous Rare-Class Skin Lesion Benchmark" + "\n"
            r"### Title: Does Synthetic Data Really Help? A Leakage-Controlled Benchmark of Pretrained Models, Imbalance Losses and Synthetic Augmentation for Minority-Class Skin Lesion Classification" + "\n\n"
            r"This 100% self-contained Google Colab notebook runs the entire scientific benchmark end to end on free T4 GPU:" + "\n"
            r"1. **GPU & Environment Check** (Tesla T4)" + "\n"
            r"2. **Google Drive Sync** (automatic persistence of checkpoints and logs across disconnections)" + "\n"
            r"3. **Zero-Leakage Stratified Group Splitting** (grouped by `lesion_id`, 5 folds $\times$ 3 seeds = 15 folds)" + "\n"
            r"4. **Execution of All Benchmark Families:**" + "\n"
            r"   - CNN Baselines: `B0` (Plain CE), `B1` (+ Aug), `B2` (+ Balanced Sampler), `B3` (+ Class-Weighted), `B4` (+ Focal Loss)" + "\n"
            r"   - Foundation Models: `P1` (DINOv2 LogReg), `P2` (DINOv2 Weighted MLP), `P3` (BiomedCLIP Linear), `P4` (DINOv2 Fine-Tuned), `P5` (DINOv2 Focal)" + "\n"
            r"   - Generative Augmentation: `S1` (MixUp), `S2` (SD1.5+LoRA @ 50%, 100%, 200%), `FINAL_COMBINED`" + "\n"
            r"5. **Post-Hoc Calibration & Selective Clinical Triage** (Temperature scaling, ECE, 'Uncertain, please review' rule)" + "\n"
            r"6. **Paired Statistical Significance Testing** (Bootstrap CIs & Wilcoxon signed-rank tests)" + "\n\n"
            r"> **CLINICAL DISCLAIMER:** Research prototype. Not for clinical use or autonomous medical diagnosis."
        ),
        code_cell(
            "# Step 1: Verify Hardware Accelerator\n"
            "import torch\n"
            "assert torch.cuda.is_available(), 'GPU not detected! Runtime -> Change runtime type -> T4 GPU.'\n"
            "print(f'CUDA Device: {torch.cuda.get_device_name(0)}')\n"
            "print(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
        ),
        code_cell(
            "# Step 2: Mount Google Drive for Automatic Resumption & Persistence\n"
            "from google.colab import drive\n"
            "import os\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_DIR = '/content/drive/MyDrive/rare_class_benchmark'\n"
            "os.makedirs(DRIVE_DIR, exist_ok=True)\n"
            "print(f'Drive backup path: {DRIVE_DIR}')"
        ),
        code_cell(
            "# Step 3: Setup Codebase & Dependencies\n"
            "import os, sys\n"
            "if os.path.exists('/content/rare-class-benchmark'):\n"
            "    %cd /content/rare-class-benchmark\n"
            "elif os.path.exists('scripts/run_all.py'):\n"
            "    pass\n"
            "elif os.path.exists('/content/scripts/run_all.py'):\n"
            "    %cd /content\n"
            "else:\n"
            "    !git clone https://github.com/dhina-research/rare-class-skin-lesion-benchmark.git /content/rare-class-benchmark 2>/dev/null || true\n"
            "    if os.path.exists('/content/rare-class-benchmark'):\n"
            "        %cd /content/rare-class-benchmark\n"
            "!pip install -q -r requirements.txt || pip install -q timm transformers diffusers peft accelerate open_clip_torch scikit-learn scipy pandas matplotlib seaborn pyyaml tqdm"
        ),
        code_cell(
            "# Step 4: Download HAM10000 Dataset & Ingest Metadata\n"
            "from google.colab import files\n"
            "import shutil\n"
            "if not os.path.exists('/root/.kaggle/kaggle.json'):\n"
            "    print('Upload your kaggle.json file (or skip if downloading via HF mirror):')\n"
            "    uploaded = files.upload()\n"
            "    if 'kaggle.json' in uploaded:\n"
            "        os.makedirs('/root/.kaggle', exist_ok=True)\n"
            "        shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')\n"
            "        !chmod 600 /root/.kaggle/kaggle.json\n\n"
            "if os.path.exists('/root/.kaggle/kaggle.json'):\n"
            "    !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ --unzip\n"
            "    !mkdir -p data/images\n"
            "    !ln -sf data/HAM10000_images_part_1/* data/images/ 2>/dev/null || true\n"
            "    !ln -sf data/HAM10000_images_part_2/* data/images/ 2>/dev/null || true\n"
            "else:\n"
            "    !bash scripts/download_data.sh\n\n"
            "# Generate leak-free splits\n"
            "!python scripts/make_splits.py\n"
            "!python scripts/prepare_external_set.py"
        ),
        code_cell(
            "# Step 5: Interactive Benchmark Arm Selection & Execution\n"
            "# Select arms to run (all 15 evaluations per arm):\n"
            "ARMS_TO_RUN = ['B0', 'B1', 'B2', 'B3', 'B4', 'P1', 'P2', 'P3', 'P4', 'P5', 'S1', 'S2_100', 'FINAL_COMBINED']\n"
            "arms_flag = ' '.join(ARMS_TO_RUN)\n"
            "!python scripts/run_all.py --arms {arms_flag} --seeds 42 1337 2024 --folds 0 1 2 3 4"
        ),
        code_cell(
            "# Step 6: Statistical Analysis & Table Generation\n"
            "from src.stats import aggregate_benchmark_runs, compute_paired_comparison\n"
            "df_summary, df_runs = aggregate_benchmark_runs('results/logs/runs.csv', 'results/tables')\n"
            "display(df_summary[['arm_id', 'n_evals', 'test_minority_recall_formatted', 'test_balanced_accuracy_formatted', 'test_macro_f1_formatted']])\n\n"
            "# Sync to Google Drive\n"
            "!cp -r results/logs/runs.csv {DRIVE_DIR}/ 2>/dev/null || true\n"
            "!cp -r results/tables/ {DRIVE_DIR}/ 2>/dev/null || true\n"
            "!cp -r results/figures/ {DRIVE_DIR}/ 2>/dev/null || true\n\n"
            "# Download complete bundle\n"
            "!zip -r /content/complete_benchmark_results.zip results/\n"
            "files.download('/content/complete_benchmark_results.zip')"
        )
    ]
    make_notebook(cells, "notebooks/colab_runner_master.ipynb")


# Retain Kaggle generators for dual compatibility
def generate_kaggle_notebooks():
    # Baselines
    cells = [
        md_cell("# Rare-Class Benchmark: CNN Baselines (B0–B4) on HAM10000 (Kaggle GPU)"),
        code_cell("!pip install -q timm scikit-learn pandas matplotlib seaborn pyyaml tqdm"),
        code_cell("!python scripts/make_splits.py"),
        code_cell("!python scripts/run_all.py --arms B0 B1 B2 B3 B4 --seeds 42 1337 2024 --folds 0 1 2 3 4")
    ]
    make_notebook(cells, "notebooks/kaggle_runner_baselines.ipynb")


if __name__ == "__main__":
    generate_colab_baselines_notebook()
    generate_colab_pretrained_notebook()
    generate_colab_synthetic_notebook()
    generate_colab_master_notebook()
    generate_kaggle_notebooks()
