# Environment & Compute Specification

## Target Architecture & Zero-Cost Constraints

This benchmark is engineered to operate strictly under zero financial cost, following a dual-tier execution model:

1. **Tier 1: Local Development & Smoke Testing (CPU-Only / Local)**
   - **Environment:** Ubuntu Linux (x86_64), Python 3.10+ / 3.14.
   - **Hardware Acceleration:** Auto-detected. If `torch.cuda.is_available() == False`, all pipelines degrade gracefully to CPU with reduced batch sizes, single workers, and tiny subsets (DermaMNIST or sub-sampled HAM10000).
   - **Memory Footprint:** Kept within standard local RAM limits (< 8GB).
   - **Testing Guardrails:** Fast smoke tests, split leakage checks, and metric parity tests execute in seconds locally.

2. **Tier 2: Production Benchmark Runs (Free Kaggle / Colab GPU)**
   - **Environment:** Kaggle Kernels (Tesla P100 / T4 × 2) or Google Colab (Tesla T4).
   - **Precision:** Mixed-precision (`torch.cuda.amp.autocast()`, fp16/bf16).
   - **Resilience to Timeouts:** Kaggle interactive sessions time out at 9 hours (or 12 hours batch). Every long experiment script incrementally commits per-epoch checkpoints (`checkpoint.pt`), logs metrics per fold/epoch directly to `results/logs/`, and supports seamless resumption.
   - **Seeds:** All random number generators (`random`, `numpy`, `torch`, `torch.cuda`) are strictly seeded per configuration.

---

## Software Dependencies & Tooling

- **Core Deep Learning:** `torch>=2.0.0`, `torchvision>=0.15.0`, `timm>=1.0.0`
- **Foundation Models:** `transformers>=4.40.0`, `open_clip_torch>=2.24.0`
- **Generative & Synthetic:** `diffusers>=0.28.0`, `peft>=0.10.0`, `accelerate>=0.29.0`
- **Data & Statistics:** `numpy>=1.24.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`, `scipy>=1.11.0`
- **Visualization & Logging:** `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `tqdm>=4.65.0`
- **Testing & Quality Assurance:** `pytest>=7.4.0`
- **Interactive Interface:** `gradio>=4.0.0`
- **Debug / Fast Prototyping:** `medmnist>=3.0.0`

---

## Device Adaptation Protocol

All code in `src/` follows the universal device dispatcher:
```python
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Batch sizes, number of workers, and mixed precision flags dynamically scale according to device capabilities.
