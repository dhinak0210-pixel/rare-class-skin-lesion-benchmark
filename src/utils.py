"""
src/utils.py
============
Universal utility functions for reproducibility, hardware adaptation, configuration loading,
and checkpoint serialization.
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Universal device selector adapting automatically to CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file, recursively merging with base_config if specified."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "base_config" in cfg and cfg["base_config"]:
        base_path = cfg["base_config"]
        # Resolve relative to project root or config directory
        if not os.path.exists(base_path):
            base_path = os.path.join(os.path.dirname(config_path), "..", base_path)
            base_path = os.path.normpath(base_path)
        base_cfg = load_yaml_config(base_path)
        # Deep merge
        cfg = deep_merge_dicts(base_cfg, cfg)

    return cfg

def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False
) -> str:
    """Saves training checkpoint to disk incrementally."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
    return filepath

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """Loads model weights, optimizer, and scheduler state from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
