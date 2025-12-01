"""
Utility functions for TinyZero
"""
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model,
    optimizer,
    step: int,
    save_dir: str,
    **kwargs
):
    """Save model checkpoint"""
    save_path = Path(save_dir) / f"checkpoint_{step}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle DDP/FSDP wrapping - unwrap if needed
    # First check if model itself is DDP/FSDP wrapped
    if hasattr(model, 'module'):
        # model is DDP/FSDP wrapped
        actual_model = model.module
    else:
        # model might be PolicyModel wrapper with .model attribute, or raw model
        actual_model = model.model if hasattr(model, 'model') else model

    checkpoint = {
        'step': step,
        'model_state_dict': actual_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str
):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)

    # Handle DDP/FSDP wrapping - unwrap if needed
    # First check if model itself is DDP/FSDP wrapped
    if hasattr(model, 'module'):
        # model is DDP/FSDP wrapped
        actual_model = model.module
    else:
        # model might be PolicyModel wrapper with .model attribute, or raw model
        actual_model = model.model if hasattr(model, 'model') else model

    actual_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']

    print(f"Checkpoint loaded from {checkpoint_path}")
    return step


class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count