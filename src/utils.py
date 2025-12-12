"""
Utility functions for the project.
"""
import yaml
import os
import torch
import numpy as np
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        use_cuda: Whether to use CUDA if available
    
    Returns:
        torch.device instance
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_output_dirs(output_dirs: list):
    """
    Create output directories if they don't exist.
    
    Args:
        output_dirs: List of directory paths to create
    """
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)

