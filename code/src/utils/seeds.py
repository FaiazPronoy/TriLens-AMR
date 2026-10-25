"""Reproducibility — seed locking across all libraries."""
from __future__ import annotations

import os
import random
import numpy as np
import torch
import lightning as L


def set_global_seed(seed: int = 42) -> None:
    """Lock all random seeds for reproducibility.
    
    Args:
        seed: Seed value to apply across Python, NumPy, PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)
    
    # Deterministic mode (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"