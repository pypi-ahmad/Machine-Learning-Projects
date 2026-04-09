"""Reproducibility seed helper.

Usage:
    from utils.seed import set_global_seed
    set_global_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available).

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
