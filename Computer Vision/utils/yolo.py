"""Shared YOLO model loader with instance caching and CUDA enforcement.

Usage::

    from utils.yolo import load_yolo

    model = load_yolo()                        # yolo26n.pt (default, on CUDA)
    model = load_yolo("yolo26n-seg.pt")        # segmentation
    model = load_yolo("/path/to/custom.pt")    # custom weights
"""

from __future__ import annotations

from functools import lru_cache

import torch
from ultralytics import YOLO

# Default device — always prefer CUDA when available
_DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=16)
def load_yolo(model: str = "yolo26n.pt", device: str | None = None) -> YOLO:
    """Load, cache, and move an Ultralytics YOLO model to GPU.

    Parameters
    ----------
    model : str
        Model name (auto-downloaded) or absolute path to custom ``.pt`` file.
        Default is ``yolo26n.pt`` (YOLO v26 nano).
    device : str | None
        Override device (e.g. ``"cpu"``). Defaults to CUDA if available.
    """
    m = YOLO(model)
    m.to(device or _DEVICE)
    return m
