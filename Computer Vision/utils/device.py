"""
Device resolver — automatically selects the best available compute device.

Usage:
    from utils.device import get_device
    device = get_device()          # auto-detect
    device = get_device("cpu")     # force CPU
"""

from __future__ import annotations

import torch


def get_device(override: str | None = None) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    override : str | None
        Force a specific device string (e.g. ``"cpu"``, ``"cuda:0"``,
        ``"cuda:1"``).  When *None*, automatically selects CUDA if a GPU
        is available, otherwise falls back to CPU.

    Returns
    -------
    torch.device
    """
    if override is not None:
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def device_summary() -> str:
    """Return a human-readable one-liner about the active device."""
    dev = get_device()
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(dev)
        mem = torch.cuda.get_device_properties(dev).total_mem / (1024 ** 3)
        return f"GPU: {name} ({mem:.1f} GB)"
    return "CPU (no CUDA GPU detected)"


if __name__ == "__main__":
    print(f"Selected device : {get_device()}")
    print(f"Summary         : {device_summary()}")
