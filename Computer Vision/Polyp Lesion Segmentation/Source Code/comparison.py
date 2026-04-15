"""Polyp Lesion Segmentation — comparison backend hooks.
"""Polyp Lesion Segmentation — comparison backend hooks.

This module defines the abstract interface for alternative segmentation
backends (e.g. MedSAM) and a registry to switch between them.  The YOLO
baseline is always available; additional backends can be registered
without modifying the core pipeline.

Usage::

    from comparison import get_backend

    seg = get_backend("yolo", config)   # PolypSegmenter
    seg = get_backend("medsam", config) # MedSAMBackend (if available)
"""
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import PolypConfig
    from segmentation import SegmentationResult

# ── abstract backend ───────────────────────────────────────


class SegmentationBackend(ABC):
    """Minimal interface every segmentation backend must implement."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load model weights / resources."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Produce a SegmentationResult from a BGR uint8 image."""

    def close(self) -> None:
        """Release resources (optional override)."""


# ── backend registry ──────────────────────────────────────

_REGISTRY: dict[str, type[SegmentationBackend]] = {}


def register_backend(name: str):
    """Decorator to register a segmentation backend class."""
    def decorator(cls: type[SegmentationBackend]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def available_backends() -> list[str]:
    """Return names of all registered backends."""
    _ensure_builtins()
    return list(_REGISTRY.keys())


def get_backend(name: str, config: PolypConfig) -> SegmentationBackend:
    """Instantiate a backend by name.
    """Instantiate a backend by name.

    Parameters
    ----------
    name : str
        Backend identifier — ``"yolo"`` (always available) or
        ``"medsam"`` (requires extra dependencies).
    config : PolypConfig
        Configuration used to initialise the backend.

    Returns
    -------
    SegmentationBackend
    """
    """
    _ensure_builtins()
    if name not in _REGISTRY:
        avail = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend '{name}'. Available: {avail}"
        )
    return _REGISTRY[name](config)


# ── built-in YOLO backend ────────────────────────────────


def _ensure_builtins() -> None:
    """Lazily register the YOLO backend on first access."""
    if "yolo" in _REGISTRY:
        return

    @register_backend("yolo")
    class YOLOBackend(SegmentationBackend):
        """YOLO26m-seg baseline backend."""

        name = "yolo"

        def __init__(self, config: PolypConfig) -> None:
            from segmentation import PolypSegmenter
            self._seg = PolypSegmenter(
                model_name=config.model_name,
                confidence=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                imgsz=config.imgsz,
            )

        def load(self) -> None:
            self._seg.load()

        def segment(self, image: np.ndarray) -> SegmentationResult:
            return self._seg.segment(image)

        def close(self) -> None:
            self._seg.close()

    # Attempt to register MedSAM backend if dependencies are present
    try:
        _register_medsam()
    except ImportError:
        pass  # MedSAM not available -- YOLO-only mode


def _register_medsam() -> None:
    """Register MedSAM backend if ``segment_anything`` is importable."""
    import importlib
    sa = importlib.import_module("segment_anything")  # noqa: F841

    @register_backend("medsam")
    class MedSAMBackend(SegmentationBackend):
        """MedSAM comparison backend.
        """MedSAM comparison backend.

        Requires ``segment_anything`` and a MedSAM checkpoint.
        This is a *comparison* backend — the YOLO baseline should be
        preferred for routine inference.
        """
        """

        name = "medsam"

        def __init__(self, config: PolypConfig) -> None:
            self._config = config
            self._model = None

        def load(self) -> None:
            print(
                "[comparison] MedSAM backend detected. "
                "Provide a MedSAM checkpoint via config for full support."
            )
            # Placeholder: real integration would load sam_model_registry
            # and the MedSAM fine-tuned checkpoint here.
            self._model = "medsam_placeholder"

        def segment(self, image: np.ndarray) -> SegmentationResult:
            if self._model is None:
                self.load()
            # Placeholder — returns empty result until fully integrated.
            from segmentation import SegmentationResult as SR
            h, w = image.shape[:2]
            return SR(
                instances=[],
                combined_mask=np.zeros((h, w), dtype=np.uint8),
                image_hw=(h, w),
            )

        def close(self) -> None:
            self._model = None
