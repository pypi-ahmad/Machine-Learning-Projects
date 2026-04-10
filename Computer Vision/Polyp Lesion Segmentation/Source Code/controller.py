"""Polyp Lesion Segmentation — main controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comparison import SegmentationBackend, get_backend
from config import PolypConfig
from metrics import PolypMetrics, compute_polyp_metrics
from segmentation import SegmentationResult


@dataclass
class FrameResult:
    """Complete result for one processed image."""

    image: np.ndarray
    segmentation: SegmentationResult
    metrics: PolypMetrics
    backend_name: str


class PolypController:
    """High-level controller: segmentation → metrics.

    Supports hot-swapping the segmentation backend between YOLO (baseline)
    and optional comparison backends (e.g. MedSAM).
    """

    def __init__(self, config: PolypConfig | None = None) -> None:
        self.cfg = config or PolypConfig()
        self._backend: SegmentationBackend | None = None

    def load(self) -> None:
        """Eagerly load the configured backend."""
        self._backend = get_backend(self.cfg.backend, self.cfg)
        self._backend.load()

    def close(self) -> None:
        """Release model resources."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def switch_backend(self, name: str) -> None:
        """Switch to a different segmentation backend at runtime."""
        self.close()
        self.cfg.backend = name
        self.load()

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else self.cfg.backend

    def process(
        self,
        frame: np.ndarray,
        *,
        source: str = "",
        gt_mask: np.ndarray | None = None,
    ) -> FrameResult:
        """Run polyp segmentation + metrics on one image.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        source : str
            Optional source label.
        gt_mask : np.ndarray | None
            Optional ground-truth mask for Dice/IoU evaluation.
        """
        if self._backend is None:
            self.load()

        seg = self._backend.segment(frame)
        metrics = compute_polyp_metrics(seg, gt_mask=gt_mask)

        return FrameResult(
            image=frame,
            segmentation=seg,
            metrics=metrics,
            backend_name=self.backend_name,
        )
