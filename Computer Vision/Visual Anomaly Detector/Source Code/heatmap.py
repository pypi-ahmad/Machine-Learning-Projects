"""Visual Anomaly Detector — anomaly heatmap generation.

Generates patch-level anomaly heatmaps by sliding a window over the
image and scoring each patch independently.

Usage::

    from heatmap import HeatmapGenerator

    gen = HeatmapGenerator(feature_extractor, scorer, patch_size=64, stride=32)
    raw_map, overlay = gen.generate(image)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from feature_extractor import FeatureExtractor
from scorer import AnomalyScorer

log = logging.getLogger("visual_anomaly.heatmap")


class HeatmapGenerator:
    """Generate patch-level anomaly heatmaps."""

    def __init__(
        self,
        extractor: FeatureExtractor,
        scorer: AnomalyScorer,
        patch_size: int = 64,
        stride: int = 32,
        alpha: float = 0.4,
    ) -> None:
        self.extractor = extractor
        self.scorer = scorer
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha

    def generate(
        self,
        image: np.ndarray,
        *,
        colormap: int = cv2.COLORMAP_JET,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate an anomaly heatmap for a single image.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(raw_heatmap, overlay)`` where raw_heatmap is float32
            score map and overlay is the image with heatmap blended.
        """
        h, w = image.shape[:2]
        ps = self.patch_size
        st = self.stride

        score_map = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - ps + 1, st):
            for x in range(0, w - ps + 1, st):
                patch = image[y : y + ps, x : x + ps]
                feat = self.extractor.extract(patch)
                s = self.scorer.score_primary(feat)

                score_map[y : y + ps, x : x + ps] += s
                counts[y : y + ps, x : x + ps] += 1.0

        counts[counts == 0] = 1.0
        score_map /= counts

        # Normalise to 0–255
        smin, smax = score_map.min(), score_map.max()
        if smax - smin > 1e-8:
            norm = ((score_map - smin) / (smax - smin) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(score_map, dtype=np.uint8)

        heatmap_color = cv2.applyColorMap(norm, colormap)
        overlay = cv2.addWeighted(image, 1.0 - self.alpha, heatmap_color, self.alpha, 0)

        return score_map, overlay

    def generate_from_path(
        self,
        path: str,
        *,
        colormap: int = cv2.COLORMAP_JET,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load an image from path and generate the heatmap."""
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.generate(image, colormap=colormap)
