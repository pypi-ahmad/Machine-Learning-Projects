"""Interactive Video Object Cutout Studio — SAM 2 segmentation engine.
"""Interactive Video Object Cutout Studio — SAM 2 segmentation engine.

Wraps SAM2ImagePredictor for single-image promptable segmentation.
This module knows nothing about UI — it only accepts numpy arrays
of prompts and returns structured mask results.
"""
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import CutoutConfig


@dataclass
class MaskResult:
    """Result of a single SAM 2 prediction."""

    masks: np.ndarray       # (N, H, W) bool -- candidate masks
    scores: np.ndarray      # (N,)        float -- quality scores
    logits: np.ndarray      # (N, H, W)  float -- raw logits

    @property
    def best_idx(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def best_mask(self) -> np.ndarray:
        return self.masks[self.best_idx]

    @property
    def best_score(self) -> float:
        return float(self.scores[self.best_idx])


class SAM2Engine:
    """Promptable image segmentation powered by SAM 2."""

    def __init__(self, cfg: CutoutConfig | None = None) -> None:
        if cfg is None:
            from config import CutoutConfig
            cfg = CutoutConfig()
        self.cfg = cfg
        self._predictor = None
        self._device = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        """Load SAM 2 image predictor (auto-downloads weights from HF)."""
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._predictor = SAM2ImagePredictor.from_pretrained(
            self.cfg.model_id, device=self._device,
        )

    def close(self) -> None:
        self._predictor = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    # ── image-level API ───────────────────────────────────

    def set_image(self, image_bgr: np.ndarray) -> None:
        """Set the current image (BGR, HWC uint8)."""
        if self._predictor is None:
            self.load()
        image_rgb = image_bgr[:, :, ::-1]
        self._predictor.set_image(image_rgb)

    def predict(
        self,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> MaskResult:
        """Run SAM 2 prediction with point and/or box prompts.
        """Run SAM 2 prediction with point and/or box prompts.

        Parameters
        ----------
        points : (N, 2) float — pixel coordinates (x, y)
        labels : (N,)   int   — 1 = foreground, 0 = background
        box    : (4,)   float — [x1, y1, x2, y2]
        """
        """
        import torch

        if self._predictor is None:
            raise RuntimeError("Call load() or set_image() first.")

        ac_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            self._device.type, dtype=ac_dtype,
        ):
            masks, scores, logits = self._predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=self.cfg.multimask_output,
            )

        return MaskResult(
            masks=masks.astype(bool),
            scores=scores,
            logits=logits,
        )
