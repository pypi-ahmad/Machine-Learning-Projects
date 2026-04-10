"""Interactive Video Object Cutout Studio — high-level controller.

Orchestrates SAM2Engine (images) and VideoPropagator (video) with
export helpers.  The controller does **not** handle prompt collection;
prompts are passed in as data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cv2

from config import CutoutConfig
from engine import MaskResult, SAM2Engine
from propagator import PropagationResult, VideoPropagator


@dataclass
class ImageResult:
    image: np.ndarray
    mask_result: MaskResult
    best_mask: np.ndarray
    score: float


@dataclass
class VideoResult:
    propagation: PropagationResult
    frames_dir: Path
    frame_count: int


class CutoutController:
    """Top-level orchestrator for image and video cutout."""

    def __init__(self, cfg: CutoutConfig | None = None) -> None:
        self.cfg = cfg or CutoutConfig()
        self._engine = SAM2Engine(self.cfg)
        self._propagator = VideoPropagator(self.cfg)
        self._loaded_image = False
        self._loaded_video = False

    # ── lifecycle ──────────────────────────────────────────

    def load_image_engine(self) -> None:
        self._engine.load()
        self._loaded_image = True

    def load_video_engine(self) -> None:
        self._propagator.load()
        self._loaded_video = True

    def close(self) -> None:
        self._engine.close()
        self._propagator.close()

    # ── image segmentation ────────────────────────────────

    def segment_image(
        self,
        image_bgr: np.ndarray,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> ImageResult:
        """Segment a single image with the given prompts."""
        if not self._loaded_image:
            self.load_image_engine()

        self._engine.set_image(image_bgr)
        result = self._engine.predict(points=points, labels=labels, box=box)

        return ImageResult(
            image=image_bgr,
            mask_result=result,
            best_mask=result.best_mask,
            score=result.best_score,
        )

    # ── video propagation ─────────────────────────────────

    def process_video(
        self,
        video_path: str | Path,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        frames_dir: Path | None = None,
    ) -> VideoResult:
        """Extract frames, add prompt on *frame_idx*, propagate masks."""
        if not self._loaded_video:
            self.load_video_engine()

        _frames_dir, count = self._propagator.extract_frames(
            video_path, frames_dir,
        )
        state = self._propagator.init_video(_frames_dir)
        self._propagator.add_prompt(
            state, frame_idx, obj_id,
            points=points, labels=labels, box=box,
        )
        prop = self._propagator.propagate(state)

        return VideoResult(
            propagation=prop,
            frames_dir=_frames_dir,
            frame_count=count,
        )

    def get_frame(self, frames_dir: Path, idx: int) -> np.ndarray | None:
        """Read extracted frame by index."""
        import cv2
        p = frames_dir / f"{idx:06d}.jpg"
        if not p.exists():
            return None
        return cv2.imread(str(p))
