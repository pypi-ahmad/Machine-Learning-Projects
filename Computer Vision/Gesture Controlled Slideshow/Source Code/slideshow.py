"""Slideshow state machine for Gesture Controlled Slideshow.

Manages an ordered collection of slide images with next/previous
navigation, pause/resume, and pointer-mode state.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GestureConfig

log = logging.getLogger("gesture.slideshow")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SlideState:
    """Current slideshow state."""

    current_index: int = 0
    total_slides: int = 0
    paused: bool = False
    pointer_mode: bool = False
    last_action: str = ""


class Slideshow:
    """Simple image slideshow with gesture-driven navigation."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._slides: list[Path] = []
        self._index: int = 0
        self._paused: bool = False
        self._pointer_mode: bool = False
        self._last_action: str = ""
        self._blank = np.full((480, 640, 3), 40, dtype=np.uint8)

    def load_slides(self, directory: str | Path | None = None) -> int:
        """Load slide images from a directory.

        Parameters
        ----------
        directory : str or Path, optional
            Folder containing images.  Falls back to
            ``cfg.slide_dir``.

        Returns
        -------
        int
            Number of slides loaded.
        """
        slide_dir = Path(directory) if directory else Path(self.cfg.slide_dir)

        if not slide_dir.is_dir():
            log.warning("Slide directory not found: %s", slide_dir)
            self._generate_demo_slides()
            return len(self._slides)

        self._slides = sorted(
            f for f in slide_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )

        if not self._slides:
            log.warning("No images in %s — using demo slides", slide_dir)
            self._generate_demo_slides()

        log.info("Loaded %d slides from %s", len(self._slides), slide_dir)
        self._index = 0
        return len(self._slides)

    def _generate_demo_slides(self) -> None:
        """Generate numbered placeholder slides for demo."""
        self._slides = []
        colors = [
            (60, 60, 180), (60, 140, 60), (140, 80, 40),
            (140, 60, 140), (40, 140, 140),
        ]
        for i, color in enumerate(colors):
            img = np.full((480, 640, 3), color, dtype=np.uint8)
            cv2.putText(
                img, f"Slide {i + 1}", (180, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3,
            )
            cv2.putText(
                img, "Demo Mode", (210, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1,
            )
            # Store as Path-like with .demo suffix for identification
            self._slides.append(Path(f"__demo_slide_{i}__"))

        # Cache the demo images
        self._demo_images = {
            f"__demo_slide_{i}__": img
            for i, (img, color) in enumerate(
                zip(
                    [np.full((480, 640, 3), c, dtype=np.uint8) for c in colors],
                    colors,
                )
            )
        }
        # Regenerate with text
        self._demo_images = {}
        for i, color in enumerate(colors):
            img = np.full((480, 640, 3), color, dtype=np.uint8)
            cv2.putText(
                img, f"Slide {i + 1}", (180, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3,
            )
            cv2.putText(
                img, "Demo Mode", (210, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1,
            )
            self._demo_images[f"__demo_slide_{i}__"] = img

    def execute(self, action: str) -> None:
        """Execute a slideshow action.

        Parameters
        ----------
        action : str
            One of: "next", "previous", "pause", "resume",
            "pointer", "first", "last".
        """
        self._last_action = action

        if action == "next":
            self._next()
        elif action == "previous":
            self._previous()
        elif action == "pause":
            self._paused = not self._paused
            log.info("Slideshow %s", "paused" if self._paused else "resumed")
        elif action == "resume":
            self._paused = False
        elif action == "pointer":
            self._pointer_mode = not self._pointer_mode
            log.info("Pointer mode %s", "ON" if self._pointer_mode else "OFF")
        elif action == "first":
            self._index = 0
        elif action == "last":
            self._index = max(0, len(self._slides) - 1)

    def _next(self) -> None:
        if not self._slides:
            return
        if self._index < len(self._slides) - 1:
            self._index += 1
        elif self.cfg.loop:
            self._index = 0

    def _previous(self) -> None:
        if not self._slides:
            return
        if self._index > 0:
            self._index -= 1
        elif self.cfg.loop:
            self._index = len(self._slides) - 1

    def current_slide(self) -> np.ndarray:
        """Get the current slide image.

        Returns
        -------
        np.ndarray
            BGR image.
        """
        if not self._slides:
            return self._blank.copy()

        slide_path = self._slides[self._index]

        # Demo slides
        if hasattr(self, "_demo_images") and str(slide_path) in self._demo_images:
            return self._demo_images[str(slide_path)].copy()

        img = cv2.imread(str(slide_path))
        if img is None:
            return self._blank.copy()
        return img

    @property
    def state(self) -> SlideState:
        return SlideState(
            current_index=self._index,
            total_slides=len(self._slides),
            paused=self._paused,
            pointer_mode=self._pointer_mode,
            last_action=self._last_action,
        )

    def reset(self) -> None:
        self._index = 0
        self._paused = False
        self._pointer_mode = False
        self._last_action = ""
