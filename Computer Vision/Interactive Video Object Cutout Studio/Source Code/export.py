"""Interactive Video Object Cutout Studio — export utilities.

Save alpha masks, transparent cutouts, and overlay visualisations
for both single images and video frame sequences.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_alpha_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a boolean mask as a single-channel 8-bit PNG (0 / 255)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(p), out)
    return p


def save_cutout(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
) -> Path:
    """Save an RGBA PNG: the object on a transparent background."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    cv2.imwrite(str(p), rgba)
    return p


def save_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
    color: tuple[int, int, int] = (255, 144, 30),
    alpha: float = 0.45,
) -> Path:
    """Save an overlay: mask region tinted with *color*."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    vis = image_bgr.copy()
    tint = np.full_like(vis, color, dtype=np.uint8)
    vis[mask] = cv2.addWeighted(vis[mask], 1 - alpha, tint[mask], alpha, 0)

    # contour outline
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(vis, contours, -1, color, 2)

    cv2.imwrite(str(p), vis)
    return p


def draw_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 144, 30),
    alpha: float = 0.45,
) -> np.ndarray:
    """Return an overlay image (does not save)."""
    vis = image_bgr.copy()
    tint = np.full_like(vis, color, dtype=np.uint8)
    vis[mask] = cv2.addWeighted(vis[mask], 1 - alpha, tint[mask], alpha, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(vis, contours, -1, color, 2)
    return vis


class VideoExporter:
    """Export masks, cutouts, and overlays for a sequence of video frames."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_masks: bool = True,
        save_cutouts: bool = True,
        save_overlays: bool = True,
        overlay_color: tuple[int, int, int] = (255, 144, 30),
        overlay_alpha: float = 0.45,
    ) -> None:
        self.root = Path(output_dir)
        self._masks_dir = self.root / "masks"
        self._cutouts_dir = self.root / "cutouts"
        self._overlays_dir = self.root / "overlays"
        self._save_masks = save_masks
        self._save_cutouts = save_cutouts
        self._save_overlays = save_overlays
        self._color = overlay_color
        self._alpha = overlay_alpha
        self._count = 0

    def export_frame(
        self,
        frame_idx: int,
        image_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Export a single frame's results."""
        name = f"{frame_idx:06d}.png"
        if self._save_masks:
            save_alpha_mask(mask, self._masks_dir / name)
        if self._save_cutouts:
            save_cutout(image_bgr, mask, self._cutouts_dir / name)
        if self._save_overlays:
            save_overlay(image_bgr, mask, self._overlays_dir / name, self._color, self._alpha)
        self._count += 1

    def finalize(self) -> dict:
        return {
            "frames_exported": self._count,
            "output_dir": str(self.root),
            "masks_dir": str(self._masks_dir) if self._save_masks else None,
            "cutouts_dir": str(self._cutouts_dir) if self._save_cutouts else None,
            "overlays_dir": str(self._overlays_dir) if self._save_overlays else None,
        }
