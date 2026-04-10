"""Building Footprint Change Detector — visual comparison rendering.

Produces:
  1. **side_by_side** — before (left) / after (right) with mask overlays.
  2. **change_overlay** — single image with new (green), demolished (red),
     unchanged (orange) regions overlaid on the *after* image.
  3. **metrics_panel** — small text panel with key numbers.
"""

from __future__ import annotations

import cv2
import numpy as np

from diff_engine import DiffResult
from metrics import ChangeMetrics


# ── colour constants (BGR) ─────────────────────────────────
_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_ORANGE = (0, 160, 230)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    alpha: float = 0.35,
    mask_color: tuple[int, int, int] = (255, 200, 0),
) -> np.ndarray:
    """Create a horizontal concat of before/after with mask overlays."""
    left = _overlay_mask(before, before_mask, mask_color, alpha)
    right = _overlay_mask(after, after_mask, mask_color, alpha)

    # Add labels
    cv2.putText(left, "BEFORE", (10, 30), _FONT, 0.8, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(right, "AFTER", (10, 30), _FONT, 0.8, _WHITE, 2, cv2.LINE_AA)

    return np.hstack([left, right])


def draw_change_overlay(
    after: np.ndarray,
    diff: DiffResult,
    *,
    alpha: float = 0.40,
    new_color: tuple[int, int, int] = _GREEN,
    demolished_color: tuple[int, int, int] = _RED,
    unchanged_color: tuple[int, int, int] = _ORANGE,
) -> np.ndarray:
    """Overlay change regions on the *after* image with a legend."""
    canvas = after.copy()

    # Layer order: unchanged first (background), then changes on top
    canvas = _overlay_mask(canvas, diff.unchanged_mask, unchanged_color, alpha * 0.6)
    canvas = _overlay_mask(canvas, diff.new_mask, new_color, alpha)
    canvas = _overlay_mask(canvas, diff.demolished_mask, demolished_color, alpha)

    # Draw bounding boxes around change regions
    for region in diff.regions:
        color = new_color if region.label == "new" else demolished_color
        x, y, w, h = region.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        tag = f"{region.label} ({region.area_px}px)"
        cv2.putText(canvas, tag, (x, y - 6), _FONT, 0.4, color, 1, cv2.LINE_AA)

    # Legend (top-right)
    _draw_legend(canvas, new_color, demolished_color, unchanged_color)

    return canvas


def draw_metrics_panel(
    m: ChangeMetrics,
    width: int = 400,
) -> np.ndarray:
    """Render a small image with metric text."""
    lines = [
        f"Before area:  {m.before_area_px:,} px  ({m.before_coverage:.2%})",
        f"After area:   {m.after_area_px:,} px  ({m.after_coverage:.2%})",
        f"New:          {m.new_area_px:,} px  ({m.num_new_regions} regions)",
        f"Demolished:   {m.demolished_area_px:,} px  ({m.num_demolished_regions} regions)",
        f"IoU:          {m.iou:.4f}",
        f"Change ratio: {m.change_ratio:.4f}",
        f"Growth ratio: {m.growth_ratio:.4f}",
    ]
    line_h = 28
    pad = 16
    height = pad * 2 + line_h * len(lines)
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    y = pad + 20
    for line in lines:
        cv2.putText(panel, line, (pad, y), _FONT, 0.50, _WHITE, 1, cv2.LINE_AA)
        y += line_h

    return panel


def compose_report(
    before: np.ndarray,
    after: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    diff: DiffResult,
    m: ChangeMetrics,
    *,
    alpha: float = 0.40,
) -> np.ndarray:
    """Compose a full visual report (side-by-side + change overlay + metrics).

    Layout::

        ┌──────────────┬──────────────┐
        │   BEFORE     │    AFTER     │
        │  (mask ovl)  │  (mask ovl)  │
        ├──────────────┴──────────────┤
        │      CHANGE OVERLAY         │
        ├─────────────────────────────┤
        │      METRICS PANEL          │
        └─────────────────────────────┘
    """
    sbs = draw_side_by_side(before, after, before_mask, after_mask, alpha=alpha)
    total_w = sbs.shape[1]

    change = draw_change_overlay(after, diff, alpha=alpha)
    change_resized = cv2.resize(
        change, (total_w, int(change.shape[0] * total_w / change.shape[1])),
        interpolation=cv2.INTER_LINEAR,
    )

    metrics_img = draw_metrics_panel(m, width=total_w)

    return np.vstack([sbs, change_resized, metrics_img])


# ── internal helpers ───────────────────────────────────────


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend *color* onto *image* where *mask* > 0."""
    out = image.copy()
    roi = mask > 127
    if not roi.any():
        return out
    overlay = out.copy()
    overlay[roi] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def _draw_legend(
    canvas: np.ndarray,
    new_c: tuple[int, int, int],
    demo_c: tuple[int, int, int],
    unch_c: tuple[int, int, int],
) -> None:
    """Draw a colour legend in the top-right corner."""
    h, w = canvas.shape[:2]
    items = [
        (new_c, "New building"),
        (demo_c, "Demolished"),
        (unch_c, "Unchanged"),
    ]
    bx = w - 170
    by = 10
    for i, (color, label) in enumerate(items):
        y = by + i * 24
        cv2.rectangle(canvas, (bx, y), (bx + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (bx + 22, y + 13), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
