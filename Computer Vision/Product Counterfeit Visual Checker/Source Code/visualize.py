"""Product Counterfeit Visual Checker — visualisation helpers.

Build comparison grids and mismatch heatmaps for screening results.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from comparator import ComparisonDetail, ScreeningResult

_RISK_COLORS = {
    "low": (80, 200, 80),      # green
    "medium": (0, 180, 255),    # orange
    "high": (0, 0, 220),        # red
}


def make_screening_grid(
    suspect_image: np.ndarray,
    result: ScreeningResult,
    *,
    thumb_size: int = 160,
    cols: int = 4,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Build a visual grid: suspect on the left, reference matches + scores."""
    sz = thumb_size
    pad = 4
    cell = sz + pad * 2
    row_h = cell + 40  # extra room for two-line labels

    n = len(result.details)
    total_cols = 1 + min(n, cols)
    rows = 1 + (n - 1) // cols if n > cols else 1

    canvas_w = total_cols * cell
    canvas_h = rows * row_h + 30  # +30 for risk banner
    canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

    # Risk banner at top
    risk_color = _RISK_COLORS.get(result.risk_level, (128, 128, 128))
    banner_text = f"MISMATCH RISK: {result.risk_level.upper()}  ({result.mismatch_risk_pct}%)"
    cv2.rectangle(canvas, (0, 0), (canvas_w, 24), risk_color, -1)
    cv2.putText(canvas, banner_text, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    y_offset = 28

    # Suspect thumbnail (top-left)
    suspect_thumb = _resize_pad(suspect_image, sz)
    _paste(canvas, suspect_thumb, pad, y_offset + pad)
    cv2.rectangle(canvas, (pad - 1, y_offset + pad - 1),
                  (pad + sz, y_offset + pad + sz), (0, 160, 255), 2)
    cv2.putText(canvas, "SUSPECT", (pad, y_offset + cell + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    # Reference thumbnails
    for i, detail in enumerate(result.details):
        col = (i % cols) + 1
        row = i // cols
        x = col * cell + pad
        y = y_offset + row * row_h + pad

        ref_img = cv2.imread(detail.reference_path)
        if ref_img is None:
            ref_img = np.full((sz, sz, 3), 128, dtype=np.uint8)
        thumb = _resize_pad(ref_img, sz)
        _paste(canvas, thumb, x, y)

        # Border colour based on composite score
        border = _score_to_color(detail.composite_score)
        cv2.rectangle(canvas, (x - 1, y - 1), (x + sz, y + sz), border, 2)

        # Labels
        label1 = f"{detail.reference_product}" if detail.reference_product else "ref"
        label2 = f"G:{detail.global_score:.2f} R:{detail.region_score:.2f} H:{detail.histogram_score:.2f}"
        label3 = f"composite: {detail.composite_score:.3f}"
        ty = y_offset + row * row_h + cell + 12
        cv2.putText(canvas, label1, (x, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label2, (x, ty + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, label3, (x, ty + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, border, 1, cv2.LINE_AA)

    return canvas


def make_region_heatmap(
    image_bgr: np.ndarray,
    region_scores: list[float],
    grid: tuple[int, int] = (3, 3),
    *,
    size: int = 320,
) -> np.ndarray:
    """Overlay a colour heatmap showing per-region similarity scores."""
    rows, cols = grid
    vis = cv2.resize(image_bgr, (size, size))
    overlay = vis.copy()
    ph, pw = size // rows, size // cols

    for idx, score in enumerate(region_scores):
        r, c = divmod(idx, cols)
        y0, y1 = r * ph, (r + 1) * ph
        x0, x1 = c * pw, (c + 1) * pw

        color = _score_to_color(score)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

        label = f"{score:.2f}"
        cx = x0 + pw // 2 - 15
        cy = y0 + ph // 2 + 5
        cv2.putText(overlay, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

    # Grid lines
    for r in range(1, rows):
        y = r * ph
        cv2.line(blended, (0, y), (size, y), (200, 200, 200), 1)
    for c in range(1, cols):
        x = c * pw
        cv2.line(blended, (x, 0), (x, size), (200, 200, 200), 1)

    return blended


def draw_risk_badge(
    image_bgr: np.ndarray,
    risk_level: str,
    composite_score: float,
) -> np.ndarray:
    """Annotate an image with a risk badge in the top-left corner."""
    vis = image_bgr.copy()
    color = _RISK_COLORS.get(risk_level, (128, 128, 128))
    text = f"{risk_level.upper()} RISK ({composite_score:.3f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(vis, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


# ── internal helpers ──────────────────────────────────────

def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 35, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    h, w = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    if y + h > ch or x + w > cw or y < 0 or x < 0:
        return
    canvas[y:y + h, x:x + w] = patch


def _score_to_color(score: float) -> tuple[int, int, int]:
    """Map a 0–1 similarity score to a BGR colour (red → yellow → green)."""
    if score >= 0.75:
        return (80, 200, 80)    # green — low risk
    elif score >= 0.55:
        return (0, 180, 255)    # orange — medium risk
    return (0, 0, 220)          # red — high risk
