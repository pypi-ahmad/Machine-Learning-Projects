"""Similar Image Finder — visualisation helpers.

Build preview grids showing query + top-k matches with scores.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from index import SearchHit


def make_result_grid(
    query_image: np.ndarray,
    hits: list[SearchHit],
    *,
    thumb_size: int = 160,
    cols: int = 4,
    border_color: tuple[int, int, int] = (80, 200, 80),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Build a visual grid: query on the left, top-k matches on the right."""
    sz = thumb_size
    pad = 4
    cell = sz + pad * 2

    query_thumb = _resize_pad(query_image, sz)

    match_thumbs: list[tuple[np.ndarray, SearchHit]] = []
    for hit in hits:
        img = cv2.imread(hit.path)
        if img is None:
            img = np.full((sz, sz, 3), 128, dtype=np.uint8)
        match_thumbs.append((_resize_pad(img, sz), hit))

    n = len(match_thumbs)
    total_cols = 1 + min(n, cols)
    rows = 1 + (n - 1) // cols if n > cols else 1

    canvas_w = total_cols * cell
    canvas_h = rows * (cell + 22)
    canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

    # Query thumbnail (top-left)
    _place_thumb(canvas, query_thumb, 0, 0, pad, (0, 160, 255))
    cv2.putText(canvas, "QUERY", (pad, cell + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    # Match thumbnails
    for i, (thumb, hit) in enumerate(match_thumbs):
        col = (i % cols) + 1
        row = i // cols
        _place_thumb(canvas, thumb, col, row, pad, border_color)
        label = f"{hit.score:.3f}"
        if hit.category:
            label = f"{hit.category} {label}"
        x = col * cell + pad
        y = row * (cell + 22) + cell + 14
        cv2.putText(canvas, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_color, 1, cv2.LINE_AA)

    return canvas


def make_category_grid(
    category_images: dict[str, list[str]],
    *,
    thumb_size: int = 96,
    max_per_cat: int = 6,
    max_cats: int = 20,
) -> np.ndarray:
    """Build a grid showing sample images per category."""
    sz = thumb_size
    pad = 3
    cell = sz + pad * 2

    cats = sorted(category_images.keys())[:max_cats]
    n_cols = max_per_cat + 1

    canvas_h = len(cats) * (cell + 18)
    canvas_w = n_cols * cell
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for row, cat in enumerate(cats):
        y_off = row * (cell + 18)
        cv2.putText(canvas, cat, (4, y_off + cell // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        paths = category_images[cat][:max_per_cat]
        for col, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                continue
            thumb = _resize_pad(img, sz)
            _paste(canvas, thumb, (col + 1) * cell + pad, y_off + pad)

    return canvas


def draw_query_overlay(
    image_bgr: np.ndarray,
    label: str,
    score: float,
    color: tuple[int, int, int] = (80, 200, 80),
) -> np.ndarray:
    """Annotate an image with the top match info."""
    vis = image_bgr.copy()
    text = f"{label} ({score:.3f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
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


def _place_thumb(
    canvas: np.ndarray,
    thumb: np.ndarray,
    col: int,
    row: int,
    pad: int,
    border_color: tuple[int, int, int],
) -> None:
    sz = thumb.shape[0]
    cell = sz + pad * 2
    row_h = cell + 22
    x = col * cell + pad
    y = row * row_h + pad
    _paste(canvas, thumb, x, y)
    cv2.rectangle(canvas, (x - 1, y - 1), (x + sz, y + sz), border_color, 1)


def _paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    h, w = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    if y + h > ch or x + w > cw or y < 0 or x < 0:
        return
    canvas[y:y + h, x:x + w] = patch
