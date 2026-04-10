"""Logo Retrieval Brand Match — visualisation helpers.

Build preview grids showing query + top-k matches with similarity
scores and brand labels.
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
    thumb_size: int = 128,
    cols: int = 5,
    border_color: tuple[int, int, int] = (0, 200, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Build a visual grid: query on the left, top-k matches on the right.

    Returns a BGR image.
    """
    sz = thumb_size
    pad = 4
    cell = sz + pad * 2

    # Prepare query thumbnail
    query_thumb = _resize_pad(query_image, sz)

    # Prepare match thumbnails
    match_thumbs = []
    for hit in hits:
        img = cv2.imread(hit.path)
        if img is None:
            img = np.full((sz, sz, 3), 128, dtype=np.uint8)
        match_thumbs.append((_resize_pad(img, sz), hit))

    n_matches = len(match_thumbs)
    total_cols = 1 + min(n_matches, cols)

    # Canvas
    canvas_w = total_cols * cell
    canvas_h = cell + 28  # extra space for labels
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    # Draw query
    _place_thumb(canvas, query_thumb, 0, 0, pad, border_color)
    cv2.putText(canvas, "QUERY", (pad, cell + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    # Draw matches
    for i, (thumb, hit) in enumerate(match_thumbs[:cols]):
        col = i + 1
        _place_thumb(canvas, thumb, col, 0, pad, border_color)
        label = f"{hit.brand} ({hit.score:.2f})"
        x = col * cell + pad
        cv2.putText(canvas, label, (x, cell + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_color, 1, cv2.LINE_AA)

    return canvas


def make_brand_grid(
    brand_images: dict[str, list[str]],
    *,
    thumb_size: int = 96,
    max_per_brand: int = 6,
    max_brands: int = 20,
) -> np.ndarray:
    """Build a grid showing sample images per brand (for index overview)."""
    sz = thumb_size
    pad = 3
    cell = sz + pad * 2

    brands = sorted(brand_images.keys())[:max_brands]
    n_cols = max_per_brand + 1  # label column + images

    canvas_h = len(brands) * (cell + 18)
    canvas_w = n_cols * cell
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for row, brand in enumerate(brands):
        y_off = row * (cell + 18)
        # Brand label
        cv2.putText(canvas, brand, (4, y_off + cell // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        paths = brand_images[brand][:max_per_brand]
        for col, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                continue
            thumb = _resize_pad(img, sz)
            x = (col + 1) * cell + pad
            y = y_off + pad
            _paste(canvas, thumb, x, y)

    return canvas


def draw_query_overlay(
    image_bgr: np.ndarray,
    brand: str,
    score: float,
    color: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    """Annotate an image with the top brand match."""
    vis = image_bgr.copy()
    label = f"{brand} ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
    cv2.putText(vis, label, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis


# ── internal helpers ──────────────────────────────────────

def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad to square."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), 40, dtype=np.uint8)
    y_off = (size - nh) // 2
    x_off = (size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
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
    x = col * cell + pad
    y = row * cell + pad
    _paste(canvas, thumb, x, y)
    cv2.rectangle(canvas, (x - 1, y - 1), (x + sz, y + sz), border_color, 1)


def _paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    h, w = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    if y + h > ch or x + w > cw or y < 0 or x < 0:
        return
    canvas[y:y + h, x:x + w] = patch
