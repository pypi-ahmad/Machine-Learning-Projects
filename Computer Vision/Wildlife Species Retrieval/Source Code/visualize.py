"""Wildlife Species Retrieval — visualisation utilities.

Renders query + result grids with species labels and similarity scores.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from index import SearchHit


def _resize_thumb(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size))


def draw_query_overlay(
    image: np.ndarray,
    label: str,
    score: float | None = None,
    font_scale: float = 0.5,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """Draw a label bar at the top of an image."""
    out = image.copy()
    h, w = out.shape[:2]
    bar_h = max(20, int(h * 0.12))
    cv2.rectangle(out, (0, 0), (w, bar_h), bg_color, -1)
    text = f"{label}  {score:.3f}" if score is not None else label
    cv2.putText(out, text, (4, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    return out


def make_result_grid(
    query_image: np.ndarray,
    hits: list[SearchHit],
    *,
    thumb_size: int = 160,
    cols: int = 4,
    border_color: tuple[int, int, int] = (80, 200, 80),
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
) -> np.ndarray:
    """Build a visual grid: query on left, matches on right."""
    sz = thumb_size

    # Query thumbnail
    q_thumb = _resize_thumb(query_image, sz * 2)
    q_thumb = draw_query_overlay(q_thumb, "QUERY", font_scale=font_scale * 1.2,
                                 text_color=text_color, bg_color=(0, 80, 160))

    # Match thumbnails
    match_thumbs = []
    for h in hits:
        img = cv2.imread(h.path)
        if img is None:
            img = np.zeros((sz, sz, 3), dtype=np.uint8)
        thumb = _resize_thumb(img, sz)
        thumb = draw_query_overlay(thumb, h.species, h.score,
                                   font_scale=font_scale, text_color=text_color,
                                   bg_color=border_color)
        match_thumbs.append(thumb)

    # Layout: query (left) | matches grid (right)
    n = len(match_thumbs)
    rows = max(1, (n + cols - 1) // cols)
    grid_w = cols * sz
    grid_h = rows * sz

    # Query area: 2×sz tall, matches area must be at least as tall
    total_h = max(sz * 2, grid_h)

    canvas = np.zeros((total_h, sz * 2 + grid_w + 4, 3), dtype=np.uint8)
    # Place query
    qh, qw = q_thumb.shape[:2]
    canvas[:qh, :qw] = q_thumb

    # Place matches
    x_off = sz * 2 + 4
    for i, mt in enumerate(match_thumbs):
        r, c = divmod(i, cols)
        y, x = r * sz, c * sz + x_off
        if y + sz <= total_h and x + sz <= canvas.shape[1]:
            canvas[y:y + sz, x:x + sz] = mt

    return canvas


def save_result_grid(
    query_image: np.ndarray,
    hits: list[SearchHit],
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Build and save a result grid."""
    grid = make_result_grid(query_image, hits, **kwargs)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), grid)
    return p
