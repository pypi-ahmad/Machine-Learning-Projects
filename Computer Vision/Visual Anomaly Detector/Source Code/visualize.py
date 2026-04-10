"""Visual Anomaly Detector — visualization overlays.

Draws anomaly labels, colored borders, score bars, and compose
heatmap overlays for inference output images.

Usage::

    from visualize import draw_result, draw_score_bar

    annotated = draw_result(image, result, cfg)
"""

from __future__ import annotations

import cv2
import numpy as np

from config import AnomalyConfig


def draw_result(
    image: np.ndarray,
    result: dict,
    cfg: AnomalyConfig | None = None,
) -> np.ndarray:
    """Draw anomaly result overlay on an image.

    Includes label, score text, colored border, and a score bar.
    """
    if cfg is None:
        cfg = AnomalyConfig()

    vis = image.copy()
    is_anomaly = result.get("is_anomaly", False)
    label = result.get("label", "UNKNOWN")
    score = result.get("anomaly_score", 0.0)
    threshold = result.get("threshold", cfg.anomaly_threshold)

    color = cfg.anomaly_color if is_anomaly else cfg.normal_color

    # Colored border
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color, cfg.border_width)

    # Label text
    text = f"{label} (score={score:.2f}, thresh={threshold:.2f})"
    _put_label(vis, text, (10, 30), color)

    # Score bar
    draw_score_bar(vis, score, threshold, y_offset=h - 40)

    return vis


def draw_score_bar(
    image: np.ndarray,
    score: float,
    threshold: float,
    *,
    y_offset: int = 0,
    bar_height: int = 20,
    margin: int = 10,
) -> None:
    """Draw a horizontal score bar at the bottom of the image."""
    h, w = image.shape[:2]
    bar_w = w - 2 * margin
    if bar_w < 20:
        return

    y1 = y_offset
    y2 = y1 + bar_height

    # Background
    cv2.rectangle(image, (margin, y1), (margin + bar_w, y2), (60, 60, 60), -1)

    # Score fill
    max_score = max(threshold * 2, score * 1.2, 1.0)
    fill_w = int(bar_w * min(score / max_score, 1.0))
    bar_color = (0, 0, 255) if score > threshold else (0, 200, 0)
    cv2.rectangle(image, (margin, y1), (margin + fill_w, y2), bar_color, -1)

    # Threshold marker
    t_x = margin + int(bar_w * min(threshold / max_score, 1.0))
    cv2.line(image, (t_x, y1 - 2), (t_x, y2 + 2), (255, 255, 255), 2)

    # Score text
    cv2.putText(
        image, f"{score:.2f}", (margin + 4, y2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )


def draw_batch_summary(
    results: list[dict],
    *,
    width: int = 800,
    row_height: int = 28,
) -> np.ndarray:
    """Draw a summary image for batch inference results."""
    n = len(results)
    height = max(row_height * (n + 2), 100)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)

    # Header
    cv2.putText(canvas, "Anomaly Detection Results", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, r in enumerate(results):
        y = row_height * (i + 2)
        name = r.get("source", f"image_{i}")
        label = r.get("label", "?")
        score = r.get("anomaly_score", 0.0)
        color = (0, 0, 255) if r.get("is_anomaly") else (0, 200, 0)

        cv2.putText(canvas, f"{name[:40]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"{label:8s} {score:.3f}", (350, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return canvas


def _put_label(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(canvas, (org[0] - 2, org[1] - th - 4),
                  (org[0] + tw + 4, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(canvas, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
