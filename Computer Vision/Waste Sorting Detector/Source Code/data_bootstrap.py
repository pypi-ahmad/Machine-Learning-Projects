"""Dataset bootstrap for Waste Sorting Detector.

Tries the shared Roboflow download infrastructure first. If that fails
(missing credentials, network error, etc.), generates a small synthetic
YOLO-format dataset so that train / evaluate / infer can still run.

Usage::

    from data_bootstrap import ensure_waste_dataset

    data_root = ensure_waste_dataset()            # idempotent
    data_root = ensure_waste_dataset(force=True)   # force re-download
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("waste_sorting.data_bootstrap")

PROJECT_KEY = "waste_sorting_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASSES = ["plastic", "paper", "cardboard", "metal", "glass", "trash"]
SPLIT_COUNTS = {"train": 60, "valid": 15, "test": 15}


def ensure_waste_dataset(*, force: bool = False) -> Path:
    """Return path to a ready YOLO-format dataset, downloading or generating it."""
    ready_marker = DATA_ROOT / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared at %s", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    # Try real download first
    try:
        from scripts.download_data import ensure_dataset as _ensure
        data_path = _ensure(PROJECT_KEY, force=force)
        if (data_path / "data.yaml").exists():
            ready_marker.parent.mkdir(parents=True, exist_ok=True)
            ready_marker.write_text("ok", encoding="utf-8")
            log.info("[%s] Real dataset ready at %s", PROJECT_KEY, data_path)
            return data_path
    except Exception as exc:
        log.warning("[%s] Real download failed (%s) -> falling back to synthetic dataset",
                    PROJECT_KEY, exc)

    # Generate synthetic dataset
    _generate_synthetic(DATA_ROOT)
    ready_marker.write_text("synthetic", encoding="utf-8")
    log.info("[%s] Synthetic dataset generated at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_ITEM_COLORS = {
    "plastic": (255, 165, 0),
    "paper": (200, 230, 255),
    "cardboard": (80, 120, 180),
    "metal": (180, 180, 180),
    "glass": (200, 255, 200),
    "trash": (60, 60, 60),
}
_BG_COLORS = [(220, 220, 210), (200, 190, 180), (230, 230, 230)]


def _generate_synthetic(root: Path) -> None:
    """Create a small synthetic waste detection dataset in YOLO format."""
    rng = random.Random(42)
    img_w, img_h = 640, 480

    for split, count in SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, labels = _make_waste_scene(rng, img_w, img_h)
            cv2.imwrite(str(img_dir / f"{split}_{i:04d}.jpg"), img)
            lbl_dir.joinpath(f"{split}_{i:04d}.txt").write_text(
                "\n".join(labels) + "\n" if labels else "", encoding="utf-8",
            )

    yaml_text = (
        f"path: {root.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _make_waste_scene(
    rng: random.Random, w: int, h: int,
) -> tuple[np.ndarray, list[str]]:
    """Render a synthetic scene with waste items on a flat surface."""
    bg = rng.choice(_BG_COLORS)
    img = np.full((h, w, 3), bg, dtype=np.uint8)

    # Add some surface texture lines
    for _ in range(rng.randint(2, 5)):
        y = rng.randint(0, h)
        shade = tuple(max(0, c - rng.randint(10, 30)) for c in bg)
        cv2.line(img, (0, y), (w, y), shade, 1)

    # Place 2-6 waste items
    n_items = rng.randint(2, 6)
    labels: list[str] = []
    for _ in range(n_items):
        cls_id = rng.randint(0, len(CLASSES) - 1)
        cls_name = CLASSES[cls_id]
        color = _ITEM_COLORS[cls_name]
        # Add some color variation
        color = tuple(min(255, max(0, c + rng.randint(-30, 30))) for c in color)

        bw = rng.randint(30, 90)
        bh = rng.randint(30, 90)
        x1 = rng.randint(5, w - bw - 5)
        y1 = rng.randint(5, h - bh - 5)
        x2, y2 = x1 + bw, y1 + bh

        # Draw item shape
        shape = rng.choice(["rect", "ellipse"])
        if shape == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(img, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, color, -1)
            cv2.ellipse(img, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, (0, 0, 0), 1)

        cx_n = (x1 + x2) / 2 / w
        cy_n = (y1 + y2) / 2 / h
        nw = bw / w
        nh = bh / h
        labels.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {nw:.6f} {nh:.6f}")

    return img, labels
