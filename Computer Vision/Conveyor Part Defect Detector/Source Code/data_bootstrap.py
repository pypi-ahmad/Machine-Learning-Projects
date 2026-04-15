"""Dataset bootstrap for Conveyor Part Defect Detector.

Tries the shared Roboflow download infrastructure first. If that fails
(missing credentials, network error, etc.), generates a small synthetic
YOLO-format dataset so that train / evaluate / infer can still run.

Usage::

    from data_bootstrap import ensure_defect_dataset

    data_root = ensure_defect_dataset()            # idempotent
    data_root = ensure_defect_dataset(force=True)   # force re-download
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

log = logging.getLogger("conveyor_defect.data_bootstrap")

PROJECT_KEY = "conveyor_part_defect_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASSES = ["scratch", "dent", "crack", "missing_part", "missing_hole",
           "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SPLIT_COUNTS = {"train": 60, "valid": 15, "test": 15}


def ensure_defect_dataset(*, force: bool = False) -> Path:
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

_BG_COLOR = (200, 200, 190)       # light grey PCB-ish background
_BOARD_COLOR = (60, 120, 60)      # green PCB board
_DEFECT_COLORS = [
    (0, 0, 180), (180, 0, 0), (0, 0, 0), (200, 200, 0),
    (128, 0, 128), (0, 128, 128), (80, 80, 80), (255, 100, 0),
]


def _generate_synthetic(root: Path) -> None:
    """Create a small synthetic industrial defect dataset in YOLO format."""
    rng = random.Random(42)
    img_w, img_h = 640, 480

    for split, count in SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, labels = _make_pcb_scene(rng, img_w, img_h)
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


def _make_pcb_scene(
    rng: random.Random, w: int, h: int,
) -> tuple[np.ndarray, list[str]]:
    """Render a synthetic PCB-like board with defect marks."""
    img = np.full((h, w, 3), _BG_COLOR, dtype=np.uint8)

    # Draw a green PCB board area
    margin = 30
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), _BOARD_COLOR, -1)

    # Add some traces (horizontal/vertical lines)
    for _ in range(rng.randint(3, 8)):
        color = (rng.randint(140, 200), rng.randint(140, 200), rng.randint(40, 80))
        if rng.random() < 0.5:
            y = rng.randint(margin + 20, h - margin - 20)
            cv2.line(img, (margin, y), (w - margin, y), color, rng.randint(1, 3))
        else:
            x = rng.randint(margin + 20, w - margin - 20)
            cv2.line(img, (x, margin), (x, h - margin), color, rng.randint(1, 3))

    # Place 1-5 defects
    n_defects = rng.randint(1, 5)
    labels: list[str] = []
    for _ in range(n_defects):
        cls_id = rng.randint(0, len(CLASSES) - 1)
        color = rng.choice(_DEFECT_COLORS)
        bw = rng.randint(15, 60)
        bh = rng.randint(15, 60)
        x1 = rng.randint(margin + 5, w - margin - bw - 5)
        y1 = rng.randint(margin + 5, h - margin - bh - 5)
        x2, y2 = x1 + bw, y1 + bh

        # Draw defect shape (rectangle, circle, or line depending on class)
        shape = rng.choice(["rect", "circle", "line"])
        if shape == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif shape == "circle":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = min(bw, bh) // 2
            cv2.circle(img, (cx, cy), r, color, -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), color, rng.randint(2, 4))

        cx_n = (x1 + x2) / 2 / w
        cy_n = (y1 + y2) / 2 / h
        nw = bw / w
        nh = bh / h
        labels.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {nw:.6f} {nh:.6f}")

    return img, labels
