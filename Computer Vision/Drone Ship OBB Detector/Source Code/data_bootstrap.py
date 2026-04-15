"""Dataset bootstrap for Drone Ship OBB Detector.

Tries the shared Roboflow download infrastructure first. If that fails
(missing credentials, network error, etc.), generates a small synthetic
YOLO-OBB-format dataset so that train / evaluate / infer can still run.

Usage::

    from data_bootstrap import ensure_obb_dataset

    data_root = ensure_obb_dataset()            # idempotent
    data_root = ensure_obb_dataset(force=True)  # force re-download
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("drone_ship_obb.data_bootstrap")

PROJECT_KEY = "drone_ship_obb_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASSES = ["ship", "large-vehicle", "small-vehicle", "plane", "helicopter",
           "harbor", "storage-tank", "container-crane"]
SPLIT_COUNTS = {"train": 60, "valid": 15, "test": 15}


def ensure_obb_dataset(*, force: bool = False) -> Path:
    """Return path to a ready YOLO-OBB-format dataset, downloading or generating it."""
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

    # Generate synthetic OBB dataset
    _generate_synthetic(DATA_ROOT)
    ready_marker.write_text("synthetic", encoding="utf-8")
    log.info("[%s] Synthetic OBB dataset generated at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic OBB data generation
# ---------------------------------------------------------------------------

_WATER_COLOR = (180, 130, 80)       # blueish water (BGR)
_LAND_COLOR = (80, 140, 90)         # greenish land
_OBJ_COLORS = {
    "ship": (200, 200, 220),
    "large-vehicle": (60, 60, 180),
    "small-vehicle": (60, 180, 60),
    "plane": (220, 220, 220),
    "helicopter": (180, 100, 180),
    "harbor": (120, 120, 120),
    "storage-tank": (160, 160, 100),
    "container-crane": (100, 200, 200),
}


def _generate_synthetic(root: Path) -> None:
    """Create a small synthetic aerial OBB dataset in YOLO-OBB format."""
    rng = random.Random(42)
    img_w, img_h = 1024, 1024

    for split, count in SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, labels = _make_aerial_scene(rng, img_w, img_h)
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


def _rotated_rect_corners(
    cx: float, cy: float, w: float, h: float, angle_rad: float,
) -> np.ndarray:
    """Return 4 corners of a rotated rectangle as (4, 2) array."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx_w, dy_w = cos_a * w / 2, sin_a * w / 2
    dx_h, dy_h = -sin_a * h / 2, cos_a * h / 2
    return np.array([
        [cx - dx_w - dx_h, cy - dy_w - dy_h],
        [cx + dx_w - dx_h, cy + dy_w - dy_h],
        [cx + dx_w + dx_h, cy + dy_w + dy_h],
        [cx - dx_w + dx_h, cy - dy_w + dy_h],
    ], dtype=np.float32)


def _make_aerial_scene(
    rng: random.Random, w: int, h: int,
) -> tuple[np.ndarray, list[str]]:
    """Render a synthetic aerial scene with rotated objects."""
    # Water background with some land patches
    img = np.full((h, w, 3), _WATER_COLOR, dtype=np.uint8)
    # Add noise for texture
    noise = np.random.RandomState(rng.randint(0, 99999)).randint(
        -15, 15, (h, w, 3), dtype=np.int16,
    )
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add a land patch in ~30% of images
    if rng.random() < 0.3:
        lx = rng.randint(0, w // 2)
        ly = rng.randint(0, h // 2)
        lw = rng.randint(w // 4, w // 2)
        lh = rng.randint(h // 4, h // 2)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), _LAND_COLOR, -1)

    # Place 2-8 rotated objects
    n_objs = rng.randint(2, 8)
    labels: list[str] = []
    for _ in range(n_objs):
        cls_id = rng.randint(0, len(CLASSES) - 1)
        cls_name = CLASSES[cls_id]
        color = _OBJ_COLORS.get(cls_name, (180, 180, 180))
        color = tuple(min(255, max(0, c + rng.randint(-20, 20))) for c in color)

        # Object size depends on class
        if cls_name == "ship":
            obj_w, obj_h = rng.randint(60, 140), rng.randint(15, 35)
        elif cls_name in ("large-vehicle", "plane"):
            obj_w, obj_h = rng.randint(40, 80), rng.randint(15, 30)
        elif cls_name in ("small-vehicle", "helicopter"):
            obj_w, obj_h = rng.randint(20, 40), rng.randint(10, 20)
        else:
            obj_w, obj_h = rng.randint(30, 70), rng.randint(30, 70)

        cx = rng.randint(obj_w, w - obj_w)
        cy = rng.randint(obj_h, h - obj_h)
        angle = rng.uniform(-math.pi / 2, math.pi / 2)

        corners = _rotated_rect_corners(cx, cy, obj_w, obj_h, angle)
        pts = corners.astype(np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (0, 0, 0), 1)

        # YOLO-OBB format: cls_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        corners_norm = corners.copy()
        corners_norm[:, 0] /= w
        corners_norm[:, 1] /= h
        coords = " ".join(f"{c:.6f}" for c in corners_norm.flatten())
        labels.append(f"{cls_id} {coords}")

    return img, labels
