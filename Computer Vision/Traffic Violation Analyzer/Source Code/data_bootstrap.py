"""Dataset bootstrap for Traffic Violation Analyzer.

Tries the shared Roboflow download infrastructure first. If that fails
(missing credentials, network error, etc.), generates a small synthetic
YOLO-format dataset so that train / evaluate / infer can still run.

Usage::

    from data_bootstrap import ensure_traffic_dataset

    data_root = ensure_traffic_dataset()            # idempotent
    data_root = ensure_traffic_dataset(force=True)   # force re-download
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

log = logging.getLogger("traffic_violation.data_bootstrap")

PROJECT_KEY = "traffic_violation_analyzer"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
SPLIT_COUNTS = {"train": 60, "valid": 15, "test": 15}


def ensure_traffic_dataset(*, force: bool = False) -> Path:
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

_VEHICLE_COLORS = [
    (200, 60, 60), (60, 60, 200), (60, 200, 60),
    (200, 200, 60), (200, 60, 200), (60, 200, 200),
    (180, 180, 180), (100, 100, 100), (255, 255, 255),
]

_ROAD_GRAY = (90, 90, 90)
_LANE_COLOR = (0, 200, 200)


def _generate_synthetic(root: Path) -> None:
    """Create a small synthetic traffic dataset in YOLO format."""
    rng = random.Random(42)
    img_w, img_h = 640, 480

    for split, count in SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, labels = _make_traffic_scene(rng, img_w, img_h)
            cv2.imwrite(str(img_dir / f"{split}_{i:04d}.jpg"), img)
            lbl_dir.joinpath(f"{split}_{i:04d}.txt").write_text(
                "\n".join(labels) + "\n" if labels else "", encoding="utf-8",
            )

    # Write data.yaml
    yaml_text = (
        f"path: {root.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _make_traffic_scene(
    rng: random.Random, w: int, h: int,
) -> tuple[np.ndarray, list[str]]:
    """Render a synthetic top-down-ish road scene with vehicles."""
    img = np.full((h, w, 3), _ROAD_GRAY, dtype=np.uint8)

    # Draw 2-3 lane lines
    n_lanes = rng.randint(2, 3)
    for li in range(1, n_lanes + 1):
        lx = w * li // (n_lanes + 1) + rng.randint(-20, 20)
        for seg_y in range(0, h, 40):
            cv2.line(img, (lx, seg_y), (lx, min(seg_y + 20, h)), _LANE_COLOR, 2)

    # Place 2-6 vehicles
    n_vehicles = rng.randint(2, 6)
    labels: list[str] = []
    for _ in range(n_vehicles):
        cls_id = rng.randint(0, len(CLASSES) - 1)
        color = rng.choice(_VEHICLE_COLORS)
        # Vehicle size varies by class
        if cls_id <= 0:  # car
            bw, bh = rng.randint(40, 70), rng.randint(50, 80)
        elif cls_id == 1:  # truck
            bw, bh = rng.randint(50, 80), rng.randint(70, 110)
        elif cls_id == 2:  # bus
            bw, bh = rng.randint(50, 75), rng.randint(80, 120)
        elif cls_id == 3:  # motorcycle
            bw, bh = rng.randint(20, 35), rng.randint(35, 55)
        else:  # bicycle
            bw, bh = rng.randint(15, 30), rng.randint(30, 50)

        x1 = rng.randint(10, w - bw - 10)
        y1 = rng.randint(10, h - bh - 10)
        x2, y2 = x1 + bw, y1 + bh

        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # YOLO format: cls cx cy w h (normalised)
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        nw = bw / w
        nh = bh / h
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return img, labels
