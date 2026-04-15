"""Dataset bootstrap for Crowd Zone Counter.

Downloads a public person/crowd detection dataset via DatasetResolver.
Falls back to generating a small synthetic YOLO-format dataset for
demo and CI purposes when the real dataset is unavailable.

Usage::

    from data_bootstrap import ensure_crowd_dataset

    data_root = ensure_crowd_dataset()            # idempotent
    data_root = ensure_crowd_dataset(force=True)  # force re-download
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("crowd_zone.data_bootstrap")

PROJECT_KEY = "crowd_zone_counter"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASS_NAMES = ["person"]


def ensure_crowd_dataset(*, force: bool = False) -> Path:
    """Download or generate the crowd/person detection dataset.

    Returns the project data root containing a ``data.yaml``.
    """
    ready_marker = DATA_ROOT / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset ready at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    # Try real download first
    try:
        from scripts.download_data import ensure_dataset as _ensure
        data_path = _ensure(PROJECT_KEY, force=force)
        data_yaml = _find_data_yaml(data_path)
        if data_yaml is not None:
            log.info("[%s] Real dataset found at %s", PROJECT_KEY, data_path)
            ready_marker.parent.mkdir(parents=True, exist_ok=True)
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return data_path
    except Exception as exc:
        log.warning("[%s] Real download failed (%s) -- generating synthetic data", PROJECT_KEY, exc)

    # Synthetic fallback
    _generate_synthetic(DATA_ROOT)

    # Metadata
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": "synthetic",
        "description": "Auto-generated crowd scenes with person bounding boxes",
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (DATA_ROOT / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

_SPLIT_COUNTS = {"train": 60, "valid": 15, "test": 15}

_SHIRT_COLOURS = [
    (200, 50, 50), (50, 50, 200), (50, 200, 50), (200, 200, 50),
    (200, 50, 200), (50, 200, 200), (150, 100, 50), (100, 50, 150),
]


def _generate_synthetic(root: Path) -> None:
    """Create a tiny YOLO-format person detection dataset."""
    random.seed(42)
    np.random.seed(42)

    for split, count in _SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img, labels = _make_crowd_scene(idx)
            cv2.imwrite(str(img_dir / f"crowd_{idx:04d}.png"), img)
            lbl_dir.joinpath(f"crowd_{idx:04d}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )

    # data.yaml
    yaml_text = (
        f"path: {root}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _make_crowd_scene(seed_offset: int) -> tuple[np.ndarray, list[str]]:
    """Generate a synthetic crowd scene with person-like rectangles."""
    h, w = 720, 1280
    # Grey floor / pavement background
    img = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)
    # Add some ground variation
    cv2.rectangle(img, (0, h // 2), (w, h), (160, 160, 160), -1)
    labels: list[str] = []

    n_persons = random.randint(3, 12)
    for i in range(n_persons):
        pw = random.randint(25, 45)
        ph = random.randint(60, 110)
        x1 = random.randint(10, w - pw - 10)
        y1 = random.randint(50, h - ph - 10)
        x2, y2 = x1 + pw, y1 + ph

        colour = _SHIRT_COLOURS[i % len(_SHIRT_COLOURS)]
        # Body
        cv2.rectangle(img, (x1, y1 + ph // 4), (x2, y2), colour, -1)
        # Head
        head_r = pw // 3
        cv2.circle(img, ((x1 + x2) // 2, y1 + head_r), head_r, (180, 160, 140), -1)
        # Legs
        cv2.rectangle(img, (x1 + 2, y2 - ph // 5), (x2 - 2, y2), (60, 60, 80), -1)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        nw = pw / w
        nh = ph / h
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return img, labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_yaml(root: Path) -> Path | None:
    if (root / "data.yaml").exists():
        return root / "data.yaml"
    for child in root.iterdir():
        if child.is_dir() and (child / "data.yaml").exists():
            return child / "data.yaml"
    for candidate in root.rglob("data.yaml"):
        return candidate
    return None
