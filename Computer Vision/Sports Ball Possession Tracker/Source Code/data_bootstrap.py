"""Dataset bootstrap for Sports Ball Possession Tracker.

Downloads a public sports detection dataset (players + ball) via
DatasetResolver.  Falls back to generating a small synthetic
YOLO-format dataset for demo and CI purposes.

Usage::

    from data_bootstrap import ensure_sports_dataset

    data_root = ensure_sports_dataset()            # idempotent
    data_root = ensure_sports_dataset(force=True)  # force re-download
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

log = logging.getLogger("sports_possession.data_bootstrap")

PROJECT_KEY = "sports_ball_possession_tracker"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASS_NAMES = ["player", "ball"]


def ensure_sports_dataset(*, force: bool = False) -> Path:
    """Download or generate the sports detection dataset.

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
        "description": "Auto-generated sports field with players and ball",
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

_PLAYER_COLOURS = [
    (200, 50, 50), (50, 50, 200), (200, 200, 50), (50, 200, 200),
    (200, 50, 200), (100, 150, 50),
]


def _generate_synthetic(root: Path) -> None:
    """Create a tiny YOLO-format sports dataset with players and a ball."""
    random.seed(42)
    np.random.seed(42)

    for split, count in _SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img, labels = _make_frame(idx)
            cv2.imwrite(str(img_dir / f"frame_{idx:04d}.png"), img)
            lbl_dir.joinpath(f"frame_{idx:04d}.txt").write_text(
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


def _make_frame(seed_offset: int) -> tuple[np.ndarray, list[str]]:
    """Generate a synthetic sports field frame with players and a ball."""
    h, w = 720, 1280
    # Green field background
    img = np.full((h, w, 3), (50, 140, 50), dtype=np.uint8)
    # Add field lines
    cv2.line(img, (w // 2, 0), (w // 2, h), (220, 220, 220), 2)
    cv2.circle(img, (w // 2, h // 2), 80, (220, 220, 220), 2)
    labels: list[str] = []

    # Players (class 0)
    n_players = random.randint(4, 10)
    for i in range(n_players):
        pw, ph = random.randint(30, 50), random.randint(60, 100)
        x1 = random.randint(20, w - pw - 20)
        y1 = random.randint(20, h - ph - 20)
        x2, y2 = x1 + pw, y1 + ph
        colour = _PLAYER_COLOURS[i % len(_PLAYER_COLOURS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
        # Head
        cv2.circle(img, ((x1 + x2) // 2, y1 - 8), 8, (180, 160, 140), -1)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        nw = pw / w
        nh = ph / h
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # Ball (class 1) — one per frame
    bx = random.randint(40, w - 40)
    by = random.randint(40, h - 40)
    br = random.randint(8, 14)
    cv2.circle(img, (bx, by), br, (255, 255, 255), -1)
    cv2.circle(img, (bx, by), br, (0, 0, 0), 1)

    cx = bx / w
    cy = by / h
    nw = (br * 2) / w
    nh = (br * 2) / h
    labels.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

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
