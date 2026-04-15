"""Dataset bootstrap for Document Layout Block Detector.

Downloads a public document-layout detection dataset via DatasetResolver.
Falls back to generating a small synthetic YOLO-format dataset for demo
and CI purposes when the real dataset is unavailable.

Usage::

    from data_bootstrap import ensure_layout_dataset

    data_root = ensure_layout_dataset()            # idempotent
    data_root = ensure_layout_dataset(force=True)  # force re-download
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

log = logging.getLogger("doc_layout.data_bootstrap")

PROJECT_KEY = "document_layout_block_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

CLASS_NAMES = [
    "title", "text", "table", "figure", "list",
    "caption", "header", "footer", "page-number", "stamp",
]


def ensure_layout_dataset(*, force: bool = False) -> Path:
    """Download or generate the document layout detection dataset.

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
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

_SPLIT_COUNTS = {"train": 60, "valid": 15, "test": 15}

# BGR colours for each class when rendering synthetic blocks
_CLASS_COLOURS = [
    (200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50),
    (200, 50, 200), (50, 200, 200), (128, 128, 50), (50, 128, 128),
    (180, 100, 50), (100, 50, 180),
]


def _generate_synthetic(root: Path) -> None:
    """Create a tiny YOLO-format document layout dataset."""
    random.seed(42)
    np.random.seed(42)

    for split, count in _SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img, labels = _make_page(idx)
            cv2.imwrite(str(img_dir / f"page_{idx:04d}.png"), img)
            lbl_dir.joinpath(f"page_{idx:04d}.txt").write_text(
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


def _make_page(seed_offset: int) -> tuple[np.ndarray, list[str]]:
    """Generate a single synthetic document page with layout blocks."""
    h, w = 1024, 768
    img = np.full((h, w, 3), 245, dtype=np.uint8)  # near-white background
    labels: list[str] = []
    n_blocks = random.randint(3, 8)

    for _ in range(n_blocks):
        cls_id = random.randint(0, len(CLASS_NAMES) - 1)
        bw = random.randint(80, w // 2)
        bh = random.randint(30, h // 4)
        x1 = random.randint(20, w - bw - 20)
        y1 = random.randint(20, h - bh - 20)
        x2, y2 = x1 + bw, y1 + bh

        colour = _CLASS_COLOURS[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
        # Add some "text lines" inside text/list blocks
        if cls_id in (1, 4):  # text, list
            for ly in range(y1 + 10, y2 - 5, 12):
                lx2 = min(x2 - 5, x1 + random.randint(60, bw))
                cv2.line(img, (x1 + 5, ly), (lx2, ly), (80, 80, 80), 1)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        nw = bw / w
        nh = bh / h
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

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
