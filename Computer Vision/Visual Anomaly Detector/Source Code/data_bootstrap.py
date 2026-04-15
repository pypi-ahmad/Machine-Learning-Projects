"""Dataset bootstrap for Visual Anomaly Detector.
"""Dataset bootstrap for Visual Anomaly Detector.

Downloads the MVTec-AD dataset from Hugging Face and prepares the
data/raw / data/processed layout.

Usage::

    from data_bootstrap import ensure_anomaly_dataset

    data_root = ensure_anomaly_dataset()             # idempotent
    data_root = ensure_anomaly_dataset(force=True)   # force re-download
"""
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("visual_anomaly.data_bootstrap")

PROJECT_KEY = "visual_anomaly_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_anomaly_dataset(*, force: bool = False) -> Path:
    """Download and prepare the anomaly detection dataset.
    """Download and prepare the anomaly detection dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset``.
    2. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project's data root.
    """
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    from scripts.download_data import ensure_dataset as _ensure
    data_path = _ensure(PROJECT_KEY, force=force)

    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Write dataset_info.json
    info = {
        "project": PROJECT_KEY,
        "source": "huggingface:alexrods/mvtec-ad",
        "description": "MVTec Anomaly Detection -- industrial product images",
        "structure": "category/train/good/ + category/test/<defect_type>/",
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = data_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def find_normal_images(data_root: Path) -> list[Path]:
    """Find normal/good images in common anomaly dataset layouts.
    """Find normal/good images in common anomaly dataset layouts.

    Searches for directories named 'good', 'normal', or 'OK' under
    train/ or directly under data_root.
    """
    """
    candidates = [
        data_root / "train" / "good",
        data_root / "good",
        data_root / "normal",
        data_root / "train" / "normal",
        data_root / "OK",
        data_root / "train" / "OK",
    ]
    for d in candidates:
        if d.exists():
            imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
            if imgs:
                return imgs

    # Fallback: recurse for any 'good' or 'normal' directory
    for name in ("good", "normal", "OK"):
        for sub in data_root.rglob(name):
            if sub.is_dir():
                imgs = [p for p in sorted(sub.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    return imgs

    return []


def find_test_images(data_root: Path) -> tuple[list[Path], list[Path]]:
    """Find test normal and anomalous images.
    """Find test normal and anomalous images.

    Returns
    -------
    tuple[list[Path], list[Path]]
        ``(test_normal, test_anomaly)`` image paths.
    """
    """
    test_normal: list[Path] = []
    test_anomaly: list[Path] = []

    for pattern in ["test/good", "test/normal", "test/OK"]:
        d = data_root / pattern
        if d.exists():
            test_normal = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
            break

    test_dir = data_root / "test"
    if test_dir.exists():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir() and sub.name.lower() not in ("good", "normal", "ok"):
                test_anomaly.extend(
                    p for p in sorted(sub.iterdir()) if p.suffix.lower() in IMAGE_EXTS
                )

    return test_normal, test_anomaly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_anomaly_dataset(force="--force" in sys.argv)
