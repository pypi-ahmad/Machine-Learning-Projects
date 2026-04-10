"""Dataset bootstrap for Number Plate Reader Pro.

Downloads and prepares the Roboflow license plate detection dataset
for training and evaluation.

Usage::

    from data_bootstrap import ensure_plate_dataset

    data_root = ensure_plate_dataset()              # idempotent
    data_root = ensure_plate_dataset(force=True)    # force re-download
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("plate_reader.data_bootstrap")

PROJECT_KEY = "number_plate_reader_pro"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_plate_dataset(*, force: bool = False) -> Path:
    """Download and prepare the license plate detection dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset``.
    2. Collects images into ``data/processed/images/``.
    3. Writes ``data/dataset_info.json`` with provenance metadata.
    4. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project data root (``data/number_plate_reader_pro/``).
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info(
            "[%s] Dataset already prepared at %s — skipping",
            PROJECT_KEY, DATA_ROOT,
        )
        return DATA_ROOT

    from scripts.download_data import ensure_dataset as _ensure

    data_path = _ensure(PROJECT_KEY, force=force)

    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    _collect_images(data_path, processed_dir)
    _write_info(data_path)

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_images(data_path: Path, processed_dir: Path) -> None:
    """Collect all plate images into processed/images/."""
    images_dir = processed_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in data_path.rglob("*"):
        if img.suffix.lower() in IMAGE_EXTS and "processed" not in img.parts:
            dst = images_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
                count += 1

    log.info(
        "[%s] Collected %d plate images into %s",
        PROJECT_KEY, count, images_dir,
    )


def _write_info(data_path: Path) -> None:
    """Write dataset provenance metadata."""
    info_path = data_path / "dataset_info.json"
    if info_path.exists():
        return

    from utils.datasets import DatasetResolver

    resolver = DatasetResolver()
    entry = resolver.registry.get(PROJECT_KEY, {})

    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": entry.get("type", "unknown"),
        "description": entry.get("description", ""),
        "source_workspace": entry.get("workspace", ""),
        "source_project": entry.get("project", ""),
        "source_version": entry.get("version", ""),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
