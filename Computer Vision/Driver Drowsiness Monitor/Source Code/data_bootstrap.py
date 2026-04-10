"""Dataset bootstrap for Driver Drowsiness Monitor.

Downloads and prepares a public drowsiness-detection dataset from
Hugging Face for evaluating the monitoring pipeline.

Usage::

    from data_bootstrap import ensure_drowsiness_dataset

    data_root = ensure_drowsiness_dataset()            # idempotent
    data_root = ensure_drowsiness_dataset(force=True)  # force
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

log = logging.getLogger("drowsiness.data_bootstrap")

PROJECT_KEY = "driver_drowsiness_monitor"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def ensure_drowsiness_dataset(*, force: bool = False) -> Path:
    """Download and prepare the drowsiness evaluation dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset``.
    2. Organises images and videos into ``data/processed/``.
    3. Writes ``data/dataset_info.json`` with provenance metadata.
    4. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project data root.
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

    _collect_media(data_path, processed_dir)
    _write_info(data_path)

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _collect_media(data_path: Path, processed_dir: Path) -> None:
    """Collect images and videos into processed/media/."""
    media_dir = processed_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    vid_count = 0
    for f in data_path.rglob("*"):
        if "processed" in f.parts:
            continue
        if f.suffix.lower() in IMAGE_EXTS:
            dst = media_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                img_count += 1
        elif f.suffix.lower() in VIDEO_EXTS:
            dst = media_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                vid_count += 1

    log.info(
        "[%s] Collected %d images, %d videos into %s",
        PROJECT_KEY, img_count, vid_count, media_dir,
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
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
