"""Dataset bootstrap for Handwritten Note to Markdown.

Downloads and prepares a public handwriting recognition dataset
for testing and benchmarking the TrOCR pipeline.

Usage::

    from data_bootstrap import ensure_handwriting_dataset

    data_root = ensure_handwriting_dataset()              # idempotent
    data_root = ensure_handwriting_dataset(force=True)    # force re-download
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

log = logging.getLogger("handwritten_note.data_bootstrap")

PROJECT_KEY = "handwritten_note_to_markdown"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_handwriting_dataset(*, force: bool = False) -> Path:
    """Download and prepare the handwriting dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset``.
    2. Organises images into ``data/processed/images/``.
    3. Writes ``data/dataset_info.json`` with provenance metadata.
    4. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project data root (``data/handwritten_note_to_markdown/``).
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_images(data_path: Path, processed_dir: Path) -> None:
    """Collect all handwriting images into processed/images/."""
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
        "[%s] Collected %d handwriting images into %s",
        PROJECT_KEY, count, images_dir,
    )


def _write_info(data_path: Path) -> None:
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
