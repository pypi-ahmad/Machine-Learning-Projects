"""Dataset bootstrap for Blink Headpose Analyzer.

Downloads and prepares a public blink/facial-landmark dataset
from Hugging Face for evaluating the analysis pipeline.

Usage::

    from data_bootstrap import ensure_blink_headpose_dataset

    data_root = ensure_blink_headpose_dataset()            # idempotent
    data_root = ensure_blink_headpose_dataset(force=True)  # force
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("blink_headpose.data_bootstrap")

PROJECT_KEY = "blink_headpose_analyzer"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_FACES_PER_IDENTITY = 4
MAX_IMAGES_PER_IDENTITY = 6


def ensure_blink_headpose_dataset(*, force: bool = False) -> Path:
    """Download and prepare the evaluation dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset``.
    2. Collects media into ``data/processed/media/``.
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
            "[%s] Dataset already prepared at %s -- skipping",
            PROJECT_KEY, DATA_ROOT,
        )
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    processed_dir = DATA_ROOT / "processed"
    raw_dir = DATA_ROOT / "raw"
    media_dir = processed_dir / "media"
    raw_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    dataset_source = "sklearn_lfw"
    data_path = DATA_ROOT
    try:
        media_count = _prepare_lfw_dataset(media_dir)
    except Exception as exc:
        log.warning(
            "[%s] LFW bootstrap failed; falling back to shared downloader: %s",
            PROJECT_KEY,
            exc,
        )
        dataset_source = "download_helper"
        from scripts.download_data import ensure_dataset as _ensure

        data_path = _ensure(PROJECT_KEY, force=force)
        raw_dir = data_path / "raw"
        processed_dir = data_path / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        _collect_media(data_path, processed_dir)
        media_count = _count_media(processed_dir / "media")

    _write_info(data_path, dataset_source=dataset_source, media_count=media_count)

    if media_count == 0:
        raise RuntimeError("Could not prepare a public blink/head-pose evaluation dataset automatically.")

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _prepare_lfw_dataset(media_dir: Path) -> int:
    from sklearn.datasets import fetch_lfw_people

    dataset = fetch_lfw_people(
        data_home=str(DATA_ROOT / "raw"),
        min_faces_per_person=MIN_FACES_PER_IDENTITY,
        color=True,
        resize=1.0,
        download_if_missing=True,
    )

    per_identity_count: dict[str, int] = {}
    for image, target in zip(dataset.images, dataset.target):
        name = _sanitize_identity_name(str(dataset.target_names[target]))
        current_count = per_identity_count.get(name, 0)
        if current_count >= MAX_IMAGES_PER_IDENTITY:
            continue

        image_uint8 = image
        if image_uint8.max() <= 1.0:
            image_uint8 = image_uint8 * 255.0
        image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)
        if image_uint8.ndim == 2:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        out_path = media_dir / f"{name}_{current_count + 1:03d}.jpg"
        cv2.imwrite(str(out_path), image_bgr)
        per_identity_count[name] = current_count + 1

    media_count = sum(per_identity_count.values())
    log.info(
        "[%s] Prepared %d images from LFW",
        PROJECT_KEY,
        media_count,
    )
    return media_count


def _collect_media(data_path: Path, processed_dir: Path) -> None:
    """Collect images and videos into processed/media/."""
    media_dir = processed_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    vid_count = 0
    for f in data_path.rglob("*"):
        if processed_dir in f.parents:
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


def _count_media(media_dir: Path) -> int:
    if not media_dir.exists():
        return 0

    count = 0
    for media_path in media_dir.iterdir():
        if media_path.is_file() and media_path.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS):
            count += 1
    return count


def _sanitize_identity_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _write_info(
    data_path: Path,
    *,
    dataset_source: str,
    media_count: int,
) -> None:
    """Write dataset provenance metadata."""
    info_path = data_path / "dataset_info.json"
    if info_path.exists():
        return

    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": dataset_source,
        "description": "Public face images prepared for blink and head-pose evaluation.",
        "media_count": media_count,
        "media_layout": str((data_path / "processed" / "media").relative_to(data_path)),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the blink/headpose evaluation dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the dataset even if it already exists",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parser.parse_args(argv)
    data_root = ensure_blink_headpose_dataset(force=args.force)
    print(data_root)


if __name__ == "__main__":
    main()
