"""Dataset bootstrap for Face Clustering Photo Organizer.

Prepares a multi-identity public face dataset for clustering evaluation.
The bootstrap keeps an ImageFolder-style layout so each identity has its
own directory under `processed/identities/`.
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

log = logging.getLogger("face_cluster.data_bootstrap")

PROJECT_KEY = "face_clustering_photo_organizer"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_FACES_PER_IDENTITY = 4
MAX_IMAGES_PER_IDENTITY = 6


def ensure_face_cluster_dataset(*, force: bool = False) -> Path:
    """Download and prepare the face clustering evaluation dataset."""
    ready_marker = DATA_ROOT / "processed" / ".ready"
    identities_dir = DATA_ROOT / "processed" / "identities"
    if ready_marker.exists() and not force:
        log.info(
            "[%s] Dataset already prepared at %s -- skipping",
            PROJECT_KEY, DATA_ROOT,
        )
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    raw_dir = DATA_ROOT / "raw"
    identities_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_source = "sklearn_lfw"
    identity_count = 0
    image_count = 0

    try:
        identity_count, image_count = _prepare_lfw_dataset(identities_dir)
    except Exception as exc:
        log.warning("[%s] LFW bootstrap failed: %s", PROJECT_KEY, exc)
        dataset_source = "download_helper"
        identity_count, image_count = _prepare_download_helper_dataset(
            identities_dir,
            force=force,
        )

    if identity_count == 0 or image_count == 0:
        raise RuntimeError(
            "Could not prepare a multi-identity face dataset automatically.",
        )

    _write_info(
        DATA_ROOT,
        dataset_source=dataset_source,
        identity_count=identity_count,
        image_count=image_count,
    )

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _prepare_lfw_dataset(identities_dir: Path) -> tuple[int, int]:
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

        identity_dir = identities_dir / name
        identity_dir.mkdir(parents=True, exist_ok=True)

        image_uint8 = image
        if image_uint8.max() <= 1.0:
            image_uint8 = image_uint8 * 255.0
        image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)
        if image_uint8.ndim == 2:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        out_path = identity_dir / f"{current_count + 1:03d}.jpg"
        cv2.imwrite(str(out_path), image_bgr)
        per_identity_count[name] = current_count + 1

    identity_count = len(per_identity_count)
    image_count = sum(per_identity_count.values())
    log.info(
        "[%s] Prepared %d identities / %d images from LFW",
        PROJECT_KEY,
        identity_count,
        image_count,
    )
    return identity_count, image_count


def _prepare_download_helper_dataset(
    identities_dir: Path,
    *,
    force: bool,
) -> tuple[int, int]:
    try:
        from scripts.download_data import ensure_dataset as ensure_dataset_helper
    except Exception as exc:
        log.warning("[%s] Download helper unavailable: %s", PROJECT_KEY, exc)
        return 0, 0

    try:
        data_path = ensure_dataset_helper(PROJECT_KEY, force=force)
    except Exception as exc:
        log.warning("[%s] Download helper failed: %s", PROJECT_KEY, exc)
        return 0, 0

    candidate_dirs: list[Path] = []
    for child in sorted(data_path.iterdir()):
        if not child.is_dir() or child.name == "processed":
            continue
        if _contains_images(child):
            candidate_dirs.append(child)
            continue
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and _contains_images(grandchild):
                candidate_dirs.append(grandchild)

    image_count = 0
    for directory in candidate_dirs:
        identity_name = _sanitize_identity_name(directory.name)
        target_dir = identities_dir / identity_name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for image_path in sorted(directory.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.is_file():
                continue
            copied += 1
            if copied > MAX_IMAGES_PER_IDENTITY:
                break
            destination = target_dir / f"{copied:03d}{image_path.suffix.lower()}"
            shutil.copy2(str(image_path), str(destination))
            image_count += 1

    identity_count = len([d for d in identities_dir.iterdir() if d.is_dir()])
    log.info(
        "[%s] Prepared %d identities / %d images from helper dataset",
        PROJECT_KEY,
        identity_count,
        image_count,
    )
    return identity_count, image_count


def _contains_images(path: Path) -> bool:
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def _sanitize_identity_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _write_info(
    data_path: Path,
    *,
    dataset_source: str,
    identity_count: int,
    image_count: int,
) -> None:
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": dataset_source,
        "description": "Multi-identity face dataset prepared for clustering evaluation.",
        "identity_count": identity_count,
        "image_count": image_count,
        "identity_layout": str((data_path / "processed" / "identities").relative_to(data_path)),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = data_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the face clustering dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the dataset even if it already exists",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parser.parse_args(argv)
    data_root = ensure_face_cluster_dataset(force=args.force)
    print(data_root)


if __name__ == "__main__":
    main()
