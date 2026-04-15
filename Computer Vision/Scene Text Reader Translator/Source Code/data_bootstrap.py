"""Dataset bootstrap for Scene Text Reader Translator.

Downloads and prepares a public scene text dataset for testing
and benchmarking the OCR + translation pipeline.

Uses a public scene-text dataset when available and falls back to
synthetic scene-text images with metadata when the download path is
unavailable.

Usage::

    from data_bootstrap import ensure_scene_text_dataset

    data_root = ensure_scene_text_dataset()              # idempotent
    data_root = ensure_scene_text_dataset(force=True)    # force re-download
"""

from __future__ import annotations

import logging
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("scene_text.data_bootstrap")

PROJECT_KEY = "scene_text_reader_translator"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_scene_text_dataset(*, force: bool = False) -> Path:
    """Download and prepare the scene-text dataset (idempotent)."""
    images_dir = DATA_ROOT / "images"
    ready_marker = DATA_ROOT / ".ready"

    if ready_marker.exists() and not force:
        log.info("[%s] Dataset ready at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    download_error = ""

    try:
        from scripts.download_data import ensure_dataset as _ensure

        data_path = _ensure(PROJECT_KEY, force=force)
        images_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for img in data_path.rglob("*"):
            if img.suffix.lower() in IMAGE_EXTS and img.parent != images_dir:
                dst = images_dir / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
                    count += 1

        if count > 0:
            _write_metadata_json({}, "")
            _write_dataset_info(
                source="public-scene-text-dataset",
                license_note="See the upstream dataset page for current licence terms.",
                sample_count=count,
                fallback_used=False,
                download_error="",
            )
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return DATA_ROOT
    except Exception as exc:
        download_error = str(exc)
        log.warning(
            "[%s] Public dataset bootstrap failed (%s); generating synthetic scene-text images",
            PROJECT_KEY,
            exc,
        )

    metadata = _generate_synthetic_scene_images(images_dir, n=12)
    _write_metadata_json(metadata, "synthetic")
    _write_dataset_info(
        source="synthetic-scene-text-v1",
        license_note="Generated locally for testing and validation.",
        sample_count=len(metadata),
        fallback_used=True,
        download_error=download_error,
    )
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


_TEXT_GROUPS = [
    ["CITY CAFE", "OPEN 24 HOURS", "ESPRESSO 3.50"],
    ["BOOK SHOP", "SALE TODAY", "BUY 2 GET 1"],
    ["METRO STATION", "LINE 2", "EXIT EAST"],
    ["PARKING", "LEVEL B2", "PAY HERE"],
    ["FRESH MART", "ORGANIC FRUIT", "BEST PRICE"],
    ["HOTEL PLAZA", "CHECK IN", "VACANCY"],
    ["BUS STOP", "ROUTE 18", "NEXT 5 MIN"],
    ["MUSEUM", "TICKETS", "ENTRY GATE"],
    ["NO PARKING", "TOW AWAY", "ZONE 4"],
    ["RIVER WALK", "FOOD COURT", "THIS WAY"],
    ["LIBRARY", "QUIET FLOOR", "ROOM 204"],
    ["WELCOME", "MAIN LOBBY", "INFORMATION"],
]


def _generate_synthetic_scene_images(out_dir: Path, n: int = 12) -> dict[str, dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    metadata: dict[str, dict[str, object]] = {}

    for index in range(n):
        width = 1280
        height = 720
        image = np.full((height, width, 3), 235, dtype=np.uint8)
        sky = np.linspace(220, 170, height // 2, dtype=np.uint8)
        for row in range(height // 2):
            image[row, :, :] = (sky[row], sky[row], 255)
        image[height // 2 :, :, :] = (80, 80, 80)

        blocks_meta: list[dict[str, object]] = []
        group = _TEXT_GROUPS[index % len(_TEXT_GROUPS)]
        x0 = 120 + (index % 3) * 180
        y0 = 130 + (index % 2) * 80

        sign_w = 500
        sign_h = 220
        sign_colour = (
            rng.randint(25, 110),
            rng.randint(55, 150),
            rng.randint(90, 190),
        )
        cv2.rectangle(image, (x0, y0), (x0 + sign_w, y0 + sign_h), sign_colour, -1)
        cv2.rectangle(image, (x0, y0), (x0 + sign_w, y0 + sign_h), (20, 20, 20), 3)

        for line_index, text in enumerate(group):
            tx = x0 + 28
            ty = y0 + 60 + line_index * 58
            cv2.putText(
                image,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (245, 245, 245),
                3,
                cv2.LINE_AA,
            )
            text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.25, 3)
            x1 = tx - 6
            y1 = ty - text_size[1] - 8
            x2 = tx + text_size[0] + 6
            y2 = ty + baseline + 6
            blocks_meta.append(
                {
                    "text": text,
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                }
            )

        image_name = f"scene_text_{index:04d}.png"
        cv2.imwrite(str(out_dir / image_name), image)
        metadata[image_name] = {"blocks": blocks_meta}

    return metadata


def _write_metadata_json(metadata: dict[str, dict[str, object]], source_type: str) -> None:
    payload = {"source_type": source_type, "images": metadata}
    (DATA_ROOT / "ocr_labels.json").write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_dataset_info(
    *,
    source: str,
    license_note: str,
    sample_count: int,
    fallback_used: bool,
    download_error: str,
) -> None:
    payload = {
        "dataset_key": PROJECT_KEY,
        "source": source,
        "license_note": license_note,
        "fallback_used": fallback_used,
        "sample_count": sample_count,
        "images_dir": str((DATA_ROOT / 'images').resolve()),
        "metadata_json": str((DATA_ROOT / 'ocr_labels.json').resolve()),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if download_error:
        payload["download_error"] = download_error
    (DATA_ROOT / "dataset_info.json").write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    path = ensure_scene_text_dataset()
    print(path)


if __name__ == "__main__":
    main()
