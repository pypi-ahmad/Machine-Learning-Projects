"""Dataset bootstrap for Gesture Controlled Slideshow.

Downloads and prepares a small public hand-gesture evaluation set
for the slideshow gesture pipeline.

Usage::

    from data_bootstrap import ensure_gesture_dataset

    data_root = ensure_gesture_dataset()            # idempotent
    data_root = ensure_gesture_dataset(force=True)  # force
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("gesture.data_bootstrap")

PROJECT_KEY = "gesture_controlled_slideshow"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY
DATASET_ID = "cj-mills/pexel-hand-gesture-test-images"

CURATED_SAMPLES = [
    ("pexels-andrea-piacquadio-3764395.jpg", "POINTING"),
    ("pexels-ketut-subiyanto-4584599.jpg", "OPEN_PALM"),
    ("pexels-joshua-roberts-12922530.jpg", "PEACE"),
    ("pexels-kevin-malik-9017379.jpg", "THUMBS_UP"),
    ("pexels-sora-shimazaki-5926368.jpg", "FIST"),
    ("pexels-medium-photoclub-4090836.jpg", "POINTING"),
]


@dataclass
class GestureSample:
    media_path: str
    expected_gesture: str
    source_name: str


def ensure_gesture_dataset(*, force: bool = False) -> Path:
    """Download and prepare the gesture evaluation dataset."""
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info(
            "[%s] Dataset already prepared at %s -- skipping",
            PROJECT_KEY, DATA_ROOT,
        )
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    raw_dir = DATA_ROOT / "raw"
    processed_dir = DATA_ROOT / "processed"
    media_dir = processed_dir / "media"
    raw_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    samples = _prepare_subset(raw_dir, media_dir, force=force)

    if not samples:
        raise RuntimeError("Could not prepare a public gesture evaluation dataset automatically.")

    manifest_path = processed_dir / "manifest.csv"
    _write_manifest(manifest_path, samples)
    _write_info(DATA_ROOT, sample_count=len(samples))

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _prepare_subset(
    raw_dir: Path,
    media_dir: Path,
    *,
    force: bool,
) -> list[GestureSample]:
    samples: list[GestureSample] = []
    for filename, expected_gesture in CURATED_SAMPLES:
        cache_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
            force_download=force,
        )

        raw_path = raw_dir / filename
        if not raw_path.exists() or force:
            shutil.copy2(cache_path, raw_path)

        output_path = media_dir / filename
        if not output_path.exists() or force:
            shutil.copy2(cache_path, output_path)

        samples.append(
            GestureSample(
                media_path=str(Path("processed") / "media" / filename),
                expected_gesture=expected_gesture,
                source_name=filename,
            ),
        )

    return samples


def _write_manifest(manifest_path: Path, samples: list[GestureSample]) -> None:
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "media_path",
                "expected_gesture",
                "source_class",
                "set_id",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "media_path": sample.media_path,
                    "expected_gesture": sample.expected_gesture,
                    "source_class": sample.source_name,
                    "set_id": "curated",
                },
            )


def _write_info(data_path: Path, *, sample_count: int) -> None:
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": "huggingface",
        "dataset_id": DATASET_ID,
        "description": "Small curated public image subset prepared for slideshow gesture smoke tests.",
        "sample_count": sample_count,
        "manifest": str((data_path / "processed" / "manifest.csv").relative_to(data_path)),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = data_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the gesture-controlled slideshow evaluation dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the dataset even if it already exists",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parser.parse_args(argv)
    data_root = ensure_gesture_dataset(force=args.force)
    print(data_root)


if __name__ == "__main__":
    main()
