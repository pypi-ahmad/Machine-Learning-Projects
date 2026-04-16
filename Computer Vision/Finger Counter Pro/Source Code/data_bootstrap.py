"""Finger Counter Pro -- idempotent dataset bootstrap."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_KEY = "finger_counter_pro"
DATASET_ID = "cj-mills/pexel-hand-gesture-test-images"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DATA_ROOT = _REPO / "data" / PROJECT_KEY

CURATED_SAMPLES = [
    ("pexels-andrea-piacquadio-3764395.jpg", 1),
    ("pexels-cottonbro-studio-6284256.jpg", 1),
    ("pexels-ketut-subiyanto-4584445.jpg", 1),
    ("pexels-joshua-roberts-12922530.jpg", 2),
    ("pexels-kampus-production-6684834.jpg", 2),
    ("pexels-kevin-malik-9017379.jpg", 1),
]


@dataclass
class SampleEntry:
    media_path: str
    expected_total: int
    source_name: str


def ensure_finger_counter_dataset(force: bool = False) -> Path:
    """Download and prepare the evaluation dataset (idempotent).

    Parameters
    ----------
    force : bool
        If *True*, delete any existing data and re-download.

    Returns
    -------
    Path
        Root of the prepared dataset directory.
    """
    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    raw_dir = DATA_ROOT / "raw"
    processed = DATA_ROOT / "processed"
    ready_marker = processed / ".ready"

    if ready_marker.exists() and not force:
        return DATA_ROOT

    raw_dir.mkdir(parents=True, exist_ok=True)
    media_dir = processed / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_media(raw_dir, media_dir, force=force)
    _write_manifest(processed / "manifest.csv", samples)
    _write_info(DATA_ROOT, len(samples))
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")

    return DATA_ROOT


def _collect_media(raw_dir: Path, media_dir: Path, *, force: bool) -> list[SampleEntry]:
    """Download a small curated public image set into processed/media/."""
    entries: list[SampleEntry] = []
    for filename, expected_total in CURATED_SAMPLES:
        cached_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
            force_download=force,
        )

        raw_path = raw_dir / filename
        media_path = media_dir / filename
        if force or not raw_path.exists():
            shutil.copy2(cached_path, raw_path)
        if force or not media_path.exists():
            shutil.copy2(cached_path, media_path)

        entries.append(
            SampleEntry(
                media_path=str(Path("processed") / "media" / filename),
                expected_total=expected_total,
                source_name=filename,
            )
        )
    return entries


def _write_manifest(manifest_path: Path, entries: list[SampleEntry]) -> None:
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["media_path", "expected_total", "source_name"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "media_path": entry.media_path,
                    "expected_total": entry.expected_total,
                    "source_name": entry.source_name,
                }
            )


def _write_info(data_root: Path, sample_count: int) -> None:
    """Write dataset_info.json with provenance metadata."""
    info = {
        "project": PROJECT_KEY,
        "dataset_id": DATASET_ID,
        "description": "Small curated public hand image subset for finger-count smoke tests.",
        "source_dir": str(data_root),
        "media_count": sample_count,
        "manifest": "processed/manifest.csv",
    }
    info_path = data_root / "processed" / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_finger_counter_dataset(force=force)
    print(f"Dataset ready at: {path}")
