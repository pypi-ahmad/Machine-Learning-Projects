"""Dataset bootstrap for Ecommerce Item Attribute Tagger.

Downloads the Fashion Product Images (Small) dataset from Kaggle
and prepares label maps from ``styles.csv``.

Usage::

    from data_bootstrap import ensure_tagger_dataset

    data_root = ensure_tagger_dataset()
    data_root = ensure_tagger_dataset(force=True)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("attribute_tagger.data_bootstrap")

PROJECT_KEY = "ecommerce_item_attribute_tagger"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

ATTRIBUTE_COLUMNS = [
    "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "usage",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_tagger_dataset(*, force: bool = False) -> Path:
    """Download and prepare the fashion product dataset.

    Returns the project data root.
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared — skipping", PROJECT_KEY)
        return DATA_ROOT

    from scripts.download_data import ensure_dataset as _ensure
    data_path = _ensure(PROJECT_KEY, force=force)

    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Build label maps from styles.csv
    styles_csv = _find_styles_csv(data_path)
    if styles_csv:
        label_maps = _build_label_maps(styles_csv, processed_dir)
        log.info("Built label maps from %s", styles_csv)

    # Write dataset_info.json
    info = {
        "project": PROJECT_KEY,
        "source": "kaggle:paramaggarwal/fashion-product-images-small",
        "license": "MIT",
        "description": "Fashion Product Images (Small) — 44k products with attribute labels",
        "attributes": ATTRIBUTE_COLUMNS,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (data_path / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8",
    )

    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _find_styles_csv(data_path: Path) -> Path | None:
    """Locate styles.csv in the dataset hierarchy."""
    for candidate in [
        data_path / "styles.csv",
        data_path / "raw" / "styles.csv",
    ]:
        if candidate.exists():
            return candidate

    # Recurse
    for p in data_path.rglob("styles.csv"):
        return p

    return None


def _build_label_maps(
    styles_csv: Path,
    processed_dir: Path,
    min_samples: int = 20,
) -> dict[str, list[str]]:
    """Parse styles.csv and produce per-attribute label lists.

    Labels with fewer than ``min_samples`` occurrences are merged
    into ``<other>``.

    Also saves ``label_maps.json`` and ``styles_clean.csv`` to
    *processed_dir*.
    """
    import csv

    rows: list[dict] = []
    with open(styles_csv, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    label_maps: dict[str, list[str]] = {}

    for col in ATTRIBUTE_COLUMNS:
        counter: Counter[str] = Counter()
        for row in rows:
            val = (row.get(col) or "").strip()
            if val:
                counter[val] += 1

        # Keep labels with enough samples
        kept = sorted(k for k, v in counter.items() if v >= min_samples)
        kept.append("<other>")
        label_maps[col] = kept

    # Save label maps
    maps_path = processed_dir / "label_maps.json"
    maps_path.write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    # Save cleaned styles with mapped labels
    clean_path = processed_dir / "styles_clean.csv"
    with open(clean_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["id"] + ATTRIBUTE_COLUMNS,
        )
        writer.writeheader()
        for row in rows:
            clean_row = {"id": row.get("id", "")}
            for col in ATTRIBUTE_COLUMNS:
                val = (row.get(col) or "").strip()
                if val and val in label_maps[col]:
                    clean_row[col] = val
                else:
                    clean_row[col] = "<other>"
            writer.writerow(clean_row)

    log.info("Saved label_maps.json and styles_clean.csv to %s", processed_dir)
    return label_maps


def find_images_dir(data_root: Path) -> Path | None:
    """Locate the images directory."""
    for candidate in [
        data_root / "images",
        data_root / "raw" / "images",
    ]:
        if candidate.exists():
            return candidate

    for p in data_root.rglob("images"):
        if p.is_dir():
            return p

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_tagger_dataset(force="--force" in sys.argv)
