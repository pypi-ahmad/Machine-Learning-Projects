"""Logo Retrieval Brand Match — idempotent dataset bootstrap.

Downloads the Popular Brand Logos dataset from Kaggle and organises
images into brand sub-folders for index building.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "logo_retrieval_brand_match"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_logo_dataset(force: bool = False) -> Path:
    """Download and prepare the logo dataset (idempotent).

    Parameters
    ----------
    force : bool
        Delete existing data and re-download.

    Returns
    -------
    Path
        Root of the prepared dataset directory.
    """
    from scripts.download_data import ensure_dataset

    raw_dir = ensure_dataset(PROJECT_KEY, force=force)

    processed = raw_dir / "processed"
    ready_marker = processed / ".ready"

    if ready_marker.exists() and not force:
        return raw_dir

    brands_dir = processed / "brands"
    if brands_dir.exists() and force:
        shutil.rmtree(brands_dir)
    brands_dir.mkdir(parents=True, exist_ok=True)

    _organise_by_brand(raw_dir, brands_dir)
    _write_info(raw_dir, brands_dir, processed)
    ready_marker.touch()

    return raw_dir


def _organise_by_brand(raw_dir: Path, brands_dir: Path) -> None:
    """Organise logo images into brand sub-folders.

    Strategy:
    1. Try to use LogoDatabase.csv if present (maps filename → brand).
    2. Otherwise, use parent directory names as brand labels.
    3. Fall back to scanning for brand-named directories.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Try CSV-based mapping first
    csv_path = _find_csv(raw_dir)
    if csv_path:
        _organise_from_csv(raw_dir, brands_dir, csv_path, exts)
        return

    # Fall back to directory-structure-based organisation
    _organise_from_dirs(raw_dir, brands_dir, exts)


def _find_csv(raw_dir: Path) -> Path | None:
    """Find LogoDatabase.csv in the raw download."""
    for f in raw_dir.rglob("*.csv"):
        if "logo" in f.name.lower():
            return f
    return None


def _organise_from_csv(
    raw_dir: Path,
    brands_dir: Path,
    csv_path: Path,
    exts: set[str],
) -> None:
    """Map images to brands using the CSV metadata file."""
    # Read CSV to get image → brand mapping
    brand_map: dict[str, str] = {}
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try common column name patterns
            name = None
            brand = None
            for col in row:
                col_lower = col.lower().strip()
                if col_lower in ("filename", "file", "image", "name", "logo"):
                    name = row[col].strip()
                if col_lower in ("brand", "label", "class", "category", "company"):
                    brand = row[col].strip()
            if name and brand:
                brand_map[name] = brand

    # Copy images with brand mapping
    copied = 0
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() not in exts or "processed" in f.parts:
            continue
        brand = brand_map.get(f.name)
        if brand is None:
            # Try to use parent directory as brand
            if f.parent.name.lower() not in ("logos", "images", "data", "raw"):
                brand = f.parent.name
            else:
                brand = "unknown"

        brand_safe = _sanitise_brand(brand)
        dst_dir = brands_dir / brand_safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied += 1

    print(f"[INFO] Organised {copied} image(s) using CSV mapping")


def _organise_from_dirs(
    raw_dir: Path,
    brands_dir: Path,
    exts: set[str],
) -> None:
    """Organise images using directory names as brand labels."""
    copied = 0
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() not in exts or "processed" in f.parts:
            continue

        # Use the immediate parent directory as the brand name
        parent = f.parent.name
        if parent.lower() in ("logos", "images", "data", "raw", ""):
            parent = "unknown"

        brand_safe = _sanitise_brand(parent)
        dst_dir = brands_dir / brand_safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied += 1

    print(f"[INFO] Organised {copied} image(s) from directory structure")


def _sanitise_brand(name: str) -> str:
    """Clean brand name for safe directory naming."""
    safe = name.strip().replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_").replace("*", "_").replace("?", "_")
    safe = safe.replace('"', "").replace("<", "").replace(">", "")
    safe = safe.replace("|", "_")
    return safe or "unknown"


def _write_info(raw_dir: Path, brands_dir: Path, processed: Path) -> None:
    brand_counts: dict[str, int] = {}
    for d in brands_dir.iterdir():
        if d.is_dir():
            count = len(list(d.iterdir()))
            brand_counts[d.name] = count

    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "brands_dir": str(brands_dir),
        "num_brands": len(brand_counts),
        "total_images": sum(brand_counts.values()),
        "brand_counts": brand_counts,
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_logo_dataset(force=force)
    print(f"Dataset ready at: {path}")
