"""Similar Image Finder — idempotent dataset bootstrap.

Downloads the Natural Images dataset from Kaggle and verifies
images are organised into category sub-folders.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "similar_image_finder"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_image_dataset(force: bool = False) -> Path:
    """Download and prepare the natural images dataset (idempotent).

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

    images_dir = processed / "images"
    if images_dir.exists() and force:
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    _organise_by_category(raw_dir, images_dir)
    _write_info(raw_dir, images_dir, processed)
    ready_marker.touch()

    return raw_dir


def _organise_by_category(raw_dir: Path, images_dir: Path) -> None:
    """Organise images into category sub-folders.

    The Natural Images dataset already has category sub-folders
    (airplane, car, cat, dog, flower, fruit, motorbike, person).
    This copies them into the processed directory for consistency.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Look for the natural_images sub-directory
    candidates = [
        raw_dir / "natural_images",
        raw_dir / "natural images",
    ]
    # Also search for any directory containing category folders
    for d in raw_dir.rglob("*"):
        if d.is_dir() and d.name.lower() in ("natural_images", "natural images"):
            candidates.append(d)

    source = None
    for c in candidates:
        if c.is_dir():
            # Check it has sub-folders with images
            sub_dirs = [s for s in c.iterdir() if s.is_dir()]
            if sub_dirs:
                source = c
                break

    if source is None:
        # Fall back to scanning raw_dir for category-like folders
        source = raw_dir

    copied = 0
    for cat_dir in sorted(source.iterdir()):
        if not cat_dir.is_dir():
            continue
        if cat_dir.name.startswith(".") or cat_dir.name == "processed":
            continue

        cat_name = cat_dir.name.lower().strip()
        dst_dir = images_dir / cat_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted(cat_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                dst = dst_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1

    print(f"[INFO] Organised {copied} image(s) into category folders")


def _write_info(raw_dir: Path, images_dir: Path, processed: Path) -> None:
    """Write dataset summary JSON."""
    cat_counts: dict[str, int] = {}
    for d in sorted(images_dir.iterdir()):
        if d.is_dir():
            count = len([f for f in d.iterdir() if f.is_file()])
            cat_counts[d.name] = count

    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "images_dir": str(images_dir),
        "num_categories": len(cat_counts),
        "total_images": sum(cat_counts.values()),
        "category_counts": cat_counts,
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_image_dataset(force=force)
    print(f"Dataset ready at: {path}")
