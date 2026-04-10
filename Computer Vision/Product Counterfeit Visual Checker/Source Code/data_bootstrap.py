"""Product Counterfeit Visual Checker — idempotent dataset bootstrap.

Downloads the Grocery Store Dataset from Kaggle and organises images
into product sub-folders suitable for reference building.

The dataset has 5,125 images across 81 fine-grained product classes
grouped into 42 coarse-grained categories — ideal for screening
because visually similar products share a coarse category.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "product_counterfeit_visual_checker"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_product_dataset(force: bool = False) -> Path:
    """Download and prepare the grocery store dataset (idempotent).

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

    products_dir = processed / "products"
    if products_dir.exists() and force:
        shutil.rmtree(products_dir)
    products_dir.mkdir(parents=True, exist_ok=True)

    _organise_by_product(raw_dir, products_dir)
    _write_info(raw_dir, products_dir, processed)
    ready_marker.touch()

    return raw_dir


def _organise_by_product(raw_dir: Path, products_dir: Path) -> None:
    """Organise images into product sub-folders.

    The Grocery Store Dataset has structure:
        dataset/
            train/ or test/ or val/
                Fruits/ or Vegetables/ or Packages/
                    ProductName/
                        image.jpg ...

    We flatten this into:
        products/
            ProductName/
                image.jpg ...
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Find the dataset directory (may be nested)
    dataset_dir = _find_dataset_dir(raw_dir)
    if dataset_dir is None:
        print("[WARN] Could not locate dataset directory; scanning raw_dir")
        dataset_dir = raw_dir

    copied = 0
    for f in sorted(dataset_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        if "processed" in f.parts or "iconic" in f.name.lower():
            continue

        # Product label from immediate parent directory
        product = f.parent.name
        if product.lower() in ("train", "test", "val", "fruits", "vegetables",
                                "packages", "dataset", "data", "images"):
            continue

        product_safe = _sanitise_name(product)
        dst_dir = products_dir / product_safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied += 1

    print(f"[INFO] Organised {copied} product image(s)")


def _find_dataset_dir(raw_dir: Path) -> Path | None:
    """Locate the dataset sub-directory within the download."""
    for d in raw_dir.rglob("*"):
        if d.is_dir() and d.name.lower() in ("dataset", "grocerystoredataset"):
            # Check for expected sub-structure (train/test/val)
            children = [c.name.lower() for c in d.iterdir() if c.is_dir()]
            if any(s in children for s in ("train", "test", "val")):
                return d
    # Try direct children
    for d in raw_dir.iterdir():
        if d.is_dir():
            children = [c.name.lower() for c in d.iterdir() if c.is_dir()]
            if any(s in children for s in ("train", "test", "val")):
                return d
    return None


def _sanitise_name(name: str) -> str:
    safe = name.strip().replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_").replace("*", "_").replace("?", "_")
    safe = safe.replace('"', "").replace("<", "").replace(">", "")
    safe = safe.replace("|", "_")
    return safe or "unknown"


def _write_info(raw_dir: Path, products_dir: Path, processed: Path) -> None:
    product_counts: dict[str, int] = {}
    for d in sorted(products_dir.iterdir()):
        if d.is_dir():
            count = len([f for f in d.iterdir() if f.is_file()])
            product_counts[d.name] = count

    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "products_dir": str(products_dir),
        "num_products": len(product_counts),
        "total_images": sum(product_counts.values()),
        "product_counts": product_counts,
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_product_dataset(force=force)
    print(f"Dataset ready at: {path}")
