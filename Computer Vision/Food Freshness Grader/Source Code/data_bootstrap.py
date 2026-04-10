"""Food Freshness Grader — idempotent dataset bootstrap.

Downloads the Fresh and Stale Images of Fruits and Vegetables
dataset from Kaggle and prepares it for training.

The dataset is already in ImageFolder format with 12 class folders:
fresh_apple, fresh_banana, fresh_bitter_gourd, fresh_capsicum,
fresh_orange, fresh_tomato, stale_apple, stale_banana,
stale_bitter_gourd, stale_capsicum, stale_orange, stale_tomato.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "food_freshness_grader"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_freshness_dataset(force: bool = False) -> Path:
    """Download and prepare the freshness dataset (idempotent).

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

    train_dir = processed / "train"
    if train_dir.exists() and force:
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    _organise_splits(raw_dir, processed)
    _write_info(raw_dir, processed)
    ready_marker.touch()

    return raw_dir


def _organise_splits(raw_dir: Path, processed: Path) -> None:
    """Organise images into train/val splits.

    The original dataset has flat class folders at the top level.
    We create train/ and val/ with the same class folders.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    import random

    random.seed(42)

    # Find the class folders
    source = _find_class_root(raw_dir)
    if source is None:
        print("[WARN] Could not locate class folders; scanning raw_dir")
        source = raw_dir

    train_dir = processed / "train"
    val_dir = processed / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith(".") or class_dir.name == "processed":
            continue

        class_name = class_dir.name.lower().strip()

        # Collect all images in this class
        images = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        )

        if not images:
            continue

        # 80/20 split
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy to train/
        dst_train = train_dir / class_name
        dst_train.mkdir(parents=True, exist_ok=True)
        for f in train_imgs:
            dst = dst_train / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                total_train += 1

        # Copy to val/
        dst_val = val_dir / class_name
        dst_val.mkdir(parents=True, exist_ok=True)
        for f in val_imgs:
            dst = dst_val / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                total_val += 1

    print(f"[INFO] Organised {total_train} train + {total_val} val images")


def _find_class_root(raw_dir: Path) -> Path | None:
    """Find the directory containing class folders (e.g. fresh_apple/)."""
    # Check if raw_dir itself has class folders
    children = [d.name.lower() for d in raw_dir.iterdir() if d.is_dir()]
    if any("fresh" in c or "stale" in c for c in children):
        return raw_dir

    # Search one level deeper
    for d in raw_dir.iterdir():
        if d.is_dir() and d.name != "processed":
            sub_children = [s.name.lower() for s in d.iterdir() if s.is_dir()]
            if any("fresh" in c or "stale" in c for c in sub_children):
                return d

    # Search two levels deep
    for d in raw_dir.rglob("*"):
        if d.is_dir():
            sub_children = [s.name.lower() for s in d.iterdir() if s.is_dir()]
            if sum(1 for c in sub_children if "fresh" in c or "stale" in c) >= 4:
                return d

    return None


def _write_info(raw_dir: Path, processed: Path) -> None:
    """Write dataset summary JSON."""
    class_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}

    for split in ("train", "val"):
        split_dir = processed / split
        if not split_dir.exists():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir():
                count = len([f for f in d.iterdir() if f.is_file()])
                class_counts[split][d.name] = count

    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "train_classes": len(class_counts["train"]),
        "val_classes": len(class_counts["val"]),
        "train_images": sum(class_counts["train"].values()),
        "val_images": sum(class_counts["val"].values()),
        "class_counts": class_counts,
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_freshness_dataset(force=force)
    print(f"Dataset ready at: {path}")
