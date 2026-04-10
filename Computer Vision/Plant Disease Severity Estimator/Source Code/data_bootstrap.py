"""Plant Disease Severity Estimator — idempotent dataset bootstrap.

Downloads PlantVillage (abdallahalidev/plantvillage-dataset) from Kaggle
and prepares it for training by extracting the ``color/`` sub-directory
with its 38 ImageFolder class folders and creating an 80/20 train/val
split.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "plant_disease_severity_estimator"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_plant_dataset(force: bool = False) -> Path:
    """Download and prepare PlantVillage (idempotent).

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


def _find_color_root(raw_dir: Path) -> Path | None:
    """Locate the ``color/`` directory containing 38 class folders.

    PlantVillage ships three directories: ``color/``, ``grayscale/``,
    and ``segmented/``.  We use only ``color/``.
    """
    # Direct child
    for d in raw_dir.iterdir():
        if d.is_dir() and d.name.lower() == "color":
            return d

    # Search recursively (handles extra nesting)
    for d in raw_dir.rglob("color"):
        if d.is_dir():
            children = [c for c in d.iterdir() if c.is_dir()]
            if len(children) >= 20:
                return d

    # Fallback: look for a directory with many class-style children
    for d in raw_dir.rglob("*"):
        if d.is_dir():
            children = [c for c in d.iterdir() if c.is_dir()]
            if len(children) >= 30:
                return d

    return None


def _organise_splits(raw_dir: Path, processed: Path) -> None:
    """Create train/val splits from the color/ ImageFolder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    random.seed(42)

    source = _find_color_root(raw_dir)
    if source is None:
        print("[WARN] Could not locate 'color/' directory; scanning raw_dir")
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

        class_name = class_dir.name

        images = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        )
        if not images:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        dst_train = train_dir / class_name
        dst_train.mkdir(parents=True, exist_ok=True)
        for f in train_imgs:
            dst = dst_train / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                total_train += 1

        dst_val = val_dir / class_name
        dst_val.mkdir(parents=True, exist_ok=True)
        for f in val_imgs:
            dst = dst_val / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                total_val += 1

    print(f"[INFO] Organised {total_train} train + {total_val} val images")


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
    path = ensure_plant_dataset(force=force)
    print(f"Dataset ready at: {path}")
