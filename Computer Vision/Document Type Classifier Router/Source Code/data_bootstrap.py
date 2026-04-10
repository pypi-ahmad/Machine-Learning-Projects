"""Document Type Classifier Router — idempotent dataset bootstrap.

Downloads the Real World Documents Collections dataset from Kaggle
and prepares it for training with an 80/20 train/val split.

Dataset: shaz13/real-world-documents-collections
- ~5,000 images across 16 document types
- ImageFolder format: docs-sm/<class>/*.jpg
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "document_type_classifier_router"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_document_dataset(force: bool = False) -> Path:
    """Download and prepare the document dataset (idempotent).

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

    if processed.exists() and force:
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)

    _organise_splits(raw_dir, processed)
    _write_info(raw_dir, processed)
    ready_marker.touch()

    return raw_dir


def _find_class_root(raw_dir: Path) -> Path | None:
    """Locate the directory containing 16 document-type sub-folders.

    The dataset ships as ``docs-sm/<class>/`` at some nesting level.
    """
    # Direct child
    for d in raw_dir.iterdir():
        if d.is_dir() and d.name.lower() in ("docs-sm", "docs_sm"):
            children = [c for c in d.iterdir() if c.is_dir()]
            if len(children) >= 10:
                return d

    # Look for any directory with 10+ sub-dirs matching known names
    known = {"invoice", "letter", "form", "email", "resume", "budget", "memo"}
    for d in raw_dir.rglob("*"):
        if d.is_dir():
            child_names = {c.name.lower() for c in d.iterdir() if c.is_dir()}
            if len(child_names & known) >= 4:
                return d

    # Fallback: directory with most sub-directories
    best, best_count = None, 0
    for d in raw_dir.rglob("*"):
        if d.is_dir():
            n = sum(1 for c in d.iterdir() if c.is_dir())
            if n > best_count:
                best, best_count = d, n
    return best


def _organise_splits(raw_dir: Path, processed: Path) -> None:
    """Create train/val splits from the ImageFolder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    random.seed(42)

    source = _find_class_root(raw_dir)
    if source is None:
        print("[WARN] Could not locate class folders; using raw_dir")
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
    path = ensure_document_dataset(force=force)
    print(f"Dataset ready at: {path}")
