"""Wildlife Species Retrieval — idempotent dataset bootstrap.

Downloads the Animal Image Dataset (90 Different Animals)
from Kaggle and prepares it for indexing / training.

Dataset: iamsouravbanerjee/animal-image-dataset-90-different-animals
- 5,400 images across 90 species
- ImageFolder format: animals/<species>/*.jpg
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "wildlife_species_retrieval"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_wildlife_dataset(force: bool = False) -> Path:
    """Download and prepare the wildlife dataset (idempotent).

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


def _find_species_root(raw_dir: Path) -> Path | None:
    """Locate the directory containing 90 species sub-folders.

    The dataset ships as ``animals/animals/<species>/`` or
    ``animals/<species>/``.
    """
    # Direct children
    for d in raw_dir.iterdir():
        if d.is_dir() and d.name.lower() == "animals":
            children = [c for c in d.iterdir() if c.is_dir()]
            if len(children) >= 50:
                return d
            # One more level: animals/animals/
            for sub in d.iterdir():
                if sub.is_dir() and sub.name.lower() == "animals":
                    grandchildren = [c for c in sub.iterdir() if c.is_dir()]
                    if len(grandchildren) >= 50:
                        return sub

    # Recursive fallback
    for d in raw_dir.rglob("*"):
        if d.is_dir():
            children = [c for c in d.iterdir() if c.is_dir()]
            if len(children) >= 80:
                return d

    return None


def _organise_splits(raw_dir: Path, processed: Path) -> None:
    """Create train/val splits from the species ImageFolder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    random.seed(42)

    source = _find_species_root(raw_dir)
    if source is None:
        print("[WARN] Could not locate species folders; using raw_dir")
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

        species_name = class_dir.name

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

        dst_train = train_dir / species_name
        dst_train.mkdir(parents=True, exist_ok=True)
        for f in train_imgs:
            dst = dst_train / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                total_train += 1

        dst_val = val_dir / species_name
        dst_val.mkdir(parents=True, exist_ok=True)
        for f in val_imgs:
            dst = dst_val / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                total_val += 1

    print(f"[INFO] Organised {total_train} train + {total_val} val images")


def _write_info(raw_dir: Path, processed: Path) -> None:
    """Write dataset summary JSON."""
    species_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}

    for split in ("train", "val"):
        split_dir = processed / split
        if not split_dir.exists():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir():
                count = len([f for f in d.iterdir() if f.is_file()])
                species_counts[split][d.name] = count

    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "train_species": len(species_counts["train"]),
        "val_species": len(species_counts["val"]),
        "train_images": sum(species_counts["train"].values()),
        "val_images": sum(species_counts["val"].values()),
        "species_counts": species_counts,
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_wildlife_dataset(force=force)
    print(f"Dataset ready at: {path}")
