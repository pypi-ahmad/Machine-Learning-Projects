"""Building Footprint Change Detector — idempotent dataset bootstrap."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "building_footprint_change_detector"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_change_dataset(force: bool = False) -> Path:
    """Download and prepare the change-detection dataset (idempotent).

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

    # Set up processed directory structure
    for sub in ("A", "B", "label"):
        (processed / sub).mkdir(parents=True, exist_ok=True)

    _organise_pairs(raw_dir, processed)
    _write_info(raw_dir, processed)
    ready_marker.touch()

    return raw_dir


def _organise_pairs(raw_dir: Path, processed: Path) -> None:
    """Locate before (A), after (B), and label directories and copy images.

    LEVIR-CD layout:  train/A/, train/B/, train/label/
                      test/A/,  test/B/,  test/label/
    Other datasets may have A/, B/, label/ at root.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    # Try to find A/B/label directories at any depth
    a_dirs: list[Path] = []
    b_dirs: list[Path] = []
    lbl_dirs: list[Path] = []

    for d in sorted(raw_dir.rglob("*")):
        if not d.is_dir() or "processed" in d.parts:
            continue
        name = d.name.lower()
        if name == "a":
            a_dirs.append(d)
        elif name == "b":
            b_dirs.append(d)
        elif name in {"label", "labels", "mask", "masks"}:
            lbl_dirs.append(d)

    if not a_dirs or not b_dirs:
        # Fallback: just copy all images into A for evaluation
        print("[data_bootstrap] No A/B structure found; copying all images")
        _copy_all_images(raw_dir, processed / "A", exts)
        return

    idx = 0
    for a_dir, b_dir in zip(a_dirs, b_dirs):
        a_files = sorted(f for f in a_dir.iterdir() if f.suffix.lower() in exts)
        b_files = {f.name: f for f in b_dir.iterdir() if f.suffix.lower() in exts}

        for af in a_files:
            bf = b_files.get(af.name)
            if bf is None:
                continue
            dst_name = f"{idx:05d}{af.suffix.lower()}"
            shutil.copy2(af, processed / "A" / dst_name)
            shutil.copy2(bf, processed / "B" / dst_name)
            idx += 1

    # Copy label masks if present
    if lbl_dirs:
        idx = 0
        for lbl_dir in lbl_dirs:
            lbl_files = sorted(f for f in lbl_dir.iterdir() if f.suffix.lower() in exts)
            for lf in lbl_files:
                dst_name = f"{idx:05d}{lf.suffix.lower()}"
                dst = processed / "label" / dst_name
                if not dst.exists():
                    shutil.copy2(lf, dst)
                idx += 1

    print(f"[data_bootstrap] Organised {idx} image pair(s)")


def _copy_all_images(raw_dir: Path, dest: Path, exts: set[str]) -> None:
    """Fallback: copy all images into a single directory."""
    idx = 0
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() in exts and "processed" not in f.parts:
            dst = dest / f"{idx:05d}{f.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(f, dst)
            idx += 1


def _write_info(raw_dir: Path, processed: Path) -> None:
    a_count = len(list((processed / "A").iterdir()))
    b_count = len(list((processed / "B").iterdir()))
    lbl_count = len(list((processed / "label").iterdir()))
    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "before_count": a_count,
        "after_count": b_count,
        "label_count": lbl_count,
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_change_dataset(force=force)
    print(f"Dataset ready at: {path}")
