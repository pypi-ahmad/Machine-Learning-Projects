"""Industrial Scratch / Crack Segmentation -- idempotent dataset bootstrap."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "industrial_scratch_crack_segmentation"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_defect_dataset(force: bool = False) -> Path:
    """Download and prepare the surface defect dataset (idempotent).
    """Download and prepare the surface defect dataset (idempotent).

    Parameters
    ----------
    force : bool
        Delete existing data and re-download.

    Returns
    -------
    Path
        Root of the prepared dataset directory.
    """
    """
    from scripts.download_data import ensure_dataset

    raw_dir = ensure_dataset(PROJECT_KEY, force=force)

    processed = raw_dir / "processed"
    ready_marker = processed / ".ready"

    if ready_marker.exists() and not force:
        return raw_dir

    media_dir = processed / "media"
    if media_dir.exists() and force:
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    _collect_media(raw_dir, media_dir)
    _write_info(raw_dir, media_dir)
    ready_marker.touch()

    return raw_dir


def _collect_media(raw_dir: Path, media_dir: Path) -> None:
    """Copy defect images from raw download into processed/media/."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    idx = 0
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() in exts and "processed" not in f.parts:
            dst = media_dir / f"{idx:05d}{f.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(f, dst)
            idx += 1


def _write_info(raw_dir: Path, media_dir: Path) -> None:
    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "media_count": len(list(media_dir.iterdir())),
    }
    info_path = raw_dir / "processed" / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_defect_dataset(force=force)
    print(f"Dataset ready at: {path}")
