"""Interactive Video Object Cutout Studio — idempotent dataset bootstrap.

Downloads DAVIS 2017 trainval (480p) for benchmarking / demo purposes.
The project also works on arbitrary local media.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "interactive_video_object_cutout_studio"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_davis_dataset(force: bool = False) -> Path:
    """Download and prepare the DAVIS 2017 benchmark dataset (idempotent).

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

    processed.mkdir(parents=True, exist_ok=True)

    # Locate DAVIS structure inside the download
    davis_root = _find_davis_root(raw_dir)
    if davis_root:
        _write_info(raw_dir, davis_root, processed)
    else:
        print("[WARN] DAVIS directory structure not found — raw files available.")
        _write_info(raw_dir, raw_dir, processed)

    ready_marker.touch()
    return raw_dir


def _find_davis_root(base: Path) -> Path | None:
    """Walk `base` looking for the DAVIS JPEGImages directory."""
    for d in base.rglob("JPEGImages"):
        if d.is_dir():
            return d.parent
    return None


def _write_info(raw_dir: Path, davis_root: Path, processed: Path) -> None:
    sequences = []
    jpeg_dir = davis_root / "JPEGImages" / "480p"
    if jpeg_dir.exists():
        sequences = sorted(d.name for d in jpeg_dir.iterdir() if d.is_dir())

    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "davis_root": str(davis_root),
        "resolution": "480p",
        "sequences": sequences,
        "num_sequences": len(sequences),
    }
    info_path = processed / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def get_sequence_frames(davis_root: Path, sequence: str) -> list[Path]:
    """Return sorted frame paths for a DAVIS sequence."""
    seq_dir = davis_root / "JPEGImages" / "480p" / sequence
    if not seq_dir.exists():
        return []
    return sorted(seq_dir.glob("*.jpg"))


def get_sequence_annotations(davis_root: Path, sequence: str) -> list[Path]:
    """Return sorted annotation mask paths for a DAVIS sequence."""
    seq_dir = davis_root / "Annotations" / "480p" / sequence
    if not seq_dir.exists():
        return []
    return sorted(seq_dir.glob("*.png"))


def list_sequences(davis_root: Path) -> list[str]:
    """List available DAVIS sequences."""
    jpeg_dir = davis_root / "JPEGImages" / "480p"
    if not jpeg_dir.exists():
        return []
    return sorted(d.name for d in jpeg_dir.iterdir() if d.is_dir())


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_davis_dataset(force=force)
    print(f"Dataset ready at: {path}")
