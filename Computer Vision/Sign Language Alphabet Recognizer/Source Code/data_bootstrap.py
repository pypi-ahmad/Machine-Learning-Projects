"""Sign Language Alphabet Recognizer — idempotent dataset bootstrap.

Expected dataset structure after download (HuggingFace):
  <raw>/<split>/<letter>/<image>.jpg

The bootstrap copies images into:
  <raw>/processed/by_letter/<A|B|...>/<image>.jpg
for easy consumption by the trainer.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_KEY = "sign_language_alphabet_recognizer"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_sign_lang_dataset(force: bool = False) -> Path:
    """Download and prepare the ASL alphabet dataset (idempotent).

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

    by_letter = processed / "by_letter"
    if by_letter.exists() and force:
        shutil.rmtree(by_letter)
    by_letter.mkdir(parents=True, exist_ok=True)

    _organise_by_letter(raw_dir, by_letter)
    _write_info(raw_dir, by_letter)
    ready_marker.touch()

    return raw_dir


def _organise_by_letter(raw_dir: Path, by_letter: Path) -> None:
    """Walk raw download and copy images into per-letter directories.

    Handles common dataset layouts:
    - <raw>/<split>/<LETTER>/images...
    - <raw>/<LETTER>/images...
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counters: dict[str, int] = {}

    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() not in exts:
            continue
        if "processed" in f.parts:
            continue

        # Try to infer label from nearest parent directory name
        label = _infer_label(f)
        if label is None:
            continue

        counters.setdefault(label, 0)
        dst_dir = by_letter / label
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / f"{counters[label]:05d}{f.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(f, dst)
        counters[label] += 1


def _infer_label(path: Path) -> str | None:
    """Infer the letter label from *path*'s parent directory."""
    from config import ASL_STATIC_LABELS

    valid = set(ASL_STATIC_LABELS)
    # Walk up from parent to grandparent
    for parent in (path.parent, path.parent.parent):
        name = parent.name.upper()
        if name in valid:
            return name
    return None


def _write_info(raw_dir: Path, by_letter: Path) -> None:
    counts = {}
    for d in sorted(by_letter.iterdir()):
        if d.is_dir():
            counts[d.name] = len(list(d.iterdir()))
    info = {
        "project": PROJECT_KEY,
        "source_dir": str(raw_dir),
        "classes": len(counts),
        "per_class": counts,
        "total_images": sum(counts.values()),
    }
    info_path = raw_dir / "processed" / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    path = ensure_sign_lang_dataset(force=force)
    print(f"Dataset ready at: {path}")
