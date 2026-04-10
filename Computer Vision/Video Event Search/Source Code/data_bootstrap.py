"""Dataset bootstrap for Video Event Search.

Downloads the Pedestrian Dataset from Kaggle (crosswalk / night /
fourway video scenes with bounding-box CSVs) and prepares the repo's
``data/raw`` / ``data/processed`` layout.

Usage::

    from data_bootstrap import ensure_video_event_dataset

    data_root = ensure_video_event_dataset()            # idempotent
    data_root = ensure_video_event_dataset(force=True)   # force re-download
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("video_event_search.data_bootstrap")

PROJECT_KEY = "video_event_search"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_video_event_dataset(*, force: bool = False) -> Path:
    """Download and prepare the pedestrian video dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset`` for the
       Kaggle download.
    2. Organises ``.avi`` / ``.csv`` files into ``data/raw/``.
    3. Writes ``data/dataset_info.json`` with provenance metadata.
    4. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project's data root.
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared at %s — skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    from scripts.download_data import ensure_dataset as _ensure
    data_path = _ensure(PROJECT_KEY, force=force)

    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Move video + CSV files into raw/
    for ext in ("*.avi", "*.csv", "*.mp4"):
        for f in data_path.rglob(ext):
            if f.parent != raw_dir:
                dest = raw_dir / f.name
                shutil.move(str(f), str(dest))
                log.info("Moved %s → %s", f.name, dest)

    # Write dataset_info.json
    info = {
        "project": PROJECT_KEY,
        "source": "kaggle:smeschke/pedestrian-dataset",
        "license": "CC0: Public Domain",
        "description": "Pedestrian crosswalk video scenes with bounding-box annotations",
        "scenes": ["crosswalk", "night", "fourway"],
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = data_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Ready marker
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_video_event_dataset(force="--force" in sys.argv)
