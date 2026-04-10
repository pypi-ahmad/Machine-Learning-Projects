"""Dataset bootstrap for Retail Shelf Stockout Detector.

Wraps the repo-level ``scripts/download_data.py:ensure_dataset`` and adds
project-specific preparation: converting the Roboflow export into the
repo's preferred ``data/raw`` / ``data/processed`` layout with a
YOLO-format ``data.yaml``.

Usage::

    from data_bootstrap import ensure_retail_dataset

    data_root = ensure_retail_dataset()            # idempotent
    data_root = ensure_retail_dataset(force=True)   # force re-download
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

# Repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("retail_shelf.data_bootstrap")

PROJECT_KEY = "retail_shelf_stockout"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_retail_dataset(*, force: bool = False) -> Path:
    """Download and prepare the retail shelf detection dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset`` for the
       actual download (Roboflow / HF / URL).
    2. Organises into ``data/raw/`` and ``data/processed/``.
    3. Writes ``data/dataset_info.json`` with provenance metadata.
    4. Idempotent — skips everything if ``.ready`` marker exists unless
       *force* is ``True``.

    Returns
    -------
    Path
        The project's data root (``data/retail_shelf_stockout/``).
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared at %s — skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    # Step 1: download via shared infrastructure
    from scripts.download_data import ensure_dataset as _ensure
    data_path = _ensure(PROJECT_KEY, force=force)

    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: locate the YOLO data.yaml from the download
    # Roboflow exports usually produce: train/, valid/, test/, data.yaml
    data_yaml = _find_data_yaml(data_path)

    if data_yaml is not None:
        # Move raw download artefacts into raw/
        _organise_raw(data_path, raw_dir, data_yaml)

        # Create processed/ with symlinks or copies pointing to proper structure
        _prepare_processed(raw_dir, processed_dir, data_yaml)
    else:
        log.warning("[%s] No data.yaml found — dataset may need manual preparation", PROJECT_KEY)

    # Step 3: write dataset_info.json
    _write_info(data_path)

    # Step 4: stamp ready marker
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _find_data_yaml(root: Path) -> Path | None:
    """Search for data.yaml in the download directory tree."""
    # Direct
    if (root / "data.yaml").exists():
        return root / "data.yaml"
    # One level deep (common with Roboflow zip extracts)
    for child in root.iterdir():
        if child.is_dir() and (child / "data.yaml").exists():
            return child / "data.yaml"
    # Recursive fallback
    for candidate in root.rglob("data.yaml"):
        return candidate
    return None


def _organise_raw(data_path: Path, raw_dir: Path, data_yaml: Path) -> None:
    """Move downloaded split directories into raw/."""
    yaml_parent = data_yaml.parent
    for split_name in ("train", "valid", "val", "test"):
        src = yaml_parent / split_name
        if src.exists() and src != raw_dir / split_name:
            dst = raw_dir / split_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))

    # Copy data.yaml into raw/
    shutil.copy2(str(data_yaml), str(raw_dir / "data.yaml"))


def _prepare_processed(raw_dir: Path, processed_dir: Path, data_yaml: Path) -> None:
    """Create processed/ that mirrors train/val/test splits + data.yaml."""
    import yaml as _y

    # Read and fix paths in data.yaml to point to processed/
    try:
        cfg = _y.safe_load((raw_dir / "data.yaml").read_text(encoding="utf-8"))
    except Exception:
        # Fallback — just copy as-is
        shutil.copy2(str(raw_dir / "data.yaml"), str(processed_dir / "data.yaml"))
        return

    # Rewrite paths relative to processed/
    for key in ("train", "val", "test"):
        if key in cfg:
            # Point to the raw split directories (they contain the actual images)
            split_dir = raw_dir / ("valid" if key == "val" and (raw_dir / "valid").exists() else key)
            if split_dir.exists():
                cfg[key] = str(split_dir / "images")

    out_yaml = processed_dir / "data.yaml"
    out_yaml.write_text(_y.dump(cfg, default_flow_style=False), encoding="utf-8")

    # Also copy data.yaml to the project data root for easy access
    shutil.copy2(str(out_yaml), str(DATA_ROOT / "data.yaml"))


def _write_info(data_path: Path) -> None:
    """Write dataset_info.json provenance metadata."""
    info_path = data_path / "dataset_info.json"
    if info_path.exists():
        return

    from utils.datasets import DatasetResolver

    resolver = DatasetResolver()
    entry = resolver.registry.get(PROJECT_KEY, {})

    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": entry.get("type", "unknown"),
        "description": entry.get("description", ""),
        "source_workspace": entry.get("workspace", ""),
        "source_project": entry.get("project", ""),
        "source_version": entry.get("version", ""),
        "format": entry.get("format", "yolov8"),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_path),
        "raw_dir": str(data_path / "raw"),
        "processed_dir": str(data_path / "processed"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("[%s] Wrote dataset_info.json", PROJECT_KEY)
