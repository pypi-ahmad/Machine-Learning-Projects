"""Dataset bootstrap for Drone Ship OBB Detector.

Downloads and prepares a public OBB-format aerial dataset.
Uses the DOTA-ship subset from Roboflow (OBB-labeled, YOLO-OBB format).

Usage::

    from data_bootstrap import ensure_obb_dataset

    data_root = ensure_obb_dataset()            # idempotent
    data_root = ensure_obb_dataset(force=True)  # force re-download
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

log = logging.getLogger("drone_ship_obb.data_bootstrap")

PROJECT_KEY = "drone_ship_obb_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_obb_dataset(*, force: bool = False) -> Path:
    """Download and prepare the OBB aerial ship dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset`` for the
       actual download (Roboflow / URL).
    2. Organises into ``data/raw/`` and ``data/processed/``.
    3. Validates that labels contain OBB-format annotations.
    4. Writes ``data/dataset_info.json`` with provenance metadata.
    5. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project data root (``data/drone_ship_obb_detector/``).
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
    data_yaml = _find_data_yaml(data_path)

    if data_yaml is not None:
        _organise_raw(data_path, raw_dir, data_yaml)
        _prepare_processed(raw_dir, processed_dir, data_yaml)
        _validate_obb_labels(raw_dir)
    else:
        log.warning("[%s] No data.yaml found — dataset may need manual preparation", PROJECT_KEY)

    # Step 3: write dataset_info.json
    _write_info(data_path)

    # Step 4: stamp ready marker
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] OBB dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_data_yaml(root: Path) -> Path | None:
    if (root / "data.yaml").exists():
        return root / "data.yaml"
    for child in root.iterdir():
        if child.is_dir() and (child / "data.yaml").exists():
            return child / "data.yaml"
    for candidate in root.rglob("data.yaml"):
        return candidate
    return None


def _organise_raw(data_path: Path, raw_dir: Path, data_yaml: Path) -> None:
    yaml_parent = data_yaml.parent
    for split_name in ("train", "valid", "val", "test"):
        src = yaml_parent / split_name
        if src.exists() and src != raw_dir / split_name:
            dst = raw_dir / split_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))
    shutil.copy2(str(data_yaml), str(raw_dir / "data.yaml"))


def _prepare_processed(raw_dir: Path, processed_dir: Path, data_yaml: Path) -> None:
    import yaml as _y

    try:
        cfg = _y.safe_load((raw_dir / "data.yaml").read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(str(raw_dir / "data.yaml"), str(processed_dir / "data.yaml"))
        return

    for key in ("train", "val", "test"):
        if key in cfg:
            split_dir = raw_dir / ("valid" if key == "val" and (raw_dir / "valid").exists() else key)
            if split_dir.exists():
                cfg[key] = str(split_dir / "images")

    out_yaml = processed_dir / "data.yaml"
    out_yaml.write_text(_y.dump(cfg, default_flow_style=False), encoding="utf-8")
    shutil.copy2(str(out_yaml), str(DATA_ROOT / "data.yaml"))


def _validate_obb_labels(raw_dir: Path) -> None:
    """Spot-check that label files contain OBB-format annotations.

    OBB labels have 9 values per line: class_id x1 y1 x2 y2 x3 y3 x4 y4
    (all normalised) — 4 corner points instead of cx cy w h.
    """
    label_dirs = list(raw_dir.rglob("labels"))
    checked = 0
    for ldir in label_dirs:
        for txt in ldir.glob("*.txt"):
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[:5]:
                parts = line.strip().split()
                if len(parts) == 9:
                    checked += 1
                elif len(parts) == 5:
                    log.warning(
                        "[%s] Label %s appears to use HBB format (5 values) "
                        "instead of OBB (9 values). Training may fail.",
                        PROJECT_KEY, txt.name,
                    )
                break
            if checked > 0:
                break
        if checked > 0:
            break

    if checked > 0:
        log.info("[%s] OBB label format validated (%d files checked)", PROJECT_KEY, checked)
    else:
        log.warning("[%s] Could not validate OBB label format — no label files found", PROJECT_KEY)


def _write_info(data_path: Path) -> None:
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
        "label_format": "yolo-obb (4 corner points, normalised)",
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
