"""Dataset bootstrap for Document Layout Block Detector.

Downloads and prepares a public document-layout detection dataset
(DocLayNet subset from Roboflow) with YOLO-format labels.

Usage::

    from data_bootstrap import ensure_layout_dataset

    data_root = ensure_layout_dataset()            # idempotent
    data_root = ensure_layout_dataset(force=True)  # force re-download
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

log = logging.getLogger("doc_layout.data_bootstrap")

PROJECT_KEY = "document_layout_block_detector"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_layout_dataset(*, force: bool = False) -> Path:
    """Download and prepare the document layout detection dataset.

    1. Delegates to ``scripts/download_data.py:ensure_dataset``.
    2. Organises into ``data/raw/`` and ``data/processed/``.
    3. Writes ``data/dataset_info.json`` with provenance metadata.
    4. Idempotent — skips if ``.ready`` marker exists unless *force*.

    Returns
    -------
    Path
        The project data root (``data/document_layout_block_detector/``).
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

    data_yaml = _find_data_yaml(data_path)

    if data_yaml is not None:
        _organise_raw(data_path, raw_dir, data_yaml)
        _prepare_processed(raw_dir, processed_dir, data_yaml)
    else:
        log.warning("[%s] No data.yaml found — dataset may need manual preparation", PROJECT_KEY)

    _write_info(data_path)

    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
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
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
