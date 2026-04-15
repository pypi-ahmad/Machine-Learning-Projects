"""Dataset bootstrap for Retail Shelf Stockout Detector.

Downloads and prepares a retail shelf product detection dataset in
YOLO format.  Uses the repo-level ``DatasetResolver`` when available
(Roboflow source), and falls back to generating a synthetic demo
dataset so the project always works out of the box.

Usage::

    from data_bootstrap import ensure_retail_dataset

    data_root = ensure_retail_dataset()            # idempotent
    data_root = ensure_retail_dataset(force=True)   # force re-download
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("retail_shelf.data_bootstrap")

PROJECT_KEY = "retail_shelf_stockout"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

# Class names for the synthetic demo dataset
_DEMO_CLASSES = ["product", "bottle", "can", "box", "bag"]


def ensure_retail_dataset(*, force: bool = False) -> Path:
    """Download and prepare the retail shelf detection dataset.

    Strategy:
    1. If already prepared and *force* is False, return immediately.
    2. Try the repo-level DatasetResolver (Roboflow download).
    3. On any failure (missing SDK, network, credentials), fall back to
       generating a synthetic demo dataset with dummy shelf images.
    4. Organise into ``data/raw/`` and ``data/processed/``.
    5. Write ``data/dataset_info.json`` with provenance metadata.

    Returns
    -------
    Path
        The project's data root (``data/retail_shelf_stockout/``).
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared at %s — skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_ROOT / "raw"
    processed_dir = DATA_ROOT / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Try shared infrastructure first ----------
    downloaded = False
    try:
        from scripts.download_data import ensure_dataset as _ensure
        data_path = _ensure(PROJECT_KEY, force=force)
        data_yaml = _find_data_yaml(data_path)
        if data_yaml is not None:
            _organise_raw(data_path, raw_dir, data_yaml)
            _prepare_processed(raw_dir, processed_dir)
            downloaded = True
            print(f"[INFO] Dataset downloaded via DatasetResolver -> {DATA_ROOT}")
    except Exception as exc:
        log.warning("[%s] DatasetResolver failed (%s) — falling back to synthetic demo", PROJECT_KEY, exc)
        print(f"[WARN] Could not download real dataset ({exc})")
        print("[INFO] Generating synthetic demo dataset instead...")

    # ---------- Fallback: synthetic demo ----------
    if not downloaded:
        _generate_synthetic_dataset(raw_dir)
        _prepare_processed(raw_dir, processed_dir)
        print(f"[INFO] Synthetic demo dataset created at {DATA_ROOT}")

    # ---------- Metadata ----------
    _write_info(DATA_ROOT, synthetic=not downloaded)

    # ---------- Ready marker ----------
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(raw_dir: Path) -> None:
    """Create a small synthetic shelf detection dataset in YOLO format.

    Generates shelf-like images with coloured rectangles as 'products'
    and writes YOLO-format label files.  Good enough for training smoke
    tests and demonstrating the full pipeline.
    """
    random.seed(42)
    np.random.seed(42)

    splits = {"train": 60, "valid": 15, "test": 15}

    for split_name, n_images in splits.items():
        img_dir = raw_dir / split_name / "images"
        lbl_dir = raw_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_images):
            img, labels = _make_shelf_image(idx)
            fname = f"shelf_{split_name}_{idx:04d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)

            with open(lbl_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
                for lbl in labels:
                    f.write(lbl + "\n")

    # Write data.yaml
    data_cfg = {
        "path": str(raw_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(_DEMO_CLASSES),
        "names": _DEMO_CLASSES,
    }
    (raw_dir / "data.yaml").write_text(yaml.dump(data_cfg, default_flow_style=False), encoding="utf-8")


def _make_shelf_image(seed_offset: int) -> tuple[np.ndarray, list[str]]:
    """Render a single synthetic shelf image with product-like rectangles."""
    W, H = 640, 480
    img = np.full((H, W, 3), (230, 225, 220), dtype=np.uint8)  # light grey shelf bg

    # Draw 2-3 shelf lines
    n_shelves = random.randint(2, 3)
    shelf_ys = sorted(random.sample(range(100, H - 60, 50), n_shelves))
    for sy in shelf_ys:
        cv2.line(img, (20, sy), (W - 20, sy), (120, 100, 80), 3)

    # Place products on shelves
    labels: list[str] = []
    for sy in shelf_ys:
        n_products = random.randint(3, 8)
        x_cursor = random.randint(30, 60)
        for _ in range(n_products):
            pw = random.randint(30, 60)
            ph = random.randint(40, 80)
            x1 = x_cursor
            y1 = sy - ph - random.randint(2, 8)
            x2 = x1 + pw
            y2 = sy - random.randint(0, 4)

            if x2 >= W - 20:
                break

            # Random product colour
            color = (
                random.randint(40, 220),
                random.randint(40, 220),
                random.randint(40, 220),
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # YOLO label: class_id cx cy w h (normalised)
            cls_id = random.randint(0, len(_DEMO_CLASSES) - 1)
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            nw = pw / W
            nh = (y2 - y1) / H
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            x_cursor = x2 + random.randint(4, 15)

    # Add mild noise
    noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img, labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_yaml(root: Path) -> Path | None:
    """Search for data.yaml in the download directory tree."""
    if (root / "data.yaml").exists():
        return root / "data.yaml"
    for child in root.iterdir():
        if child.is_dir() and (child / "data.yaml").exists():
            return child / "data.yaml"
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
    shutil.copy2(str(data_yaml), str(raw_dir / "data.yaml"))


def _prepare_processed(raw_dir: Path, processed_dir: Path) -> None:
    """Create processed/ data.yaml with absolute paths to raw splits."""
    src_yaml = raw_dir / "data.yaml"
    if not src_yaml.exists():
        return

    try:
        cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(str(src_yaml), str(processed_dir / "data.yaml"))
        return

    # Rewrite paths to absolute
    for key in ("train", "val", "test"):
        if key in cfg:
            split_name = key
            if key == "val" and (raw_dir / "valid").exists():
                split_name = "valid"
            split_dir = raw_dir / split_name / "images"
            if split_dir.exists():
                cfg[key] = str(split_dir)

    cfg["path"] = str(raw_dir)

    out_yaml = processed_dir / "data.yaml"
    out_yaml.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    shutil.copy2(str(out_yaml), str(DATA_ROOT / "data.yaml"))


def _write_info(data_path: Path, *, synthetic: bool = False) -> None:
    """Write dataset_info.json provenance metadata."""
    info_path = data_path / "dataset_info.json"

    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": "synthetic_demo" if synthetic else "roboflow",
        "description": (
            "Synthetic demo shelf product dataset (coloured rectangles)"
            if synthetic
            else "Retail shelf product detection — Roboflow Universe"
        ),
        "format": "yolov8",
        "synthetic": synthetic,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_path),
        "raw_dir": str(data_path / "raw"),
        "processed_dir": str(data_path / "processed"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("[%s] Wrote dataset_info.json", PROJECT_KEY)
