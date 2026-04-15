"""Dataset bootstrap for Parking Occupancy Monitor.

Downloads and prepares a parking/vehicle detection dataset in YOLO
format.  Uses the repo-level ``DatasetResolver`` when available
(Roboflow source), and falls back to generating a synthetic demo
dataset so the project always works out of the box.

Usage::

    from data_bootstrap import ensure_parking_dataset

    data_root = ensure_parking_dataset()            # idempotent
    data_root = ensure_parking_dataset(force=True)   # force re-download
"""

from __future__ import annotations

import json
import logging
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

log = logging.getLogger("parking_occupancy.data_bootstrap")

PROJECT_KEY = "parking_occupancy_monitor"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

_DEMO_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]


def ensure_parking_dataset(*, force: bool = False) -> Path:
    """Download and prepare the parking / vehicle detection dataset.

    Strategy:
    1. If already prepared and *force* is False, return immediately.
    2. Try the repo-level DatasetResolver (Roboflow download).
    3. On any failure, fall back to generating a synthetic demo dataset.
    4. Organise into ``data/raw/`` and ``data/processed/``.
    5. Write ``data/dataset_info.json``.

    Returns
    -------
    Path
        ``data/parking_occupancy_monitor/``
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared -- skipping", PROJECT_KEY)
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
        log.warning("[%s] DatasetResolver failed (%s) -- falling back to synthetic demo", PROJECT_KEY, exc)
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
    """Create a synthetic parking-lot vehicle dataset in YOLO format."""
    random.seed(42)
    np.random.seed(42)

    splits = {"train": 60, "valid": 15, "test": 15}

    for split_name, n_images in splits.items():
        img_dir = raw_dir / split_name / "images"
        lbl_dir = raw_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_images):
            img, labels = _make_parking_image()
            fname = f"parking_{split_name}_{idx:04d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)
            with open(lbl_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
                for lbl in labels:
                    f.write(lbl + "\n")

    data_cfg = {
        "path": str(raw_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(_DEMO_CLASSES),
        "names": _DEMO_CLASSES,
    }
    (raw_dir / "data.yaml").write_text(
        yaml.dump(data_cfg, default_flow_style=False), encoding="utf-8"
    )


def _make_parking_image() -> tuple[np.ndarray, list[str]]:
    """Render a synthetic parking lot overhead image."""
    W, H = 640, 480
    # Asphalt background
    img = np.full((H, W, 3), (90, 90, 90), dtype=np.uint8)

    # Draw parking lines (2 rows of 4 slots)
    slot_w, slot_h = 90, 130
    labels: list[str] = []

    for row_idx, y0 in enumerate([60, 250]):
        for col in range(4):
            x0 = 60 + col * 120
            # Slot lines (white)
            cv2.rectangle(img, (x0, y0), (x0 + slot_w, y0 + slot_h), (200, 200, 200), 1)

            # Randomly place a vehicle in slot
            if random.random() < 0.6:
                # Vehicle rectangle slightly smaller than slot
                vx1 = x0 + random.randint(5, 15)
                vy1 = y0 + random.randint(5, 15)
                vx2 = x0 + slot_w - random.randint(5, 15)
                vy2 = y0 + slot_h - random.randint(5, 15)
                # Car colour
                color = (
                    random.randint(40, 220),
                    random.randint(40, 220),
                    random.randint(40, 220),
                )
                cv2.rectangle(img, (vx1, vy1), (vx2, vy2), color, -1)
                cv2.rectangle(img, (vx1, vy1), (vx2, vy2), (30, 30, 30), 1)
                # Mostly cars, sometimes trucks/motorcycles
                cls_id = random.choices([0, 1, 3], weights=[0.7, 0.2, 0.1])[0]
                cx = ((vx1 + vx2) / 2) / W
                cy = ((vy1 + vy2) / 2) / H
                w = (vx2 - vx1) / W
                h = (vy2 - vy1) / H
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Road area
    cv2.rectangle(img, (0, 200), (W, 240), (70, 70, 70), -1)
    # Dashed center line
    for x in range(0, W, 40):
        cv2.line(img, (x, 220), (x + 20, 220), (220, 220, 220), 1)

    # Noise
    noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img, labels


# ---------------------------------------------------------------------------
# Helpers
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


def _prepare_processed(raw_dir: Path, processed_dir: Path) -> None:
    src_yaml = raw_dir / "data.yaml"
    if not src_yaml.exists():
        return

    try:
        cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(str(src_yaml), str(processed_dir / "data.yaml"))
        return

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
    info_path = data_path / "dataset_info.json"
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": "synthetic_demo" if synthetic else "roboflow",
        "description": (
            "Synthetic parking lot vehicle dataset (overhead view)"
            if synthetic
            else "Parking lot vehicle detection -- Roboflow Universe"
        ),
        "format": "yolov8",
        "synthetic": synthetic,
        "classes": _DEMO_CLASSES,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_path),
        "raw_dir": str(data_path / "raw"),
        "processed_dir": str(data_path / "processed"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
