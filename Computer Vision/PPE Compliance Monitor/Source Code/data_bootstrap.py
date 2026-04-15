"""Dataset bootstrap for PPE Compliance Monitor.

Downloads and prepares a PPE detection dataset in YOLO format.
Uses the repo-level ``DatasetResolver`` when available (Roboflow source),
and falls back to generating a synthetic demo dataset so the project
always works out of the box.

Usage::

    from data_bootstrap import ensure_ppe_dataset

    data_root = ensure_ppe_dataset()            # idempotent
    data_root = ensure_ppe_dataset(force=True)   # force re-download
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

log = logging.getLogger("ppe_compliance.data_bootstrap")

PROJECT_KEY = "ppe_compliance_monitor"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

# Synthetic dataset class names — matches the real dataset
_DEMO_CLASSES = ["person", "helmet", "safety_vest", "gloves", "goggles", "boots"]


def ensure_ppe_dataset(*, force: bool = False) -> Path:
    """Download and prepare the PPE detection dataset.

    Strategy:
    1. If already prepared and *force* is False, return immediately.
    2. Try the repo-level DatasetResolver (Roboflow download).
    3. On any failure, fall back to generating a synthetic demo dataset.
    4. Organise into ``data/raw/`` and ``data/processed/``.
    5. Write ``data/dataset_info.json``.

    Returns
    -------
    Path
        ``data/ppe_compliance_monitor/``
    """
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset already prepared at %s -- skipping", PROJECT_KEY, DATA_ROOT)
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
    """Create a synthetic PPE detection dataset in YOLO format.

    Generates images with simplified person silhouettes and PPE item
    rectangles.  Sufficient for pipeline smoke tests.
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
            img, labels = _make_ppe_image(idx)
            fname = f"ppe_{split_name}_{idx:04d}"
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
    (raw_dir / "data.yaml").write_text(
        yaml.dump(data_cfg, default_flow_style=False), encoding="utf-8"
    )


def _make_ppe_image(seed_offset: int) -> tuple[np.ndarray, list[str]]:
    """Render a synthetic construction-site-like image with persons + PPE."""
    W, H = 640, 480
    # Construction site background colour (grey concrete)
    img = np.full((H, W, 3), (180, 175, 170), dtype=np.uint8)
    # Ground line
    cv2.rectangle(img, (0, H - 60), (W, H), (100, 110, 90), -1)

    labels: list[str] = []
    n_persons = random.randint(1, 4)

    for _ in range(n_persons):
        # Person body
        px = random.randint(60, W - 100)
        py = random.randint(100, H - 120)
        pw = random.randint(50, 80)
        ph = random.randint(120, 200)
        x1, y1 = px, py
        x2, y2 = px + pw, py + ph

        # Draw person (dark rectangle)
        body_color = (random.randint(30, 80), random.randint(30, 80), random.randint(60, 120))
        cv2.rectangle(img, (x1, y1), (x2, y2), body_color, -1)
        labels.append(_yolo_label(0, x1, y1, x2, y2, W, H))  # class 0 = person

        # Randomly add PPE items
        # Helmet (above head)
        if random.random() < 0.7:
            hx1 = x1 + pw // 4
            hy1 = y1 - random.randint(15, 25)
            hx2 = hx1 + pw // 2
            hy2 = y1 + 5
            hy1 = max(0, hy1)
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 200, 255), -1)  # yellow helmet
            labels.append(_yolo_label(1, hx1, hy1, hx2, hy2, W, H))  # class 1 = helmet

        # Safety vest (on upper torso)
        if random.random() < 0.6:
            vx1 = x1 + 3
            vy1 = y1 + ph // 5
            vx2 = x2 - 3
            vy2 = y1 + ph // 2
            cv2.rectangle(img, (vx1, vy1), (vx2, vy2), (0, 140, 255), -1)  # orange vest
            labels.append(_yolo_label(2, vx1, vy1, vx2, vy2, W, H))  # class 2 = safety_vest

        # Gloves (on hands)
        if random.random() < 0.4:
            gx1 = x1 - 8
            gy1 = y1 + ph * 2 // 3
            gx2 = x1 + 8
            gy2 = gy1 + 20
            cv2.rectangle(img, (max(0, gx1), gy1), (gx2, gy2), (200, 200, 0), -1)
            labels.append(_yolo_label(3, max(0, gx1), gy1, gx2, gy2, W, H))  # class 3 = gloves

        # Goggles (on face area)
        if random.random() < 0.3:
            ox1 = x1 + pw // 4
            oy1 = y1 + 10
            ox2 = x1 + 3 * pw // 4
            oy2 = oy1 + 12
            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (200, 100, 0), -1)
            labels.append(_yolo_label(4, ox1, oy1, ox2, oy2, W, H))  # class 4 = goggles

        # Boots (at feet)
        if random.random() < 0.35:
            bx1 = x1 + 5
            by1 = y2 - 15
            bx2 = x2 - 5
            by2 = y2 + 5
            by2 = min(H, by2)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
            labels.append(_yolo_label(5, bx1, by1, bx2, by2, W, H))  # class 5 = boots

    # Add mild noise
    noise = np.random.randint(0, 10, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img, labels


def _yolo_label(cls_id: int, x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> str:
    """Format a YOLO normalised label line."""
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = abs(x2 - x1) / W
    h = abs(y2 - y1) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


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
            "Synthetic PPE detection demo (persons + helmet/vest/gloves/goggles/boots)"
            if synthetic
            else "Construction-site PPE detection -- Roboflow Universe"
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
