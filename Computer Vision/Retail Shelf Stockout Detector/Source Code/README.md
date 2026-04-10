# Retail Shelf Stockout Detector

> **Task:** Detection + Zone Counting &nbsp;|&nbsp; **Key:** `retail_shelf_stockout` &nbsp;|&nbsp; **Framework:** YOLO26m

---

## Overview

Production-style computer vision project that detects products on retail shelves, counts items per configurable zone polygon, and flags low-stock conditions with alerts, event logs, and annotated snapshots. Supports image, video, and webcam inference.

## Architecture

```
Retail Shelf Stockout Detector/
└── Source Code/
    ├── modern.py           # CVProject subclass — @register("retail_shelf_stockout")
    ├── train.py            # YOLO training CLI (auto-downloads dataset)
    ├── infer.py            # Inference: image / video / webcam
    ├── evaluate.py         # Model evaluation + metrics report
    ├── config.py           # StockoutConfig + ZoneConfig dataclasses + loader
    ├── data_bootstrap.py   # Dataset download, preparation, idempotent caching
    ├── zones.py            # ZoneCounter logic — point-in-polygon, alerts
    ├── visualize.py        # OverlayRenderer — boxes, zones, dashboard
    ├── export.py           # EventExporter — CSV, JSON, alert snapshots
    ├── zones.yaml          # Sample zone configuration
    ├── requirements.txt    # Project-level dependencies
    ├── outputs/            # Inference outputs (created at runtime)
    └── README.md           # This file
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Dataclasses for all settings, YAML/JSON loader, sample config factory |
| `data_bootstrap.py` | Wraps repo's `DatasetResolver`, organises raw/processed, writes metadata |
| `zones.py` | Pure logic — point-in-polygon, per-zone counts, low-stock alerts |
| `visualize.py` | OpenCV overlay rendering — boxes, filled zone polygons, dashboard |
| `export.py` | CSV/JSON event accumulation, alert snapshot saving with cooldown |
| `infer.py` | Orchestrates model → zones → render → export pipeline |
| `evaluate.py` | YOLO `val()` wrapper + per-class metrics + JSON report |
| `modern.py` | Thin CVProject adapter for the repo's global registry |

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) + zone counting |
| **Model** | YOLO26m (Ultralytics) |
| **Zone Logic** | Configurable polygon regions with per-zone thresholds |
| **Export** | CSV, JSON event logs, alert snapshot images |
| **Viz** | Semi-transparent zone fills, per-zone dashboard, alert banners |

## Dataset

| Field | Value |
|-------|-------|
| **Source** | Roboflow Universe — retail shelf product detection |
| **Type** | YOLO-format object detection (bounding boxes) |
| **License** | See [dataset page](https://universe.roboflow.com/) for terms |
| **Download** | Automatic via `DatasetResolver` on first run |
| **Structure** | `data/retail_shelf_stockout/raw/` (original) + `processed/` (ready) |
| **Metadata** | `data/retail_shelf_stockout/dataset_info.json` |

The dataset downloads automatically when you run `train.py` or `infer.py` for the first time. No manual placement required. Use `--force-download` to re-download.

## Quick Start

### 1. Training

```bash
cd "Retail Shelf Stockout Detector/Source Code"

# Default: auto-downloads dataset + trains YOLO26m
python train.py

# Custom settings
python train.py --model yolo26m.pt --epochs 80 --batch 16 --imgsz 640

# Force dataset re-download
python train.py --force-download
```

### 2. Inference

```bash
# Single image
python infer.py --source shelf_photo.jpg

# Video
python infer.py --source store_footage.mp4 --save-video

# Webcam
python infer.py --source 0

# With zone configuration
python infer.py --source shelf.jpg --config zones.yaml

# Headless (no GUI)
python infer.py --source video.mp4 --no-display --export-dir results/
```

### 3. Evaluation

```bash
# Evaluate with trained weights (auto-finds runs/retail_shelf_detect/weights/best.pt)
python evaluate.py

# Custom weights
python evaluate.py --weights path/to/best.pt --conf 0.25
```

### 4. Python API

```python
from core import discover_projects
from core.registry import PROJECT_REGISTRY

discover_projects()

proj = PROJECT_REGISTRY["retail_shelf_stockout"]()
proj.load()

# Configure zones
proj.set_zones([
    {"name": "Shelf-A", "polygon": [(50,100), (400,100), (400,350), (50,350)], "low_stock_threshold": 3},
    {"name": "Shelf-B", "polygon": [(420,100), (780,100), (780,350), (420,350)], "low_stock_threshold": 5},
])

# Run on image
result = proj.predict("path/to/shelf.jpg")
print(result["zone_counts"])    # {"Shelf-A": 7, "Shelf-B": 2}
print(result["alerts"])          # ["LOW STOCK: Shelf-B (2/5 items)"]

# Visualize
annotated = proj.visualize("path/to/shelf.jpg", result)

# Export
files = proj.export_events()
```

## Zone Configuration

Zones are defined in a YAML file (see `zones.yaml`):

```yaml
zones:
  - name: "Shelf-A"
    polygon: [[50, 100], [400, 100], [400, 350], [50, 350]]
    low_stock_threshold: 3
    # classes: ["bottle", "can"]  # optional — only count these classes

  - name: "Shelf-B"
    polygon: [[420, 100], [780, 100], [780, 350], [420, 350]]
    low_stock_threshold: 5
```

Each zone specifies:
- **name**: Display label on the overlay
- **polygon**: List of `[x, y]` pixel coordinates forming a closed polygon
- **low_stock_threshold**: Count below which an alert fires
- **classes** (optional): Only count detections of these class names

## Outputs

After inference, the `outputs/` directory contains:

```
outputs/
├── events.csv            # All low-stock events
├── events.json           # Same events in JSON
├── summary.json          # Aggregate stats
├── result_<name>.jpg     # Annotated output (image mode)
├── output.mp4            # Annotated video (--save-video)
└── snapshots/
    └── alert_Shelf-B_2026-04-08T14-30-00.jpg
```

## Dependencies

```
pip install ultralytics>=8.4.0 opencv-python>=4.10.0 numpy>=1.26.0 pyyaml>=6.0
```

Or from repo root:
```
pip install -e "."
```
