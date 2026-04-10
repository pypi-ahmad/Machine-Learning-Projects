# Drone Ship OBB Detector

Detect ships, vehicles, and other objects in aerial and satellite imagery using **oriented bounding boxes (OBB)**.

---

## Why Oriented Bounding Boxes?

Standard axis-aligned bounding boxes (HBB) are a poor fit for aerial imagery:

| Problem | HBB | OBB |
|---|---|---|
| **Rotated objects** | Box includes large background areas, reducing precision | Tight fit regardless of rotation |
| **IoU accuracy** | Two ships at different angles may overlap heavily in HBB space | Rotated IoU correctly distinguishes nearby objects |
| **Dense scenes** | Overlapping HBBs cause false NMS suppression | OBB NMS preserves closely-packed objects |
| **Thin, elongated objects** | A 200×20 px ship at 45° needs a 156×156 HBB (12× area) | OBB covers exactly 200×20 px |

Ships in port, aircraft on runways, and containers in yards are canonical examples where OBB dramatically outperforms axis-aligned detection.

This project uses **YOLO26m-OBB** — a purpose-built oriented detection head that predicts four corner points per object instead of centre + width + height.

---

## Features

| Feature | Description |
|---|---|
| **Oriented bounding boxes** | 4-corner polygon detection with rotation angle |
| **Per-class counting** | Real-time class distribution |
| **JSON export** | Full detection details (corners, angle, confidence) |
| **TXT export** | YOLO-OBB format for downstream evaluation |
| **Dashboard overlay** | On-screen counts and oriented box rendering |
| **Image / video / webcam** | Unified inference pipeline |
| **Auto dataset download** | One-command training data bootstrap |

---

## Architecture

```
Drone Ship OBB Detector/
└── Source Code/
    ├── config.py          # OBBConfig dataclass + YAML loader
    ├── data_bootstrap.py  # Dataset download & OBB label validation
    ├── detector.py        # YOLO-OBB inference + corner extraction
    ├── visualize.py       # Oriented box renderer + dashboard
    ├── export.py          # JSON + YOLO-OBB TXT export
    ├── infer.py           # CLI inference pipeline
    ├── train.py           # Training (delegates to train/train_obb.py)
    ├── evaluate.py        # OBB evaluation with rotated-box IoU metrics
    ├── modern.py          # Registry entry (@register)
    ├── obb_config.yaml    # Sample configuration
    └── requirements.txt   # Python dependencies

train/
└── train_obb.py           # Shared OBB training pipeline (new)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r "Drone Ship OBB Detector/Source Code/requirements.txt"
```

### 2. Run inference

```bash
# Single aerial image
python "Drone Ship OBB Detector/Source Code/infer.py" \
    --source aerial_port.jpg

# Video with config
python "Drone Ship OBB Detector/Source Code/infer.py" \
    --source drone_footage.mp4 \
    --config "Drone Ship OBB Detector/Source Code/obb_config.yaml"

# Headless with export
python "Drone Ship OBB Detector/Source Code/infer.py" \
    --source satellite.png \
    --no-display \
    --export-json results.json \
    --export-txt output/labels/
```

### 3. Train a custom OBB model

```bash
python "Drone Ship OBB Detector/Source Code/train.py" \
    --epochs 200 \
    --batch 4 \
    --imgsz 1024
```

### 4. Evaluate

```bash
python "Drone Ship OBB Detector/Source Code/evaluate.py" \
    --model runs/obb_ship/train/weights/best.pt
```

---

## Configuration

Edit `obb_config.yaml` to customise:

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `yolo26m-obb.pt` | YOLO-OBB model weights |
| `conf_threshold` | `float` | `0.35` | Minimum detection confidence |
| `imgsz` | `int` | `1024` | Input image size (larger = better for aerial) |
| `target_classes` | `list[str]` | `[]` | Class filter (empty = all) |
| `export_json` | `str` | `""` | JSON output path |
| `export_txt` | `str` | `""` | TXT output directory (YOLO-OBB format) |
| `show_conf` | `bool` | `true` | Display confidence on labels |
| `show_angle` | `bool` | `true` | Display rotation angle on labels |

---

## Export Formats

### JSON

```json
{
  "frame_idx": 0,
  "total": 3,
  "class_counts": {"ship": 2, "large-vehicle": 1},
  "detections": [
    {
      "class": "ship",
      "confidence": 0.92,
      "corners": [[100, 50], [300, 60], [298, 90], [98, 80]],
      "centre": [199, 70],
      "angle_deg": 3.4
    }
  ]
}
```

### TXT (YOLO-OBB format)

One `.txt` file per image:
```
ship 0.0521 0.0694 0.1563 0.0833 0.1552 0.1250 0.0510 0.1111
```

Format: `class_name x1 y1 x2 y2 x3 y3 x4 y4` (normalised corner coordinates)

---

## Dataset

The training dataset is auto-downloaded from Roboflow via the shared dataset infrastructure (`configs/datasets/drone_ship_obb_detector.yaml`).

**Source**: ShipRSImageNet OBB subset — satellite imagery with oriented ship annotations.

**License**: Refer to the dataset page on Roboflow Universe for terms of use. The dataset is intended for research and educational purposes.

To force a re-download:

```bash
python "Drone Ship OBB Detector/Source Code/train.py" --force-download
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit (video/webcam mode) |
