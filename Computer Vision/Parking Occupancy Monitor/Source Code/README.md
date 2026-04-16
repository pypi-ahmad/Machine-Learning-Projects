# Parking Occupancy Monitor

Real-time **parking-lot occupancy monitoring** using YOLO object
detection.  Detects vehicles (car, truck, bus, motorcycle, bicycle),
evaluates which configurable parking-slot polygons are occupied, and
exports per-frame occupancy summaries to CSV / JSON.

---

## Architecture

```
Source Code/
├── config.py          # ParkingConfig / SlotConfig dataclasses, YAML/JSON loader
├── data_bootstrap.py  # Dataset download & preparation (Roboflow)
├── slots.py           # Detection dataclass, SlotEvaluator (polygon containment + IoU)
├── visualize.py       # Overlay renderer (slot polygons, vehicle boxes, dashboard)
├── export.py          # CSV / JSON occupancy logging
├── infer.py           # Full inference pipeline (image / video / webcam)
├── train.py           # YOLO training entry-point
├── evaluate.py        # YOLO val() + per-class metrics report
├── modern.py          # CVProject adapter for repo registry
├── slots.yaml         # Sample 8-slot configuration
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (optional — download + fine-tune)

```bash
cd "Parking Occupancy Monitor/Source Code"
python train.py --epochs 30 --batch 16
```

### 3. Run inference

```bash
# Video file
python infer.py --source video.mp4 --config slots.yaml

# Webcam (lower-latency model)
python infer.py --source 0 --config slots.yaml --live

# Single image (headless)
python infer.py --source frame.jpg --config slots.yaml --no-display
```

### 4. Evaluate

```bash
python evaluate.py --model runs/parking_occupancy_monitor/train/weights/best.pt
```

## Configuration

Edit `slots.yaml` to define your parking lot layout:

| Key | Description |
|-----|-------------|
| `model` | YOLO weights for batch / video inference |
| `model_live` | Lighter YOLO weights for webcam mode (`--live`) |
| `conf_threshold` | Detection confidence threshold |
| `vehicle_classes` | List of class names treated as vehicles |
| `occupancy_iou_threshold` | Min box overlap to consider a slot occupied |
| `slots` | List of named polygon slots |
| `export_dir` | Output directory for CSV / JSON logs |

### Slot definition

Each slot is a named polygon with four or more `[x, y]` vertices:

```yaml
slots:
  - name: A1
    polygon:
      - [50, 60]
      - [140, 60]
      - [140, 190]
      - [50, 190]
```

Adjust coordinates to match your camera's perspective.

## Pipeline Flow

```
Camera / Video
     │
     ▼
  YOLO Detection
     │
     ▼
  ┌──────────────────┐
  │  Slot Evaluator   │  ← center-in-polygon + IoU check
  │                    │  ← per-slot occupied / free status
  └────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  Visualize   Export
  (overlays)  (CSV / JSON)
```

## Outputs

| File | Content |
|------|---------|
| `outputs/occupancy.csv` | Per-frame occupancy summary with slot details |
| `outputs/occupancy.json` | Same data in JSON format |
| `outputs/eval_report.json` | Model evaluation metrics |

## Model Variants

| Model | Use Case |
|-------|----------|
| `yolo26m.pt` | Default — best accuracy for video / batch |
| `yolo26s.pt` | `--live` flag — lower latency for webcam |

## Registration

Registered as `parking_occupancy_monitor` in the repo's project registry:

```python
from core.registry import list_projects
print(list_projects())  # includes "parking_occupancy_monitor"
```

## Dataset

Uses the **Parking Lot Detection** dataset from Roboflow Universe,
auto-downloaded via `configs/datasets/parking_occupancy_monitor.yaml`.

If the Roboflow SDK is not installed (`pip install roboflow`), the system
automatically generates a synthetic demo dataset with overhead parking-lot
images containing vehicle rectangles in labelled slots. This allows the
full pipeline to work out of the box for testing and development.

Dataset is stored in `data/parking_occupancy_monitor/` with `raw/` and
`processed/` subdirectories plus `dataset_info.json`. Use `--force-download`
to re-download.

Classes: `car`, `truck`, `bus`, `motorcycle`, `bicycle`.
