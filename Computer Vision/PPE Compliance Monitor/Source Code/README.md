# PPE Compliance Monitor

Real-time **Personal Protective Equipment** compliance monitoring using
YOLO object detection.  Detects persons and their PPE items (helmet,
safety vest, gloves, goggles, boots), associates each item with the
nearest person, and evaluates compliance against configurable per-zone
rules.

---

## Architecture

```
Source Code/
├── config.py          # PPEConfig / ZoneConfig dataclasses, YAML/JSON loader
├── data_bootstrap.py  # Dataset download & preparation (Roboflow)
├── compliance.py      # Person ↔ PPE association & compliance rule engine
├── zones.py           # Zone-based monitoring with polygon containment
├── visualize.py       # Overlay renderer (zones, boxes, alerts, dashboard)
├── export.py          # CSV / JSON event logging, violation snapshots
├── infer.py           # Full inference pipeline (image / video / webcam)
├── train.py           # YOLO training entry-point
├── evaluate.py        # YOLO val() + per-class metrics report
├── modern.py          # CVProject adapter for repo registry
├── zones.yaml         # Sample zone configuration
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (optional — download + fine-tune)

```bash
cd "PPE Compliance Monitor/Source Code"
python train.py --epochs 30 --batch 16
```

### 3. Run inference

```bash
# Video file
python infer.py --source video.mp4 --config zones.yaml

# Webcam
python infer.py --source 0 --config zones.yaml

# Single image (headless)
python infer.py --source frame.jpg --config zones.yaml --no-display
```

### 4. Evaluate

```bash
python evaluate.py --model runs/ppe_compliance_monitor/train/weights/best.pt
```

## Configuration

Edit `zones.yaml` to define:

| Key | Description |
|-----|-------------|
| `model` | YOLO weights path |
| `conf_threshold` | Detection confidence threshold |
| `required_ppe` | Global list of required PPE items |
| `zones` | List of named polygon zones with per-zone `required_ppe` |
| `alert_cooldown_sec` | Minimum seconds between alerts for the same zone |
| `export_dir` | Output directory for events and snapshots |

## Pipeline Flow

```
Camera / Video
     │
     ▼
  YOLO Detection
     │
     ▼
  ┌─────────────────┐
  │ Compliance Check │  ← associate PPE items with persons
  │                   │  ← check required items per person
  └────────┬──────────┘
           │
           ▼
  ┌─────────────────┐
  │  Zone Monitor    │  ← assign persons to polygon zones
  │                   │  ← override per-zone PPE requirements
  │                   │  ← generate alerts with cooldown
  └────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  Visualize   Export
  (overlays)  (CSV/JSON/snapshots)
```

## Outputs

| File | Content |
|------|---------|
| `outputs/events.csv` | Per-frame, per-zone compliance log |
| `outputs/events.json` | Same data in JSON format |
| `outputs/violations/` | Snapshot images of violation frames |
| `outputs/eval_report.json` | Model evaluation metrics |

## Registration

Registered as `ppe_compliance_monitor` in the repo's project registry:

```python
from core.registry import list_projects
print(list_projects())  # includes "ppe_compliance_monitor"
```

## Dataset

Uses the **Construction Site Safety** dataset from Roboflow Universe,
auto-downloaded via `configs/datasets/ppe_compliance_monitor.yaml`.

Classes: `person`, `helmet`, `safety_vest`, `gloves`, `goggles`, `boots`.
