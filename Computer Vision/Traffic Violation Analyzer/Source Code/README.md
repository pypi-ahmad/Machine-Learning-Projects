# Traffic Violation Analyzer

Real-time **traffic violation detection** using YOLO object detection,
built-in multi-object tracking (ByteTrack / BoT-SORT), and a configurable
rule engine.  Detects vehicles, tracks them across frames, counts virtual
line crossings, and flags wrong-way movement.

---

## Architecture

```
Source Code/
├── config.py          # TrafficConfig / LineConfig / ZoneConfig, YAML/JSON loader
├── data_bootstrap.py  # Dataset download & preparation (Roboflow)
├── detector.py        # Detection dataclass (shared across modules)
├── tracker.py         # TrackManager — wraps YOLO .track(), maintains trails
├── rules.py           # RuleEngine — line crossing + wrong-way detection
├── visualize.py       # Overlay renderer (lines, zones, trails, alerts, dashboard)
├── export.py          # CSV / JSON event logging
├── infer.py           # Full inference pipeline (image / video / webcam)
├── train.py           # YOLO training entry-point
├── evaluate.py        # YOLO val() + per-class metrics report
├── modern.py          # CVProject adapter for repo registry
├── traffic.yaml       # Sample line & zone configuration
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (optional — download + fine-tune)

```bash
cd "Traffic Violation Analyzer/Source Code"
python train.py --epochs 30 --batch 16
```

### 3. Run inference

```bash
# Video file (recommended — tracking works best on video)
python infer.py --source traffic.mp4 --config traffic.yaml

# Webcam
python infer.py --source 0 --config traffic.yaml

# Single image (detection only, no tracking)
python infer.py --source frame.jpg --no-display

# Save annotated output
python infer.py --source traffic.mp4 --config traffic.yaml --save-video --no-display
```

### 4. Evaluate

```bash
python evaluate.py --model runs/traffic_violation_analyzer/train/weights/best.pt
```

## Configuration

Edit `traffic.yaml` to define virtual lines and zones:

### Virtual lines

| Field | Description |
|-------|-------------|
| `name` | Unique line identifier |
| `pt1`, `pt2` | Start and end `[x, y]` coordinates |
| `direction` | Allowed crossing direction: `up`, `down`, `left`, `right`, or `any` |

When `direction` is set to a cardinal direction, any vehicle crossing in
the **opposite** direction triggers a **wrong-way** violation event.
Set `direction: any` for pure counting lines.

### Zones

Named polygons drawn on the frame. Currently used for visualisation;
the rule engine can be extended to detect zone intrusions.

### Other settings

| Key | Description |
|-----|-------------|
| `model` | YOLO weights path |
| `tracker` | Tracker config (`bytetrack.yaml` or `botsort.yaml`) |
| `conf_threshold` | Detection confidence threshold |
| `vehicle_classes` | List of class names treated as vehicles |
| `export_dir` | Output directory for CSV / JSON events |

## Pipeline Flow

```
Camera / Video
     │
     ▼
  YOLO Detection + Tracking
  (model.track, persist=True)
     │
     ▼
  ┌──────────────────┐
  │  Track Manager    │  ← parse IDs, maintain per-track trails
  └────────┬──────────┘
           │
           ▼
  ┌──────────────────┐
  │  Rule Engine      │  ← line crossing (cross-product sign change)
  │                    │  ← wrong-way detection (direction mismatch)
  └────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  Visualize   Export
  (overlays)  (CSV / JSON events)
```

## Rule Engine Details

### Line crossing detection

Uses the **cross-product sign-change** method: for a line defined by
points A→B, the engine computes the sign of `(B-A) × (P-A)` for the
vehicle's previous and current center positions.  A sign change means
the vehicle crossed the line.  Each track ID can cross each line at
most once (de-duplicated).

### Wrong-way detection

After detecting a crossing, the engine computes the dominant cardinal
direction of vehicle movement.  If a line specifies `direction: up`
but the vehicle crossed **downward**, a `wrong_way` event is emitted
alongside the `line_cross` event.

## Outputs

| File | Content |
|------|---------|
| `outputs/events.csv` | Per-event log (line crossings, wrong-way) |
| `outputs/events.json` | Same data in JSON format |
| `outputs/output.mp4` | Annotated video (with `--save-video`) |
| `outputs/eval_report.json` | Model evaluation metrics |

## Registration

Registered as `traffic_violation_analyzer` in the repo's project registry:

```python
from core.registry import list_projects
print(list_projects())  # includes "traffic_violation_analyzer"
```

## Dataset

Uses the **Traffic Detection** dataset from Roboflow Universe,
auto-downloaded via `configs/datasets/traffic_violation_analyzer.yaml`.
The dataset contains road-scene images with vehicle classes annotated
in YOLO format.

### Dataset bootstrap behaviour

- **Idempotent** — skips download if `.ready` marker exists
- **`--force-download`** — forces re-download and re-preparation
- Layout: `data/traffic_violation_analyzer/raw/` + `data/traffic_violation_analyzer/processed/`
- Provenance: `data/traffic_violation_analyzer/dataset_info.json`
