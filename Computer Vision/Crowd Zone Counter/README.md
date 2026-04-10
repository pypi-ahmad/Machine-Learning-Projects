# Crowd Zone Counter

Zone-based people counting for crowd analytics and overcrowding alerts.  Define polygon zones, set capacity thresholds, and get real-time counts with automatic warnings when limits are exceeded.

---

## Features

| Feature | Description |
|---|---|
| **Person detection** | YOLO26m filtered to the "person" class |
| **Configurable zones** | Arbitrary polygon regions with names and colours |
| **Per-zone counting** | Foot-point-in-polygon test for accurate assignment |
| **Overcrowding alerts** | Configurable `max_capacity` per zone with cooldown |
| **CSV / JSON export** | Per-frame zone counts + alert log |
| **Zone overlays** | Transparent fills, borders, count labels, alert banners |
| **Dashboard** | Real-time panel with zone counts and totals |
| **Image / video / webcam** | Unified inference pipeline |
| **Auto dataset download** | One-command training data bootstrap |

---

## How It Works

1. **Detect** all persons in the frame using YOLO26m (COCO class 0).
2. **Assign** each person to a zone by testing their **foot-point** (bottom-centre of bounding box) against each zone polygon using `cv2.pointPolygonTest`.
3. **Count** persons per zone; persons outside all zones are counted as "unzoned".
4. **Alert** if any zone's count exceeds its `max_capacity`.  Alerts are suppressed for `alert_cooldown_frames` to avoid flooding.

The foot-point is used instead of the bbox centre because it better represents where a person is standing — critical for accurate zone assignment in perspective views.

---

## Architecture

```
Crowd Zone Counter/
└── Source Code/
    ├── config.py           # CrowdConfig / ZoneConfig + YAML loader
    ├── data_bootstrap.py   # Dataset download & preparation
    ├── detector.py         # PersonDetector — YOLO person-only detection
    ├── zone_counter.py     # ZoneCounter — assignment + overcrowding logic
    ├── visualize.py        # Overlay renderer (zones, boxes, alerts, dashboard)
    ├── export.py           # JSON + CSV zone count / alert export
    ├── infer.py            # CLI inference (image / video / webcam)
    ├── train.py            # Training (delegates to train/train_detection.py)
    ├── evaluate.py         # Evaluation with per-class mAP
    ├── modern.py           # Registry entry (@register)
    ├── crowd_config.yaml   # Sample configuration
    └── requirements.txt    # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r "Crowd Zone Counter/Source Code/requirements.txt"
```

### 2. Run inference

```bash
# Video with config
python "Crowd Zone Counter/Source Code/infer.py" \
    --source crowd_video.mp4 \
    --config "Crowd Zone Counter/Source Code/crowd_config.yaml"

# Webcam (default)
python "Crowd Zone Counter/Source Code/infer.py"

# Image
python "Crowd Zone Counter/Source Code/infer.py" --source crowd.jpg

# Headless with export
python "Crowd Zone Counter/Source Code/infer.py" \
    --source footage.mp4 \
    --no-display \
    --save-video output.mp4 \
    --export-json counts.json \
    --export-csv counts.csv
```

### 3. Train a custom model

```bash
python "Crowd Zone Counter/Source Code/train.py" --epochs 100 --batch 16
```

### 4. Evaluate

```bash
python "Crowd Zone Counter/Source Code/evaluate.py" \
    --model runs/crowd_zone/train/weights/best.pt
```

---

## Configuration

Edit `crowd_config.yaml` to customise:

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `yolo26m.pt` | YOLO model weights |
| `conf_threshold` | `float` | `0.30` | Detection confidence threshold |
| `person_class_id` | `int` | `0` | Class index for "person" |
| `zones` | `list` | `[]` | Zone polygon definitions |
| `alert_cooldown_frames` | `int` | `30` | Frames between repeated alerts |
| `show_zone_fill` | `bool` | `true` | Transparent zone fill |
| `zone_alpha` | `float` | `0.25` | Fill transparency |

### Zone Definition

```yaml
zones:
  - name: "Entrance"
    polygon: [[50, 50], [400, 50], [400, 400], [50, 400]]
    max_capacity: 15       # 0 = no limit

  - name: "Main-Area"
    polygon: [[450, 50], [900, 50], [900, 650], [450, 650]]
    max_capacity: 40
    colour: [0, 200, 0]   # optional BGR override
```

Adjust polygon coordinates to match your camera view.  Use any number of vertices to define irregular zones.

---

## JSON Export Format

```json
{
  "total_frames": 1800,
  "zones": [
    {"name": "Entrance", "max_capacity": 15},
    {"name": "Main-Area", "max_capacity": 40}
  ],
  "alerts": [
    {
      "frame_idx": 342,
      "zone": "Entrance",
      "count": 18,
      "max_capacity": 15
    }
  ],
  "timeline": [
    {
      "frame_idx": 0,
      "total_persons": 25,
      "unzoned": 3,
      "zone_counts": {"Entrance": 8, "Main-Area": 14},
      "overcrowded_zones": []
    }
  ]
}
```

---

## Dataset

The training dataset is auto-downloaded from Roboflow via the shared dataset infrastructure (`configs/datasets/crowd_zone_counter.yaml`).

**Source**: Crowd person detection dataset with single-class (person) bounding-box annotations from crowd scenes.

**License**: Refer to the dataset page on Roboflow Universe for terms of use.

To force a re-download:

```bash
python "Crowd Zone Counter/Source Code/train.py" --force-download
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit (video/webcam mode) |
| Any key | Next (image mode) |
