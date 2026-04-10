# Sports Ball Possession Tracker

Detect players and the ball in sports video, track them across frames, and estimate ball possession over time using a transparent nearest-player heuristic.

---

## How Possession Is Estimated

The algorithm is deliberately **simple and auditable**:

1. **Detect** players and the ball each frame using YOLO26m.
2. **Track** all objects with ByteTrack for stable IDs across frames.
3. **Find nearest player** — compute Euclidean distance from ball centre to every tracked player centre.
4. **Assign possession** if the nearest player is within `possession_radius_px` (default: 120 px).
5. **Sticky hold** — possession is maintained for `possession_hold_frames` (default: 5) after the ball leaves the radius, preventing flickering during dribbles and partial occlusions.
6. **Accumulate** — per-player frame counts are converted to possession percentages.

No black-box ML model is involved in the possession logic. The detection and tracking are ML-based (YOLO + ByteTrack), but the possession assignment is pure geometry — fully inspectable and tuneable via config.

---

## Features

| Feature | Description |
|---|---|
| **Player + ball detection** | YOLO26m with auto-detected class IDs |
| **Multi-object tracking** | ByteTrack (or BoTSORT) for stable player IDs |
| **Possession estimation** | Nearest-player heuristic with sticky hold |
| **Possession timeline** | Per-frame CSV/JSON export of who holds the ball |
| **Possession bar** | Real-time horizontal bar showing cumulative split |
| **Player trails** | Fading movement trails per tracked player |
| **Possession link** | Visual line connecting ball to current holder |
| **Auto class detection** | Auto-maps "person"/"player" and "ball" classes |
| **Auto dataset download** | One-command bootstrap via Roboflow |

---

## Architecture

```
Sports Ball Possession Tracker/
└── Source Code/
    ├── config.py               # PossessionConfig dataclass + YAML loader
    ├── data_bootstrap.py       # Dataset download & preparation
    ├── tracker.py              # YOLO model.track() wrapper → FrameDetections
    ├── possession.py           # Nearest-player possession estimator
    ├── visualize.py            # Overlay renderer (boxes, trails, bar, link)
    ├── export.py               # JSON + CSV possession timeline export
    ├── infer.py                # CLI video inference pipeline
    ├── train.py                # Training (delegates to train/train_detection.py)
    ├── evaluate.py             # Evaluation with per-class mAP
    ├── modern.py               # Registry entry (@register)
    ├── possession_config.yaml  # Sample configuration
    └── requirements.txt        # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r "Sports Ball Possession Tracker/Source Code/requirements.txt"
```

### 2. Run on a match video

```bash
# With display
python "Sports Ball Possession Tracker/Source Code/infer.py" \
    --source match.mp4

# With config
python "Sports Ball Possession Tracker/Source Code/infer.py" \
    --source match.mp4 \
    --config "Sports Ball Possession Tracker/Source Code/possession_config.yaml"

# Headless with export
python "Sports Ball Possession Tracker/Source Code/infer.py" \
    --source match.mp4 \
    --no-display \
    --save-video output.mp4 \
    --export-json possession.json \
    --export-csv possession.csv
```

### 3. Train a custom model

```bash
python "Sports Ball Possession Tracker/Source Code/train.py" \
    --epochs 100 --batch 8 --imgsz 1280
```

### 4. Evaluate

```bash
python "Sports Ball Possession Tracker/Source Code/evaluate.py" \
    --model runs/sports_possession/train/weights/best.pt
```

---

## Configuration

Edit `possession_config.yaml` to customise:

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `yolo26m.pt` | YOLO model weights |
| `conf_threshold` | `float` | `0.30` | Detection confidence threshold |
| `imgsz` | `int` | `1280` | Input image size |
| `player_class_id` | `int` | `-1` | Player class index (-1 = auto-detect) |
| `ball_class_id` | `int` | `-1` | Ball class index (-1 = auto-detect) |
| `tracker_type` | `str` | `bytetrack` | Tracker: `bytetrack` or `botsort` |
| `possession_radius_px` | `int` | `120` | Max ball→player distance for possession |
| `possession_hold_frames` | `int` | `5` | Sticky hold duration |
| `min_ball_conf` | `float` | `0.20` | Minimum ball detection confidence |
| `show_trails` | `bool` | `true` | Render player movement trails |
| `show_possession_bar` | `bool` | `true` | Show cumulative possession bar |

---

## JSON Export Format

```json
{
  "summary": {
    "total_frames": 1800,
    "contested_frames": 320,
    "contested_pct": 17.8,
    "player_possession": {
      "player_3": {"frames": 540, "pct": 30.0},
      "player_7": {"frames": 480, "pct": 26.7},
      "player_12": {"frames": 460, "pct": 25.6}
    }
  },
  "timeline": [
    {
      "frame_idx": 0,
      "ball_detected": true,
      "holder_id": 3,
      "holder_name": "Player #3",
      "distance_px": 45.2,
      "num_players": 8
    }
  ]
}
```

---

## Class Auto-Detection

The tracker automatically identifies player and ball classes from the model's class names:

| Role | Keywords searched | COCO fallback |
|---|---|---|
| **Player** | `person`, `player`, `goalkeeper`, `referee` | Class 0 (`person`) |
| **Ball** | `ball`, `sports ball`, `football`, `soccer ball`, `basketball`, `tennis ball` | Class 32 (`sports ball`) |

Override with `player_class_id` and `ball_class_id` in the config if your model uses different indices.

---

## Dataset

The training dataset is auto-downloaded from Roboflow via the shared dataset infrastructure (`configs/datasets/sports_ball_possession_tracker.yaml`).

**Source**: Football player + ball detection dataset with bounding-box annotations.

**License**: Refer to the dataset page on Roboflow Universe for terms of use.

To force a re-download:

```bash
python "Sports Ball Possession Tracker/Source Code/train.py" --force-download
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit |
