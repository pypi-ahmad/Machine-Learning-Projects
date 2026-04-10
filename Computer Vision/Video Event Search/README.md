# Video Event Search

Detect, track, and search structured events in video streams using YOLO26m + ByteTrack.

## Overview

This project processes video files (or live camera feeds) through a YOLO detection + ByteTrack tracking pipeline and generates **structured, searchable events**:

| Event Type | Description |
|---|---|
| `appear` | A tracked object appears for the first time |
| `disappear` | A tracked object leaves the scene |
| `zone_enter` | Object centroid enters a defined polygon zone |
| `zone_exit` | Object centroid exits a polygon zone |
| `line_cross` | Object centroid crosses a virtual line |
| `dwell` | Object stays in a zone longer than a threshold |

Events are stored as JSON/CSV logs and can be queried by type, time range, track ID, zone, class, and more.

## Pipeline

```
Video → YOLO26m Detect+Track → TrackManager → EventGenerator → EventStore
                                                                    ↓
                                                              Query Interface
```

## Dataset

**Pedestrian Dataset** (Kaggle: `smeschke/pedestrian-dataset`)
- 3 scenes: crosswalk (12s), night (25s), fourway (42s)
- `.avi` video files with bounding-box `.csv` annotations
- CC0 Public Domain license
- ~53 MB total

Auto-download:
```bash
python data_bootstrap.py
```

## Usage

### Process a Video

```bash
cd "Video Event Search/Source Code"

# Process with default config (pedestrian crosswalk)
python infer.py process --source ../../data/video_event_search/raw/crosswalk.avi

# Process with custom config
python infer.py process --source video.mp4 --config my_config.yaml

# Headless mode + save annotated video
python infer.py process --source video.mp4 --no-display --save-video

# Auto-download dataset first
python infer.py process --source ../../data/video_event_search/raw/crosswalk.avi --download
```

### Query Events

```bash
# Show event summary
python infer.py query --summary

# Filter by event type
python infer.py query --event-type zone_enter

# Filter by track ID
python infer.py query --track-id 3

# Filter by time range
python infer.py query --time-start 5.0 --time-end 20.0

# Filter by zone
python infer.py query --zone crosswalk_zone

# Combine filters
python infer.py query --event-type dwell --zone crosswalk_zone --class-name person
```

### Python API

```python
from modern import VideoEventSearchModern
from config import EventSearchConfig, LineConfig, ZoneConfig

# Configure zones and lines
cfg = EventSearchConfig(
    lines=[LineConfig(name="entry_line", pt1=(50, 300), pt2=(590, 300))],
    zones=[ZoneConfig(name="lobby", polygon=[[40, 200], [600, 200], [600, 400], [40, 400]])],
    target_classes=["person"],
    dwell_threshold=5.0,
)

# Run
project = VideoEventSearchModern(config=cfg)
project.load()

# Process frames
import cv2
cap = cv2.VideoCapture("video.mp4")
project.set_fps(cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = project.predict(frame)
    annotated = project.visualize(frame, result)

project.flush_events()

# Query
summary = project.event_summary()
dwell_events = project.query_events(event_type="dwell", min_dwell=10.0)
```

## Event Schema

Each event is a JSON object:

```json
{
  "event_type": "zone_enter",
  "track_id": 3,
  "class_name": "person",
  "frame_idx": 142,
  "timestamp_sec": 5.68,
  "confidence": 0.87,
  "center_x": 320,
  "center_y": 280,
  "zone_or_line": "crosswalk_zone",
  "direction": "",
  "dwell_seconds": 0.0,
  "metadata": {}
}
```

## Configuration

YAML or JSON config file:

```yaml
model: yolo26m.pt
conf_threshold: 0.30
tracker: bytetrack.yaml
target_classes: [person]
dwell_threshold: 5.0  # seconds

lines:
  - name: crossing_line
    pt1: [50, 300]
    pt2: [590, 300]
    direction: any

zones:
  - name: waiting_area
    polygon: [[40, 200], [600, 200], [600, 400], [40, 400]]
```

## Project Structure

```
Video Event Search/
├── Source Code/
│   ├── config.py           # Configuration + event schema + EventType enum
│   ├── detector.py         # Detection dataclass
│   ├── tracker.py          # TrackManager — YOLO tracking wrapper + trail history
│   ├── event_generator.py  # EventGenerator — zone/line/dwell/appear/disappear logic
│   ├── event_store.py      # EventStore — JSON + CSV persistence
│   ├── query.py            # EventQuery — filter/search interface
│   ├── visualize.py        # OverlayRenderer — annotated frame drawing
│   ├── export.py           # Re-export utilities
│   ├── infer.py            # CLI: process video + query events
│   ├── modern.py           # CVProject adapter (registered as video_event_search)
│   └── data_bootstrap.py   # Dataset download from Kaggle
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- ultralytics >= 8.0
- opencv-python >= 4.8
- numpy >= 1.24
- pyyaml >= 6.0

## Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLO26m (Ultralytics) |
| Tracking | ByteTrack (built-in) |
| Event Engine | Custom rule-based generator |
| Storage | JSON + CSV append-only logs |
| Query | In-memory filter engine |
| Visualisation | OpenCV overlay renderer |
