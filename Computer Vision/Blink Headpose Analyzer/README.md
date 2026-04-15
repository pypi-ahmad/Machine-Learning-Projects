# Blink Headpose Analyzer

Reusable face-landmark analytics for **blink counting** and
**head pose estimation** using MediaPipe Face Landmarker.

Provides a `shared/` utility package that other face-analysis
projects can import directly.

---

## Features

| Capability | Module | Metric |
|------------|--------|--------|
| Blink counting | `blink_counter.py` | Eye Aspect Ratio (EAR) |
| Head pose | `pose_estimator.py` | Yaw / Pitch / Roll via solvePnP |
| Per-frame export | `export.py` | CSV and JSON metrics |
| Real-time overlay | `visualize.py` | Eye contours, EAR bar, pose text |
| Automatic bootstrap | `data_bootstrap.py` | Public LFW face dataset |

## Reusable Utilities (`shared/`)

The `shared/` subpackage contains **pure-function utilities** with
no project-specific dependencies.  Import them from any project:

```python
# EAR computation
from shared.ear import compute_ear_from_points, compute_ear_bilateral

pts = [(x1,y1), (x2,y2), ..., (x6,y6)]  # 6 eye contour points
ear = compute_ear_from_points(pts)
left, right, avg = compute_ear_bilateral(left_pts, right_pts)

# Head pose estimation
from shared.head_pose_math import solve_head_pose, rotation_to_euler

yaw, pitch, roll = solve_head_pose(image_points_6x2, frame_w, frame_h)

# Landmark constants
from shared.landmarks import (
    LEFT_EYE, RIGHT_EYE,
    POSE_LANDMARKS, MODEL_POINTS_3D,
    pixel_coords, extract_eye_points, extract_pose_points,
)
```

### `shared/ear.py`

| Function | Description |
|----------|-------------|
| `compute_ear_from_points(pts)` | EAR from 6 (x, y) eye-contour points |
| `compute_ear_bilateral(left, right)` | Both eyes + average |

### `shared/head_pose_math.py`

| Function | Description |
|----------|-------------|
| `solve_head_pose(img_pts, w, h)` | Full solvePnP → (yaw, pitch, roll) |
| `rotation_to_euler(R)` | 3×3 rotation matrix → Euler angles |

### `shared/landmarks.py`

| Symbol | Description |
|--------|-------------|
| `LEFT_EYE` / `RIGHT_EYE` | 6-point eye contour indices |
| `POSE_LANDMARKS` | 6 head-pose landmark indices |
| `MODEL_POINTS_3D` | 3D reference model (6 points, mm) |
| `pixel_coords()` | MediaPipe landmark → (x, y) pixels |
| `extract_eye_points()` | Eye contour → pixel coords list |
| `extract_pose_points()` | Pose landmarks → (N, 2) ndarray |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Prepare the public evaluation dataset
cd "Blink Headpose Analyzer/Source Code"
python train.py --max-frames 20

# Webcam (press 'q' to quit)
python infer.py --source 0

# Video file
python infer.py --source driving_video.mp4

# Headless with CSV export
python infer.py --source 0 --no-display --export-csv metrics.csv

# JSON export
python infer.py --source video.mp4 --export-json metrics.json
```

## Dataset

- Default bootstrap: `sklearn.datasets.fetch_lfw_people` (LFW), prepared automatically into `Computer Vision/data/blink_headpose_analyzer/processed/media`.
- Helper fallback: if the local sklearn download path fails, the project can still fall back to the repo's shared dataset helper using a public LFW-style Hugging Face source.
- Rebuild: pass `--force-download` to `train.py` or `infer.py`.

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source          '0' for webcam, path to video or image

Optional:
  --config          Path to YAML/JSON config file
  --ear-threshold   Override EAR threshold (default: 0.21)
  --yaw-threshold   Override yaw threshold (default: 30°)
  --no-display      Disable GUI windows
  --export-csv      CSV export path for per-frame metrics
  --export-json     JSON export path for per-frame metrics
  --save-annotated  Save annotated frames to output dir
  --output-dir      Output directory (default: output/)
  --force-download  Force dataset re-download
```

## Configuration

All tunables are configurable via YAML/JSON:

```yaml
# MediaPipe Face Landmarker
max_num_faces: 1
min_detection_confidence: 0.5
min_presence_confidence: 0.5
min_tracking_confidence: 0.5

# Blink detection
ear_threshold: 0.21
blink_consec_frames: 2

# Head pose
yaw_threshold: 30.0
pitch_threshold: 25.0
```

Load a config: `python infer.py --source 0 --config my_config.yaml`

## Detection Rules

### EAR — Eye Aspect Ratio

Computed from 6 eye-contour landmarks per eye:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

A blink is counted when EAR drops below `ear_threshold` for at
least `blink_consec_frames` consecutive frames, then rises back.

### Head Pose

Six facial landmarks are matched to a canonical 3D face model
via `cv2.solvePnP`.  The resulting rotation matrix is decomposed
into yaw (left/right), pitch (up/down), and roll.

## Project Structure

```
Blink Headpose Analyzer/
├── Source Code/
│   ├── shared/               ← Reusable utilities
│   │   ├── __init__.py
│   │   ├── ear.py            ← EAR computation
│   │   ├── head_pose_math.py ← solvePnP + Euler
│   │   └── landmarks.py      ← Landmark indices + helpers
│   ├── config.py             ← AnalyzerConfig dataclass
│   ├── landmark_engine.py    ← MediaPipe Face Landmarker wrapper
│   ├── blink_counter.py      ← EAR-based blink counting
│   ├── pose_estimator.py     ← Head pose tracker
│   ├── analyzer.py           ← Pipeline orchestrator
│   ├── validator.py          ← Quality-check validator
│   ├── visualize.py          ← Overlay renderer
│   ├── export.py             ← Per-frame CSV/JSON export
│   ├── infer.py              ← CLI entry point
│   ├── modern.py             ← CVProject registry entry
│   ├── train.py              ← Dataset evaluation script
│   └── data_bootstrap.py     ← Idempotent dataset download
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- MediaPipe ≥ 0.10.14
- OpenCV ≥ 4.10
- NumPy ≥ 1.26
- scikit-learn ≥ 1.5
- PyYAML ≥ 6.0

## License

See repository root LICENSE.
