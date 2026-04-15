# Driver Drowsiness Monitor

Real-time driver drowsiness detection using **MediaPipe Face Landmarker** landmarks.
Monitors eye closure (EAR), yawning (MAR), head pose, and generates
time-stamped alerts with configurable cooldowns.

---

## ⚠️ Safety Disclaimers

> **This project is a research prototype and educational demo.**
> It is **NOT** a certified safety system and must **NOT** be used as
> the sole or primary drowsiness detection mechanism in any vehicle.
>
> - **No warranty of accuracy.** False positives and false negatives
>   will occur under varying lighting, head accessories, sunglasses,
>   and camera placement.
> - **Do not drive drowsy.** If you feel tired, pull over and rest.
>   No software can replace responsible driving behavior.
> - **Supplemental use only.** This tool may be used alongside —
>   never instead of — proper rest, caffeinated breaks, or a
>   co-driver.
> - **Not ADAS / ISO 26262 compliant.** This project has not
>   undergone automotive safety certification.

---

## Features

| Signal | Metric | Default Threshold |
|--------|--------|-------------------|
| Blink detection | Eye Aspect Ratio (EAR) | 0.21 |
| Prolonged eye closure | Consecutive low-EAR frames | 15 frames |
| PERCLOS | % eye-closure in rolling window | 40 % over 60 s |
| Yawn detection | Mouth Aspect Ratio (MAR) | 0.55 |
| Head pose / distraction | Yaw / Pitch via solvePnP | 30° / 25° |
| Alert deduplication | Per-type cooldown | 10 s |

## Architecture

```
landmark_detector.py   MediaPipe Face Landmarker (dense face landmarks)
        │
   ┌────┴────┬──────────┐
   ▼         ▼          ▼
blink_tracker  yawn_tracker  head_pose
   │         │          │
   └────┬────┘──────────┘
        ▼
   alert_manager   (cooldown-based dedup)
        │
   ┌────┴────┐
   ▼         ▼
export     visualize
```

## Dataset

- **Source:** Hugging Face dataset `ckcl/driver-safety-dataset`
- **Prepared Path:** `data/driver_drowsiness_monitor/processed/media/`
- **Download:** Automatic on first `python train.py` or `python data_bootstrap.py` run
- **Force re-download:** `python train.py --force-download` or `python data_bootstrap.py --force`
- **Practical note:** the public dataset is image-based, while the project itself supports webcam, single-image, and video inputs at inference time.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Webcam (press 'q' to quit)
cd "Driver Drowsiness Monitor/Source Code"
python infer.py --source 0

# Prepare the evaluation dataset locally
python data_bootstrap.py

# Video file
python infer.py --source ~/driving_clip.mp4

# Headless with CSV export
python infer.py --source 0 --no-display --export-csv log.csv

# JSON metrics export
python infer.py --source video.mp4 --export-json metrics.json
```

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source          '0' for webcam, path to video/image

Optional:
  --config          Path to YAML/JSON config file
  --ear-threshold   Override EAR threshold (default 0.21)
  --mar-threshold   Override MAR threshold (default 0.55)
  --yaw-threshold   Override yaw distraction threshold (default 30°)
  --no-display      Disable GUI windows
  --export-csv      CSV export path for per-frame metrics
  --export-json     JSON export path for per-frame metrics
  --save-annotated  Save annotated frames to output dir
  --output-dir      Output directory (default: output/)
  --force-download  Force dataset re-download
```

## Configuration

All thresholds are tunable via YAML/JSON config:

```yaml
ear_threshold: 0.21
blink_consec_frames: 2
drowsy_eye_frames: 15
perclos_window_sec: 60.0
perclos_threshold: 0.40

mar_threshold: 0.55
yawn_consec_frames: 10
yawn_cooldown_sec: 5.0

yaw_threshold: 30.0
pitch_threshold: 25.0
distraction_consec_frames: 15

alert_cooldown_sec: 10.0
```

Load a config: `python infer.py --source 0 --config my_config.yaml`

## Detection Rules

### EAR — Eye Aspect Ratio

Computed from 6 eye-contour landmarks per eye. Drops below threshold
when eyelids close. A blink is registered when EAR stays below
`ear_threshold` for `blink_consec_frames` then rises.

### PERCLOS — Percentage of Eye Closure

Fraction of time eyes are closed over a rolling window
(default 60 s). Values above `perclos_threshold` (40 %) trigger a
drowsiness alert.

### MAR — Mouth Aspect Ratio

Ratio of vertical lip distances to horizontal lip width. Values
above `mar_threshold` sustained for `yawn_consec_frames` register
a yawn-like mouth-opening event. This is a fatigue proxy, not a clinical
yawn detector.

### Head Pose

6 facial landmarks projected to a 3D model via `cv2.solvePnP`.
Yaw or pitch exceeding thresholds for `distraction_consec_frames`
triggers a distraction alert.

## Project Structure

```
Driver Drowsiness Monitor/
├── Source Code/
│   ├── config.py             # DrowsinessConfig dataclass
│   ├── landmark_detector.py  # MediaPipe Face Landmarker wrapper
│   ├── blink_tracker.py      # EAR, blink count, PERCLOS
│   ├── yawn_tracker.py       # MAR, yawn detection
│   ├── head_pose.py          # solvePnP head-pose estimation
│   ├── alert_manager.py      # Cooldown-based alert dedup
│   ├── parser.py             # DrowsinessPipeline orchestrator
│   ├── validator.py          # Quality-check validator
│   ├── visualize.py          # Overlay renderer
│   ├── export.py             # Per-frame CSV/JSON export
│   ├── infer.py              # CLI entry point
│   ├── modern.py             # CVProject registry entry
│   ├── train.py              # Dataset eval script
│   └── data_bootstrap.py     # Idempotent dataset download
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- MediaPipe ≥ 0.10.14
- OpenCV ≥ 4.10
- NumPy ≥ 1.26
- PyYAML ≥ 6.0
- huggingface-hub ≥ 0.30

## License

See repository root LICENSE.
