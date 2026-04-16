# Finger Counter Pro

Robust **multi-hand finger counting** using MediaPipe Hand Landmarker.
Counts 0–5 fingers per hand (0–10 total across two hands) with
per-finger state detection, left/right handedness support, and
temporal smoothing for stable output.

---

## Limitations

> **Heuristic finger-state detection, not a trained classifier.**
>
> - **Geometric method only.**  Finger "extended" is determined by
>   comparing tip vs PIP joint y-coordinates (x for thumb).
>   No learned gesture model is used.
> - **Two hands maximum.**  MediaPipe supports up to 2 simultaneous
>   hands; more will be ignored.
> - **Background clutter.**  Skin-tone objects or complex backgrounds
>   may cause false detections.
> - **Lighting.**  Low-light or strong backlighting degrades landmark
>   accuracy.
> - **Camera angle.**  Best results with a front-facing camera at
>   roughly arm's length.
> - **Smoothing adds latency.**  EMA + majority vote stabilises the
>   count but introduces ~150 ms response delay.

---

## Features

| Capability | Module | Detail |
|------------|--------|--------|
| Hand detection | `hand_detector.py` | MediaPipe Hand Landmarker, up to 2 hands |
| Finger counting | `finger_counter.py` | Per-finger tip vs PIP comparison |
| Thumb detection | `finger_counter.py` | x-axis, handedness-aware |
| Smoothing | `smoother.py` | EMA + majority vote per hand |
| Overlay | `visualize.py` | Skeleton, per-finger state, count |
| Per-frame export | `export.py` | CSV and JSON metrics |

## Sample Dataset

`train.py` auto-downloads a small public sample test set from
`cj-mills/pexel-hand-gesture-test-images` using an idempotent bootstrap.
Run `python train.py --force-download` to refresh the sample set.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Webcam — both hands (press 'q' to quit, 'r' to reset smoothing)
cd "Finger Counter Pro/Source Code"
python infer.py --source 0

# Video file
python infer.py --source recording.mp4

# Single image
python infer.py --source hand.jpg

# Headless with CSV export
python infer.py --source 0 --no-display --export-csv counts.csv

# Disable smoothing (raw counts)
python infer.py --source 0 --no-smoothing
```

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source            '0' for webcam, or path to video/image

Optional:
  --config            Path to YAML/JSON config file
  --no-smoothing      Disable EMA + majority-vote smoothing
  --no-display        Disable GUI windows (headless mode)
  --export-csv        CSV export path for per-frame metrics
  --export-json       JSON export path for per-frame metrics
  --save-annotated    Save annotated frames to output dir
  --output-dir        Output directory (default: output/)
  --force-download    Force dataset re-download
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset smoothing state |

## Configuration

All tunables via YAML/JSON:

```yaml
# MediaPipe
max_num_hands: 2
model_complexity: 1
min_detection_confidence: 0.6
min_presence_confidence: 0.5
min_tracking_confidence: 0.5

# Finger detection
finger_up_margin: 0.02    # y-gap threshold

# Smoothing
enable_smoothing: true
ema_alpha: 0.35           # higher = more responsive
vote_window: 5            # majority-vote window

# Display
show_landmarks: true
show_finger_state: true
show_count: true
show_handedness: true
```

Load a config: `python infer.py --source 0 --config my_config.yaml`

## How It Works

### Hand Landmark Detection

MediaPipe Hand Landmarker detects 21 3D landmarks per hand.
Key landmarks for finger state:

- **Fingertips:** 8 (index), 12 (middle), 16 (ring), 20 (pinky)
- **PIP joints:** 6, 10, 14, 18
- **Thumb:** tip (4), IP (3), MCP (2)
- **Wrist:** 0

### Finger State Detection

A finger is **extended** when its tip is above its PIP joint:

$$
\text{extended}_i = y_{\text{tip}} < y_{\text{PIP}} - \text{margin}
$$

The **thumb** uses x-axis comparison, adjusted for handedness:

$$
\text{thumb\_extended} = \begin{cases}
x_{\text{tip}} < x_{\text{IP}} & \text{if right hand} \\
x_{\text{tip}} > x_{\text{IP}} & \text{if left hand}
\end{cases}
$$

### Smoothing

Two layers reduce jitter in the finger count:

1. **EMA** (exponential moving average) with configurable
   $\alpha$ on the raw count per hand.
2. **Majority vote** over a sliding window of $N$ rounded
   EMA values.

### Testability

The core counting functions (`is_finger_extended`,
`is_thumb_extended`, `detect_fingers`, `count_fingers`) are
pure functions that take plain coordinates and return booleans
or integers — no MediaPipe dependency, easy to unit-test.

## Project Structure

```
Finger Counter Pro/
├── Source Code/
│   ├── config.py           # FingerCounterConfig dataclass
│   ├── hand_detector.py    # MediaPipe Hand Landmarker (multi-hand)
│   ├── finger_counter.py   # Core finger-state logic (testable)
│   ├── smoother.py         # EMA + majority-vote smoothing
│   ├── controller.py       # Pipeline orchestrator
│   ├── validator.py        # Quality-check validator
│   ├── visualize.py        # Overlay renderer
│   ├── export.py           # Per-frame CSV/JSON export
│   ├── infer.py            # CLI entry point
│   ├── modern.py           # CVProject registry entry
│   ├── train.py            # Dataset evaluation script
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- MediaPipe ≥ 0.10.14
- OpenCV ≥ 4.10
- NumPy ≥ 1.26
- huggingface_hub ≥ 0.30
