# Gaze Direction Estimator

Coarse gaze-direction estimation using **MediaPipe Face Mesh** iris
landmarks.  Classifies gaze into **LEFT / RIGHT / UP / DOWN / CENTER**
with optional calibration and temporal smoothing.

---

## Approximation Limits

> **This is a heuristic estimator, not a precision eye tracker.**
>
> - **Coarse output only.** The system outputs one of five cardinal
>   directions — it does not estimate a continuous gaze point or
>   screen coordinate.
> - **Iris-ratio method.** Gaze is inferred from the position of
>   the iris center relative to the eye-contour bounding box.
>   This is a geometric proxy, not a learned gaze model.
> - **Head-pose coupling.** Head rotation shifts the iris position
>   in the same way as eye movement.  The classifier cannot
>   distinguish between a head turn and a gaze shift without
>   explicit head-pose compensation (not included by default).
> - **Lighting sensitivity.** Iris detection degrades under
>   strong side-lighting, backlighting, or low-contrast conditions.
> - **Glasses and contacts.** Reflective lenses, thick frames, or
>   tinted lenses can occlude or distort iris landmarks.
> - **Camera angle matters.** Best results are achieved with a
>   front-facing camera at eye level, approximately 40–80 cm away.
> - **Calibration helps.** Use `--calibrate` to capture personal
>   baselines and improve classification for your specific setup.
> - **Not suitable for accessibility or medical use** without
>   independent validation.

---

## Features

| Capability | Module | Method |
|------------|--------|--------|
| Iris position | `iris_locator.py` | MediaPipe iris landmarks 468–477 |
| Gaze classification | `gaze_classifier.py` | Threshold-based ratio classification |
| Calibration | `calibrator.py` | 5-position baseline capture |
| Smoothing | `smoother.py` | EMA on ratios + majority vote |
| Per-frame export | `export.py` | CSV and JSON metrics |
| Real-time overlay | `visualize.py` | Iris markers, direction label, ratios |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Webcam (press 'q' to quit)
cd "Gaze Direction Estimator/Source Code"
python infer.py --source 0

# With calibration (webcam only)
python infer.py --source 0 --calibrate

# Video file
python infer.py --source recording.mp4

# Single image
python infer.py --source face.jpg

# Headless with CSV export
python infer.py --source 0 --no-display --export-csv gaze.csv
```

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source            '0' for webcam, or path to video/image

Optional:
  --config            Path to YAML/JSON config file
  --calibrate         Run interactive 5-position calibration (webcam only)
  --calibration-file  Path to save/load calibration JSON
  --no-smoothing      Disable EMA + majority-vote smoothing
  --no-display        Disable GUI windows
  --export-csv        CSV export path for per-frame metrics
  --export-json       JSON export path for per-frame metrics
  --save-annotated    Save annotated frames to output dir
  --output-dir        Output directory (default: output/)
  --force-download    Force dataset re-download
```

## Calibration

Interactive calibration captures iris position baselines at five
gaze positions (center, left, right, up, down).  This compensates
for individual eye geometry and camera placement.

```bash
# Calibrate and save offsets
python infer.py --source 0 --calibrate --calibration-file cal.json

# Reuse saved calibration
python infer.py --source 0 --calibration-file cal.json
```

During calibration:
1. On-screen prompt shows the target direction
2. Look in the indicated direction
3. Press **SPACE** to begin collecting samples
4. Hold gaze until the counter reaches the target
5. Repeat for all five positions

## Configuration

All tunables are configurable via YAML/JSON:

```yaml
# Gaze classification thresholds
horiz_left_threshold: 0.38
horiz_right_threshold: 0.62
vert_up_threshold: 0.38
vert_down_threshold: 0.62

# Smoothing
enable_smoothing: true
ema_alpha: 0.4        # higher = more responsive, more jittery
vote_window: 5        # majority-vote window size

# MediaPipe
refine_landmarks: true   # REQUIRED for iris landmarks
```

Load a config: `python infer.py --source 0 --config my_config.yaml`

## How It Works

### Iris Ratio Method

The iris center (MediaPipe landmark 468 / 473) is located within
the eye-contour bounding box.  Position is expressed as a ratio:

- **Horizontal ratio** ($h$): 0.0 = leftmost, 0.5 = center, 1.0 = rightmost
- **Vertical ratio** ($v$): 0.0 = topmost, 0.5 = center, 1.0 = bottommost

Both eyes are averaged for robustness.

### Classification

Direction is assigned by threshold comparison:

$$
\text{direction} = \begin{cases}
\text{LEFT}   & \text{if } h < 0.38 \\
\text{RIGHT}  & \text{if } h > 0.62 \\
\text{UP}     & \text{if } v < 0.38 \\
\text{DOWN}   & \text{if } v > 0.62 \\
\text{CENTER} & \text{otherwise}
\end{cases}
$$

Horizontal shifts take priority when both axes are off-center,
because lateral iris displacement is more reliably detected.

### Smoothing

Two layers reduce jitter:

1. **EMA** (exponential moving average) with configurable
   $\alpha$ on the raw ratios.
2. **Majority vote** over a sliding window of $N$ classified
   directions.

## Project Structure

```
Gaze Direction Estimator/
├── Source Code/
│   ├── config.py           # GazeConfig dataclass
│   ├── landmark_engine.py  # MediaPipe Face Mesh (478 landmarks)
│   ├── iris_locator.py     # Iris center → eye-box ratios
│   ├── gaze_classifier.py  # Ratio → LEFT/RIGHT/UP/DOWN/CENTER
│   ├── calibrator.py       # Optional 5-position calibration
│   ├── smoother.py         # EMA + majority-vote smoothing
│   ├── analyzer.py         # GazePipeline orchestrator
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
- MediaPipe ≥ 0.10.14 (with `refine_landmarks=True` for iris)
- OpenCV ≥ 4.10
- NumPy ≥ 1.26

## License

See repository root LICENSE.
