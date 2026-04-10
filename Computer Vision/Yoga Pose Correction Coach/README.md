# Yoga Pose Correction Coach

Identifies **yoga poses** from body landmarks and provides
**rule-based correction hints** to help practitioners improve their
alignment.  Uses MediaPipe Pose (33 landmarks) for body detection
and heuristic angle-template matching for classification.

---

## Coaching Quality Disclaimer

> **This is not a substitute for a qualified yoga instructor.**
>
> - **Heuristic classifier.**  Poses are identified by matching
>   joint-angle signatures against hard-coded templates, not by a
>   trained model.  Accuracy is limited.
> - **Five poses only.**  Mountain, Warrior I, Warrior II,
>   Tree, and Downward Dog.
> - **Rule-based corrections.**  Hints are derived from simple
>   angle-vs-target comparisons.  They are generic suggestions,
>   not personalised coaching.
> - **Camera angle matters.**  Side-on or frontal views at
>   full-body distance give best results.
> - **Not suitable for medical or therapeutic guidance** without
>   independent validation by a professional.
> - **Single person.**  Only one body is analysed per frame.

---

## Features

| Capability | Module | Detail |
|------------|--------|--------|
| Pose detection | `pose_detector.py` | MediaPipe Pose (33 landmarks) |
| Angle computation | `angle_calculator.py` | Pure math — 3-point, vertical, horizontal |
| Pose classification | `pose_classifier.py` | Angle-template matching with Gaussian scoring |
| Correction hints | `correction_engine.py` | Per-pose joint-angle rule checks |
| Smoothing | `smoother.py` | Majority-vote on pose labels |
| Overlays | `visualize.py` | Skeleton, pose label, confidence, hints panel |

## Supported Poses

| Pose | Sanskrit | Key Angles Checked |
|------|----------|-------------------|
| Mountain | Tadasana | Knees straight, torso upright |
| Warrior I | Virabhadrasana I | Front knee ~90°, arms raised |
| Warrior II | Virabhadrasana II | Front knee ~90°, arms horizontal |
| Tree | Vrksasana | Standing leg straight, torso upright, arms up |
| Downward Dog | Adho Mukha Svanasana | Arms/legs straight, hips high (inverted V) |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Webcam (press 'q' to quit, 'r' to reset)
cd "Yoga Pose Correction Coach/Source Code"
python infer.py --source 0

# Video file
python infer.py --source yoga_session.mp4

# Single image
python infer.py --source warrior.jpg

# Headless with CSV export
python infer.py --source 0 --no-display --export-csv yoga.csv
```

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source            '0' for webcam, or path to video/image

Optional:
  --config            YAML/JSON config path
  --no-smoothing      Disable majority-vote smoothing
  --no-display        Headless mode
  --export-csv        CSV export path
  --export-json       JSON export path
  --save-annotated    Save annotated frames
  --output-dir        Output directory (default: output/)
  --force-download    Re-download dataset
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset smoother |

## Configuration

```yaml
# MediaPipe Pose
model_complexity: 2          # 0, 1, or 2
min_detection_confidence: 0.6
min_tracking_confidence: 0.5

# Classification
min_visibility: 0.5
confidence_threshold: 0.4

# Smoothing
enable_smoothing: true
vote_window: 7

# Correction hints
angle_tolerance: 15        # degrees — deviation before suggesting a fix
max_hints: 3               # maximum hints per frame

# Display
show_skeleton: true
show_angles: true
show_pose_label: true
show_confidence: true
show_corrections: true
```

## How It Works

### Angle-Template Classification

For each supported pose, a template defines expected joint angles
with weights.  The classifier computes all angles from the current
pose and scores each template using Gaussian similarity:

$$
s_i = \exp\!\left(-\frac{(\theta_{\text{measured}} - \theta_{\text{expected}})^2}{\sigma^2}\right)
$$

The overall score for a template is the weighted average of its
checkpoint scores.  The pose with the highest score above the
confidence threshold wins.

Symmetric poses are handled by checking both left and right sides
for each checkpoint and taking the better match.

### Correction Engine

Once a pose is identified, the correction engine evaluates each
checkpoint's angle against the target:

$$
\text{hint triggered if } |\theta - \theta_{\text{target}}| > \text{tolerance}
$$

Each checkpoint carries two hint texts — one for "too low" and one
for "too high" — so the feedback is directional.  Hints are sorted
by severity (major > minor) and capped at `max_hints`.

### Temporal Smoothing

A majority vote over the last $N$ frames (default 7) prevents
rapid flicker between similar poses, giving stable on-screen labels.

## Adding a New Pose

1. **Define the template** in `pose_classifier.py`'s `_TEMPLATES`:
   list of `(angle_key, expected_degrees, weight)` tuples.

2. **Add correction rules** in `correction_engine.py`'s
   `_CORRECTION_RULES`: list of dicts with `key`, `joint`,
   `expected`, `too_low`, and `too_high` texts.

3. **Register the name** in `config.py`'s `YOGA_POSES` list.

## Project Structure

```
Yoga Pose Correction Coach/
├── Source Code/
│   ├── config.py              # YogaConfig dataclass
│   ├── pose_detector.py       # MediaPipe Pose (33 landmarks)
│   ├── angle_calculator.py    # Pure angle math (reusable)
│   ├── pose_classifier.py     # Angle-template pose classifier
│   ├── correction_engine.py   # Rule-based correction hints
│   ├── smoother.py            # Majority-vote smoothing
│   ├── controller.py          # Pipeline orchestrator
│   ├── validator.py           # Quality-check validator
│   ├── visualize.py           # Overlay renderer
│   ├── export.py              # Per-frame CSV/JSON export
│   ├── infer.py               # CLI entry point
│   ├── modern.py              # CVProject registry entry
│   ├── train.py               # Dataset evaluation script
│   └── data_bootstrap.py      # Idempotent dataset download
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- MediaPipe ≥ 0.10.14
- OpenCV ≥ 4.10
- NumPy ≥ 1.26
