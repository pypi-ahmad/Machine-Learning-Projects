# Exercise Rep Counter

Pose-based **exercise repetition counter** using MediaPipe Pose.
Tracks joint angles through exercise stages and counts reps for
squats, push-ups, and bicep curls with EMA smoothing and
stability filtering.

---

## Limitations

> **Rule-based stage detection, not a learned exercise classifier.**
>
> - **Three exercises only.**  Squat, push-up, and bicep curl are
>   supported.  Adding a new exercise requires defining its joint
>   angle and thresholds in `exercise_rules.py`.
> - **Angle-threshold method.**  Stage transitions are determined by
>   comparing a single joint angle against fixed thresholds.  Partial
>   reps or unusual form may be miscounted.
> - **Single person.**  Only one body is analysed per frame.
> - **Camera angle matters.**  Best results with a side-on view
>   (perpendicular to the exercise plane) at full-body distance.
> - **Lighting and occlusion.**  Poor lighting, loose clothing, or
>   self-occlusion degrade pose detection.
> - **Not a form checker.**  The system counts reps but does not
>   evaluate exercise form quality.

---

## Features

| Capability | Module | Detail |
|------------|--------|--------|
| Pose detection | `pose_detector.py` | MediaPipe Pose (33 landmarks) |
| Angle computation | `angle_calculator.py` | Pure math: 3-point angle (reusable) |
| Exercise rules | `exercise_rules.py` | Stage logic per exercise (extensible) |
| Rep counting | `rep_counter.py` | State machine: down→up = 1 rep |
| Angle smoothing | `smoother.py` | EMA on raw angles before stage detection |
| Overlays | `visualize.py` | Skeleton, angle label, rep count, stage |
| Per-frame export | `export.py` | CSV and JSON metrics |

## Supported Exercises

| Exercise | Joint Angle | Down Threshold | Up Threshold |
|----------|-------------|:--------------:|:------------:|
| **Squat** | Hip–Knee–Ankle | ≤ 90° | ≥ 160° |
| **Push-Up** | Shoulder–Elbow–Wrist | ≤ 90° | ≥ 160° |
| **Bicep Curl** | Shoulder–Elbow–Wrist | ≥ 160° (extended) | ≤ 40° (curled) |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Webcam — squat counting (press 'q' to quit, 'r' to reset)
cd "Exercise Rep Counter/Source Code"
python infer.py --source 0 --exercise squat

# Push-ups
python infer.py --source 0 --exercise pushup

# Bicep curls (right arm)
python infer.py --source 0 --exercise bicep_curl --side right

# Video file
python infer.py --source workout.mp4 --exercise squat

# Headless with CSV export
python infer.py --source workout.mp4 --exercise squat --no-display --export-csv reps.csv
```

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source            '0' for webcam, or path to video/image

Optional:
  --exercise          squat | pushup | bicep_curl (default: squat)
  --side              left | right (default: left)
  --config            YAML/JSON config path
  --no-smoothing      Disable EMA angle smoothing
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
| `r` | Reset rep count and smoother |

## Configuration

```yaml
# MediaPipe Pose
model_complexity: 1
min_detection_confidence: 0.6
min_tracking_confidence: 0.5

# Exercise
exercise: squat

# Stage thresholds (degrees)
squat_down_angle: 90
squat_up_angle: 160
pushup_down_angle: 90
pushup_up_angle: 160
curl_down_angle: 160
curl_up_angle: 40

# Smoothing
enable_smoothing: true
ema_alpha: 0.4
stable_frames: 2

# Display
show_skeleton: true
show_angles: true
show_rep_count: true
show_stage: true
show_exercise: true
```

## How It Works

### Joint Angle Computation

The angle at the vertex $b$ between points $a$, $b$, $c$ is:

$$
\theta = \arccos\left(\frac{\vec{ba} \cdot \vec{bc}}{|\vec{ba}||\vec{bc}|}\right)
$$

This is computed in `angle_calculator.py` as a pure function —
no MediaPipe dependency, fully testable with synthetic points.

### Stage Detection

Each exercise defines a joint and two angle thresholds.
For a standard exercise (squat, push-up):

$$
\text{stage} = \begin{cases}
\text{down} & \text{if } \theta \leq \theta_{\text{down}} \\
\text{up}   & \text{if } \theta \geq \theta_{\text{up}} \\
\text{unknown} & \text{otherwise}
\end{cases}
$$

For bicep curls the logic is inverted (small angle = "up").

### Rep Counting

A rep is counted on a **down → up** transition (completion
of the concentric phase).  A stability filter requires the
same stage to persist for $N$ consecutive frames before
confirming the transition, preventing single-frame noise.

### Angle Smoothing

An EMA (exponential moving average) with configurable $\alpha$
is applied to the raw joint angle *before* stage detection:

$$
\hat{\theta}_t = \alpha \cdot \theta_t + (1 - \alpha) \cdot \hat{\theta}_{t-1}
$$

This smooths out landmark jitter without adding significant latency.

## Adding a New Exercise

1. Define the analysis function in `exercise_rules.py`:
   - Pick three landmarks forming the joint angle
   - Set down/up angle thresholds
   - Use `_analyse_angle_exercise()` for standard logic

2. Add thresholds to `config.py`

3. Register in `EXERCISE_REGISTRY`

4. Add the exercise name to `infer.py`'s `--exercise` choices

## Project Structure

```
Exercise Rep Counter/
├── Source Code/
│   ├── config.py              # ExerciseConfig dataclass
│   ├── pose_detector.py       # MediaPipe Pose (33 landmarks)
│   ├── angle_calculator.py    # Pure 3-point angle math (reusable)
│   ├── exercise_rules.py      # Exercise definitions + stage logic
│   ├── rep_counter.py         # Rep-counting state machine
│   ├── smoother.py            # EMA angle smoothing
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
