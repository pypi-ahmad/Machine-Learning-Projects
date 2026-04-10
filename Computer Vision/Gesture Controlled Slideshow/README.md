# Gesture Controlled Slideshow

Control a slide presentation with **hand gestures** using MediaPipe
Hands.  Recognises five gestures from finger state and maps each to
a slideshow action — no special hardware required, just a webcam.

---

## Limitations

> **This is a heuristic gesture classifier, not a trained model.**
>
> - **Finger-state only.**  Gestures are inferred from which fingers
>   are extended or curled.  No temporal motion (swipe) is detected.
> - **Single hand.**  Only the first detected hand is processed.
> - **Background clutter.**  Complex backgrounds or skin-tone objects
>   may cause false detections.
> - **Lighting sensitivity.**  Low-light or strong backlighting
>   degrades hand-landmark accuracy.
> - **Camera angle.**  Front-facing webcam at roughly arm's length
>   gives the best results.
> - **Debouncing provides stability** but adds a small reaction
>   delay (~0.8 s default).

---

## Gesture Vocabulary

| Gesture | Fingers | Default Action |
|---------|---------|----------------|
| OPEN_PALM | All 5 extended | Next slide |
| FIST | 0 fingers extended | Previous slide |
| PEACE | Index + middle | Pause / toggle |
| POINTING | Index only | Pointer mode |
| THUMBS_UP | Thumb only (upright) | Resume |

All mappings are configurable via `gesture_map` in the config.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Demo mode — generates coloured placeholder slides (press 'q' to quit)
cd "Gesture Controlled Slideshow/Source Code"
python infer.py --source 0

# With your own slides
python infer.py --source 0 --slides /path/to/slide/images/

# Camera-only mode (no slide view, just gesture overlay)
python infer.py --source 0 --cam-only

# Video file input
python infer.py --source gestures.mp4

# Single image
python infer.py --source photo.jpg
```

## CLI Reference

```
python infer.py --source SOURCE [options]

Required:
  --source            '0' for webcam, or path to video/image

Optional:
  --slides            Directory of slide images (PNG/JPG)
  --cam-only          Show camera feed with gesture overlay only
  --config            Path to YAML/JSON config file
  --no-display        Disable GUI windows
  --export-csv        CSV export path for per-frame metrics
  --export-json       JSON export path for per-frame metrics
  --save-annotated    Save annotated frames to output dir
  --output-dir        Output directory (default: output/)
  --force-download    Force dataset re-download
```

## Keyboard Fallback

Keys work alongside gestures:

| Key | Action |
|-----|--------|
| `n` | Next slide |
| `p` | Previous slide |
| `SPACE` | Pause / resume |
| `t` | Toggle pointer mode |
| `q` | Quit |

## Configuration

All tunables are configurable via YAML/JSON:

```yaml
# MediaPipe hand detection
max_num_hands: 1
model_complexity: 1
min_detection_confidence: 0.6
min_tracking_confidence: 0.5

# Debouncing
debounce_sec: 0.8          # cooldown between actions
stable_frames: 3           # consecutive same-gesture frames required
confidence_threshold: 0.6  # minimum detection confidence

# Gesture mapping (gesture name → slideshow action)
gesture_map:
  OPEN_PALM: next
  FIST: previous
  PEACE: pause
  POINTING: pointer
  THUMBS_UP: resume

# Slideshow
auto_advance_sec: 0.0      # 0 = manual only
loop: true

# Keyboard mapping
key_next: n
key_prev: p
key_pause: " "
key_pointer: t
```

Load a config: `python infer.py --source 0 --config my_config.yaml`

## How It Works

### Hand Landmark Detection

MediaPipe Hands produces 21 3D landmarks per detected hand.
Key landmarks used for finger-state detection:

- **Fingertips:** landmarks 8, 12, 16, 20
- **PIP joints:** landmarks 6, 10, 14, 18
- **Thumb:** tip (4), IP (3), MCP (2), wrist (0)

### Finger State Detection

A finger is considered **extended** when its tip is above
(lower y) its PIP joint:

$$
\text{extended}_i = \begin{cases}
\text{true} & \text{if } y_{\text{tip}} < y_{\text{PIP}} \\
\text{false} & \text{otherwise}
\end{cases}
$$

The thumb uses x-axis comparison adjusted for handedness
(left vs right hand).

### Gesture Classification

The gesture is determined by the pattern of extended fingers:

| Finger Count | Pattern | Gesture |
|:---:|---------|---------|
| 5 | All extended | OPEN_PALM |
| 0 | All curled | FIST |
| 2 | Index + middle | PEACE |
| 1 | Index only | POINTING |
| 1 | Thumb only + upright | THUMBS_UP |

### Debouncing

Two mechanisms prevent spurious triggers:

1. **Stability filter:** the same gesture must persist for
   `stable_frames` consecutive frames (default: 3).
2. **Cooldown timer:** after an action fires, no new action
   can trigger for `debounce_sec` seconds (default: 0.8).

### Display Modes

- **Slideshow mode** (default): slide image fills the window
  with a small camera PiP in the corner showing hand landmarks.
- **Camera-only mode** (`--cam-only`): full-frame camera feed
  with gesture overlay, slide counter, and action banners.

## Project Structure

```
Gesture Controlled Slideshow/
├── Source Code/
│   ├── config.py              # GestureConfig dataclass
│   ├── hand_detector.py       # MediaPipe Hands wrapper
│   ├── gesture_recognizer.py  # Finger state → gesture classifier
│   ├── debouncer.py           # Temporal debouncing + action mapping
│   ├── slideshow.py           # Slideshow state machine
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
