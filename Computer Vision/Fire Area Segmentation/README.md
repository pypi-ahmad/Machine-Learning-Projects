# Fire Area Segmentation

Segment **fire** and optionally **smoke** regions in images and video using **YOLO26m-seg** instance segmentation — with per-frame area estimation, cross-frame trend tracking, and configurable alert levels.

---

## Features

| Capability | Details |
|---|---|
| **Fire Segmentation** | Pixel-level fire region masks from images, video, or webcam |
| **Smoke Segmentation** | Optional smoke regions (disable with `--no-smoke`) |
| **Per-Frame Metrics** | Fire/smoke area (px), coverage ratio, instance count, mean confidence |
| **Trend Tracking** | Rolling-window summaries: avg/peak coverage, growth rate per frame |
| **Alert Levels** | `none → low → medium → high → critical` based on coverage + growth |
| **Visual Reports** | Fire/smoke overlays with legend, bounding boxes, trend panel |
| **Structured Export** | JSON per frame, CSV batch, binary mask PNGs |
| **Public Dataset** | Auto-download from Kaggle (`diversisai/fire-segmentation-image-dataset`) |

---

## Pipeline Architecture

```
Frame → FireSegmenter (YOLO26m-seg)
          ↓
      SegmentationResult (fire_mask, smoke_mask, instances)
          ↓
      FrameMetrics (area, coverage, counts)
          ↓
      TrendTracker (rolling window)
          ↓
      AlertEvaluator (independent module)
          ↓
      FrameResult (segmentation + metrics + trend + alert)
```

Segmentation and alert logic are **separate modules** — alert thresholds and upgrade rules can be changed without touching the segmentation pipeline.

---

## Metrics

### Per-Frame

$$\text{fire\_coverage} = \frac{\text{fire\_area\_px}}{\text{total\_image\_px}}$$

### Cross-Frame Trend

$$\text{fire\_growth\_rate} = \frac{\text{coverage}_{t} - \text{coverage}_{t-N}}{N}$$

where $N$ is the rolling window size (default 30 frames).

### Alert Levels

| Level | Fire Coverage Threshold |
|---|---|
| `critical` | ≥ 25% |
| `high` | ≥ 10% |
| `medium` | ≥ 3% |
| `low` | ≥ 0.5% |
| `none` | < 0.5% |

Alert level upgrades by one step when fire growth rate exceeds +0.5%/frame.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single image
python "Source Code/infer.py" --source fire.jpg

# Webcam (live)
python "Source Code/infer.py" --source 0

# Video with export
python "Source Code/infer.py" --source wildfire.mp4 \
    --save-annotated --export-csv fire_stats.csv

# Directory batch
python "Source Code/infer.py" --source images/ \
    --save-annotated --save-masks --export-json reports/

# Fire only (no smoke)
python "Source Code/infer.py" --source fire.jpg --no-smoke
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | `0` for webcam, or path to image/video/directory |
| `--config` | — | YAML/JSON config override |
| `--no-display` | off | Headless mode |
| `--output-dir` | `output` | Output directory |
| `--export-json` | — | JSON report path |
| `--export-csv` | — | CSV export path |
| `--save-annotated` | off | Save annotated images/video |
| `--save-masks` | off | Save binary fire/smoke mask PNGs |
| `--no-smoke` | off | Disable smoke segmentation |
| `--force-download` | off | Re-download dataset |

---

## Training

```bash
# Fine-tune YOLO26m-seg on fire data
python "Source Code/train.py" --data path/to/data.yaml --epochs 80

# Evaluate on dataset images
python "Source Code/train.py" --eval

# Force re-download
python "Source Code/train.py" --eval --force-download
```

---

## Project Structure

```
Fire Area Segmentation/
├── Source Code/
│   ├── config.py           # FireConfig dataclass + YAML/JSON loader
│   ├── segmentation.py     # FireSegmenter — YOLO26m-seg wrapper
│   ├── metrics.py          # Per-frame FrameMetrics computation
│   ├── trend.py            # TrendTracker — rolling window summaries
│   ├── alert.py            # AlertLevel evaluation (separate from seg)
│   ├── visualize.py        # Overlay rendering + trend panel
│   ├── export.py           # JSON / CSV / mask export
│   ├── validator.py        # Input validation
│   ├── controller.py       # FireController (seg → metrics → trend → alert)
│   ├── infer.py            # CLI entry point
│   ├── train.py            # Training + evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**Fire Segmentation Image Dataset** — [Kaggle](https://www.kaggle.com/datasets/diversisai/fire-segmentation-image-dataset)

- 27,500+ fire images with pixel-level segmentation masks
- 11,400+ negative (no-fire) images
- CC0 Public Domain license
- ~508 MB

```bash
# Auto-download via bootstrap
python "Source Code/data_bootstrap.py"

# Or force re-download
python "Source Code/data_bootstrap.py" --force-download
```

---

## Configuration

All parameters are tuneable via `FireConfig` or a YAML/JSON file:

```yaml
model_name: yolo26m-seg.pt
confidence_threshold: 0.30
iou_threshold: 0.45
imgsz: 640
class_names: [fire, smoke]
enable_smoke: true
trend_window: 30
mask_alpha: 0.45
```

---

## Requirements

- Python 3.10+
- ultralytics >= 8.3.0
- opencv-python >= 4.10.0
- numpy >= 1.26.0
- torch >= 2.0.0
