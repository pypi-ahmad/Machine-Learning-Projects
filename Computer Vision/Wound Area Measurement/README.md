# Wound Area Measurement

Segment wound regions and estimate **relative wound area** from images using **YOLO26m-seg** instance segmentation — with optional multi-image change tracking to monitor wound size over time.

> **DISCLAIMER — NOT A MEDICAL DEVICE**
>
> This software produces **relative pixel-based area estimates only**.
> It is provided for **informational and research purposes** and is
> **NOT** intended to be used as a medical device, diagnostic tool, or
> substitute for professional medical advice.  **Do not** use outputs
> from this tool to make clinical decisions.  Always consult a qualified
> healthcare professional for wound assessment and treatment.
>
> All area measurements are expressed in pixels unless an external
> calibration reference is provided.  No absolute physical units
> (cm², mm²) are derived by this tool.

---

## Features

| Capability | Details |
|---|---|
| **Wound Segmentation** | Pixel-level wound masks from images, video, or webcam |
| **Per-Image Metrics** | Wound area (px), coverage ratio, instance count, mean confidence |
| **Change Tracking** | Multi-image series: per-step deltas, net change, peak area |
| **Visual Reports** | Wound mask overlay with contours, bboxes, stats panel, change summary |
| **Structured Export** | JSON per image, series summary JSON, CSV batch, mask PNGs |
| **Public Dataset** | Auto-download from Kaggle (`leoscode/wound-segmentation-images`) |

---

## Metrics

### Per-Image

$$\text{wound\_coverage} = \frac{\text{wound\_area\_px}}{\text{total\_image\_px}}$$

### Multi-Image Change Tracking

When processing a directory with `--track-changes`, the tool records a series:

$$\Delta_i = \text{area}_i - \text{area}_{i-1}$$

$$\text{net\_change\_ratio} = \frac{\text{area}_{final} - \text{area}_{initial}}{\text{area}_{initial}}$$

A negative net change ratio indicates wound area reduction (healing trend); positive indicates expansion.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single image
python "Source Code/infer.py" --source wound.jpg

# Webcam
python "Source Code/infer.py" --source 0

# Directory with change tracking
python "Source Code/infer.py" --source patient_images/ \
    --track-changes --save-annotated --export-csv wound_stats.csv

# Full export
python "Source Code/infer.py" --source wound.jpg \
    --export-json report.json --save-annotated --save-masks
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
| `--save-masks` | off | Save binary mask PNGs |
| `--track-changes` | off | Track wound area changes across images in a directory |
| `--force-download` | off | Re-download dataset |

---

## Change Tracking

When `--track-changes` is used with a directory source, images are processed in sorted filename order.  The tool assumes all images depict the **same wound** across time points.

Outputs include:
- Per-image delta (change vs. previous image)
- Series summary: initial area, final area, net change, peak area
- `series_summary.json` — full tracking report

---

## Training

```bash
# Fine-tune YOLO26m-seg on wound data
python "Source Code/train.py" --data path/to/data.yaml --epochs 80

# Evaluate on dataset images
python "Source Code/train.py" --eval

# Force re-download
python "Source Code/train.py" --eval --force-download
```

---

## Project Structure

```
Wound Area Measurement/
├── Source Code/
│   ├── config.py           # WoundConfig dataclass + YAML/JSON loader
│   ├── segmentation.py     # WoundSegmenter — YOLO26m-seg wrapper
│   ├── metrics.py          # Per-image WoundMetrics computation
│   ├── change_tracker.py   # Multi-image change tracking + summary
│   ├── visualize.py        # Overlay rendering + change summary panel
│   ├── export.py           # JSON / CSV / mask export
│   ├── validator.py        # Input validation
│   ├── controller.py       # WoundController (seg → metrics → tracking)
│   ├── infer.py            # CLI entry point
│   ├── train.py            # Training + evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**Wound Images Segmentation** — [Kaggle](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images)

- 2,760 wound images with pixel-level segmentation masks
- Sources: Medetec, FUSeg Challenge, WSNet
- Pre-split: 2,208 train / 552 test
- 512×512 resolution, MIT licence
- ~715 MB

```bash
# Auto-download via bootstrap
python "Source Code/data_bootstrap.py"

# Force re-download
python "Source Code/data_bootstrap.py" --force-download
```

---

## Configuration

```yaml
model_name: yolo26m-seg.pt
confidence_threshold: 0.25
iou_threshold: 0.45
imgsz: 640
mask_alpha: 0.40
```

---

## Requirements

- Python 3.10+
- ultralytics >= 8.3.0
- opencv-python >= 4.10.0
- numpy >= 1.26.0
- torch >= 2.0.0

---

## Disclaimer

This project is for **educational and research purposes only**.
It does not provide medical advice.  All wound area values are
**relative pixel counts** that depend on image resolution, camera
distance, angle, and lighting.  No absolute physical measurements
are computed.  **Do not rely on this tool for clinical decisions.**
