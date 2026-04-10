# Waterbody & Flood Extent Segmentation

Detect water bodies and compare before/after flood extent in satellite and aerial imagery using **YOLO26m-seg** instance segmentation.

---

## Features

| Capability | Details |
|---|---|
| **Water Detection** | Segment water bodies from a single image, video, or webcam |
| **Flood Comparison** | Before/after analysis: classify every pixel as *new flooding*, *receded*, *permanent water*, or *dry* |
| **Coverage Metrics** | Per-image water area, coverage ratio, instance count, mean confidence |
| **Change Metrics** | IoU, flood expansion ratio, recession ratio, net change ratio, region counts |
| **Visual Reports** | Side-by-side overlay, change-map coloring, metrics panel |
| **Structured Export** | JSON per image/pair, CSV batch export, binary mask PNGs |
| **Public Dataset** | Auto-download from Kaggle (`franciscoescobar/satellite-images-of-water-bodies`) |

---

## Modes of Operation

### 1. Single-Image / Video / Webcam

Detects and segments water bodies, computes coverage:

$$\text{coverage\_ratio} = \frac{\text{water\_area\_px}}{\text{total\_image\_px}}$$

### 2. Before / After Flood Comparison

Pixel-level diff of two water masks with morphological cleanup:

| Class | Definition |
|---|---|
| `flooded_new` | Water in *after* image only |
| `receded` | Water in *before* image only |
| `permanent` | Water in both images |
| `dry` | No water in either image |

$$\text{IoU} = \frac{|\text{before} \cap \text{after}|}{|\text{before} \cup \text{after}|}$$

$$\text{net\_change\_ratio} = \frac{\text{after\_water\_px} - \text{before\_water\_px}}{\text{total\_image\_px}}$$

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single image — detect water
python "Source Code/infer.py" --source satellite.jpg

# Webcam
python "Source Code/infer.py" --source 0

# Before/after flood pair
python "Source Code/infer.py" --before pre.jpg --after post.jpg

# Batch comparison
python "Source Code/infer.py" --before-dir images/pre --after-dir images/post

# Full export
python "Source Code/infer.py" --source satellite.jpg \
    --export-json report.json --export-csv stats.csv \
    --save-annotated --save-masks
```

---

## CLI Reference

### Single-Mode Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | — | `0` for webcam, or path to image/video/directory |
| `--config` | — | YAML/JSON config override |
| `--no-display` | off | Headless mode |
| `--output-dir` | `output` | Output directory |
| `--export-json` | — | JSON report path |
| `--export-csv` | — | CSV export path |
| `--save-annotated` | off | Save annotated images/video |
| `--save-masks` | off | Save binary mask PNGs |
| `--force-download` | off | Re-download dataset |

### Comparison-Mode Arguments

| Argument | Description |
|---|---|
| `--before` / `--after` | Single image pair |
| `--before-dir` / `--after-dir` | Batch: matched filenames in two directories |

---

## Training

```bash
# Fine-tune on water body data
python "Source Code/train.py" --data path/to/data.yaml --epochs 80

# Evaluate on dataset images
python "Source Code/train.py" --eval

# Force re-download dataset
python "Source Code/train.py" --eval --force-download
```

---

## Project Structure

```
Waterbody Flood Extent Segmentation/
├── Source Code/
│   ├── config.py           # FloodConfig dataclass + YAML/JSON loader
│   ├── segmentation.py     # WaterSegmenter — YOLO26m-seg wrapper
│   ├── flood_compare.py    # Before/after pixel-level comparison engine
│   ├── coverage.py         # CoverageMetrics + FloodChangeMetrics
│   ├── visualize.py        # Overlay rendering + comparison reports
│   ├── export.py           # JSON / CSV / mask export
│   ├── validator.py        # Input validation (image/video/pair/directory)
│   ├── controller.py       # WaterFloodController (single + pair modes)
│   ├── infer.py            # CLI entry point
│   ├── train.py            # Training + evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**Satellite Images of Water Bodies** — [Kaggle](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)

Satellite imagery with annotated water body masks for training and evaluation.

```bash
# Auto-download via bootstrap
python "Source Code/data_bootstrap.py"

# Or force re-download
python "Source Code/data_bootstrap.py" --force-download
```

---

## Configuration

All parameters are tuneable via `FloodConfig` or a YAML/JSON file:

```yaml
model_name: yolo26m-seg.pt
confidence_threshold: 0.30
iou_threshold: 0.45
imgsz: 640
morph_kernel_size: 5
min_change_area: 200
mask_alpha: 0.40
```

---

## Requirements

- Python 3.10+
- ultralytics >= 8.3.0
- opencv-python >= 4.10.0
- numpy >= 1.26.0
- torch >= 2.0.0
