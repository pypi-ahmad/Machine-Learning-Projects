# Cell Counting Instance Segmentation

Segment and **count** individual cells or nuclei in microscopy images using **YOLO26m-seg** instance segmentation, with post-processing to handle touching objects.

---

## Features

| Capability | Details |
|---|---|
| **Cell Segmentation** | Pixel-level instance masks from images, video, or webcam |
| **Automated Counting** | Per-image cell count with centroid markers |
| **Touching-Cell Splitting** | Watershed-based separation of merged objects |
| **Small-Object Filtering** | Configurable minimum area threshold |
| **Overlap Merging** | IoU-based deduplication of double-counted cells |
| **Per-Image Statistics** | Count, coverage, mean/median/min/max area, confidence |
| **Visual Reports** | Mask overlay with contours, centroids, IDs, count badge |
| **Structured Export** | JSON per image, CSV batch, mask PNGs |
| **Public Dataset** | Auto-download from Kaggle (`gangadhar/nuclei-segmentation-in-microscope-cell-images`) |

---

## Counting Pipeline

```
YOLO26m-seg  →  Raw Instances
                    │
            ┌───────┴───────┐
            │ 1. Filter     │  remove instances < min_area_px
            │ 2. Merge      │  merge IoU ≥ merge_overlap
            │ 3. Watershed  │  split touching cells
            └───────┬───────┘
                    │
             Post-processed Instances  →  count + metrics
```

### Post-Processing Steps

1. **Small-object filter** — discard detections with area < `min_area_px` (default 64).
2. **Overlap merging** — merge pairs whose mask IoU ≥ `merge_overlap` (default 0.60) to prevent double-counting.
3. **Watershed splitting** — for masks containing multiple connected components after erosion, apply OpenCV watershed to separate touching cells into individual instances.

---

## Metrics

$$\text{cell\_coverage} = \frac{\sum \text{cell\_area\_px}}{\text{total\_image\_px}}$$

Per-image output includes:

- **cell_count** — number of segmented instances after post-processing
- **raw_count** — number before post-processing (for comparison)
- **mean / median / min / max cell area** in pixels
- **cell_coverage** — fraction of image occupied by cells
- **mean_confidence** — average YOLO confidence across instances

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single image
python "Source Code/infer.py" --source cells.png

# Webcam
python "Source Code/infer.py" --source 0

# Directory with full export
python "Source Code/infer.py" --source images/ \
    --save-annotated --save-masks --export-csv counts.csv

# Disable watershed (faster, may under-separate touching cells)
python "Source Code/infer.py" --source cells.png --no-watershed
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
| `--no-watershed` | off | Disable watershed splitting |
| `--force-download` | off | Re-download dataset |

---

## Training

```bash
# Fine-tune YOLO26m-seg on cell/nuclei data
python "Source Code/train.py" --data path/to/data.yaml --epochs 80

# Evaluate on dataset images
python "Source Code/train.py" --eval

# Force re-download
python "Source Code/train.py" --eval --force-download
```

---

## Project Structure

```
Cell Counting Instance Segmentation/
├── Source Code/
│   ├── config.py           # CellConfig dataclass + loader
│   ├── segmentation.py     # CellSegmenter — YOLO26m-seg wrapper
│   ├── counting.py         # Post-processing: filter, merge, watershed
│   ├── metrics.py          # CellMetrics computation
│   ├── visualize.py        # Overlay rendering + count badge
│   ├── export.py           # JSON / CSV / mask export
│   ├── validator.py        # Input validation
│   ├── controller.py       # CellController (seg → counting → metrics)
│   ├── infer.py            # CLI entry point
│   ├── train.py            # Training + evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**Nuclei Segmentation in Microscope Cell Images** — [Kaggle](https://www.kaggle.com/datasets/gangadhar/nuclei-segmentation-in-microscope-cell-images)

- Collection of nuclei segmentation datasets in COCO format
- Sources: DSB 2018, BBBC006/007/018/020, TNBC, ISBI 2009, and more
- ~8,500 files with instance-level annotations
- ~337 MB
- Used to achieve 10th place in 2018 Data Science Bowl

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

# Counting post-processing
min_area_px: 64
merge_overlap: 0.60
watershed_split: true

# Visualisation
mask_alpha: 0.35
```

---

## Requirements

- Python 3.10+
- ultralytics >= 8.3.0
- opencv-python >= 4.10.0
- numpy >= 1.26.0
- torch >= 2.0.0
