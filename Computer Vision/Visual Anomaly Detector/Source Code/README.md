# Visual Anomaly Detector

> **Task:** Anomaly Detection &nbsp;|&nbsp; **Key:** `visual_anomaly_detector` &nbsp;|&nbsp; **Framework:** ResNet + Mahalanobis / k-NN

---

## Overview

Unsupervised visual anomaly detection for industrial images. Trains exclusively on normal (non-defective) images using a one-class approach: extract CNN features, model the normal distribution, then score new images by distance from the learned normal. Supports Mahalanobis distance, k-NN scoring, automatic threshold selection, and patch-level anomaly heatmaps.

## Architecture

```
┌──────────┐    ┌───────────────────┐    ┌──────────────┐    ┌────────────┐
│  Image   │───▶│ FeatureExtractor  │───▶│ AnomalyScorer│───▶│ Threshold  │
│          │    │ (ResNet backbone)  │    │ (Mahal/k-NN) │    │ Decision   │
└──────────┘    └───────────────────┘    └──────┬───────┘    └─────┬──────┘
                                                │                  │
                                                ▼                  ▼
                                         ┌────────────┐    ┌────────────┐
                                         │  Heatmap   │    │ Visualize  │
                                         │ Generator  │    │  & Export  │
                                         └────────────┘    └────────────┘
```

| Component | Responsibility |
|---|---|
| `feature_extractor.py` | ResNet backbone, image → feature vector (512 or 2048-D) |
| `scorer.py` | Mahalanobis + k-NN anomaly scoring, model persistence (NPZ) |
| `threshold.py` | Percentile, mean+σ, F1-optimal threshold selection, AUROC, sweep |
| `heatmap.py` | Patch-level sliding-window anomaly heatmaps |
| `visualize.py` | Annotated frame overlays, score bars, batch summaries |
| `config.py` | Dataclass configuration with YAML/JSON loading |

## Dataset

**MVTec Anomaly Detection** (Hugging Face: `alexrods/mvtec-ad`)
- Industrial product images: textures and objects
- Structure: `train/good/` (normal) + `test/good/` + `test/<defect_type>/`
- Standard benchmark for unsupervised anomaly detection
- Auto-downloads on first run

## Quick Start

### Training

```bash
cd "Visual Anomaly Detector/Source Code"

# Train with auto-download + auto-threshold
python train.py

# Custom backbone and scoring
python train.py --backbone resnet50 --scoring knn

# Fixed threshold (disable auto)
python train.py --threshold 4.0

# Force re-download dataset
python train.py --force-download

# Custom dataset path
python train.py --data /path/to/anomaly/dataset
```

### Inference — Single Image

```bash
# Score a single image (with display)
python infer.py --source image.png

# With anomaly heatmap
python infer.py --source image.png --heatmap

# Save annotated result
python infer.py --source image.png --save-dir output/ --no-display
```

### Inference — Folder

```bash
# Score all images in a folder
python infer.py --source test_images/

# Export results
python infer.py --source test_images/ --export results.json
python infer.py --source test_images/ --export results.csv --no-display

# With custom threshold
python infer.py --source test_images/ --threshold 5.0 --limit 100
```

### Python API

```python
from modern import VisualAnomalyDetectorModern
from config import AnomalyConfig

# Load
cfg = AnomalyConfig(backbone="resnet18", scoring_method="mahalanobis")
proj = VisualAnomalyDetectorModern(config=cfg)
proj.load()

# Predict
result = proj.predict("image.png")
print(f"{result['label']}: score={result['anomaly_score']:.2f}")

# Heatmap
overlay = proj.generate_heatmap("image.png")

# Visualize
annotated = proj.visualize("image.png", result)
```

### Threshold Utilities

```python
from threshold import ThresholdSelector
import numpy as np

sel = ThresholdSelector()

# Percentile-based (normal scores only)
threshold = sel.percentile(normal_scores, pct=95)

# Mean + n*sigma
threshold = sel.mean_std(normal_scores, n_sigma=3.0)

# F1-optimal (needs both normal and anomaly scores)
threshold = sel.f1_optimal(normal_scores, anomaly_scores)

# AUROC
auc = sel.auroc(normal_scores, anomaly_scores)

# Full threshold sweep with metrics at each step
report = sel.sweep(normal_scores, anomaly_scores, steps=50)
```

## Configuration

YAML/JSON config file:

```yaml
backbone: resnet18           # resnet18 | resnet50 | wide_resnet50_2
imgsz: 224
scoring_method: mahalanobis  # mahalanobis | knn | combined
knn_k: 5
anomaly_threshold: 3.0
auto_threshold: true
auto_threshold_percentile: 95.0
patch_size: 64               # heatmap patch size
patch_stride: 32             # heatmap stride
heatmap_alpha: 0.4
output_dir: output
model_save_path: runs/anomaly_model.npz
```

## Project Structure

```
Visual Anomaly Detector/
└── Source Code/
    ├── config.py              # AnomalyConfig dataclass + loader
    ├── feature_extractor.py   # ResNet feature extraction (batch support)
    ├── scorer.py              # Mahalanobis + k-NN scoring + NPZ persistence
    ├── threshold.py           # Threshold selection: percentile, F1, AUROC, sweep
    ├── heatmap.py             # Patch-level anomaly heatmap generator
    ├── visualize.py           # Overlay renderer: labels, borders, score bars
    ├── export.py              # JSON/CSV result export
    ├── data_bootstrap.py      # MVTec-AD dataset download (Hugging Face)
    ├── train.py               # Training pipeline + evaluation
    ├── infer.py               # CLI: image + folder inference
    ├── modern.py              # CVProject adapter (@register)
    ├── requirements.txt
    └── README.md
```

## Requirements

- Python 3.10+
- torch >= 2.0
- torchvision >= 0.15
- opencv-python >= 4.8
- numpy >= 1.24

## Tech Stack

| Component | Technology |
|---|---|
| Feature Extraction | ResNet18/50 (pretrained ImageNet, fc removed) |
| Anomaly Scoring | Mahalanobis distance + k-NN |
| Threshold Selection | Percentile / F1-optimal / mean+σ |
| Heatmap | Patch-level sliding window |
| Model Format | NumPy NPZ (compressed) |
| Dataset | MVTec-AD (Hugging Face) |
