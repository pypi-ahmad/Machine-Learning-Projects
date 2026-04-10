# Wildlife Image Classification

> **Task:** Classification &nbsp;|&nbsp; **Key:** `wildlife_classification` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Classifies wildlife species from camera-trap and nature images. Legacy code used a custom CNN notebook. Modern wrapper uses YOLO26-cls.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (cls) |
| **Legacy Stack** | Custom CNN (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | iNaturalist 2021 (manual download) |
| **Key Metrics** | accuracy, top-5 accuracy |

## Project Structure

```
Wildlife Image Classification/
└── wildlife image classification/
    ├── modern.py                  # Unified YOLO26-cls inference wrapper
    ├── train.py                   # Per-project training entry point
    ├── Wildlife detection.ipynb   # Original notebook
    └── How to run.txt             # Original instructions
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("wildlife_classification", "path/to/image.jpg")
```

### Training

```bash
cd "Wildlife Image Classification/wildlife image classification"
python train.py --epochs 25
```

Or from the repo root:

```python
from train.train_classification import train_classification
train_classification(data_dir="data/wildlife_classification", epochs=25)
```

### Dataset

The dataset configuration is in `configs/datasets/wildlife_classification.yaml`.

```bash
# Download dataset (if configured)
python -m utils.data_downloader wildlife_classification
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project wildlife_classification
```

## Configuration

Dataset config: [`configs/datasets/wildlife_classification.yaml`](../../configs/datasets/wildlife_classification.yaml)

## Links

- [Root README](../../README.md)
- [Dataset Configs](../../configs/datasets/README.md)
- [Training Guide](../../train/)
