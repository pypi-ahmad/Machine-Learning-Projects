# Pedestrian Detection

> **Task:** Detection &nbsp;|&nbsp; **Key:** `pedestrian_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26 (person detection, classes=[0])

---

## Overview

Detects pedestrians using YOLO26 (COCO class 0 = person). Legacy code used OpenCV HOG + SVM descriptor.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | OpenCV HOG + SVM |
| **Modern Stack** | Ultralytics YOLO26 |
| **Dataset** | Penn-Fudan / COCO person subset (manual download) |
| **Key Metrics** | mAP50, mAP50-95 |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Pedestrian Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("pedestrian_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("pedestrian_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("pedestrian_detection", "detect")` → `load_yolo(weights)`.  Default pretrained: `yolo26n.pt`.  Inference: `self.model(input_data, classes=[0], verbose=False)` (COCO class 0 = person).  Visualize with `output[0].plot()`.

### Training

```bash
cd "Pedestrian Detection/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_detection.train_detection()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_detection import train_detection
train_detection(data_yaml="data/pedestrian_detection/data.yaml", model="yolo26n.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/pedestrian_detection.yaml`

> **Manual download required.** Visit [https://data.caltech.edu/records/f6rph-90m20](https://data.caltech.edu/records/f6rph-90m20), then place files into `data/pedestrian_detection/`.

```bash
python -m utils.data_downloader pedestrian_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project pedestrian_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/pedestrian_detection.yaml)
