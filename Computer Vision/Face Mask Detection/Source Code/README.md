# Face Mask Detection

> **Task:** Detection &nbsp;|&nbsp; **Key:** `face_mask_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26 (person detection, classes=[0])

---

## Overview

Detects persons wearing / not wearing face masks via YOLO26 (COCO class 0). Legacy code used a Keras/TF CNN notebook.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | Keras/TF CNN (notebook) |
| **Modern Stack** | Ultralytics YOLO26 |
| **Dataset** | Kaggle: face-mask-detection (Prajnasb) |
| **Key Metrics** | mAP50, mAP50-95 |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Face Mask Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("face_mask_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("face_mask_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("face_mask_detection", "detect")` → `load_yolo(weights)`.  Default pretrained: `yolo26n.pt`.  Inference: `self.model(input_data, classes=[0], verbose=False)` (COCO class 0 = person).  Visualize with `output[0].plot()`.

### Training

```bash
cd "Face Mask Detection/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_detection.train_detection()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_detection import train_detection
train_detection(data_yaml="data/face_mask_detection/data.yaml", model="yolo26n.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/face_mask_detection.yaml`

```bash
python -m utils.data_downloader face_mask_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project face_mask_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/face_mask_detection.yaml)
