# Emotion Recognition from Facial Expression

> **Task:** Detection &nbsp;|&nbsp; **Key:** `emotion_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26 (person detection, classes=[0])

---

## Overview

Detects persons via YOLO26 (COCO class 0). Legacy code used Haar cascades for face detection + Keras CNN for 7-class emotion classification.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | Haar cascade + Keras CNN |
| **Modern Stack** | Ultralytics YOLO26 |
| **Dataset** | Kaggle: fer2013 (FER-2013) |
| **Key Metrics** | mAP50 |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Emotion Recognition from facial expression/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("emotion_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("emotion_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("emotion_recognition", "detect")` → `load_yolo(weights)`.  Default pretrained: `yolo26n.pt`.  Inference: `self.model(input_data, classes=[0], verbose=False)` (COCO class 0 = person).  Visualize with `output[0].plot()`.

### Training

```bash
cd "Emotion Recognition from facial expression/Source Code"
python train.py --epochs 25 --batch 32
```

Delegates to `train.train_classification.train_classification()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/emotion_recognition", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/emotion_recognition.yaml`

```bash
python -m utils.data_downloader emotion_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project emotion_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/emotion_recognition.yaml)
