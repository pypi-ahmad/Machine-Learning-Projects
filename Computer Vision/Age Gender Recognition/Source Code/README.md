# Age & Gender Recognition

> **Task:** Detection &nbsp;|&nbsp; **Key:** `age_gender_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26 (person detection, classes=[0])

---

## Overview

Detects persons using YOLO26 (COCO class 0), originally paired with Caffe age/gender classifiers via OpenCV DNN.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | Caffe DNN face detector + Caffe age/gender nets |
| **Modern Stack** | Ultralytics YOLO26 |
| **Dataset** | UTKFace / Adience (manual download) |
| **Key Metrics** | mAP50 |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Age Gender Recognition/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("age_gender_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("age_gender_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("age_gender_recognition", "detect")` → `load_yolo(weights)`.  Default pretrained: `yolo26n.pt`.  Inference: `self.model(input_data, classes=[0], verbose=False)` (COCO class 0 = person).  Visualize with `output[0].plot()`.

### Training

```bash
cd "Age Gender Recognition/Source Code"
python train.py --epochs 25 --batch 32
```

Delegates to `train.train_classification.train_classification()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/age_gender_recognition", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/age_gender_recognition.yaml`

> **Manual download required.** Visit [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/), then place files into `data/age_gender_recognition/`.

```bash
python -m utils.data_downloader age_gender_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project age_gender_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/age_gender_recognition.yaml)
