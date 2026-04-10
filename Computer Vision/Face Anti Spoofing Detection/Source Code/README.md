# Face Anti-Spoofing Detection

> **Task:** Detection &nbsp;|&nbsp; **Key:** `face_anti_spoofing` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26 (person detection, classes=[0])

---

## Overview

Detects persons via YOLO26 (COCO class 0). Legacy code used Haar cascades + Keras CNN for liveness detection (real vs. spoof).

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | Haar cascade + Keras anti-spoofing CNN |
| **Modern Stack** | Ultralytics YOLO26 |
| **Dataset** | CelebA-Spoof / NUAA (manual download) |
| **Key Metrics** | mAP50 |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Face Anti Spoofing Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("face_anti_spoofing")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("face_anti_spoofing", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("face_anti_spoofing", "detect")` → `load_yolo(weights)`.  Default pretrained: `yolo26n.pt`.  Inference: `self.model(input_data, classes=[0], verbose=False)` (COCO class 0 = person).  Visualize with `output[0].plot()`.

### Training

```bash
cd "Face Anti Spoofing Detection/Source Code"
python train.py --epochs 25 --batch 32
```

Delegates to `train.train_classification.train_classification()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/face_anti_spoofing", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/face_anti_spoofing.yaml`

> **Manual download required.** Visit [http://parnec.nuaa.edu.cn/xtan/data/NUAAImposterDB.html](http://parnec.nuaa.edu.cn/xtan/data/NUAAImposterDB.html), then place files into `data/face_anti_spoofing/`.

```bash
python -m utils.data_downloader face_anti_spoofing       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project face_anti_spoofing
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/face_anti_spoofing.yaml)
