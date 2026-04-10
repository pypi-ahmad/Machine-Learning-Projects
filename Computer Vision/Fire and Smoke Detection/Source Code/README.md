# Fire and Smoke Detection

> **Task:** Detection &nbsp;|&nbsp; **Key:** `fire_smoke_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO (custom weights)

---

## Overview

Detects fire and smoke in images/video. Resolves custom-trained `best.pt` via the model registry, falling back to Ultralytics pretrained weights.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | YOLOv5 (subprocess) |
| **Modern Stack** | Ultralytics YOLO (custom weights) |
| **Dataset** | D-Fire Dataset (GitHub) |
| **Key Metrics** | mAP50, mAP50-95 |
| **Download** | git (enabled: yes) |

## Project Structure

```
Fire and Smoke Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("fire_smoke_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("fire_smoke_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("fire_smoke_detection", "detect")` → `load_yolo(weights)`. Falls back to legacy `best.pt` from `models/fire_and_smoke_detection/` if registry has no custom weights, then to Ultralytics pretrained `yolo26n.pt`. Inference: `self.model(input_data, verbose=False)`. Visualize with `output[0].plot()`.

### Training

```bash
cd "Fire and Smoke Detection/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_detection.train_detection()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_detection import train_detection
train_detection(data_yaml="data/fire_smoke_detection/data.yaml", model="yolo26n.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/fire_smoke_detection.yaml`

```bash
python -m utils.data_downloader fire_smoke_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project fire_smoke_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/fire_smoke_detection.yaml)
