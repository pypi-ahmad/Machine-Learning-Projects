# Real-time Object Tracking

> **Task:** Tracking &nbsp;|&nbsp; **Key:** `realtime_object_tracking` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO (custom weights)

---

## Overview

Real-time multi-object tracking in video streams. Uses `model.track(data, persist=True)` with a fallback to plain detection if the `lap` package is unavailable. Legacy code used YOLOv5 subprocess + Flask.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Tracking (`tracking`) |
| **Legacy Stack** | YOLOv5 (subprocess) + Flask |
| **Modern Stack** | Ultralytics YOLO (custom weights) |
| **Dataset** | MOT17 (manual download from motchallenge.net) |
| **Key Metrics** | mAP50, MOTA, FPS |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Real-time Object Tracking/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("realtime_object_tracking")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("realtime_object_tracking", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("realtime_object_tracking", "detect")` → `load_yolo(weights)`. Falls back to legacy `best.pt`, then to Ultralytics pretrained. Inference: `self.model.track(input_data, persist=True, verbose=False)` for frame-persistent multi-object tracking. Falls back to plain `self.model(input_data, verbose=False)` if tracking deps (`lap`) are unavailable. Visualize with `output[0].plot()`.

### Training

```bash
cd "Real-time Object Tracking/Souce Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_detection.train_detection()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_detection import train_detection
train_detection(data_yaml="data/realtime_object_tracking/data.yaml", model="yolo26n.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/realtime_object_tracking.yaml`

> **Manual download required.** Visit [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/), then place files into `data/realtime_object_tracking/`.

```bash
python -m utils.data_downloader realtime_object_tracking       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project realtime_object_tracking
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/realtime_object_tracking.yaml)
