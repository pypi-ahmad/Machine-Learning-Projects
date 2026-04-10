# Licence Plate Detector

> **Task:** Detection &nbsp;|&nbsp; **Key:** `licence_plate_detector` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO (custom weights) + EasyOCR

---

## Overview

Detects and reads vehicle licence plates. Resolves custom-trained weights via the model registry + EasyOCR for text extraction. Legacy code used YOLOv5 subprocess + EasyOCR.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) |
| **Legacy Stack** | YOLOv5 (subprocess) + EasyOCR |
| **Modern Stack** | Ultralytics YOLO (custom weights) + EasyOCR |
| **Dataset** | Custom ANPR dataset (manual download) |
| **Key Metrics** | mAP50, plate read accuracy |
| **Download** | git (enabled: no) |

## Project Structure

```
Licence Plate Detector/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("licence_plate_detector")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("licence_plate_detector", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("licence_plate_detector", "detect")` → `load_yolo(weights)`. Falls back to legacy `best.pt` from `models/licence_plate_detector/`, then to Ultralytics pretrained `yolo26n.pt`. Inference: `self.model(input_data, verbose=False)`. Visualize with `output[0].plot()`.

### Training

```bash
cd "Licence Plate Detector/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_detection.train_detection()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_detection import train_detection
train_detection(data_yaml="data/licence_plate_detector/data.yaml", model="yolo26n.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/licence_plate_detector.yaml`

> Automatic download is **disabled** for this project.

```bash
python -m utils.data_downloader licence_plate_detector       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project licence_plate_detector
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/licence_plate_detector.yaml)
