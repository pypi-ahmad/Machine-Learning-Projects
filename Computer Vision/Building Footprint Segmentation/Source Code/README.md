# Building Footprint Segmentation

> **Task:** Segmentation &nbsp;|&nbsp; **Key:** `building_footprint_segmentation` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-seg

---

## Overview

Pixel-level segmentation of building footprints from aerial imagery (Massachusetts Buildings Dataset).

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Segmentation (`seg`) |
| **Legacy Stack** | Custom segmentation (notebook) |
| **Modern Stack** | Ultralytics YOLO26-seg |
| **Dataset** | Kaggle: massachusetts-buildings-dataset |
| **Key Metrics** | mIoU, Dice |
| **Download** | zip (enabled: no) |

## Project Structure

```
Building Footprint Segmentation/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("building_footprint_segmentation")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("building_footprint_segmentation", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("building_footprint_segmentation", "seg")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-seg.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Building Footprint Segmentation/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_segmentation.train_segmentation()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_segmentation import train_segmentation
train_segmentation(data_yaml="data/building_footprint_segmentation/data.yaml", model="yolo26n-seg.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/building_footprint_segmentation.yaml`

> Automatic download is **disabled** for this project.

```bash
python -m utils.data_downloader building_footprint_segmentation       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project building_footprint_segmentation
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/building_footprint_segmentation.yaml)
