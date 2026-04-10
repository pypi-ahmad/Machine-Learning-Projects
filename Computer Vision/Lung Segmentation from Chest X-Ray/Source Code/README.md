# Lung Segmentation From Chest X-Ray

> **Task:** Segmentation &nbsp;|&nbsp; **Key:** `lung_segmentation` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-seg

---

## Overview

Pixel-wise segmentation of lung regions from chest X-ray images for medical imaging analysis.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Segmentation (`seg`) |
| **Legacy Stack** | U-Net (notebook) |
| **Modern Stack** | Ultralytics YOLO26-seg |
| **Dataset** | Kaggle: chest-xray-masks-and-labels |
| **Key Metrics** | mIoU, Dice |
| **Download** | zip (enabled: yes) |

## Project Structure

```
Lung Segmentation From Chest X-Ray/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("lung_segmentation")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("lung_segmentation", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("lung_segmentation", "seg")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-seg.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Lung Segmentation From Chest X-Ray/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_segmentation.train_segmentation()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_segmentation import train_segmentation
train_segmentation(data_yaml="data/lung_segmentation/data.yaml", model="yolo26n-seg.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/lung_segmentation.yaml`

```bash
python -m utils.data_downloader lung_segmentation       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project lung_segmentation
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/lung_segmentation.yaml)
