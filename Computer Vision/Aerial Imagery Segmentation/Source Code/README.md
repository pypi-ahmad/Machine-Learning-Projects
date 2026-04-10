# Aerial Imagery Segmentation

> **Task:** Segmentation &nbsp;|&nbsp; **Key:** `aerial_imagery_segmentation` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-seg

---

## Overview

Semantic segmentation of aerial/satellite imagery into land-use classes. Legacy code used a custom U-Net notebook.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Segmentation (`seg`) |
| **Legacy Stack** | Custom U-Net (notebook) |
| **Modern Stack** | Ultralytics YOLO26-seg |
| **Dataset** | Kaggle: semantic-segmentation-of-aerial-imagery |
| **Key Metrics** | mIoU, mAP50 |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Aerial Imagery Segmentation/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("aerial_imagery_segmentation")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("aerial_imagery_segmentation", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("aerial_imagery_segmentation", "seg")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-seg.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Aerial Imagery Segmentation/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_segmentation.train_segmentation()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_segmentation import train_segmentation
train_segmentation(data_yaml="data/aerial_imagery_segmentation/data.yaml", model="yolo26n-seg.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/aerial_imagery_segmentation.yaml`

> **Manual download required.** Visit [https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx), then place files into `data/aerial_imagery_segmentation/`.

```bash
python -m utils.data_downloader aerial_imagery_segmentation       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project aerial_imagery_segmentation
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/aerial_imagery_segmentation.yaml)
