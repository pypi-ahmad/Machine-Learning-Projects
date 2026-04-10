# Cell Nuclei Segmentation

> **Task:** Segmentation &nbsp;|&nbsp; **Key:** `cell_nuclei_segmentation` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-seg

---

## Overview

Instance segmentation of cell nuclei in microscopy images (Data Science Bowl 2018).

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Segmentation (`seg`) |
| **Legacy Stack** | U-Net (notebook) |
| **Modern Stack** | Ultralytics YOLO26-seg |
| **Dataset** | Kaggle: data-science-bowl-2018 |
| **Key Metrics** | mIoU, Dice |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Cell Nuclei Segmentation/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("cell_nuclei_segmentation")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("cell_nuclei_segmentation", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("cell_nuclei_segmentation", "seg")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-seg.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Cell Nuclei Segmentation/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_segmentation.train_segmentation()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_segmentation import train_segmentation
train_segmentation(data_yaml="data/cell_nuclei_segmentation/data.yaml", model="yolo26n-seg.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/cell_nuclei_segmentation.yaml`

> **Manual download required.** Visit [https://bbbc.broadinstitute.org/BBBC038](https://bbbc.broadinstitute.org/BBBC038), then place files into `data/cell_nuclei_segmentation/`.

```bash
python -m utils.data_downloader cell_nuclei_segmentation       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project cell_nuclei_segmentation
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/cell_nuclei_segmentation.yaml)
