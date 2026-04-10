# Aerial Cactus Identification

> **Task:** Classification &nbsp;|&nbsp; **Key:** `aerial_cactus_identification` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Binary classification of aerial imagery patches to identify columnar cacti. Originally a Kaggle competition notebook using a custom Keras CNN.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Custom CNN (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | Kaggle: aerial-cactus-identification |
| **Key Metrics** | accuracy, F1 |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Aerial Cactus Identification/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("aerial_cactus_identification")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("aerial_cactus_identification", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("aerial_cactus_identification", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Aerial Cactus Identification/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/aerial_cactus_identification", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/aerial_cactus_identification.yaml`

```bash
python -m utils.data_downloader aerial_cactus_identification       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project aerial_cactus_identification
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/aerial_cactus_identification.yaml)
