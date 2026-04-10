# Skin Cancer Detection

> **Task:** Classification &nbsp;|&nbsp; **Key:** `skin_cancer_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls (placeholder)

---

## Overview

7-class classification of skin lesion images from the HAM10000 dataset. Original notebook has missing dependencies.

> **Status:** `needs_fix` — original dependencies are broken; modern wrapper works with pretrained YOLO26 weights.


## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Custom CNN (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls (placeholder) |
| **Dataset** | Kaggle: HAM10000 (skin-cancer-mnist-ham10000) |
| **Key Metrics** | accuracy, balanced accuracy |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Skin Cancer Detection/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("skin_cancer_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("skin_cancer_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("skin_cancer_detection", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Skin Cancer Detection/Souce Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/skin_cancer_detection", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/skin_cancer_detection.yaml`

```bash
python -m utils.data_downloader skin_cancer_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project skin_cancer_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/skin_cancer_detection.yaml)
