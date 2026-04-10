# Brain Tumour Detection

> **Task:** Classification &nbsp;|&nbsp; **Key:** `brain_tumour_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Binary classification of brain MRI scans to detect presence of tumours. Includes a data augmentation pipeline in the legacy notebook.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Custom CNN + data augmentation (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | Kaggle: brain-mri-images-for-brain-tumor-detection |
| **Key Metrics** | accuracy, F1 |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Brain Tumour Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("brain_tumour_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("brain_tumour_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("brain_tumour_detection", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Brain Tumour Detection/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/brain_tumour_detection", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/brain_tumour_detection.yaml`

```bash
python -m utils.data_downloader brain_tumour_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project brain_tumour_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/brain_tumour_detection.yaml)
