# Wildlife Image Classification

> **Task:** Classification &nbsp;|&nbsp; **Key:** `wildlife_classification` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Classifies wildlife species from camera-trap and nature images using transfer learning.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Custom CNN (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | Kaggle: animals-10 / iNaturalist subset (manual download) |
| **Key Metrics** | accuracy, top-5 accuracy |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Wildlife Image Classification/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("wildlife_classification")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("wildlife_classification", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("wildlife_classification", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Wildlife Image Classification/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/wildlife_classification", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/wildlife_classification.yaml`

> **Manual download required.** Visit [https://www.tensorflow.org/datasets/catalog/i_naturalist2021](https://www.tensorflow.org/datasets/catalog/i_naturalist2021), then place files into `data/wildlife_classification/`.

```bash
python -m utils.data_downloader wildlife_classification       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project wildlife_classification
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/wildlife_classification.yaml)
