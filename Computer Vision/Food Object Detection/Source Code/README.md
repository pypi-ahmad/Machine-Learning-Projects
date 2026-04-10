# Food Object Detection

> **Task:** Classification &nbsp;|&nbsp; **Key:** `food_object_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Classifies food items from images. Legacy code used InceptionV3 (Keras) with a Streamlit frontend. Modern wrapper uses YOLO26-cls.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | InceptionV3 (Keras) + Streamlit |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | UECFOOD-256 (manual download) |
| **Key Metrics** | accuracy |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Food Object Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("food_object_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("food_object_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("food_object_detection", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Food Object Detection/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/food_object_detection", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/food_object_detection.yaml`

> **Manual download required.** Visit [https://mm.cs.uec.ac.jp/uecfood256.html](https://mm.cs.uec.ac.jp/uecfood256.html), then place files into `data/food_object_detection/`.

```bash
python -m utils.data_downloader food_object_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project food_object_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/food_object_detection.yaml)
