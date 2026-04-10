# Food Image Recognition & Calories Estimation

> **Task:** Classification &nbsp;|&nbsp; **Key:** `food_image_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Classifies food images into categories with optional calorie estimation. Legacy code used InceptionV3/EfficientNet with a Flask web interface.

> **Status:** `needs_fix` — original dependencies are broken; modern wrapper works with pretrained YOLO26 weights.


## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | InceptionV3/EfficientNet (Flask) |
| **Modern Stack** | Ultralytics YOLO26-cls (placeholder) |
| **Dataset** | Kaggle: food-101 (ETH Food-101) |
| **Key Metrics** | accuracy, top-5 accuracy |
| **Download** | tar (enabled: yes) |

## Project Structure

```
Food Image Recognition & Calories Estimation/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("food_image_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("food_image_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("food_image_recognition", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Food Image Recognition & Calories Estimation/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/food_image_recognition", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/food_image_recognition.yaml`

```bash
python -m utils.data_downloader food_image_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project food_image_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/food_image_recognition.yaml)
