# Plant Disease Prediction

> **Task:** Classification &nbsp;|&nbsp; **Key:** `plant_disease_prediction` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls (placeholder)

---

## Overview

Classifies plant leaf images to identify disease types (38 classes) from the PlantVillage dataset. Original notebook has missing data/dependencies.

> **Status:** `broken` — original notebook has missing data/dependencies; modern wrapper is a placeholder.


## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Custom CNN (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls (placeholder) |
| **Dataset** | Kaggle: plantvillage-dataset |
| **Key Metrics** | accuracy, per-class F1 |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Plant Disease Predicton/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("plant_disease_prediction")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("plant_disease_prediction", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("plant_disease_prediction", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Plant Disease Predicton/Souce Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/plant_disease_prediction", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/plant_disease_prediction.yaml`

```bash
python -m utils.data_downloader plant_disease_prediction       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project plant_disease_prediction
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/plant_disease_prediction.yaml)
