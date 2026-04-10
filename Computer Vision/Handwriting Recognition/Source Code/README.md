# Handwriting Recognition

> **Task:** Classification &nbsp;|&nbsp; **Key:** `handwriting_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Classifies handwritten characters/digits. Legacy code used a custom TF/Keras HTR CNN+RNN. Modern wrapper uses YOLO26-cls.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Custom TF/Keras HTR CNN+RNN |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | EMNIST (manual download) |
| **Key Metrics** | accuracy, CER |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Handwriting Recognition/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("handwriting_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("handwriting_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("handwriting_recognition", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Handwriting Recognition/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/handwriting_recognition", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/handwriting_recognition.yaml`

> **Manual download required.** Visit [https://www.nist.gov/itl/products-and-services/emnist-dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), then place files into `data/handwriting_recognition/`.

```bash
python -m utils.data_downloader handwriting_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project handwriting_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/handwriting_recognition.yaml)
