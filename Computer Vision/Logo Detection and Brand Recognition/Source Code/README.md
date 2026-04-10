# Logo Detection and Brand Recognition

> **Task:** Classification &nbsp;|&nbsp; **Key:** `logo_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Classifies brand logos from images. Legacy code used MobileNetV2 in a notebook.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | MobileNetV2 (notebook) |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | FlickrLogos-32 / custom (manual download) |
| **Key Metrics** | accuracy |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Logo Detection and Brand Recognition/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("logo_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("logo_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("logo_detection", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Logo Detection and Brand Recognition/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/logo_detection", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/logo_detection.yaml`

> **Manual download required.** Visit [https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/dataSets/flickrlogos/](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/dataSets/flickrlogos/), then place files into `data/logo_detection/`.

```bash
python -m utils.data_downloader logo_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project logo_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/logo_detection.yaml)
