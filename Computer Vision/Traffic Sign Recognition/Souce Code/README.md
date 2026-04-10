# Traffic Sign Recognition

> **Task:** Classification &nbsp;|&nbsp; **Key:** `traffic_sign_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls (placeholder)

---

## Overview

Multi-class classification of traffic signs (43 classes, GTSRB). Original Flask app has dependency issues.

> **Status:** `needs_fix` — original dependencies are broken; modern wrapper works with pretrained YOLO26 weights.


## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Keras CNN + Flask |
| **Modern Stack** | Ultralytics YOLO26-cls (placeholder) |
| **Dataset** | Kaggle: GTSRB (German Traffic Sign) |
| **Key Metrics** | accuracy, per-class F1 |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Traffic Sign Recognition/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("traffic_sign_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("traffic_sign_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("traffic_sign_recognition", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Traffic Sign Recognition/Souce Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/traffic_sign_recognition", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/traffic_sign_recognition.yaml`

> **Manual download required.** Visit [https://benchmark.ini.rub.de/gtsrb_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html), then place files into `data/traffic_sign_recognition/`.

```bash
python -m utils.data_downloader traffic_sign_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project traffic_sign_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/traffic_sign_recognition.yaml)
