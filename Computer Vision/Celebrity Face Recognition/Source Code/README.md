# Celebrity Face Recognition

> **Task:** Classification &nbsp;|&nbsp; **Key:** `celebrity_face_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-cls

---

## Overview

Multi-class recognition of celebrity faces (105 classes). Originally used a Keras CNN with hardcoded labels and custom `.h5` weights.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Classification (`cls`) |
| **Legacy Stack** | Keras CNN + hardcoded 105-class labels |
| **Modern Stack** | Ultralytics YOLO26-cls |
| **Dataset** | Kaggle: pins-face-recognition |
| **Key Metrics** | accuracy, top-5 accuracy |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Celebrity Face Recognition/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("celebrity_face_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("celebrity_face_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("celebrity_face_recognition", "cls")` to find the best available weights (custom-trained first, then Ultralytics `yolo26n-cls.pt` default). It then loads the model via `load_yolo(weights)` and runs `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Celebrity Face Recognition/Source Code"
python train.py --epochs 25 --model resnet18
```

Delegates to `train.train_classification.train_classification()` (torchvision transfer learning).
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_classification import train_classification
train_classification(data_dir="data/celebrity_face_recognition", model_name="resnet18", epochs=25)
```

### Dataset

Config: `configs/datasets/celebrity_face_recognition.yaml`

> **Manual download required.** Visit [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), then place files into `data/celebrity_face_recognition/`.

```bash
python -m utils.data_downloader celebrity_face_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project celebrity_face_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/celebrity_face_recognition.yaml)
