# Sign Language Recognition

> **Task:** Pose Estimation &nbsp;|&nbsp; **Key:** `sign_language_recognition` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-pose

---

## Overview

Pose estimation for sign-language gesture recognition. Legacy code used MediaPipe Hands + Keras LSTM.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Pose Estimation (`pose`) |
| **Legacy Stack** | MediaPipe Hands + Keras LSTM |
| **Modern Stack** | Ultralytics YOLO26-pose |
| **Dataset** | Kaggle: sign-language-mnist / ASL |
| **Key Metrics** | accuracy, per-gesture F1 |
| **Download** | kaggle (enabled: yes) |

## Project Structure

```
Sign Language Recognition/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("sign_language_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("sign_language_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("sign_language_recognition", "pose")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-pose.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Sign Language Recognition/Souce Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_pose.train_pose()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_pose import train_pose
train_pose(data_yaml="data/sign_language_recognition/data.yaml", model="yolo26n-pose.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/sign_language_recognition.yaml`

```bash
python -m utils.data_downloader sign_language_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project sign_language_recognition
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/sign_language_recognition.yaml)
