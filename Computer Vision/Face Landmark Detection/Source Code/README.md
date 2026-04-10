# Face Landmark Detection

> **Task:** Pose Estimation &nbsp;|&nbsp; **Key:** `face_landmark_detection` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-pose

---

## Overview

Detects facial keypoints (eyes, nose, mouth, jawline). Legacy code used dlib HOG + `shape_predictor_68_face_landmarks.dat`.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Pose Estimation (`pose`) |
| **Legacy Stack** | dlib HOG + shape_predictor_68_face_landmarks |
| **Modern Stack** | Ultralytics YOLO26-pose |
| **Dataset** | iBUG 300-W / dlib shape predictor |
| **Key Metrics** | NME (Normalised Mean Error) |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Face Landmark Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("face_landmark_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("face_landmark_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("face_landmark_detection", "pose")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-pose.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Face Landmark Detection/Source Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_pose.train_pose()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_pose import train_pose
train_pose(data_yaml="data/face_landmark_detection/data.yaml", model="yolo26n-pose.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/face_landmark_detection.yaml`

> **Manual download required.** Visit [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), then place files into `data/face_landmark_detection/`.

```bash
python -m utils.data_downloader face_landmark_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project face_landmark_detection
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/face_landmark_detection.yaml)
